"""第5回の宿題。Geminiの配布索引とmultilingual-e5-smallの実験用索引を比較する。"""

import argparse
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import onnxruntime as ort
import sqlite_vec
from tokenizers import Tokenizer

from app import retrieval, tools
from app.agent import agent, build_model, print_trace
from app.config import DISTRIBUTED_RAG_DB, MODEL
from scripts.build_rag_index import embedding_text


QUESTION = "第2回の説明では、LoRAは何を固定し何を学習しますか？ QLoRAは何を加えますか？"
SEARCH_QUERY = "LoRA QLoRA 違い"


class E5Embedder:
    def __init__(self, model_dir: Path) -> None:
        ort.disable_telemetry_events()
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("<pad>"), pad_token="<pad>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer.encode_batch(texts)
        mask = np.array([item.attention_mask for item in tokens], dtype=np.int64)
        inputs = {
            "input_ids": np.array([item.ids for item in tokens], dtype=np.int64),
            "attention_mask": mask,
            "token_type_ids": np.array([item.type_ids for item in tokens], dtype=np.int64),
        }
        feed = {item.name: inputs[item.name] for item in self.session.get_inputs()}
        hidden, = self.session.run(["last_hidden_state"], feed)
        # モデルカードの手順どおり、パディングを除いた平均をL2正規化する。
        weights = mask[..., None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / weights.sum(axis=1)
        return pooled / np.linalg.norm(pooled, axis=1, keepdims=True)

    def embed_query(self, text: str, _task_type: str) -> list[float]:
        return self.encode([f"query: {text}"])[0].tolist()


def make_e5_index(index_path: Path, e5: E5Embedder) -> None:
    if index_path.exists():
        raise FileExistsError(f"実験用DBは新しいパスを指定してください: {index_path}")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pages = retrieval.load_pages()
    passages = [f"passage: {embedding_text(page)}" for page in pages]
    with TemporaryDirectory(prefix=".build-e5-", dir=index_path.parent) as directory:
        staged = Path(directory) / "index.sqlite3"
        with retrieval.open_index(DISTRIBUTED_RAG_DB) as source, closing(sqlite3.connect(staged)) as index, index:
            page_ids = {row[0] for row in source.execute("SELECT page_id FROM chunks")}
            if page_ids != {page["page_id"] for page in pages}:
                raise ValueError("配布索引と教材JSONのページ集合が一致しません。")
            # ページ情報と本文をそのままコピーし、埋め込み列だけを書き換える。
            source.backup(index)
            for start in range(0, len(pages), 12):
                vectors = e5.encode(passages[start:start + 12])
                if vectors.shape[1] != 384 or not np.isfinite(vectors).all():
                    raise ValueError("実験用索引のベクトルが有効な384次元ではありません。")
                index.executemany(
                    "UPDATE chunks SET embedding = ? WHERE page_id = ?",
                    (
                        (sqlite_vec.serialize_float32(vector.tolist()), page["page_id"])
                        for vector, page in zip(vectors, pages[start:start + len(vectors)], strict=True)
                    ),
                )
        # 完成したファイルだけを公開する。既存DBがあれば上書きせずに失敗する。
        index_path.hardlink_to(staged)


def run_trial(label: str, answer_model, index_path: Path, embedder) -> None:
    def search_materials(query: str) -> str:
        # モデルが提案した検索語をログに残し、順位比較には共通の検索語を使う。
        hits = retrieval.search(SEARCH_QUERY, index_path=index_path, embedder=embedder)
        print(f"[query] proposed={query!r} used={SEARCH_QUERY!r}")
        print("[pages] " + " / ".join(hit["page_id"] for hit in hits))
        return json.dumps({"results": hits}, ensure_ascii=False)

    search_materials.__doc__ = tools.search_materials.__doc__
    print(f"\n=== {label} ===", flush=True)
    start = perf_counter()
    with agent.override(tools=[search_materials]):
        result = agent.run_sync(QUESTION, model=answer_model)
    elapsed = perf_counter() - start
    print(result.output)
    print_trace(result, elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e5-dir", type=Path, required=True, help="E5のmodel.onnxとtokenizer.jsonの場所")
    parser.add_argument("--search-only", metavar="QUERY", help="指定した検索語でページ順位だけを比べる")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="aie05-e5-") as directory:
        e5 = E5Embedder(args.e5_dir)
        experiment = Path(os.environ["AIE_RAG_DB"]) if "AIE_RAG_DB" in os.environ else Path(directory) / "index.sqlite3"
        make_e5_index(experiment, e5)
        if args.search_only is not None:
            for label, path, embedder in (
                ("gemini-embedding-001", DISTRIBUTED_RAG_DB, None),
                ("multilingual-e5-small", experiment, e5.embed_query),
            ):
                hits = retrieval.search(args.search_only, index_path=path, embedder=embedder)
                print(f"[{label}] " + " / ".join(hit["page_id"] for hit in hits))
            return
        answer_model = build_model()
        print(f"[answer model] {MODEL}")
        run_trial("gemini-embedding-001", answer_model, DISTRIBUTED_RAG_DB, None)
        run_trial("multilingual-e5-small", answer_model, experiment, e5.embed_query)


if __name__ == "__main__":
    main()
