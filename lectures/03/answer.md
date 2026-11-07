# 第3回 解答

## 演習

- 第1回（素のLLM）とエージェントの回答の違い：同じモデル（`gemini-3.5-flash-lite`）に「AIE2026の講座の宿題にある『3レベル』とは何ですか？」と質問した。`app.ask` は一般的な講座を想定して「基礎・入門編」「応用・実践編」「発展・挑戦編」と答えた。`app.agent` は用語集を検索し、講座での区分は「基礎」「実装」「発展」で、どれか1つに取り組めばその回の宿題が完了すると答えた。後者は用語集の `homework-levels` の定義と一致した。
- `--trace` で見えたツール呼び出し（ツール名と引数）：3レベルの質問では `search_glossary({'query': '3レベル'})`。「LoRAとQLoRAの違いは？」では `search_glossary({'query': 'LoRA'})` と `search_glossary({'query': 'QLoRA'})` が呼ばれた。「第2回の内容から確認問題を1問出して」では `make_quiz({'lecture': 2})` が呼ばれ、「エポック」の問題と答えが返った。`[tool call]` はAIからツールへの依頼、`[tool return]` はツールからAIへの結果の返却を表す。
- 1つの質問での LLM 呼び出し回数（`requests`）：今回の記録では、3レベルの質問は `requests=2, tool_calls=1`、LoRAとQLoRAの質問は `requests=2, tool_calls=2`、確認問題の依頼は `requests=2, tool_calls=1`。LoRAとQLoRAの質問では、モデルが一度に2つの検索を依頼し、両方の結果を受け取って回答した。
- 追加したツールと、docstring に書いた説明：`make_quiz(lecture: int)` を追加し、指定した回の用語を1つ選んで問題と答えを返すようにした。docstringには「復習問題を求められたときに使う」「講義の回を整数で指定する」と記載した。`--list-tools` への表示と、`pytest` の全11件の成功を確認した。

## 宿題

- 取り組んだレベル：基礎 / 実装 / 発展
- やったこと：
- 結果：
- 気づき・疑問：
