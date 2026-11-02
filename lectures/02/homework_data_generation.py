"""宿題で使用したデータを生成する。モデルの学習は行わない。"""
from pathlib import Path
WORK = Path(__file__).resolve().parent / "homework_data"
WORK.mkdir(exist_ok=True)
import json
FAQ = {
'idea':'アイデアがなくても参加できます。アイデアを生み出すためのサポートを受けられます。',
'retry':'審査に落ちた場合も再エントリーできます。再挑戦を推奨しています。',
'documents':'VDSやコストシートの書き方は運営に相談できます。VDSの解説動画もあります。',
'group':'チームで起案する場合、代表者1名の所属部署と氏名を記載し、その他のメンバーは提出するコストシートのアサイニー欄に加えます。',
'delay':'審査通過後すぐに検証できない場合、所属マネージャーと開始時期を調整する、または他のメンバーによる対応が可能な場合などは起案者の関与を軽減して実施する対応を予定しています。'
}
GROUNDED = {
'idea': (['アイデアがなくても参加できますか','考えがまだ固まっていなくても応募できますか','アイデアを考える支援はありますか','何も思いついていない状態でも参加可能ですか'], ['参加','サポート']),
'retry': (['審査に落ちた場合は再挑戦できますか','不採択後にもう一度応募できますか','審査に通らなかった案も再応募できますか','選考で落選したら改めてエントリーできますか'], ['再エントリー']),
'documents': (['VDSの書き方を相談できますか','コストシートの作り方が分かりません','VDSの資料作成を教えてもらえますか','VDSやコスト表の記入方法が分からないときはどうしますか'], ['相談','VDS']),
'group': (['チームで起案するときはどう書きますか','複数人で応募する場合の記入方法は何ですか','チーム応募時の代表者とメンバーの記載方法は','複数メンバーの名前と所属はどこに記載しますか'], ['代表者','アサイニー']),
'delay': (['審査を通ってもすぐ検証できない場合はどうしますか','忙しくて採択後すぐに動けないときの対応は','ピッチ通過後の検証開始を調整できますか','採択後すぐ実証に着手できない場合の選択肢は何ですか'], ['調整','軽減'])}
UNGROUNDED = ['採択されたら必ず専任になりますか','応募できる年齢の上限はありますか','ピッチの持ち時間は何分ですか','応募資格に勤続年数の条件はありますか','応募締切はいつですか','採択後に上司の承認は必須ですか','審査員は誰ですか','検証に使える予算はいくらですか']
TEST_EXTRA_GROUNDED = {
'idea':['事業案がない人もプログラムに参加して支援を受けられますか','テーマが未決定でも企画を考える支援を受けられますか'],
'retry':['審査で不合格になったアイデアは再度提出できますか','落選後の再申請は認められますか'],
'documents':['VDSの作成で困った際に動画や相談先はありますか','事業案のVDSやコストシートの記入を相談できますか'],
'group':['代表者以外の氏名はどの書類のどこに書きますか','共同提案の代表者情報と他の参加者の記載先を教えてください'],
'delay':['審査を通過しても時間が取れない場合、開始時期や関与はどう調整しますか','ピッチ通過後すぐ動けない場合、検証の進め方はどうなりますか']}
TEST_EXTRA_UNGROUNDED = ['選考は年間何回実施しますか','応募後に交通費は支給されますか','応募用資料のページ数に上限はありますか','検証期間は必ず3か月ですか']
FAQ_TEXT = '\n'.join('- '+v for v in FAQ.values())
UNGROUNDED_ANSWER='このFAQには記載がありません。運営に確認が必要です。'
def make_example(question,answer,needs_review):
    return {'question':question,'faq':FAQ_TEXT,'target':json.dumps({'answer':answer,'needs_review':needs_review},ensure_ascii=False),'needs_review':needs_review}
train=[]; valid=[]; test=[]; gold={}
for key,(questions,words) in GROUNDED.items():
    for q in questions[:2]: train.append(make_example(q,FAQ[key],False))
    for q in questions[2:3]: valid.append(make_example(q,FAQ[key],False))
    for q in questions[3:4]+TEST_EXTRA_GROUNDED[key]:
        test.append(make_example(q,FAQ[key],False)); gold[q]={'needs_review':False,'must_include':words}
for i,q in enumerate(UNGROUNDED):
    ex=make_example(q,UNGROUNDED_ANSWER,True)
    if i<4: train.append(ex)
    elif i<6: valid.append(ex)
    else: test.append(ex); gold[q]={'needs_review':True,'must_include':['確認']}
for q in TEST_EXTRA_UNGROUNDED:
    test.append(make_example(q,UNGROUNDED_ANSWER,True)); gold[q]={'needs_review':True,'must_include':['確認']}
assert len({e['question'] for e in train+valid+test}) == len(train+valid+test)
assert len(train)==14 and len(valid)==7 and len(test)==21
(WORK/'test.jsonl').write_text('\n'.join(json.dumps(e,ensure_ascii=False) for e in test),encoding='utf-8')
(WORK/'train.jsonl').write_text('\n'.join(json.dumps(e,ensure_ascii=False) for e in train),encoding='utf-8')
(WORK/'gold.json').write_text(json.dumps(gold,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'train={len(train)} valid={len(valid)} test={len(test)} gold={len(gold)}')

(WORK/'valid.jsonl').write_text('\n'.join(json.dumps(e,ensure_ascii=False) for e in valid),encoding='utf-8')
