# TODO

- [ ] 実装: 生成系列の平均対数尤度を confidence score として記録する
  - HF Transformers の `model(**inputs, labels=...)` で生成応答の log-prob を再計算し、サンプルごと平均対数尤度（=負の cross-entropy）を保存する。
  - エントロピーと同様に per-example CSV / post-hoc 集計で活用できる形にする。

- [ ] 実装: 自己検証プロンプトによる出力チェック
  - 生成後の回答に対して「この答えは一貫していますか？」などのセルフチェックプロンプトを追加し、Yes/No 判定を confidence 指標として保存する。
  - 別回しで 1 トークン追加生成するだけで判定できるよう CLI オプション化する。
