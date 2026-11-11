---
name: kaggle-benchmarks
description: kaggle_benchmarksライブラリでLLMの評価タスクを設計・実装・検証する。タスクの入力、構造化出力、アサーション、データセット評価を扱う。
---

# Kaggle Benchmarks

<!-- Where: .agents/skills/kaggle-benchmarks. What: LLM評価タスクの作成。 Why: 合否判定を再現可能にする。 -->

1. 測りたい能力、入力、期待する出力、合否条件を明確にし、利用する `kaggle_benchmarks` のバージョンを確認する。
2. `@kbench.task` でタスクを作り、必要に応じて構造化出力と `kbench.assertions` を使う。客観条件はコードで判定し、判定根拠が追えるようにする。
3. 既知の合格例と失敗例で評価ロジックを確かめる。データセット評価では一部で試し、実行失敗とモデルの不合格を区別する。
4. Kaggleへの実行・公開を依頼された場合は `$kaggle-cli` の `benchmarks tasks` を参照する。

[上流スキルとコード例](https://github.com/Kaggle/kaggle-benchmarks/blob/5ffefc57696e26b172fc3626392731ebecbb1195/skills/kaggle-benchmarks/SKILL.md)
