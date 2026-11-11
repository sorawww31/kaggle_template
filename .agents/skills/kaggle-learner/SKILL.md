---
name: kaggle-learner
description: 過去のKaggle上位解法の知識本文を読み、CV、特徴量、モデル、学習・推論、アンサンブルを現在の実験へ転用する。
---

Kaggleへのアクセスは `$kaggle-cli` の実行方法に従う。Docker Composeが使える場合は `docker compose run --rm kaggle kaggle ...` を優先し、本文中の `kaggle ...` / `uv run kaggle ...` も同じ引数で置き換える。Dockerが使えない場合だけuvへフォールバックする。

<!-- 配置: .agents/skills/kaggle-learner。内容: 実行手順。目的: ローカル本文で再現する。 -->

# Kaggle Learner

## 知識本文を選ぶ

大会名・分野から `references/knowledge/` の実ファイルを列挙し、該当する大会ノートの本文を読む。

| 分野 | ローカル本文 |
|---|---|
| NLP | `references/knowledge/nlp/` |
| 画像 | `references/knowledge/cv/` |
| 時系列・音声 | `references/knowledge/time-series/` |
| 表形式 | `references/knowledge/tabular/` |
| マルチモーダル | `references/knowledge/multimodal/` |

存在しないカテゴリや大会ノートは作り話で埋めない。BirdCLEFはtime-series内の2023、2024、plus-2025の各ノートを比較できる。ノートの記述は過去時点の解法知識であり、現在のルールの根拠にはしない。

## 解法を読む・比較する

1. タスク、データ規模、評価指標、順位、作者、出典日を抽出する。
2. 上位解法ごとにCV分割、モデル、外部データ、特徴量、loss、augmentation、推論窓、TTA、後処理、アンサンブルを整理する。
3. コード例は該当ノート内の本文を読み、現在のデータschemaとパッケージに合わせる。数値効果や順位はノートの出典付きで示し、未検証の例を再現済みとしない。
4. 大会間で共通する方法と大会固有の仮定を分ける。必要なデータ・計算量、リークやLB依存、失敗条件を記す。
5. 現在の固定CVで確かめる最小実験に落とす。ベースとの差、変更する設定、期待する観測、採否基準を示す。

新たなKaggle情報の収集は `$kaggle-cli`、`$kaggle-discussion-research`、`$kaggle-notebook-research` のCLI手順だけを使う。自動更新を行う未導入エージェントは前提にしない。知識ノートを追加する依頼では、概要、元解法の要約、主要手法の詳細、再利用コード、失敗条件、出典・日付を本文に含める。
