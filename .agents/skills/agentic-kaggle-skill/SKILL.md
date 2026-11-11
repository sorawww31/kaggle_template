---
name: agentic-kaggle-skill
description: KaggleコンペのCVと評価指標を設計し、ベースライン、OOF、特徴量、モデル、アンサンブルを改善する。実験の設計・実装・比較を依頼されたときに使う。
---

# Agentic Kaggle Modeling

<!-- Where: .agents/skills/agentic-kaggle-skill. What: 検証設計とモデル改善。 Why: CVで採否を判断できる実験を組む。 -->

1. コンペ指標、提出形式、学習・テストの分布、リークの懸念を確認する。グループ、時系列、層化などの分割を選び、foldとseedを実験間で固定する。
2. 既存実験からの派生作成を依頼された場合は `$new-exp` を使う。まず単純なベースラインで指標とデータの読み込みを検証し、OOFを保存する。
3. 特徴量、モデル、後処理を検証可能な単位で比較する。fold別スコアとばらつきを記録し、スタッキングの学習にはfold外の予測を使う。
4. 訓練と推論を別Notebookに分ける必要があれば、設定、fold、チェックポイント、OOF、生成物のバージョンを対応付ける。Kaggle上の実行、枠確認、ログ回収、提出を依頼された場合は `$kaggle-cli` を使う。
5. ベースラインとの差、変更内容、再現手順、次の実験をまとめる。

[上流のCVと指標](https://github.com/FrankS-IntelLab/agentic-kaggle-skill/blob/3cdd820356c80fbdd946138d2f86cbc2bd5e91b9/references/cross-validation-and-metrics.md) · [アンサンブルと再現性](https://github.com/FrankS-IntelLab/agentic-kaggle-skill/blob/3cdd820356c80fbdd946138d2f86cbc2bd5e91b9/references/ensembling-and-reproducibility.md)
