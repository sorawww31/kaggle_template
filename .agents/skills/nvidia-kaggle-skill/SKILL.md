---
name: nvidia-kaggle-skill
description: 指定されたKaggle Notebookをローカルで再現する。Notebookのメタデータから入力、モデル、上流Notebook出力、依存関係を解決し、ローカル実行に必要なパスを対応付ける。
---

# NVIDIA Kaggle Notebook Reproduction

<!-- Where: .agents/skills/nvidia-kaggle-skill. What: Notebookのローカル再現。 Why: Kaggleの入力マウントと実行環境を手元で再現する。 -->

1. `owner/notebook` と対象バージョンを確認し、Notebook本体とメタデータを取得する。CLIを使う場合は `$kaggle-cli` の `kernels pull` を参照する。
2. メタデータの `competition_sources`、`dataset_sources`、`model_sources`、`kernel_sources` と、コード上の `/kaggle/input/` を突き合わせる。Notebookソースと上流Notebookの実行出力を区別する。
3. テンプレート内に置く入力は `$import-kaggle-resource` の配置規則に従い、すでにある同じバージョンの入力を再利用する。取得対象のコンペと `COMPETITION` が一致しない場合は、別コンペのデータを代用せず未解決として記録する。取得できない依存やバージョン差も記録する。
4. ローカル実行用のコピーまたはパス設定で入力を解決する。依存パッケージ、GPU、インターネットの前提を確認し、外部Notebookのコードとインストールセルを読んでから、依頼された範囲で実行する。
5. 元Notebookの参照、入力の対応表、実行コマンド、確認できた出力、再現できなかった項目を記録する。

[上流のローカル再現手順](https://github.com/NVIDIA/nvidia-kaggle/blob/2b78cf29f5f30680764292a6592de8d53d4147a8/skills/nvidia-kaggle-skill/kernel-setup.md)
