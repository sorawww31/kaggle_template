---
name: kaggle-cli
description: Kaggle公式CLIのコマンド、認証、Notebook実行・ログ・出力、提出、GPU/TPU枠、Dataset・Model操作を扱う。CLIでの操作やトラブルシューティングを依頼されたときに使う。
---

# Kaggle CLI

<!-- Where: .agents/skills/kaggle-cli. What: 公式CLI操作。 Why: Kaggleコマンドの参照先を統一する。 -->

1. このテンプレートでは `uv run kaggle --version` と `uv run kaggle <group> <command> --help` で実際に使えるコマンドとフラグを確認する。固定した上流ガイドと差があれば、インストール済みのCLIに合わせる。
2. 認証には設定済みの `KAGGLE_API_TOKEN` などを使う。トークンの値を表示せず、認証方式が不明なら上流の認証リファレンスを参照する。
3. Notebookを実行するときは `kernels init` の雛形と既存のメタデータを確認し、対象、公開範囲、アクセラレータ、入力を特定してから `kernels push` を使う。`kernels status` で結果を確認し、必要に応じて `kernels logs` と `kernels output` を使う。
4. コンペ提出ではファイル提出とNotebookバージョン提出を区別する。ユーザーが依頼した対象だけに提出し、`competitions submissions` で受理・評価状況を確認する。
5. テンプレートの `input/` 配置まで依頼された場合は `$import-kaggle-resource` を使う。コンペの一括オンボーディングは `$kaggle-fetch-competition` を使う。

[上流スキルとコマンド別リファレンス](https://github.com/Kaggle/kaggle-cli/blob/fdc788faffe3f29ae317480325c44080a00a5434/skills/SKILL.md)
