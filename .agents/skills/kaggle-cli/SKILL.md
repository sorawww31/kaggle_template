---
name: kaggle-cli
description: Kaggle公式CLIで認証、検索、取得、Notebook実行、提出、Discussion、Model、Benchmarkを操作する。コマンド仕様とトラブルシュートに使う。
---

<!-- 配置: .agents/skills/kaggle-cli。内容: 実行手順。目的: ローカル本文で再現する。 -->

# Kaggle CLI

Use this skill to answer or operate on the `kaggle` command-line tool. Treat
this skill and its references as the available command guide.

## このプロジェクトでの実行

Kaggleへのアクセスは公式CLIに統一する。コマンド仕様は下記のローカル参照本文を読み、実行前に使用中のCLIの `--help` と照合する。本文にあるコマンドでも、使用中の版に存在しなければ使わない。

まずDocker Composeの `kaggle` サービスと構築済みイメージが使えるか確認する。使える場合は、ホストのuvより次を優先する。

```bash
docker compose config --services
docker compose images kaggle
docker compose run --rm --no-build --pull never kaggle kaggle --version
docker compose run --rm kaggle kaggle quota
```

Composeは `env_file: .env` から認証情報を渡し、リポジトリをコンテナにマウントする。取得先にはコンテナ内の作業ディレクトリからの相対パスを使う。CLIの版とヘルプはコンテナ内で確認する。ホストで見た版と同じと仮定しない。

Dockerが利用不可、イメージ未構築、GPU起動不可の場合は次のuv手順へフォールバックする。CLIの確認だけのために自動で大規模なpull/buildを始めない。ビルド自体を依頼された場合はその範囲で行う。

Dockerを使えない場合はリポジトリルートで、CLIだけの軽量環境を使って確認する。`.env` は `python-dotenv` で読み、shell予約変数の影響を避ける。末尾の `quota` を `--version`、`--help`、必要なコマンド引数へ置き換える。

```bash
uv run --no-project --python 3.12 --with kaggle==2.2.2 --with six --with python-dotenv python -c 'from dotenv import load_dotenv; load_dotenv(".env", override=True); from kaggle.cli import main; main()' quota
```

2.2.2は現在の `uv.lock` と同じ版で、不足する `six` を明示している。CLIの版を変更するときは、下の参照本文と新しいヘルプを照合する。

プロジェクトの依存をセットアップ済みなら、次も使える。通常の `uv run` はML・CUDA依存も同期するため、CLI確認だけのために未構築環境で実行しない。

```bash
uv run python -c 'from dotenv import load_dotenv; load_dotenv(".env", override=True); from kaggle.cli import main; main()' quota
```

以降の `kaggle ...` / `uv run kaggle ...` は、Dockerが使えるなら `docker compose run --rm kaggle kaggle ...`、使えなければ上記の認証済みuv呼び出しに同じ引数を渡す表記。認証情報を既に環境へ読み込んでいる場合に限り、そのまま実行してよい。キーの値を表示せず、引数や一時ファイルへ書かない。

認証の確認には `quota` を使う。認証失敗時にキーを表示するコマンドを実行しない。規約未同意は認証エラーと区別し、ブラウザでの同意が必要ならそのコンペのrules URLを案内する。

コンペはユーザーの明示slugを優先し、なければロードした `COMPETITION` を使う。空なら推測せず確認する。明示指定を使う場合は後続のimportにも同じslugを渡す。

2.2.2の `competitions submissions` は `--page-token` を受け付けるが、読み取り検証とローカル実装確認では次ページtokenを出力しなかった。`--page-size 200` で取得範囲を広げ、上限に達したら全履歴取得済みとせず未取得がある可能性を報告する。tokenのない出力からtokenを捏造しない。

一覧はCSVを `csv` モジュール等で読む。ヘルプにない `--format json` を付けない。page番号とpage tokenはコマンドごとに異なるので、返された継続情報で全ページを辿る。

入力配置は `$import-kaggle-resource`、オンボーディングは `$kaggle-fetch-competition` を使う。公開範囲・実行対象・提出先を確認し、依頼範囲だけを実行する。`kernels push` は保存だけでなく実行も起こす。提出は `competitions submissions` で状態を確認する。

CLIにないプロフィール・バッジ・Agent Exam等の操作を、非公開APIや別の接続方式で代行しない。取得できない項目は未確認として報告する。

以下は取り込んだ公式スキル本文。詳細な公式ドキュメント本文も `references/official-docs/` に同梱している。通常は用途別の `references/*.md` を先に読む。

## Quick Start

```bash
docker compose run --rm kaggle kaggle --version
docker compose run --rm kaggle kaggle --help
```

Authentication options:

```bash
kaggle auth login
# or set KAGGLE_API_TOKEN
# or place an access token in ~/.kaggle/access_token
# or use legacy ~/.kaggle/kaggle.json credentials
```

## Command Tree

```text
kaggle
├── competitions | c
│   ├── list, files, download, submit, submissions, leaderboard
│   ├── team-submissions, episodes, replay, logs, pages
│   └── topics {list, show}, topic-messages
├── datasets | d
│   ├── list, files, download, init, create, version
│   ├── metadata, status, delete
│   └── topics {list, show}
├── kernels | k
│   └── list, files, init, push|update, pull|get, output, status, logs, delete
├── models | m
│   ├── list, init, create, get, update, delete
│   ├── topics {list, show}
│   └── variations | instances | v | i
│       ├── get, init, create, files, list, update, delete
│       └── versions | v {list, create, download, files, delete}
├── files {upload}
├── forums | f {list, topics {list, show}}
├── benchmarks | b
│   ├── auth, init
│   ├── tasks | t {push, run, list, status, download, log|logs, models, delete, publish}
│   └── topics {list, show}
├── config {view, set, unset}
├── auth {login, print-access-token, revoke}
├── quota
└── search
```

Note: the CLI accepts aliases such as `kernels get` for `kernels pull` and
`kernels update` for `kernels push`. Do not recommend
`models variations versions init`; use `models variations init` for variation
metadata instead.

## Reference Map

Read only the reference needed for the user's task:

- [Competitions](references/competitions.md) - competition discovery, files, downloads, submissions, leaderboards, simulations, pages, topics.
- [Datasets](references/datasets.md) - dataset search, files, downloads, metadata, create/version/status/delete, topics.
- [Kernels](references/kernels.md) - notebook/script discovery, metadata, push/pull, outputs, status, logs, delete.
- [Models](references/models.md) - model records, metadata, create/get/update/delete, model topics.
- [Model Variations](references/model_variations.md) - create and manage framework-specific model variations.
- [Model Variation Versions](references/model_variations_versions.md) - create, list, download, inspect, and delete variation versions.
- [Files](references/files.md) - inbox uploads, resumable uploads, directory compression behavior.
- [Forums](references/forums.md) - global discussion forums, topics, and comments.
- [Benchmarks](references/benchmarks.md) - benchmark auth/init, task push/run/status/download/log/model flows, benchmark topics.
- [Configuration](references/configuration.md) - config file, default path, proxy, default competition.
- [Authentication](references/auth.md) - OAuth login, access token printing, revocation, token/key sources.
- [Quota](references/quota.md) - weekly GPU/TPU accelerator quota.
- [Search](references/search.md) - unified cross-content search over competitions, datasets, notebooks, models, users, and discussions.

## Operating Guidance

- Read only the reference needed for the user's task.
- Prefer `kaggle <group> --help` or `kaggle <group> <command> --help` when a flag
  is uncertain in the installed CLI.
- For metadata files, prefer the relevant `init` command to generate a starter
  file before editing it.
- Do not invent commands that are not listed in this skill. If live `--help`
  output differs, report it as a version-specific difference.
