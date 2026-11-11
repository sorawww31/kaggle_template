---
name: kaggle-badges
description: Kaggleバッジの条件本文を確認し、CLIで行える操作と手動操作を分けて案内する。バッジを指定されたときに使う。
---

Kaggleへのアクセスは `$kaggle-cli` の実行方法に従う。Docker Composeが使える場合は `docker compose run --rm kaggle kaggle ...` を優先し、本文中の `kaggle ...` / `uv run kaggle ...` も同じ引数で置き換える。Dockerが使えない場合だけuvへフォールバックする。

<!-- 配置: .agents/skills/kaggle-badges。内容: 実行手順。目的: ローカル本文で再現する。 -->

# Kaggle Badges

1. `references/badge-catalog.md` の該当バッジ条件と注意事項を読む。名称とカテゴリを確認する。
2. `$kaggle-cli` の本文で対応コマンドを確認し、必要なNotebook実行、提出、Dataset作成などの操作を具体化する。バッジ取得のためだけに公開・提出・枠消費を勝手に追加しない。
3. プロフィール編集、投票、フォロー、取得済みバッジ確認などCLIで扱えない操作は手動手順として案内する。直接APIやブラウザ自動操作に置き換えない。
4. CLI操作の成功とバッジ付与は区別する。付与を確認できなければ未確認と報告する。
