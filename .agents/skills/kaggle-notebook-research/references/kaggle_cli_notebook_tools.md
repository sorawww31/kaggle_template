<!-- 配置: Notebook調査のreferences。内容: CLI操作。目的: 候補収集と本文比較を再現する。 -->
# Notebook CLIの使い分け

操作前に `../../kaggle-cli/references/kernels.md`、metadataが必要なら `../../kaggle-cli/references/official-docs/kernels_metadata.md` を読む。

| 目的 | コマンド | 注意 |
|---|---|---|
| 一覧 | `kernels list --competition <slug> -v` | voteCount/hotness/dateRun/commentCountとページを変える |
| 既知の親の派生 | `kernels list --parent <owner/parent> -v` | 任意の子から親を逆引きする機能とは別 |
| 本文と入力 | `kernels pull <owner/slug/version> -m -p <tmp>` | code_fileとsourcesを読む |
| 出力一覧 | `kernels files <owner/slug/version> -v` | page tokenで全ページ確認 |
| 小さい出力 | `kernels output <owner/slug/version> -p <tmp> --file-pattern <regex>` | 正規表現。globと混同しない |
| コメント | `kernels topics list <owner/slug>`、`forums topics show <topic-id>` | ヘルプで対応確認。返信も読む |

CLIの出力にないscoreや親URLを必須フィールドとしない。本文の自己申告scoreと公式取得score、コード類似と確定したforkを区別する。同じtitleだけで同一実装と決めない。

入力・モデル・CV・OOF・後処理・推論高速化の差を優先し、titleのscore変更やbatch sizeだけの差を別手法として数えない。取得日、version、検索範囲、未取得情報を報告する。
