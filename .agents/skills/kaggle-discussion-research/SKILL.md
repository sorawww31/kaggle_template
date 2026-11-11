---
name: kaggle-discussion-research
description: Kaggle公式CLIでKaggle Discussion、Competition Solution、コメント、返信を調査し、docs/discussion/*.mdへ根拠付きで要約する。ユーザーが「XXXの1位解法を調査」「YYYの上位解法をまとめて」「参加中コンペの有用Discussionを調査」「このコンペのdiscussionから実装ヒントを拾って」など、Kaggle discussionやsolution writeupの調査・整理を依頼したときに使う。
---

Kaggleへのアクセスは `$kaggle-cli` の実行方法に従う。Docker Composeが使える場合は `docker compose run --rm kaggle kaggle ...` を優先し、本文中の `kaggle ...` / `uv run kaggle ...` も同じ引数で置き換える。Dockerが使えない場合だけuvへフォールバックする。

<!--
SKILL.md
Where: .agents/skills/kaggle-discussion-research.
What: Research Kaggle discussion topics and summarize findings into docs/discussion.
Why: Preserve actionable competition knowledge with source links, including important comments.
-->

# Kaggle Discussion Research

このSkillは、Kaggle公式CLIでDiscussion本文だけでなくコメント・返信まで読み、実装に使える知見を`docs/discussion/*.md`へ残すための手順を定める。
Discussion検索はノイズが混ざるため、取得元URL、topic ID、comment ID、取得日を残し、推測と確認済み情報を分ける。

## 前提とCLI本文

最初に `$kaggle-cli` を読み、`.env` の認証をロードする。コマンド仕様は `../kaggle-cli/references/competitions.md` と `../kaggle-cli/references/forums.md` のローカル本文を読む。以下の `kaggle` は `uv run kaggle` として実行する。

明示slugを優先し、なければ `COMPETITION` と `competitions list --group entered -v` から対象を確認する。通常は `docs/discussion/`、オンボーディングから呼ばれたら `docs/competitions/<slug>/discussion.md` に保存する。既存ファイルは先に全文を読む。

## CLIで調査する

### 候補収集

```bash
kaggle competitions list -s <slug> -v
kaggle competitions topics list <slug> --sort-by top -p 1 -v
kaggle competitions topics list <slug> --sort-by hot -p 1 -v
kaggle competitions topics list <slug> --sort-by recent -p 1 -v
kaggle competitions topics list <slug> --sort-by active -p 1 -v
kaggle forums topics list --category competition_write_ups --search "<slug>" --sort-by top -v
kaggle forums topics list --category competitions --search "<slug> CV" --sort-by relevance -v
```

コンペ別一覧はページ番号、forumsはpage tokenで継続する。使用中のヘルプと返却値を確認する。通常は各軸2〜3ページ、重複除去後50〜100件程度を候補とし、依頼の広さに合わせて増減する。検索結果が対象コンペの投稿であることを本文とURLで確認する。

検索語は `1st place solution`、`writeup`、`baseline`、`CV`、`leak`、`external data`、`inference`、`ensemble`、`postprocess` と分野固有語で補う。同じ集合ばかり返る場合は後続ページと別の語へ広げる。

候補表にtopic ID、title、取得できたURL、votes、comments、日時、発見元とread/skim/skip理由を残す。投稿者名・tier・host属性が表示されない場合は推測しない。host告知、ルール訂正、データ不備、CV/LB乖離、再現可能な実装を優先する。

個人の環境エラー、チーム募集、雑談、単発のLB自慢は原則優先度を下げる。ただし高コメント・高vote投稿は先頭コメントを確認し、host回答や汎用的な回避策があれば読む。

### 本文とコメント・返信

```bash
kaggle forums topics show <topic-id> --page-size 100
```

本文だけでなく、表示されたコメントと返信を読む。継続tokenがある場合は `--page-token` で取得し、コメント不足が残れば未取得と記す。本文の訂正、追加設定、Notebookリンク、推論時間、外部データ条件を拾う。重要コメントのIDが取れればそのリンクを付け、取れないIDは作らずtopic URLを出典にする。

解法調査では `competition_write_ups` の検索とDiscussionを照合する。CLIで独立WriteUp本文を取得できない場合はDiscussion本文・コメントに基づく要約と明記する。順位・score・team名は投稿者の申告と公式に確認できた情報を区別する。

通常は上位10〜20件を本文・コメント込みで読む。網羅調査では候補数、取得ページ数、未読理由を残す。

## Markdownへ保存する

出力先は`docs/discussion/<YYYY-MM-DD>-<competition-slug>-<short-topic>.md`を基本にする。
複数topicを横断する調査なら`docs/discussion/<YYYY-MM-DD>-<competition-slug>-discussion-research.md`にまとめる。

新規Markdownには冒頭コメントを置く。

```markdown
<!--
<filename>
Where: docs/discussion.
What: Kaggle discussion research summary for <competition>.
Why: Preserve actionable ideas and source links from topics, comments, and writeups.
-->
```

本文テンプレート:

```markdown
# <調査タイトル>

- 取得日: <YYYY-MM-DD>
- 対象: <competition title> (`<slug>`)
- 依頼: <user request>
- 調査範囲: <Top/Recent/WriteUp/search terms/pages>
- 候補数: <collected/read/skipped>
- 注意: <未取得コメント、CLIの出力制約、推測の有無>

## 結論
- <実装や判断に直結する要点>

## 実装ヒント
- <model/data/training/inference/validation/ensembleなど>

## 候補一覧
| 判定 | 優先 | topic id | title | 発見元 | votes | comments | last activity | 理由 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| read | high | ... | ... | Top, Hot | ... | ... | ... | 高voteかつコメント多 |
| skip | low | ... | Notebook error | Recent | ... | ... | ... | 個人環境の実行エラーのみ |

## Topic別メモ
### <topic title>
- URL: <Kaggle URL>
- topic id: <id>
- votes/comments: <n>/<n>
- 要点:
  - <OPとコメントを統合した要約>
- 重要コメント:
  - <author> (<date>, comment <id>): <要約> <URL#comment_id>
- 信頼度: <高/中/低> - <理由>

## 未確認・リスク
- <閉じたtestで使えるか、ルール確認が必要か、再現性不足など>

## Source Index
| 種別 | id | title/comment | URL | メモ |
| --- | --- | --- | --- | --- |
| topic | ... | ... | ... | ... |
| comment | ... | ... | ... | ... |
```

## 要約ルール

- 原文の長い引用は避け、要約中心にする。必要な短いフレーズ、score、ファイル名、Notebook名だけ引用する。
- OPだけで結論を出さず、コメント・返信で訂正や実装詳細が出ていないか確認する。
- 「参加者の推測」「host/organizerの確定情報」「自分の推定」を分けて書く。
- 実装に使う場合は、Notebookリンク、モデル名、学習データ、外部データ、推論時間、LB/CV、コンペルールとの関係を分けて記録する。
- 現在開催中コンペではLB投稿を過信しない。closed/private testで成立する条件を必ず書く。

## 検証

1. 作成・更新したMarkdownを読み直し、リンク、見出し、表、未確認事項を確認する。
2. skill自体を編集した場合は、skill-creatorの`quick_validate.py`で対象skillを検証する。
3. 共有skillを追加した場合は、生成アダプタを同期し、関連テストを追加または更新する。
4. 最後に`git diff -- docs/discussion .agents/skills/kaggle-discussion-research`を確認し、調査範囲と未取得情報を報告する。
