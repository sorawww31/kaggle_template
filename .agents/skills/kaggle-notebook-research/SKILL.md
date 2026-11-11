---
name: kaggle-notebook-research
description: Kaggle公式CLIで公開Notebook/Kernelを検索し、fork親子関係、重複、派生、public score、利用データ・モデル・出力物、コード本文を根拠付きで調査してMarkdownにまとめる。ユーザーが「公開ノートブックを調査」「似たNotebookの違いを見たい」「BirdCLEFなど参加中コンペのnotebookから実装ヒントを拾って」「高スコアNotebookの派生元を追って」「Kaggle Notebookの重複・fork・改善点を整理して」などを依頼したときに使う。
---

Kaggleへのアクセスは `$kaggle-cli` の実行方法に従う。Docker Composeが使える場合は `docker compose run --rm kaggle kaggle ...` を優先し、本文中の `kaggle ...` / `uv run kaggle ...` も同じ引数で置き換える。Dockerが使えない場合だけuvへフォールバックする。

<!--
SKILL.md
Where: .agents/skills/kaggle-notebook-research.
What: Research public Kaggle notebooks with fork and duplicate awareness.
Why: Turn noisy public notebook streams into actionable, source-grounded experiment ideas.
-->

# Kaggle Notebook Research

このSkillは、Kaggle公式CLIで公開Notebookを広く集め、単なるvote順ではなく、fork系譜・重複・小改変・実装上の差分まで見て、実験に使える知見を`docs/notebooks/*.md`へ残すための手順を定める。

Notebook調査は同名fork、丸写し、小さな後処理追加、古い高vote、低voteだが新しい高scoreが混ざる。候補収集、系譜整理、本文確認、要約を分けて進める。

操作前に `$kaggle-cli` と `references/kaggle_cli_notebook_tools.md` のローカル本文を読む。以下の `kaggle` は認証済みの `uv run kaggle` として実行する。

## 前提と候補収集

ユーザーの明示slugを優先し、なければ `COMPETITION` と参加済みコンペ一覧で確認する。通常は `docs/notebooks/`、オンボーディングでは `docs/competitions/<slug>/notebooks.md` に保存する。既存文書を先に全文読む。

```bash
kaggle kernels list --competition <slug> --sort-by voteCount -p 1 -v
kaggle kernels list --competition <slug> --sort-by hotness -p 1 -v
kaggle kernels list --competition <slug> --sort-by dateRun -p 1 -v
kaggle kernels list --competition <slug> --sort-by commentCount -p 1 -v
kaggle kernels list --competition <slug> --search "<keyword>" --sort-by relevance -p 1 -v
kaggle kernels list --parent <owner/parent> -p 1 -v
```

後続ページを巡回し、refで重複を除く。検索語はbaseline/training/inference/ensemble/OOF/postprocess/ONNXと分野固有語で補う。候補表にはref、title、取得できたauthor、votes、更新日、発見元を残す。score順は指標の最適化方向を確認して使い、並び順から数値を逆算しない。

CLIに表示されないtier、medal、親URL、public scoreは未確認とする。既知の親の `--parent` 検索で子を確認できた場合だけ親子関係を確定する。本文のリンク・引用からの系譜と、コード類似による推定を区別する。

## 系譜と重複を整理する

1. 同じNotebookファミリーをまとめる。
   - CLIで確認した親URLがあるものは親URLでつなぐ。
   - 同名title、slugの共通prefix、同じデータソース、同じモデル名、同じscore表記を補助特徴にする。
   - 親が取れないNotebookは、同名でも別実装の可能性を残す。
2. 候補を分類する。
   - `root`: 親が見えない、またはシリーズの起点。
   - `direct-fork`: Kaggle上の親URLがある。
   - `copy-or-near-copy`: 親URLはないがtitle/構成/データソースがほぼ同じ。
   - `small-mod`: 閾値、TTA、後処理、batch size、LB向け係数など小変更。
   - `substantial-mod`: 新モデル、学習手順、pseudo label、外部データ、特徴抽出、CV設計など実装の芯が変わる。
   - `eda-or-utility`: 実験案よりデータ理解、可視化、変換、提出補助が主目的。
3. 優先順位を付ける。
   - 高public score、親よりscoreが上がった派生、最近更新、高comment、高vote、上位tier、利用データ・モデルが新しいものを優先。
   - 高voteでも古く、派生に取り込まれているだけならrootとして読む。
   - 低voteでもrecentでscoreや新規手法が強いものは読む。

## Notebook本文を読む

全候補を取得せず、起点、高scoreを主張する派生、新手法、コメントの多いもの、自分の実験に近いものを代表に選ぶ。

```bash
kaggle kernels pull <owner/notebook/version> -m -p <tmp>
kaggle kernels files <owner/notebook/version> -v
kaggle kernels output <owner/notebook/version> -p <tmp> --file-pattern "<必要ファイルの正規表現>"
kaggle kernels topics list <owner/notebook> --sort-by top -v
```

`kernel-metadata.json` の `code_file`、`competition_sources`、`dataset_sources`、`kernel_sources`、`model_sources`、GPU/internet設定を読む。本文はセルに分け、見出し、設定、主要関数、保存処理、親との差分を確認する。version未指定の最新版を固定version扱いにしない。

ファイル一覧は全ページを確認し、必要な小さい出力だけ取得する。コードやインストールセルは取得しただけで実行しない。実行履歴やスコアなどCLIで確認できない情報は未確認とする。

## 実装差分を抽出する

Notebookファミリーごとに、親から何が変わったかを要約する。

- データ: 外部dataset、pseudo label、OOF cache、metadata、taxonomy、過年度データ。
- モデル: backbone、Perch/embedding、SED、SSM/RNN/Transformer、ONNX変換、ensemble。
- 学習: fold、group split、loss、mixup/cutmix、sampling、early stopping、SWA、蒸留。
- 推論: window長、overlap、batch、TTA、postprocess、threshold、rank/file-level scaling。
- 検証: CV、OOF、public score、LB依存、hidden testで壊れそうな仮定。
- 再利用性: 自分のrepoに移植しやすいか、依存や実行時間が重いか、ルール確認が必要か。

「scoreが上がった」と書く場合は、Kaggle公式CLIで確認できたpublic scoreまたはNotebook内の明記を出典付きで残す。推測なら推測と書く。

## Markdownへ保存する

出力先は`docs/notebooks/<YYYY-MM-DD>-<competition-slug>-notebook-research.md`を基本にする。
単一ファミリーだけなら`docs/notebooks/<YYYY-MM-DD>-<competition-slug>-<family>.md`にする。

新規Markdownには冒頭コメントを置く。

```markdown
<!--
<filename>
Where: docs/notebooks.
What: Kaggle public notebook research summary for <competition>.
Why: Preserve actionable implementation ideas with fork, duplicate, and source context.
-->
```

本文テンプレート:

```markdown
# <調査タイトル>

- 取得日: <YYYY-MM-DD>
- 対象: <competition title> (`<slug>`)
- 依頼: <user request>
- 調査範囲: <sort/search terms/pages/families/read notebooks>
- 注意: <CLIの出力制約、未取得本文、推測の有無>

## 結論
- <実装判断に直結する要点>

## Notebookファミリー
| family | 代表 | root/parent | 最高public score | votes | 判定 | 実装差分 | 優先度 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ... | ... | ... | ... | ... | small-mod | TTA追加 | high |

## 候補一覧
| 判定 | ref | title | author | votes | score | updated | parent | 発見元 | 理由 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## 詳細メモ
### <notebook title>
- URL: <Kaggle URL>
- ref/id/version: <owner/slug>, <id>, v<version>
- 親: <fork parent or none>
- metadata: <GPU/internet/data/model sources>
- 要点:
  - <親との差分と再利用可能な実装>
- リスク:
  - <LB依存、外部データ、時間、再現性、ルール確認>

## 実装候補
- <repoへ移すなら何をどこに入れるか>

## 未確認・リスク
- <読めていない候補、CLIの出力制約、推測>

## Source Index
| 種別 | id/ref | title | URL | メモ |
| --- | --- | --- | --- | --- |
```

## 要約ルール

- 原文の長い引用は避け、要約中心にする。Notebook名、score、ファイル名、関数名、設定値は必要最小限だけ引用する。
- vote順だけで結論を出さない。fork親、public score、更新日、本文差分を一緒に見る。
- 同じtitleのNotebookを別物として扱う前に、親URLとコード構成を確認する。
- 開催中コンペではpublic LB最適化を過信しない。closed/private testで成立する条件を書く。
- ルールや外部データが絡む場合は、Kaggle公式ルールまたはcompetition情報を確認してから実装候補に入れる。

## 検証

1. 作成・更新したMarkdownを読み直し、リンク、表、未確認事項、推測表現を確認する。
2. skill自体を編集した場合は、skill-creatorの`quick_validate.py`で対象skillを検証する。
3. 共有skillを追加した場合は、生成アダプタを同期し、関連テストを追加または更新する。
4. 最後に`git diff -- docs/notebooks .agents/skills/kaggle-notebook-research tests`を確認し、調査範囲と未取得情報を報告する。
