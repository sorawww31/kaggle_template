# BirdCLEF+ 2026 実験リポジトリ

## コンペ概要

- 取得日: 2026-04-26
- Kaggle slug: `birdclef-2026`
- 公式URL: https://www.kaggle.com/competitions/birdclef-2026
- 主催: Google Research & Cornell Lab of Ornithology

BirdCLEF+ 2026 は、ブラジル・パンタナル湿地で収録された連続音声から、鳥類・両生類・哺乳類・爬虫類・昆虫などの発声種を識別する音響分類コンペです。目的は、広域かつ継続的な受動的音響モニタリングを機械学習で支援し、生物多様性の変化や保全活動の効果を把握しやすくすることです。

### データ

- `train_audio/`: Xeno-canto と iNaturalist 由来の短い個別録音。32 kHz、`ogg`形式。
- `train_soundscapes/`: テスト録音に近い地点の追加音声。一部は`train_soundscapes_labels.csv`で5秒区間ごとの専門家ラベル付き。
- `test_soundscapes/`: 提出Notebook採点時に隠しテストとして約600本の1分音声が配置される。
- `train.csv`: `primary_label`、`secondary_labels`、緯度経度、録音者、filename、rating、collectionなどのメタデータ。
- `taxonomy.csv`: 234クラスの分類情報。提出列の`primary_label`に対応する。
- `sample_submission.csv`: `row_id`と234種ID列を持つ提出例。各行は5秒窓の確率予測。

### 評価・提出制約

- 評価指標: 真陽性がないクラスをスキップするmacro ROC-AUC系指標。
- 提出形式: 各`row_id`について、各species列に存在確率を出すmulti-label予測。
- Code Competition: Notebook提出のみ。CPU Notebookは90分以内、GPU提出は実質無効、Internet disabled、`submission.csv`が必要。
- 期限: Entry/Team Merger Deadline は2026-05-27 23:59 UTC、Final Submission Deadline は2026-06-03 23:59 UTC。

### 重要ルール

- 1日あたり最大5 submission、最終提出は最大2件選択。
- チーム上限は5人。
- データ利用はCC BY-NC-SA系で、非商用・コンペ参加・Kaggle forum・学術/教育用途が中心。
- 外部データや外部モデルは、参加者全員が合理的にアクセスでき、コンペ固有ルールに反しないものに限る。
- 入賞時は、最終モデルのコード・ドキュメント提出と、勝者コードのOpen Sourceライセンス要件がある。

### 開催履歴メモ

- 2026: ブラジル・パンタナル。鳥類以外も含む234クラスの音響種識別。
- 2025: コロンビア Middle Magdalena Valley / El Silencio Natural Reserve。鳥類・両生類・哺乳類・昆虫など複数分類群。
- 2024: インド Western Ghats。主に鳥類、希少・固有・夜行性種の音響識別。
- 2023: 東アフリカ/ケニアの鳥類音声。264 bird species、padded cmAP。
- 2022: ハワイの希少・絶滅危惧鳥類を対象にした音響識別。

### 取得元

Kaggle MCPで`search_content`、`list_competition_pages`、`list_competition_data_files`を確認しました。DiscussionはMCPのTopic絞り込みでKernel/Benchmarkが混在したため、現時点ではREADMEへ未検証情報を採用していません。

## 特徴
- Docker によるポータブルなKaggleと同一の環境
- Hydra による実験管理
- 実験用スクリプトファイルを major バージョンごとにフォルダごとに管理 & 実験パラメータ設定を minor バージョンとしてファイルとして管理
   - 実験用スクリプトと実験パラメータ設定を同一フォルダで局所的に管理して把握しやすくする
- dataclass を用いた config 定義を用いることで、エディタの補完機能を利用できるように
## フォーク源から追加した機能
- 実験管理が容易に
  - ```experiments/{major_exp_name}```単位での実験管理
  - モデル、ソースコード、実験ログを一括でKaggle Datasetにして実験を管理
  - ```tools/upload_dataset.py```で一括データセット化
- 環境変数管理
  - ```.env.example```を追加
    - 環境変数にてAPI_KEY等を管理
    - それに伴い```compose.yaml```を修正

### 変更コードまとめ
* 実験スクリプト
    * ```experiments/exp000_sample```直下のディレクトリ構造
    * ```experiments/exp000_sample/run.py```
* 環境構築
  * ```Dockerfile```
  * ```Dockerfile.cpu```
  * ```compose.yaml```
  * ```compose.cpu.yaml```
  * ```.env.example```
* その他
  * ```tools/upload_dataset.py```

### Hydra による Config 管理
- Config は yamlとdictで定義するのではなく、dataclass を用いて定義することで、エディタの補完などの機能を使いつつタイポを防止できるようにする
- 各スクリプトに共通する環境依存となる設定は utils/env.py の EnvConfig で定義される
- 各スクリプトによって変わる設定は、実行スクリプトのあるフォルダ(`{major_exp_name}`)の中に `exp/{minor_exp_name}.yaml` として配置することで管理。
    - 実行時に `exp={minor_exp_name}` で上書きする
    - `{major_exp_name}` と `{minor_exp_name}` の組み合わせで実験が再現できるようにする

## Structure(罫線表記)
```text
.
├── experiments
│    └── exp000_sample
│        ├── exp
│        ├── utils
│        ├── src
│        └── run.py
├── input
├── notebook
├── tools
├── outputs
├── Dockerfile
├── Dockerfile.cpu
├── LICENSE
├── Makefile
├── README.md
├── .env.example
├── compose.cpu.yaml
└── compose.yaml

```
## 環境変数の設定
```sh
cp .env.example .env
```
を行い、`.env`に必要事項を記入

## AIエージェント設定

Codex, Cursor, Claude Code, GitHub Copilot, Gemini CLI 向けの指示・skill・command・MCP設定を用意しています。

- 共通指示: `AGENTS.md`
- 共通skill: `.agents/skills/*/SKILL.md`
- 共通command: `.agents/commands/*.md`
- 同期: `uv run python tools/sync_agent_assets.py`
- 厳密同期: `uv run python tools/sync_agent_assets.py --prune`
- 検証: `uv run python tools/sync_agent_assets.py --check`

新しいskillやcommandを追加する場合は、まず `.agents/` 以下の共通ソースへ追加し、その後に同期します。
通常同期は必要な生成物を作成・更新します。`.agents/` 以下から削除したskillやcommandに対応する生成物も削除したい場合は、`--prune` 付きで厳密同期します。

```sh
# skillを追加する場合
.agents/skills/<skill-name>/SKILL.md

# commandを追加する場合
.agents/commands/<command-name>.md

# 各エージェント用ファイルへ反映
uv run python tools/sync_agent_assets.py

# 共通ソースと同じ形にそろえ、不要な生成物も削除
uv run python tools/sync_agent_assets.py --prune

# 同期漏れを確認
uv run python tools/sync_agent_assets.py --check

# 不要な生成物も含めて厳密確認
uv run python tools/sync_agent_assets.py --check --prune
```

`CLAUDE.md`, `GEMINI.md`, `.claude/commands`, `.cursor/commands`, `.github/prompts`, `.gemini/commands`, `.claude/skills` は同期生成物です。通常は直接編集せず、`AGENTS.md` または `.agents/` 以下を編集してから同期してください。

Kaggle MCPを使う場合は、`KAGGLE_API_TOKEN` を環境変数として設定してください。詳細は `docs/agent-integrations.md` を参照してください。

## Docker による環境構築
Dockerが利用できない方は、同md下部のuvによる環境設定を参照してください。
```sh
# imageのbuild
make build

# bash に入る場合
make bash

# jupyter lab を起動する場合
make jupyter

# CPUで起動する場合はCPU=1やCPU=True などをつける
```
### スクリプトの実行方法
```sh
make bash
python experiments/exp000_sample/run.py exp=001
```

`experiments/exp000_sample/run.py` は実行時に出力ディレクトリを自動作成します。

- 出力先のベース: `env.output_dir`（デフォルト: `outputs`）
- 実験ごとの出力先: `{env.output_dir}/{major_exp_name}/{minor_exp_name}`
  - `major_exp_name`: 実行スクリプトの親ディレクトリ名（例: `exp000_sample`）
  - `minor_exp_name`: `exp=...` で選んだ設定名（例: `001`）
- `init_output_dir`関数によって、適切な`cfg.env.exp_output_dir`が設定される
- `exp.name` を指定すると末尾に `_{exp.name}` が付きます
### つまりcfg.env.exp_output_dir を利用すればいい
例:
```sh
python experiments/exp000_sample/run.py exp=001 exp.name=baseline
# -> outputs/exp000_sample/001_baseline/
```

## Kaggle データセットの作成

### 1. 任意の1ディレクトリをアップロードする
```sh
# Kaggle API Keyが必要
# major_virsion_nameでそのまま提出
# -t: タイトル, -d: ディレクトリ
python tools/upload_dataset.py --title exp000 --dir experiments/exp000_sample
```

### 2. `--exp` で実験関連ディレクトリをまとめてアップロードする（新機能）
`--exp` を指定すると、次のパターンに一致するディレクトリを探して1つのDatasetにまとめます。

- `experiments/<exp>_*`
- `outputs/<exp>_*`

Dataset内では `experiments/` と `outputs/` のサブディレクトリとして保存されます。

`-t/--title` を明示指定した場合は、その値をDataset名として使用します。
未指定の場合は、最初に見つかった対象ディレクトリ名から自動でDataset名を決定します。

```sh
# 例: experiments/exp000_sample と output/exp000_sample をまとめてアップロード
python tools/upload_dataset.py --exp exp000

# 例: --exp 利用時にタイトルを明示指定
python tools/upload_dataset.py --exp exp000 --title exp000-custom
```

## uvによる環境構築
### uvのインストール
詳しくは[こちら](https://docs.astral.sh/uv/getting-started/installation/v)を参照
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv version # インストールの確認
```
### .venv 仮想環境の作成
```sh
# セットアップ
make uv-setup

# jupyter lab を起動する場合
make uv-jupyter

```
適宜追加したいモジュールは、以下のコマンドで追加してください。その他利用方法は
```sh
uv add numpy
```
その他利用方法は[こちら](https://docs.astral.sh/uv/)を参照してください。

**注意事項**
* pythonバージョン, numpyバージョンはkaggle kernelに合わせ最新版ではない
  * ```python==3.11.13```
  * ```numpy==1.26.4```
* ```torch, cuda```は、各環境に合わせインストールしてください。
### スクリプトの実行方法
```sh
uv run experiments/exp000_sample/run.py exp=001
```
