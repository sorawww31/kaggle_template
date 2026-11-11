# Kaggle テンプレート

## コンペを始める
### 0. GitHubからリポジトリを作成する
GitHubページの右上「Use this template」から、リポジトリを作成する。
その後、自分の環境に git cloneする。
### 1. 環境変数を設定する

`.env.example`をコピーし、Kaggleユーザー名、API token、出場するコンペのslugを`.env`へ記入する。

```sh
cp .env.example .env
```

```dotenv
KAGGLE_USER_NAME=your_kaggle_user_name
KAGGLE_API_TOKEN=your_kaggle_api_token
COMPETITION=birdclef-2026
```

秘密情報を含む`.env`はcommitしない。AIエージェントを起動する前に、同じshellで環境変数を読み込む。

```sh
set -a
source .env
set +a
```

### 2. AIでオンボーディングする

Competitionデータ、公式ルール、Discussion、公開Notebook、EDAを一括で準備する場合は、次のSkillを使う。

```text
$kaggle-fetch-competition を使ってCOMPETITIONのコンペをオンボーディングして
```

データ取得だけを依頼する場合は`$import-kaggle-resource`を使う。

### 3. 手動でデータを取り込む

AIやKaggle MCPを使わず、Competitionデータを直接取得する場合は次を実行する。

```sh
uv run python tools/download_dataset.py competition "$COMPETITION"
```

Kaggle Datasetを追加する場合は`owner/dataset`形式で指定する。

```sh
uv run python tools/download_dataset.py dataset owner/dataset-slug
```

既存の保存先を置き換える必要がある場合だけ`--force`を付ける。

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

### Skillsの管理

Skillはすべて `.agents/skills/` に集約して管理します。追加・更新・削除はこのディレクトリで行います。

```sh
.agents/skills/<skill-name>/SKILL.md
```

### Kaggle向け公開Skill

公開Skillの手順をこのテンプレートに合わせて採用しています。各Skillに参照元のコミットを記載しています。スター数は配布元リポジトリの2026年11月7日時点の表示です。

| Skill | 上流スター | 担当 |
|---|---:|---|
| [`kaggle-cli`](.agents/skills/kaggle-cli/SKILL.md) | 約7,600 | Kaggle公式CLIの認証、リモート実行、ログ取得、提出 |
| [`kaggle-learner`](.agents/skills/kaggle-learner/SKILL.md) | 約6,000 | 過去コンペの上位解法から学ぶ |
| [`create-harness`](.agents/skills/create-harness/SKILL.md)、[`onboard-open-spiel-game`](.agents/skills/onboard-open-spiel-game/SKILL.md) | 459 | 対戦するLLMの実装、OpenSpielゲームの追加 |
| [`nvidia-kaggle-skill`](.agents/skills/nvidia-kaggle-skill/SKILL.md) | 349 | 指定Notebookの入力と実行環境をローカルで再現 |
| [`kaggle-benchmarks`](.agents/skills/kaggle-benchmarks/SKILL.md)、[`kaggle-standardized-agent-exam`](.agents/skills/kaggle-standardized-agent-exam/SKILL.md) | 204 | LLM評価タスクの作成、Kaggle公式のAgent Exam |
| [`agentic-kaggle-skill`](.agents/skills/agentic-kaggle-skill/SKILL.md) | 191 | CV、ベースライン、OOF、モデル比較、アンサンブル |
| [`kaggle-badges`](.agents/skills/kaggle-badges/SKILL.md) | 108 | バッジの取得条件と確認方法 |

CLI操作は公式の `kaggle-cli` に集約しています。テンプレート固有の初期準備は `kaggle-fetch-competition`、MCP経由での入力配置は `import-kaggle-resource`、参加中コンペの投稿やNotebookの系譜の調査は既存の調査Skillを使います。

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
### ビルドキャッシュの利用

ビルドには BuildKit が必要です（現在の Docker Desktop / Docker Engine では標準）。
通常は `make build` を使うと、変更のないインストール層が再利用されます。
追加パッケージを変更して層を再実行するときも、BuildKit の pip キャッシュから
ダウンロード済みファイルやビルド済み wheel を再利用できます。
キャッシュはイメージには含まれず、同じ builder 内に保持されます。
Compose の実行時ボリューム `./.cache` とは別のキャッシュです。

`.dockerignore` は Dockerfile と自身だけをビルド対象にします。
データ・出力・仮想環境・認証情報は転送しません。
ソースは実行時のボリュームで共有します。将来 `COPY` / `ADD` を追加する場合は、
必要なファイルを `.dockerignore` の許可リストにも追加してください。
現在は `COPY` がなく、依存インストールの層が独立しているため、層の分割は不要です。

時間とキャッシュの利用状況は、同じコマンドを2回実行して確認できます。
2回目の pip インストール工程が `CACHED` になれば層が再利用されています。

```sh
time docker compose --progress=plain build
time docker compose --progress=plain build

# CPU 用 Dockerfile を直接ビルドする場合
time docker build --progress=plain -f Dockerfile.cpu -t kaggle-env:cpu .
```

初回の Kaggle ベースイメージの取得時間は、このキャッシュでは短縮できません。
通常の再ビルドでは `--no-cache` を付けず、ビルドキャッシュを削除しないでください。
調査元: [Docker のキャッシュ最適化](https://docs.docker.com/build/cache/optimize/)、
[pip のキャッシュ](https://pip.pypa.io/en/stable/topics/caching/)、
[参考記事](https://zenn.dev/engharu/articles/b3aa073c3694de)。

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
