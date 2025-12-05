# MLコンペ用実験テンプレート

## 特徴
- Docker によるポータブルなKaggleと同一の環境
- Hydra による実験管理
- 実験用スクリプトファイルを major バージョンごとにフォルダごとに管理 & 実験パラメータ設定を minor バージョンとしてファイルとして管理
   - 実験用スクリプトと実験パラメータ設定を同一フォルダで局所的に管理して把握しやすくする
- dataclass を用いた config 定義を用いることで、エディタの補完機能を利用できるように

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
│        ├── output
│        ├── utils
│        ├── src
│        └── run.py
├── input
├── notebook
├── tools
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
を行い、必要事項を記入
## Docker による環境構築

```sh
# imageのbuild
make build

# bash に入る場合
make bash
make bash exp={major_exp_name}

# jupyter lab を起動する場合
make jupyter

# CPUで起動する場合はCPU=1やCPU=True などをつける
```

## スクリプトの実行方法

```sh

# make bash; python experiments/{major_version_name}/run.py exp={minor_version_name}
make bash exp=000
python run.py
python run.py exp=001
```

## Kaggle データセットの作成
```sh
# Kaggle API Keyが必要
# major_virsion_nameでそのまま提出
python tools/upload_model.py -t exp000 -d experiments/{major_virsion_name}
```