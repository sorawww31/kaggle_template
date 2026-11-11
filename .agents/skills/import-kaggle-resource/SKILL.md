---
name: import-kaggle-resource
description: Kaggle公式CLIでCompetition、Dataset、Model、Notebookソースと出力を取得し、input配下へ安全に配置する。
---

Kaggleへのアクセスは `$kaggle-cli` の実行方法に従う。Docker Composeが使える場合は `docker compose run --rm kaggle kaggle ...` を優先し、本文中の `kaggle ...` / `uv run kaggle ...` も同じ引数で置き換える。Dockerが使えない場合だけuvへフォールバックする。

<!-- 配置: .agents/skills/import-kaggle-resource。内容: 実行手順。目的: ローカル本文で再現する。 -->

# Import Kaggle Resource

最初に `$kaggle-cli` の認証・実行手順を適用する。取得種別に応じて `../kaggle-cli/references/competitions.md`、`datasets.md`、`kernels.md`、`model_variations_versions.md` の本文と該当コマンドのヘルプを読む。以下の `kaggle` は認証環境を引き継いだ `uv run kaggle` を表す。

## 種別と保存先

| 種別 | handle | 保存先 |
|---|---|---|
| Competition | competition | `input/competitions/{competition}/` |
| Dataset | owner/dataset | `input/datasets/{owner}/{dataset}/` |
| Model | owner/model/framework/variation/version | `input/models/{owner}/{model}/{framework}/{variation}/{version}/` |
| Notebook出力 | owner/notebook | `input/notebook-outputs/{owner}/{notebook}/` |
| Notebookソース | owner/notebook | `input/notebooks/{owner}/{notebook}/` |

ユーザーの明示slug、次にロード済みの `COMPETITION` を使う。両方空なら確認する。owner/slugだけで種別が曖昧なら依頼文やCLIのメタデータで確認する。「Notebookをinputへ追加」は実行出力、source/code/ipynb指定はソースを意味する。

## 事前確認

- 正規handle、指定version、容量、必要な公開範囲を確認する。CLIで取れない情報は未確認とする。
- Competitionは `competitions list --group entered -s <slug> -v` の全ページで完全一致を確認する。取得できない場合は未参加と断定しない。規約同意が必要ならrulesページでの手動同意を案内する。
- 保存先は `input/` 内に限定し、既存内容とversionを確認する。同じversionが揃っていれば再利用し、別versionや不明な内容を無断で上書きしない。
- ファイル一覧を全ページ取得して必要容量と空き容量を確認する。アーカイブと展開分の容量を見込む。

## 取得コマンド

`<tmp>` は一時ディレクトリ。メタデータはデータ用の展開先とは別に保存する。

```bash
kaggle competitions files <competition> -v
kaggle competitions download <competition> -p <tmp>
kaggle datasets files <owner/dataset> -v
kaggle datasets metadata <owner/dataset> -p <metadata-tmp>
kaggle datasets download <owner/dataset> -p <tmp>
kaggle models variations versions list <owner/model/framework/variation> -v
kaggle models variations versions files <owner/model/framework/variation/version> -v
kaggle models variations versions download <owner/model/framework/variation/version> -p <tmp>
kaggle kernels pull <owner/notebook/version> -m -p <tmp>
kaggle kernels files <owner/notebook/version> -v
kaggle kernels output <owner/notebook/version> -p <tmp>
```

Datasetのversion指定は、同梱の公式 `../kaggle-cli/references/official-docs/datasets.md` にある `owner/dataset/version` 形式を、使用中CLIで確認して使う。対応していなければ最新版にすり替えず、未対応として止める。Notebookはversionを省略すると最新版になる。Modelはversions listでversionを確定し、URLのframework表記を保存先にも使う。

## 展開・検証

1. ZIP/tarは一時領域へ取得する。事前検査のため取得時の `--unzip` / `--untar` は使わない。
2. 全entryについて絶対パス、`..`、展開先外へのパス、symlink/hardlink、特殊ファイルを検査する。安全な通常ファイルとディレクトリだけを空の一時展開先へ展開する。
3. 全ページのファイル一覧と、展開後の相対パス・件数・取得可能なサイズを比較する。Notebook出力も全ページを確認する。総件数だけで一致としない。
4. Notebookソースは `kernel-metadata.json` の `code_file` を使い、ipynbならJSONとcellsを検証する。メタデータとソースを一緒に保存する。ダウンロードしたコードは取得だけでは実行しない。
5. 一致したものだけ所定の保存先へ移す。欠落・失敗・version不明は成功にしない。一時ファイルは検証成功後に片付ける。
6. 正規handle、version、保存先、件数、容量、未確認事項を報告する。DatasetやNotebook出力は安定パスを保ち、version違いを混在させない。
