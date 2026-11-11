---
name: nvidia-kaggle-skill
description: 指定Kaggle Notebookのソース・入力・上流出力を公式CLIで取得し、ローカルのパスと依存を対応付けて再現する。
---

Kaggleへのアクセスは `$kaggle-cli` の実行方法に従う。Docker Composeが使える場合は `docker compose run --rm kaggle kaggle ...` を優先し、本文中の `kaggle ...` / `uv run kaggle ...` も同じ引数で置き換える。Dockerが使えない場合だけuvへフォールバックする。

<!-- 配置: .agents/skills/nvidia-kaggle-skill。内容: 実行手順。目的: ローカル本文で再現する。 -->

# NVIDIA Kaggle Notebook Reproduction

以下は上流Kernel Setupの本文をこのテンプレートへ適合したもの。先に `$kaggle-cli` の認証と `../kaggle-cli/references/kernels.md` を読む。`kaggle` は `uv run kaggle` として実行する。

入力の永続保存先と取得・展開は `$import-kaggle-resource` の配置規則を優先する。下記の `{folder}/input` は既存入力へのローカル対応先とし、同じデータを重複取得しない。versionを指定できない依存は最新版へ黙って置換しない。取得した外部コードは内容確認後、依頼された範囲で実行する。

# Kernel Setup

Use this workflow to reproduce a public Kaggle kernel locally by downloading the notebook and all input data.

## Inputs

| Input | Required | Description |
|---|---|---|
| Kernel URL or `owner/kernel-slug/version` | Yes | Source Kaggle notebook and requested version. |
| Folder | No | Defaults to the kernel slug. |
| `skip-competition` flag | No | Skip competition data downloads. |
| Kaggle CLI auth or `KAGGLE_API_TOKEN` | Yes | Required for `kaggle kernels pull` and downloads. |

## Workspace

Create this structure:

```text
{folder}
├── input
├── tmp
├── scripts
│   └── __init__.py
└── working
```

Pull the kernel and metadata:

```bash
kaggle kernels pull {owner}/{kernel_slug}/{version} -p {folder}/working/
kaggle kernels pull {owner}/{kernel_slug}/{version} -m -p {folder}/tmp/
cat {folder}/tmp/kernel-metadata.json
```

Use `kernel-metadata.json` as the authoritative input source list:

- `dataset_sources`
- `competition_sources`
- `model_sources`
- `kernel_sources`

## Downloads

Download all missing sources under `{folder}/input/`. Inform the user which items will be downloaded and which will be skipped.

Competitions:

```bash
kaggle competitions download {competition} -p {folder}/input/
```

Skip if `skip-competition` was specified. Extract archives and remove large zip files after successful extraction.

Datasets: 指定versionへの対応をimport手順で確認する。versionが特定できない場合は再現未完了とし、最新版を代用しない。

```bash
kaggle datasets download {owner}/{dataset}/{dataset_version} -p {folder}/input/{dataset}
```

Models:

```bash
kaggle models instances versions download {owner}/{model}/{framework}/{variation}/{version} -p {folder}/input/{model}
```

Kernel outputs:

```bash
kaggle kernels output {owner}/{kernel}/{version} -p {folder}/input/{kernel}
```

If kernel output downloads create nested folders, flatten only when the notebook expects the files at a higher path level.

## Post-processing

- If a `pm-{digits}-at-{date}` folder contains `input_requirements.txt`, create `{folder}/requirements.txt` without `pip install` prefixes and move that folder to `{folder}/tmp/wheels`.
- Move dependency script folders containing only a `.log` and `.py` into `{folder}/scripts`, renaming folders as needed for Python imports.
- Scan the notebook for `/kaggle/input/...` paths and create `{folder}/create_symlinks.sh` to map those paths to local downloads.
- Do not run `python -m pip install -r requirements.txt`; just document that command.

## README

Write `{folder}/README.md` with:

- input source summary table: source type, slug, local path, symlink path, status
- symlink script instructions
- dependency install instructions if `requirements.txt` exists
- import guidance for copied utility scripts
- metadata fields: `language`, `kernel_type`, `is_private`, `enable_gpu`, `enable_tpu`, `enable_internet`
- GPU hint to check torch and CUDA versions when GPU is enabled
- downloaded file tree summary

## Workflow-Specific Troubleshooting

認証とコマンド仕様は `$kaggle-cli`、規約・容量・展開・検証は `$import-kaggle-resource` のローカル本文を読む。

| Symptom | Action |
|---|---|
| 403 on competition data | Direct the user to accept rules at `https://www.kaggle.com/competitions/{competition}/rules`, then retry the specific download. |
| 403 on dataset | Search with `kaggle datasets list -s "<name>"`; the owner may differ from the notebook metadata. |
| Notebook paths cannot be mapped | Record the unresolved paths and uncertainty in the generated README. |
