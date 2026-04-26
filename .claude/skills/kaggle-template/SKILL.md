---
name: kaggle-template
description: Use when modifying, running, or reviewing Kaggle experiments, dataset upload/download workflows, Hydra configs, or environment setup in this repository.
---

<!--
.agents/skills/kaggle-template/SKILL.md
Where: shared agent skill source.
What: Repository-specific workflow guidance for the Kaggle experiment template.
Why: Let supported agents load detailed project workflow only when relevant.
-->

# Kaggle Template Skill

Use this skill for changes involving experiments, competition data, outputs, Docker/uv setup, or Kaggle dataset tooling.

## Workflow

- Read `README.md`, `Makefile`, `pyproject.toml`, and the touched experiment/tool files before editing.
- Keep changes inside the narrowest relevant folder, usually `experiments/<major_exp_name>/`, `tools/`, `utils/`, or docs.
- Preserve the major/minor experiment convention: code lives under `experiments/<major_exp_name>/`, configs live under `exp/<minor_exp_name>.yaml`.
- Keep reproducible settings in config files or dataclasses. Avoid hidden constants in training/runtime code.
- Do not read or write `.env`, `.netrc`, Kaggle credentials, or private tokens unless the user explicitly asks.

## Useful Commands

- `make uv-setup`
- `uv run experiments/exp000_sample/run.py exp=001`
- `uv run pytest`
- `uv run ruff check .`
- `make bash`
- `make jupyter`

## Verification

- Run the smallest check that proves the change.
- For code changes, add or update at least one focused test when practical.
- For workflow/config changes, validate JSON/TOML/YAML syntax or run the documented command if local data and credentials are available.
- Report missing local prerequisites explicitly instead of guessing.

