<!--
.agents/commands/kaggle-experiment.md
Where: shared agent command source.
What: Prompt for making or running a Kaggle experiment in this template.
Why: Keep experiment workflow commands consistent across supported agents.
-->

# Kaggle Experiment

Handle the requested Kaggle experiment change in this repository.

- Read `README.md`, `Makefile`, and the relevant `experiments/<exp>/` files before editing.
- Keep the experiment self-contained under its major experiment directory.
- Put tuneable runtime values in the experiment config, not as hidden constants in code.
- Prefer `uv run experiments/exp000_sample/run.py exp=001` style execution for local verification.
- Run focused tests or explain why the requested experiment cannot be executed locally.
- Summarize changed files, verification, and any remaining risks.

