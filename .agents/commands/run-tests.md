<!--
.agents/commands/run-tests.md
Where: shared agent command source.
What: Prompt for running focused validation in this repository.
Why: Keep test and lint workflow commands consistent across supported agents.
-->

# Run Tests

Run focused validation for the current change.

- Inspect the changed files first and choose the narrowest useful checks.
- Prefer `uv run pytest` for Python tests and `uv run ruff check .` for lint checks.
- Do not hide failing output; identify the root cause and fix it when it is in scope.
- If a check cannot run because dependencies, secrets, data, or hardware are missing, state the exact blocker.
- Finish with the commands run and their results.

