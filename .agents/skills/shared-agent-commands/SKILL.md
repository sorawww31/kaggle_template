---
name: shared-agent-commands
description: Use when a user asks for a repository shared command from .agents/commands, especially in clients without documented project custom slash-command support.
---

<!--
.agents/skills/shared-agent-commands/SKILL.md
Where: shared agent skill source.
What: Adapter guidance for command prompts stored under .agents/commands.
Why: Keep command workflows usable in agents that do not natively load this repository's command files.
-->

# Shared Agent Commands

The canonical command prompts live in `.agents/commands/*.md`.

## How To Use

- Match the user request to a command filename, for example `review`, `run-tests`, or `kaggle-experiment`.
- Read the matching `.agents/commands/<name>.md` file.
- Follow that command prompt directly, treating any user text after the command name as scope or arguments.
- If no command matches, ask for clarification or proceed with normal repository instructions.

## Current Commands

- `kaggle-experiment`: make or run a Kaggle experiment change.
- `review`: review the current worktree or selected files.
- `run-tests`: run focused validation for the current change.

