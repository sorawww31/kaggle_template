<!--
.agents/commands/review.md
Where: shared agent command source.
What: Prompt for reviewing the current worktree or selected files.
Why: Keep review criteria consistent across supported agents.
-->

# Review Current Work

Review the current worktree or selected files with a code-review stance.

- Lead with concrete findings ordered by severity.
- Focus on bugs, behavioral regressions, missing tests, secret leakage, and unsafe workflow changes.
- Ground every finding in file and line references when possible.
- Keep summaries secondary to findings.
- If no issues are found, say that directly and name any remaining test gaps.

