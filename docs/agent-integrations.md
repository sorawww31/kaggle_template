<!--
docs/agent-integrations.md
Where: project documentation.
What: Explains shared AI-agent instructions, skills, commands, and Kaggle authentication.
Why: Help contributors keep Codex, Cursor, Claude Code, GitHub Copilot, and Gemini aligned.
-->

# AI Agent Integrations

このリポジトリでは、AIエージェント向けの指示・skill・commandを共通ソースで管理します。

## 共通ソース

- `AGENTS.md`: 全エージェントの共通作業方針。
- `.agents/skills/*/SKILL.md`: 共通 Agent Skills。
- `.agents/commands/*.md`: 共通 command prompt。

## Kaggle接続

`KAGGLE_API_TOKEN` を環境変数として設定してください。トークン値は設定ファイルに直接書かないでください。

Kaggle操作はCLIを使い、このリポジトリにはKaggle MCPの接続設定を配置しません。
一部の既存skillにはMCPを前提とした手順が残っているため、CLIへの手順移行が必要です。

## Commands

共通commandは `.agents/commands/*.md` に書きます。ツール固有のcommandを配置する場合は次のパスを使います。

| Tool | Native path |
| --- | --- |
| Claude Code | `.claude/commands/*.md` |
| Cursor | `.cursor/commands/*.md` |
| GitHub Copilot | `.github/prompts/*.prompt.md` |
| Gemini CLI | `.gemini/commands/*.toml` |

Codex CLIについては、公式ドキュメント上でプロジェクトローカルのcustom slash commandファイルが確認できませんでした。そのため、Codexでは `.agents/commands` を共通ソースとして参照するか、`shared-agent-commands` skillを使って同じpromptを実行します。

## Skills

共通skillは `.agents/skills/*/SKILL.md` に集約します。Skillの追加・更新・削除はこのディレクトリで行います。
