<!--
docs/agent-integrations.md
Where: project documentation.
What: Explains shared AI-agent instructions, skills, commands, and MCP settings.
Why: Help contributors keep Codex, Cursor, Claude Code, GitHub Copilot, and Gemini aligned.
-->

# AI Agent Integrations

このリポジトリでは、AIエージェント向けの指示・skill・commandを共通ソースで管理します。

## 共通ソース

- `AGENTS.md`: 全エージェントの共通作業方針。
- `.agents/skills/*/SKILL.md`: 共通 Agent Skills。
- `.agents/commands/*.md`: 共通 command prompt。

## MCP設定

`KAGGLE_API_TOKEN` を環境変数として設定してください。トークン値は設定ファイルに直接書かないでください。

| Tool | File | Notes |
| --- | --- | --- |
| Claude Code | `.mcp.json` | Project-scoped MCP設定。 |
| Codex | `.codex/config.toml` | Kaggle MCPを `bearer_token_env_var` で参照。 |
| Cursor | `.cursor/mcp.json` | `${env:KAGGLE_API_TOKEN}` を参照。 |
| GitHub Copilot in VS Code | `.vscode/mcp.json` | VS Code input promptでトークンを受け取る。 |
| Gemini CLI | `.gemini/settings.json` | `httpUrl` と `headers` でKaggle MCPを参照。 |

GitHub Copilot CLIのユーザー単位MCPは `~/.copilot/mcp-config.json` です。これは個人環境の設定なので、このリポジトリにはコミットしません。

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
