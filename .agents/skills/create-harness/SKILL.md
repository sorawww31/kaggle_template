---
name: create-harness
description: 既存のkaggle-environmentsゲームに対して、LLMが対戦するためのハーネス、プロンプト、応答パーサーを作成・更新・検証する。
---

# Create an LLM Harness

<!-- Where: .agents/skills/create-harness. What: ゲーム用LLMハーネス。 Why: モデルの出力をゲームの合法手に変換する。 -->

1. 対象ゲームと作業先を特定する。使用中の `kaggle-environments` のインタープリター、観測データ、合法手、既存ハーネスと `core_harness.py` を読む。
2. 列挙型、自由入力型、混合型のどれかを確認し、バージョンに合う `get_legal_moves`、`make_prompt`、`parse_response` を実装する。
3. 合法手、形式崩れ、不正手、曖昧な応答をテストする。対象ゲームのローカル対戦で手が適用されることを確認する。
4. ゲーム自体のOpenSpiel登録や観測プロキシの追加は `$onboard-open-spiel-game` を使う。

[上流の実装例とテスト手順](https://github.com/Kaggle/kaggle-environments/blob/998b15c14b1b77aafb1843bed63858455035affd/.agents/skills/create-harness/SKILL.md)
