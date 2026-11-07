---
name: onboard-open-spiel-game
description: OpenSpielのゲームをkaggle-environmentsに追加する。ゲーム登録、観測プロキシ、カスタムゲームの実装を依頼されたときに使う。
---

# Onboard an OpenSpiel Game

<!-- Where: .agents/skills/onboard-open-spiel-game. What: OpenSpielゲームの登録。 Why: Kaggle環境からゲームを使えるようにする。 -->

1. ユーザーが変更する `kaggle-environments` のソース作業先を特定し、OpenSpiel側にゲームが存在するか確認する。
2. 既存ゲームは共有インタープリターへ登録し、必要に応じて構造化観測を返すプロキシを追加する。独自ゲームは対象バージョンの既存実装に沿って作る。
3. 登録名で `make()` が通ること、観測と合法手が正しいこと、ローカル対戦が終局することを確かめる。
4. 登録済みゲームをプレイするLLMの実装は `$create-harness` を使う。

[上流の登録・プロキシ実装手順](https://github.com/Kaggle/kaggle-environments/blob/998b15c14b1b77aafb1843bed63858455035affd/.agents/skills/onboard-open-spiel-game/SKILL.md)
