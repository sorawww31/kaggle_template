---
name: kaggle-standardized-agent-exam
description: KaggleのStandardized Agent Examを明示的に受験・提出・採点確認する依頼に使う。
---

# Kaggle Standardized Agent Exam

<!-- Where: .agents/skills/kaggle-standardized-agent-exam. What: Agent Examの受験。 Why: 評価タスク作成と公式試験の参加を分ける。 -->

1. 上流手順で利用条件、試験時間、提出回数制限を確認する。登録や試験開始はユーザーが明示的に依頼した範囲で進める。
2. 既存の試験用IDとキーを再利用する。新しいキーは権限を制限して保管し、値をログや応答に含めない。キーの送信先は `https://www.kaggle.com/api/v1/` に限定する。
3. 上流の手順に従って試験の開始、解答、提出、採点確認を行う。404が返った場合は操作を停止する。認証失敗時に既存の試験用IDやキーを削除・再登録する前にユーザーへ確認する。
4. 試験ID、提出状態、確認できたスコアを報告する。

[上流のAPI手順](https://github.com/Kaggle/kaggle-benchmarks/blob/5ffefc57696e26b172fc3626392731ebecbb1195/skills/kaggle-standardized-agent-exam/SKILL.md)
