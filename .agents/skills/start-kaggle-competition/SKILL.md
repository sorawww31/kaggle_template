---
name: start-kaggle-competition
description: 環境変数COMPETITIONで指定されたKaggleコンペの参加状態をKaggle MCPで確認し、全入力データをリポジトリの./input/competitions/{COMPETITION}/へ取得する。ユーザーが「コンペを開始」「Kaggleコンペに参加」「COMPETITIONのデータを取得」「inputへコンペデータを保存」などを依頼したときに使う。
---

<!--
SKILL.md
Where: .agents/skills/start-kaggle-competition.
What: Start the competition selected by COMPETITION using Kaggle MCP.
Why: Keep participation checks and input data placement consistent.
-->

# Start Kaggle Competition

Kaggle MCPで参加状態を確認し、コンペの全データを`./input/competitions/{COMPETITION}/`へ保存する。
Kaggle CLI、Kaggle Python API、KaggleHub、`tools/download_dataset.py`は使わない。

## 手順

1. リポジトリルートで`COMPETITION`を取得する。
   - プロセス環境の値だけを使い、空なら停止して設定を依頼する。
   - slugを推測したり、`.env.example`の値で代用したりしない。
2. Kaggle MCPの`get_competition(competitionName=COMPETITION)`を呼ぶ。
   - 対象slugと`user_has_entered`を確認する。
   - 認証エラーならKaggle MCPの認証を案内し、認証後に再実行する。
3. `user_has_entered == true`なら参加済みとして続行する。
   - `false`ならコンペURLを提示して停止する。
   - Kaggle MCPに参加・規約同意ツールがない場合は参加済みと偽らず、ユーザーへKaggle上での規約同意を依頼する。完了後に手順2を再実行する。
4. `get_competition_data_files_summary`でファイル数と合計容量を確認する。
   - 容量と空きディスクを報告する。
   - `./input/competitions/{COMPETITION}/`に既存データがある場合は上書きせず、衝突内容を確認する。
5. `download_competition_data_files(competitionName=COMPETITION)`を呼ぶ。
   - 返された一時URLをKaggle MCPの成果物として使う。
   - URLを回答やログへ転記しない。
6. 一時URLからZIPをローカルの一時領域へ取得し、`./input/competitions/{COMPETITION}/`へ展開する。
   - `curl --fail --location --retry 3`など、失敗を検知できる方法を使う。
   - 展開時は既存ファイルを上書きしない。
   - Kaggleへのアクセスに別のCLI、API、ライブラリを使わない。
7. 展開後に検証する。
   - 保存先配下のローカルファイル数をMCPの`total_file_count`と比較する。
   - 主要ディレクトリ、保存容量、保存先`./input/competitions/{COMPETITION}/`を報告する。
   - 不一致なら成功扱いにせず、欠落数と再試行方針を示す。

## 完了条件

- `COMPETITION`とKaggle MCPで確認したslugが一致する。
- `user_has_entered == true`である。
- MCPが示した全ファイルが`./input/competitions/{COMPETITION}/`に存在する。
- 既存ファイルやユーザーの変更を破壊していない。
