# Colabで進める卒業研究

プロジェクト名は **Hexapod Transport RL** です。研究題目の例は「六足ロボットの強化学習による協調物資運搬」です。
名前・Pythonパッケージ・実行コマンドを機体の固有名から切り離しました。
共有先は [hashimoto-robotics-lab/hexapod-transport-rl](https://github.com/hashimoto-robotics-lab/hexapod-transport-rl) です。
ノートブックは実行手順、リポジトリはコード・形状・モデルを管理します。長い埋め込みデータはありません。

## 最初に理解すること

研究の目的は、学習済みの歩行能力を使い、複数機が荷物を協調して運ぶ方法を学ぶことです。
ノートブックの冒頭で研究の動機を読み、準備後のStep 3で1台へ速度コマンドを送り、歩行を動画で確かめます。
その後、荷物・目標・相手の観測から2台へのコマンドを決める運搬方策をMAPPOで学習します。
固定する歩行モデルと、今回学習する運搬方策を分けて理解してください。
歩行体験では `WalkingSimulation` の速度指令APIを使います。同じAPIで1台と4台を動かす例を実行します。
指定したコマンドは、運搬学習の教師には使いません。

## GitHubからColabで開く

1. READMEの [Open in Colab](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/hexapod_transport_rl_colab.ipynb) を開く。
2. Googleアカウントでログインし、CPUランタイムを使う。
3. 上から順に実行する。Step 1でコード・形状・モデルを自動取得する。

**GitHubへのログイン、リポジトリへの招待、アクセストークンは不要**です。
同じランタイムでStep 1を再実行すると、取得済みの教材を使い、学生の編集や学習結果を上書きしません。

## 教材の版と共有

教材はこのリポジトリのルートで管理します。教員が変更したコードとノートブックをcommitして共有してください。
`runs/` と `.venv/` はGitで共有しません。

学生用ノートブックは `main` の教材を取得します。版を選ぶ設定はありません。
`experiment.json` に取得したコミットSHAと実行時のソース・資産のハッシュを記録します。
コードを編集した場合は、別の実験名で学習してください。
更新した教材を取得する際は、新しいランタイムでStep 1から始めます。
論文の再現用に特定の版を配布する場合は、教員側で教材の取得先を固定してください。

## 学生が編集するコード

歩行体験では `set_velocity()` の速度と `run_for()` の時間を変えます。
運搬実験では `lesson.configure()` の実験名・mode・独立世界数・seed・押す方策の初期化を変えます。
学習と評価の流れは、次の呼び出しで表します。

```python
pusher = lesson.prepare_pusher()
navigator = lesson.train_navigation(pusher)
adapted_pusher = lesson.train_handover(pusher)
model = lesson.select_model(navigator, adapted_pusher)
test_results = lesson.evaluate(model)
lesson.show_results()
lesson.show_learned_video(model)
lesson.save_results()
```

インストール、設定の検証、別プロセスの起動、モデルの再開、記録・保存は
`tools/colab_runtime.py` が担当します。学習曲線・CSVの生成は `tools/colab_analysis.py` にあります。
学生は最初からこれらの内部実装を読む必要はありません。
歩行と環境APIはセル内で直接実行します。コードを文字列にして補助APIへ渡す必要はありません。
環境APIの `reset()` / `step()` と歩行指令は、ノートブックにコードを残しています。

## 上から順に実行する

| 手順 | 学生が確認すること | 生成物 |
|---|---|---|
| 1. GitHubから取得 | 公開されたコード・形状・モデルを取得する | プロジェクトとコミットSHA |
| 2. ライブラリ準備 | 直接importするライブラリと、固定した並列学習環境 | セルと学習の実行環境 |
| 3. 歩行コマンドの体験 | 速度指令APIで1台と4台の歩行を試す | 歩行動画 |
| 4. 実験設定 | quick/research、乱数seed、独立世界数 | `experiment.json` |
| 5. API確認 | 観測、行動、チーム報酬、終了条件 | 入出力の表示 |
| 6. 参考方策の再生 | 学習済みの運搬動作と部位別接触 | 参考動画・評価JSON |
| 7. 押す方策の準備 | 既存の報酬学習モデルを使うか、新規学習するか | 初期pusher |
| 8. 回り込み学習 | カリキュラムとMAPPOの更新 | navigatorと指標 |
| 9. 引き継ぎ位置への適応 | 押す方策の追加強化学習 | adapted pusher |
| 10. 検証用seedでモデル選定 | 回り込み成功と運搬成功を区別する | 結合モデルと選定記録 |
| 11. 未使用seedでテスト | 初期配置を変えて性能を確認する | 前方・側方評価JSON |
| 12. 図表 | 学習曲線と成功率・接触・誤差 | PNG、PDF、CSV |
| 13. 新しい方策の録画 | 参考動画と自分の結果を区別する | 自分の方策の動画 |
| 14. 保存 | 設定・重み・ログ・動画を持ち帰る | 結果ZIP |

ColabのCPUランタイムを使います。GPUによる学習高速化はこの実装には組み込んでいません。
歩行と環境APIは、ColabのPythonで通常の `import` とAPI呼び出しを使って操作します。
Python 3.12・3.13に対応し、既存のNumPy・PyTorchが対応範囲内なら、そのまま利用します。
ウィンドウなしの描画設定は、MuJoCoをimportする前に準備APIが行います。
歩行は `record=True` でRGBフレームを集め、`media.show_video(sim.frames, fps=5)` で表示します。
参考方策と学習後の運搬動画もmediapyを使います。FFmpegは同梱のimageio-ffmpegから自動設定します。
表示方法は [MuJoCo公式チュートリアル](https://github.com/google-deepmind/mujoco/blob/main/python/tutorial.ipynb) と共通です。
並列学習と運搬評価は、uvで用意したPython 3.12とlockfileの専用環境で実行します。
セルで使う版は `runtime.json`、学習で使う版は `training_runtime.json` に分けて記録します。
MuJoCoの動画はOSMesaによるソフトウェア描画、標準640×480・5 fpsにして描画負荷を下げています。
影・反射を省く `--fast-video` も使用します。
物理更新は200 Hzのままです。
[uvのPython管理](https://docs.astral.sh/uv/guides/install-python/)、[MuJoCoのPython描画](https://mujoco.readthedocs.io/en/stable/python.html)

## 短い確認と本学習

`quick` は各方策を数回だけ更新し、最後まで実行できることを確認します。
新しく学習する方策の成功は期待せず、卒論の成功率として報告しません。評価時間も短縮します。
参考方策の動画だけは、事前に学習された成功モデルの運搬を表示します。

`research` では、回り込み409,600、引き継ぎ適応153,600チームステップを目安にします。
並列世界数に合わせて反復数を計算します。元の16並列の実験と収集・更新のバッチ構成が変わるため、
同じseedでも保存済みモデルと同じ重み・成功率になるとは限りません。
新規の押す方策を学習する場合は追加で2段階の押す学習を実行します。

ノートブックでは、実測した学習速度から本学習に必要な時間の概算を表示します。
時間はCPUや並列数、検証頻度で変わります。最初は `quick` で一巡し、本学習には新しい実験名を指定してください。

## 保存と再開

最初の一巡では保存用の追加設定は不要です。最後の `lesson.save_results()` でZIPを保存します。
長い本学習でDriveへ途中保存したいときだけ、Step 4の `lesson.configure()` に
`save_to_drive=True` を追加してください。Google Driveへの接続後、途中の重みとログを定期的にコピーします。
シミュレーションはColab内で実行し、Driveはバックアップに使います。
同じ実験名・設定でノートブックを再実行すると、保存済みの最新checkpointから未完了分を追加学習します。
中断後に再開する場合、すでに保存した更新は引き継ぎますが、中断時のエピソード・乱数列を完全に復元する再開ではありません。

Driveを使わない場合は最終セルのZIPを保存してください。
Step 4の `lesson.configure()` に `restore_results=True` を追加すると、保存したZIPをアップロードして再開できます。
結果ZIPには使用したソース・ロボット資産のスナップショット、SHA-256、ライブラリの版も含めます。
乱数seed・実験条件を変える場合は `lesson.configure()` の `name` を変更し、別の実験として保存します。

Colabのランタイムは削除されることがあり、資源や実行時間にも変動があります。
GitHubで教材を共有していても、ランタイム内の未保存の学習結果は失われます。
[Colab公式FAQ](https://research.google.com/colaboratory/faq.html)

## 卒論で扱う範囲

固定した歩行方策の上に、協調運搬の上位方策を学習する研究です。
脚の歩行そのものを一から学習する研究とは区別して、論文で構成を説明してください。

検証用seedでモデルを選び、別のテスト用seedで性能を報告します。
記録する項目は、運搬成功率、回り込み成功率、位置・向きの誤差、胴体接触、機体同士の接触、転倒、所要時間です。
成功率にはWilson法の95%区間も表示します。これは初期配置の試行に対する区間であり、学習seed間のばらつきを表しません。
本評価は異なる学習seedで少なくとも3回繰り返し、平均とばらつきも報告することを推奨します。

初期角度、初期配置、報酬、カリキュラム、学習量を変える比較実験ができます。
まず一度に1つの条件を変え、同じテストseed集合で比較してください。
研究で変更する定義は [コードガイド](code-guide.md) にあります。
学習曲線のtraining successは、その時点のカリキュラムの初期配置での結果です。運搬全体のテスト成功率とは別の指標です。
今回の回り込み環境は2台、平坦な床、T字物体1個、他の障害物なし、シミュレータから位置・姿勢を取得する条件です。
台数・障害物・観測誤差・実機への拡張は、別に環境を拡張して検証する課題として扱います。
