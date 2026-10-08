# MuJoCo Warpで物理と学習をGPUへ移す

現在の課題は[等長2棒・T→ゴール→ロボットからの回り込み](pose-task.md)です。
学習・検証ではGPU物理を使い、最終評価は通常のCPU MuJoCoでも行います。

ColabではT4 GPUを選び、準備セルから順に実行します。準備セルがCUDA版PyTorchを維持し、
`mujoco-warp==3.11.0`・`warp-lang==1.15.0`をインストールします。
`TorchRLTransportEnv(config, num_envs=256, backend="auto")`はGPUがあればWarpを選びます。
GPUなしの場合はCPUを選び、教材は2世界並列へ切り替えます。

| 処理 | Warpでの実行先 |
|---|---|
| 元の関節・DCモーター・床・Tの物理 | GPU |
| 固定歩行モデル 25 Hz・電圧フィードバック 200 Hz | GPU |
| 毎物理ステップの脚・足・胴体の接触計測 | GPU |
| リセット・観測・報酬・終了判定 | GPU |
| Collectorの方策・GAE・MAPPO更新 | GPU |
| 別seedのカリキュラム検証 | 独立したGPU環境 |
| 最終Gym評価・25 fps録画 | CPU物理・GPU描画（GPUなしはOSMesa） |

CPUとWarpは同じロボット資産・Tの形・モーター設定・歩行モデルを使います。
Warpの物理演算はfloat32なので、CPU MuJoCoの軌跡とは完全には一致しません。
保存モデルをCPUでも評価して、普通のGym環境へ方策を戻せるかを確認します。
GPUの学習速度から、CPUへの転送後の成功率が同じと仮定しません。

## 学生が変更する部分

`PoseRewardWeights`の係数をノートブックで変更します。
式を変える場合は `pose_rewards.py`の `pose_reward_terms()` がCPU・GPUの共通定義です。
報酬の変更にWarpカーネルの編集は不要です。行動の教師・デモの軌跡・別の押すモデルは使いません。

`NUM_ENVS`は独立した世界の数で、ロボット台数ではありません。
GPUは256世界・horizon 64で、1回に16,384チームステップを収集します。
ミニバッチ512・4 epochs・学習率3e-4、gamma 0.995・GAE lambda 0.99です。
世界数だけ増やすと更新回数が減るため、horizon・総予算も合わせて設定します。
各軸の7速度候補（停止を含む）から指令を選ぶ確率を学びます。探索ノイズの段階別調整はありません。

最初の環境作成時にGPUカーネルをコンパイルしてウォームアップし、高位ステップ全体をCUDA graphへ収録します。
ウォームアップの物理状態は学習の最初のresetで捨てます。キャッシュ済みの2回目と初回の時間は区別します。
毎ステップのCPUへの状態転送はありません。明示的な `envs.render(world=0)` だけ、選んだ世界の実際の状態をコピーして描画します。
mediapyの `media.show_video()` でそのフレーム列を表示できます。

終了したレーンだけ歩行履歴・状態・成功維持時間をリセットします。
新しい段階の目標距離・角度・許容誤差は次のresetから適用し、他レーンの状態は保持します。
接触・制約バッファの不足と非有限値はGPU上で記録し、ロールアウト境界で検査します。
途中のresetによって異常の記録が消えることを防ぎます。
4台の実際のGPU更新・保存・CPU読み込みもテストしますが、4台の運搬性能を2台の結果から保証しません。

## 計測を読む際の注意

学習CSVの時間は収集・GAE・MAPPO更新・カリキュラム検証を含みます。
最初のGPU環境作成・カーネルコンパイルと、最後のCPU評価・録画は別に測ります。
ローカルRTX A6000の値をGoogleホストのColab/T4の実測とは扱いません。
GPUを共有する他の処理によっても時間は変わります。

改善後の新規学習は **3,145,728チームステップ・30.1分（約1,743ステップ/秒）** でした。
学習は2台、ローカルRTX A6000、256世界・horizon 64です。環境初期化はキャッシュ利用で約1.9秒でした。
未使用50配置でCPU MuJoCo・Warpとも44件成功しました。同じ成功数でも成功した配置や軌跡は完全には一致しません。
[全設定・試行](results/goal_side_improved_20261008.json)と[学習CSV](results/goal_side_improved_20261008_progress.csv)を公開しています。
Colab/T4の時間は学生の実行CSVで計測してください。この30.1分は前の試行錯誤を含みません。

初回の新配置学習は約2分49秒・0/12成功で、成功する予算ではありませんでした。
[当時の全記録](results/goal_side_training_20261008.json)を保存しています。
旧形状・近傍開始の[Warp測定記録](results/warp_training_20261008.json)は別の課題です。
同記録の3.8分／条件やCPUとの約4倍の比較を、新しい回り込み課題の時間やColab/T4の速度として使わないでください。

動画は普通のGym APIの `render_mode="rgb_array_list"` で、実際の歩行制御周期25 Hzの状態を描画します。
上位方策は5 Hzのまま、補間・複製や追加の物理操作はしません。
ColabでGPUがあればEGL、なければOSMesaを使います。描画は学習ループから外します。

実装は[MuJoCo Warp公式文書](https://mujoco.readthedocs.io/en/3.11.0/mjwarp/)の
並列世界・接触容量・CUDA graphと、[WarpのPyTorch連携](https://nvidia.github.io/warp/latest/user_guide/interoperability/pytorch.html)を使います。
