# GPUの実験ループ

`train_mappo()`はCUDAがあればMuJoCo Warpを使います。
物理・固定歩行・モーター・接触・観測・面積・報酬・TorchRL更新をGPUで実行します。
学生用の初期構成は512世界、horizon 32、ミニバッチ1024、2 epochsです。

上位の行動は各機の世界XY目標で、自分の位置を基準に連続分布を探索します。
固定歩行は25 Hz、モーター・物理は200 Hzです。solverと接触容量は元の設定です。
GPUからCPUへの毎ステップの状態転送や、物資への外力は使いません。

実装は[MuJoCo Warp公式文書](https://mujoco.readthedocs.io/en/3.11.0/mjwarp/)の並列世界・CUDA graphを使います。
ロールアウトごとに接触容量不足と非有限値を確認し、部分resetは他の世界の歩行履歴を保ちます。
float32のWarpとCPU MuJoCoは初期状態を揃えて評価しますが、軌跡・成功数は一致するとは限りません。

学習時間、評価時間、動画作成時間を別に保存します。学習は360秒を目安に終了します。
初回のコンパイル・GPUの共有状況により時間は変わります。ローカルA6000をColab/T4の実測とは扱いません。
[新しい実測](results/compact_experiments_20261013.json)と、[以前の記録](archive/warp-training.md)は別の結果です。

学生用の運搬ではXY目標を0.8秒保ちます（`high_level_decimation=20`）。標準Gym設定は0.2秒です。
歩行25 Hz・物理200 Hzは同じで、記録する動画は実際の25 Hzフレームです。
