# 六足1台の学生用API

報酬・観測・行動を変更し、同じ六足機体で比較実験するための窓口です。
元の18関節・固定歩行モデル・モーターモデル・MuJoCo/Warp・TorchRLの`ClipPPOLoss`とGAEを引き継ぎます。
学生は`CylinderPushTask`と小さなPython関数を編集します。

標準課題は、直径50 cmの円柱を目標位置まで押す問題です。
開始位置は円柱から0.65〜1.00 mの円環、機体の水平な向きは全周、目標は距離0.30〜0.60 m・全方向でランダムです。
成功は位置誤差5 cm未満・円柱速度0.06 m/s未満を0.4秒保持、制限時間40秒です。
物理200 Hz・歩行25 Hz・上位行動5 Hzを使います。報酬や方策の観測を変えても成功判定は同じです。

## Colabから学ぶ

[Step 0〜13のノートブック](../notebooks/student_api_colab.ipynb)を用意しています。
最初のセルでGitHubから教材をcloneします。開き方と取得するブランチの指定は[README](../README.md#最初はcolabで学ぶ)を参照してください。
Step 0〜7では、実際に学習した標準条件と報酬1項目の比較を読みます。
Step 8〜10は仮説の記述から共通の運搬モデルでの追加学習・複数seed比較・図の作成、Step 11は独自の報酬関数を追加して学習する例です。
[実測結果](../results/reward_study/discussion.md)と[実験の手引き](experiment-guide.md)も同梱しています。
Colabに入っているPyTorchを保持するので、準備のためにPyTorchを入れ替える必要はありません。

## 最初に動かす

リポジトリ直下で実行します。通常の`.venv`はCPU版です。

```bash
uv sync --locked
PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python tools/train_student.py \
  examples/student/baseline.py --output runs/student_smoke --smoke
```

`--smoke`は2並列・16ステップ・制限時間0.4秒の接続確認です。
学習・保存・再読込・評価が実行できるかを確認します。この結果を運搬の学習性能として使いません。
出力先は毎回新しい名前にします。既存の重みを上書きしません。

安定した報酬実験の設定と初期モデルは[学習方法の説明](training-method.md)を参照してください。
以下のAPIの既定動作は新規学習です。

本学習は[READMEのGPU環境](../README.md#ローカルで実行する)で行います。

```bash
MUJOCO_GL=egl python tools/run_reward_study.py \
  --condition baseline --seeds 42 --output runs/student_stable/main
```

GPUの指定が必要なら`CUDA_VISIBLE_DEVICES=1`などを環境変数に追加します。
動画のヘッドレス描画に使うGPUは、必要に応じて`MUJOCO_EGL_DEVICE_ID`で指定します。

## Pythonから使う

以下は観測・行動も変更する新規学習の例です。完成例と同じ追加学習は[READMEの例](../README.md#安定した運搬を出発点にする)を使います。

```python
from hexapod_transport_rl.student import (
    CylinderPushTask, DiscreteDirections, train,
)

task = CylinderPushTask(
    observations=("robot_to_cargo", "cargo_to_goal", "robot_heading"),
    actions=DiscreteDirections(count=4, include_stop=True),
    rewards={
        "goal_progress": 20.0,
        "time": -0.02,
        "success": 10.0,
        "failure": -10.0,
    },
)
run = train(task, "runs/my_experiment", seed=42, steps=655_360)
report = run.evaluate(episodes=256, seed=240000)
run.video("runs/my_experiment/episode.mp4")
```

観測数・離散行動数に合わせてActor/Criticの入力・出力層を自動で構成します。
ActorとCriticはともに、選んだ観測だけを使います。PPOやTensorDictを直接編集する必要はありません。
`checkpoint_selection="best"`（既定）では、`run.checkpoint`は検証で選んだ`best.pt`です。
`checkpoint_selection="latest"`では、最後の更新の`latest.pt`です。`run.evaluate()`、`run.video()`、`compare()`もこの指定に従います。
完成例の追加学習は`latest`を使い、更新前の初期方策へ差し替えません。
最良の検証結果が初期方策だった場合は、`best.pt`に初期方策が残ります。

`steps`は全並列世界の上位行動数の合計です。`num_envs * horizon`の正の整数倍にします。
512並列・horizon 64なら、655,360ステップは20ロールアウトです。
`max_seconds`の既定値は`None`です。卒論の比較では同じステップ数を使い、時間上限による打切りは別に記録します。
PPOの設定を変える場合は、既存の`PPOSettings`を`settings=`に渡せます。

## サンプルの使い分け

| ファイル | 変更するもの |
|---|---|
| [baseline.py](../examples/student/baseline.py) | 安定した運搬モデルと同じ14入力・停止＋8方向・標準報酬・回転対称化 |
| [reward_study.py](../examples/student/reward_study.py) | 完成例の2条件・3 seed・学習量・評価条件 |
| [reward_only.py](../examples/student/reward_only.py) | 目標付近で円柱の速度を抑える独自報酬を追加 |
| [observations.py](../examples/student/observations.py) | 位置・向きの6入力へ変更。位置の正規化は基準と共通 |
| [actions.py](../examples/student/actions.py) | 停止＋4方向へ変更。16方向・連続行動の例も記載 |
| [custom_observation.py](../examples/student/custom_observation.py) | 自作のスカラー観測を追加 |
| [pretrained.py](../examples/student/pretrained.py) | 既存の選択済みモデルと同じ回転対称化を有効にする |
| [compare.py](../examples/student/compare.py) | 観測・行動を変える新規学習の接続例。追加学習の高成功率は引き継がない |

例えば報酬のサンプルは、次のように実行します。

```bash
PYTHONPATH=src MUJOCO_GL=egl python tools/train_student.py \
  examples/student/reward_only.py --output runs/student_reward_s42 \
  --backend warp --seed 42 --steps 655360
```

## 観測を選ぶ・作る

文字列で選ぶか、`ObservationTerm`に関数を渡します。
各項は選択順に連結します。通常の項は物理量をそのまま返し、`scale`で割ってネットワークへ渡します。

| 組込観測 | 次元 | 意味・単位（scaleを適用する前） |
|---|---:|---|
| `robot_to_cargo` | 2 | ロボットから円柱へ向かう世界XYベクトル [m] |
| `cargo_to_robot` | 2 | 円柱からロボットへ向かう世界XYベクトル [m] |
| `cargo_to_goal` | 2 | 円柱から目標へ向かう世界XYベクトル [m] |
| `robot_heading` | 2 | 機体の絶対yawの`[sin, cos]` |
| `robot_velocity` | 2 | 機体の世界XY速度 [m/s] |
| `cargo_velocity` | 2 | 円柱の世界XY速度 [m/s] |
| `robot_yaw_rate` | 1 | 機体のyaw角速度 [rad/s] |
| `cargo_yaw_rate` | 1 | 円柱のyaw角速度 [rad/s] |
| `goal_distance` | 1 | 円柱中心から目標までの距離 [m] |
| `surface_gap` | 1 | 機体中心から円柱外周までの距離、最小0 [m] |
| `command_velocity` | 2 | 直前に要求した世界XY移動速度 [m/s]。実測速度は`robot_velocity` |
| `baseline` | 14 | 過去の学習済みモデルと同一の入力・正規化 |

`baseline`には、位置課題で使わない姿勢の固定値も互換性のため残っています。
順序は、`cargo_to_robot / 円柱直径`、`cargo_to_goal / 円柱直径`、固定値`[0,1]`、
`robot_velocity / 0.2`、`cargo_velocity / 0.2`、固定値`0`、`robot_heading`、`robot_yaw_rate`です。
名前付きの観測を組み合わせる場合、位置の同じ正規化には`scale=task.config.width`を使います。

独自関数は、次のように`.py`ファイルのトップレベルへ書きます。

```python
from hexapod_transport_rl.student import ObservationTerm, PushState

def speed_squared(state: PushState):
    return state.cargo_velocity.square().sum(dim=-1)

term = ObservationTerm(speed_squared, name="cargo_speed_squared", scale=0.04)
```

戻り値は`torch.Tensor`の`(B,)`または`(B,D)`です。`B`は並列環境数で、スカラー項も入力数を自動推定します。
`enabled=False`なら同じ次元のゼロを入力します。

```python
ObservationTerm("cargo_velocity", scale=0.2, enabled=False)
```

入力項を削除するとネットワークのパラメータ数も変わります。
情報量の効果を調べる場合は、同じ観測一覧で`enabled`だけを変える比較も使えます。
回転対称化を使う既存モデルの入力を変える場合は、後述の`rotation`の条件も確認します。

## 報酬を選ぶ・作る

`rewards`は「ログに記録する名前→重みまたは`RewardTerm`」です。
数値だけを書いた場合は同名の組込項を使います。辞書を渡すと、含めた項だけが報酬になります。

| 組込項 | 計算 | 標準重み | 時間積分 |
|---|---|---:|---|
| `goal_progress` | 前後の目標距離の減少 [m] | 20 | なし |
| `approach_progress` | 前後の回り込みコストの減少 [m] | 2 | なし |
| `goal_distance` | 現在の目標距離 [m] | -4 | `dt`を掛ける |
| `approach_distance` | 現在の回り込みコスト [m] | -1 | `dt`を掛ける |
| `time` | 1 | -0.02 | `dt`を掛ける |
| `success` | 成功時1 | 10 | なし |
| `failure` | 転倒・範囲外等の失敗時1 | -10 | なし |
| `cargo_speed` | 円柱のXY速度の大きさ [m/s] | 未使用 | `dt`を掛ける |
| `command_change` | 前後の要求XY速度差の二乗和 | 未使用 | なし |

`approach_distance`は、表面までの距離に押す側への回り込み角度の項を加えた距離換算コストです。
角度の項は目標から見た円柱の反対側で小さくなり、目標10 cm以内では弱まります。純粋な表面距離は`state.surface_gap`で取得できます。
標準報酬は`20Δd + 2Δa - (4d + a + 0.02)dt + 10成功 - 10失敗`です。

```python
from dataclasses import replace
from hexapod_transport_rl.student import CylinderPushTask, RewardTerm

def near_goal_speed(before, after):
    return after.cargo_speed.square() * (after.goal_distance < 0.10)

baseline = CylinderPushTask()
task = replace(baseline, rewards={
    **baseline.rewards,
    "near_goal_speed": RewardTerm(near_goal_speed, weight=-2.0, per_second=True),
})
```

独自報酬は`(before, after) -> Tensor[B]`です。`per_second=True`なら重みのほかに`dt`を掛けます。
進捗差・一度限りのボーナスでは通常`False`、速度や距離の継続コストでは通常`True`を使います。
報酬を全て0にしても、物理状態による成功・失敗・時間切れの判定は動きます。

## 行動を選ぶ

```python
from hexapod_transport_rl.student import DiscreteDirections, ContinuousVelocity

DiscreteDirections(count=4, include_stop=True)  # 5択
DiscreteDirections(count=8, include_stop=True)  # 9択
DiscreteDirections(count=16, include_stop=True) # 17択
ContinuousVelocity(frame="world", speed=0.15)  # 連続XYの2出力
ContinuousVelocity(frame="body", speed=0.15)   # 機体座標で解釈
```

離散方向は+Xから反時計回りです。停止ありなら0が停止、1以降が移動方向です。
`frame="world"`は世界座標、`"body"`は機体の前方を+X・左側を+Yとします。観測の座標系は自動で変えません。
連続行動は各軸[-1,1]を出力し、ノルム1以下へ投影してから`speed`を掛けます。斜め移動で速度は増えません。

いずれも既存の方向指令→歩行指令変換を使い、固定歩行モデルが移動方向へ機体を旋回させます。
`speed`は要求速度の上限で、最大0.20 m/sです。実際の歩行指令には従来の速度・旋回制限がかかります。
連続XYは、関節18個や独立したyaw指令を直接学習する行動ではありません。

## 独自関数に渡る状態

`PushState`は、名前付きの物理量を持つスナップショットです。
位置・速度は世界XY、yawはrad、角速度はrad/s、時間は秒です。
スカラーは`(B,)`、XYベクトルは`(B,2)`、テンソルはCPUまたはGPU上にあります。

主なフィールドは`robot_position`、`cargo_position`、`goal_position`、`robot_yaw`、`cargo_yaw`、
`robot_velocity`、`cargo_velocity`、`robot_yaw_rate`、`cargo_yaw_rate`、`goal_distance`、
`approach_distance`、`success`、`failed`、`elapsed_seconds`、`command_velocity`、`dt`です。
派生量は`robot_to_cargo`、`cargo_to_robot`、`cargo_to_goal`、`robot_heading`、`cargo_speed`、`surface_gap`です。

関数は状態を読み取り、PyTorchのテンソル演算で計算します。状態を書き換えたり、乱数や外部の可変状態へ依存させたりしません。
バッチに対してPythonの`if`や`.item()`を使わず、比較、掛け算、`torch.where`などを使います。
CPU用とGPU用に別の関数を書く必要はありません。

独自関数は名前付きのトップレベル関数にします。lambda・入れ子関数・セル内定義は保存対象にできません。
ノートブックからは、`.py`の課題ファイルを作り、`load_task_file("my_task.py")`で読み込めます。
課題ファイルは`TASK = CylinderPushTask(...)`を定義します。補助関数も同じファイルに置くか、導入済みのパッケージを使います。
実行処理を同じファイルに置く場合は`if __name__ == "__main__":`で囲みます。

## 既存モデルを引き継ぐ

Colabの報酬実験では、この方法と学習率`3e-5`・エントロピー係数`0.001`を共通に使います。
ActorとCriticの初期重みは全seedで共通、Adamは毎回新規です。新規学習の3 seedとは区別して解釈します。
`episode_seconds`も変更できるため、短い操作練習でも同じモデルから開始できます。

新規学習が既定です。既存の99.5%モデルと同じ構成には`CylinderPushTask.baseline()`を使います。
ここでは`rotation="goal_octant"`が有効で、世界座標の回転対称性を利用します。

```bash
PYTHONPATH=src MUJOCO_GL=egl python tools/train_student.py \
  examples/student/pretrained.py --output runs/student_finetune_s42 \
  --backend warp --steps 655360 \
  --initial-checkpoint checkpoints/cylinder_push.pt \
  --learning-rate 3e-5 --entropy-coeff 0.001 --checkpoint-selection latest
```

Actor・Criticの重みを引き継ぎ、Adamは新規に作ります。中断した学習の完全な再開とは異なります。
`load_task(checkpoint)`で元の設定を取り出し、`replace(task, rewards=...)`で報酬だけ変えることもできます。

観測の順序・正規化・関数・マスク、行動の数・座標系・速度が違う場合は、同じ次元でも重みの継承を拒否します。
その場合は`initial_checkpoint`を省略し、新しい入力・出力層で学習します。
旧形式のチェックポイントも読み込めます。独自関数を含む新形式は保存時のソースから復元します。

`goal_octant`は既存の`baseline`14入力と停止＋8方向の世界座標行動に対応します。
観測・行動の実験では`rotation="none"`を使います。`compare.py`の観測・行動の新規学習例は、標準側も`none`に揃えています。
回転対称化の有無も実験条件なので、観測の効果を比較するときは両条件で揃えます。

## 複数seedと卒論用の比較

同じ課題ファイルを複数seedで実行できます。

```bash
PYTHONPATH=src MUJOCO_GL=egl python tools/train_student.py \
  examples/student/reward_only.py --output runs/student_reward_seeds \
  --backend warp --steps 655360 --seeds 42 43 44 --episodes 256
```

複数条件の例は次のコマンドです。

```bash
PYTHONPATH=src MUJOCO_GL=egl python examples/student/compare.py \
  --output runs/student_comparison --backend warp
```

まず`--smoke`を付ければ、3条件×2seedの短い接続試験になります。
Pythonからは`compare({"baseline": task_a, "changed": task_b}, output, seeds=(42,43,44), steps=655360)`を使えます。
物理、開始分布、制限時間、成功条件が違う組み合わせは比較APIが拒否します。
報酬・観測・行動と回転対称化は、それぞれの実験で明示的に選びます。

検証seedは131000、最終評価の既定seedは240000です。同じ評価配置を各条件・各学習seedで使います。
最終評価は貪欲な離散行動または連続分布の決定的行動で、各並列環境の最初の1エピソードだけを数えます。
Actorの観測を減らしても、評価器は別の物理状態を使い、位置誤差・成功率を同じ定義で測ります。
最終評価を見てモデルを選び直しません。

CSVは条件とseedごとの成功数、成功率、位置誤差、失敗を含む平均エピソード時間、学習ステップ・時間を持ちます。
JSONには学習seed間の成功率の平均と標本標準偏差も保存します。1seedでは標準偏差は`null`です。
CPUとGPUは乱数生成と接触数値が違うため、同じseedでも同一軌跡とは限りません。比較には同じbackendを使います。

## 保存物と実装

| 保存物 | 内容 |
|---|---|
| `initial.pt` / `latest.pt` / `best.pt` | 設定・ネットワーク形状・Actor・Critic・Adam・学習履歴 |
| `task.json` | 観測・報酬・行動・正規化・観測列の範囲 |
| `task_sources/` | 自作関数を定義したファイルとハッシュ |
| `source_snapshot/` / `environment.json` | 実験時のライブラリソースと依存バージョン |
| `progress.json` | 各更新の報酬合計、項ごとの寄与の平均、成功数、ステップ数 |
| `validations.json` | 選択用の検証結果 |
| `evaluation_*.json` | 未使用配置の結果とエピソードごとの報酬内訳 |
| `episode.mp4` / `.json` | 実際の25 Hz動画と結果 |
| `comparison.csv` / `.json` | 複数条件・seedの一覧と集計 |

独自関数を復元する際は保存したPythonソースを読み込みます。実験ディレクトリ全体を保管・移動します。
物理モデル・固定歩行モデルと依存パッケージは従来どおり必要です。

[student.py](../src/hexapod_transport_rl/student.py)が公開API、
[student_terms.py](../src/hexapod_transport_rl/student_terms.py)が状態と組込項、
[student_env.py](../src/hexapod_transport_rl/student_env.py)が既存環境との接続です。
PPO更新は既存の[torchrl_ppo.py](../src/hexapod_transport_rl/torchrl_ppo.py)を使います。
`StudentPushEnv`はTorchRL環境でもあり、必要なら`task.make_env()`で直接接続できます。
