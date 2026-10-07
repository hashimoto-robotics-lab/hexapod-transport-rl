# 学習環境API

実行手順は [README](../README.md)、内部構造は [コードガイド](code-guide.md) を参照してください。
学習ではGymnasium形式の `reset` / `step` を使います。報酬は2台で共有する1つの値です。

## 押す環境

```python
from hexapod_transport_rl import PushConfig, HexapodPushEnv

with HexapodPushEnv(
    PushConfig(shape="T"), flatten=False
) as env:
    obs, info = env.reset(seed=42)
    action = env.action_space.sample()  # 実際の学習ではactorの出力
    next_obs, reward, terminated, truncated, info = env.step(action)
```

| 項目 | 2台の場合 |
|---|---|
| 観測 | `float32 (2, 20)` |
| 行動 | `float32 (2, 3)`、各値 `[-1, 1]` |
| 報酬 | チーム共通のfloat |
| 中央critic入力 | `env.state()` の `(40,)` |
| 制御周期 | 1 step = 0.2秒、歩行25 Hz、MuJoCo物理200 Hz |

各機の行動は機体座標の `[前後, 左右, 旋回]` です。
前進上限0.20 m/s、後退上限0.15 m/s、左右0.10 m/s、旋回0.60 rad/sへ変換します。
ゼロ指令は停止です。関節は固定した学習済み歩行モデルが制御します。

コンストラクタの既定は `flatten=True` で、観測 `(40,)`、行動 `(6,)` になります。
`PushConfig` の既定形状は箱のため、APIを直接使う場合は **`shape="T"` を指定**してください。
`run.sh` の学習コマンドは2台のT字運搬を選びます。

局所観測の添字は以下です。位置ベクトルは観測する機体の座標へ変換します。

| 添字 | 内容 |
|---|---|
| `0:2` | 自機→荷物のXY / 3 m |
| `2:4` | 荷物→目標のXY / 3 m |
| `4:7` | 自機のXY速度とyaw角速度 |
| `7:9` | 荷物と自機のyaw差のsin・cos |
| `9:11` | 目標と自機のyaw差のsin・cos |
| `11:14` | 直前の速度指令 / `[0.20, 0.10, 0.60]` |
| `14` | 自機の上向き軸と世界Z軸の内積 |
| `15` | 担当する押し位置の横座標 / 荷物幅 |
| `16:20` | 相手の相対XY / 3 mとyaw差のsin・cos |

## 回り込み環境

```python
from hexapod_transport_rl.approach import ApproachConfig, ApproachEnv

env = ApproachEnv(ApproachConfig(layout="front"))
try:
    obs, info = env.reset(seed=42)
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
finally:
    env.close()
```

観測は `(2, 10)`、行動は `(2, 3)` です。
`observe_approach()` がT基準の位置・向き・自機速度・前回指令を作り、左右の機体を共通の座標へ反転します。
`physical_actions()` がactorの出力を実際の左右・旋回方向へ戻します。この変換は `ApproachEnv.step()` 内で行います。

回り込みの成功は、2台がT後方の担当位置 `x=-1.05 m, y=±0.325 m` へ近づいて向きを合わせることです。
回り込み中の成功判定と、Tを目標まで運ぶ成功判定は別です。
観測の正確な並びと正規化は `approach.py` の `observe_approach()` を参照してください。

## 終了、診断、並列化

`reset(seed=...)` は `(obs, info)`、`step(action)` は `(obs, reward, terminated, truncated, info)` を返します。
`terminated` は成功・失敗、`truncated` は時間切れです。最終観測を返した後、自動resetはしません。
押す環境での成功は位置誤差0.18 m未満、yaw誤差0.25 rad未満、速度0.06 m/s未満、yaw速度0.10 rad/s未満を0.5秒以上保つことです。

`info` に成功、目標までの距離、yaw誤差、接触、報酬内訳を返します。
押す環境の `reward_terms` は距離・向き・押す位置への接近、指令変化、時間、衝突、胴体接触、終了報酬を含みます。
`body_normal_impulse_ns`、`leg_link_normal_impulse_ns`、`foot_normal_impulse_ns` は各機の部位別の法線力積です。

押す学習の `make_vector_env(B, flatten=False, asynchronous=True)` はB個の独立世界を別プロセスで動かします。
観測 `(B, 2, 20)`、行動 `(B, 2, 3)`、報酬 `(B,)` になります。
回り込み学習は `make_approach_vector()` を使い、観測が `(B, 2, 10)` になります。
終了した世界だけ `reset(options={"reset_mask": terminated | truncated})` でリセットします。
reset前に最終観測から価値を計算する処理は `mappo.py` に共通化しています。

別プロセスを使う自作スクリプトでは `if __name__ == "__main__":` で起動処理を囲み、最後に `close()` を呼んでください。
`HexapodPushEnv(render_mode="rgb_array")` はRGB画像、`render_mode="human"` はGUIを提供します。
運搬全体の25 fps録画は `./run.sh eval --video-dir ... --episodes 1` を使ってください。
