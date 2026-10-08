# Colabで報酬設計を比較する

[学生用Colab](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/hexapod_transport_rl_colab.ipynb)を開き、「ランタイム → ランタイムのタイプを変更」で **T4 GPU** を選び、上から実行します。
GPUが利用できればMuJoCo Warp・固定歩行モデル・MAPPOへ使い、利用できなければCPUへ切り替えます。準備セルと学習セルが実際の学習先を表示します。
GitHubの認証は不要です。既に取得したコードは学生の編集を残すため上書きしません。
古い教材を使っている場合は、新規ランタイムで始めてください。

研究の動機、学習済み歩行モデルへの指令、運搬Gym API、報酬、MAPPO学習、評価・動画の順です。
`with`構文や独自の実行クラスは使わず、普通のimportとセルで進めます。動画はmediapyで表示します。

## 課題と学習量

横棒・縦棒の中心線が等長（2台用で両方1.3 m）のTを2台が脚・足で押します。
初期配置はT→ゴール→ロボットです。押す側への回り込みと、位置・向き・停止を学びます。歩行モデルだけを固定します。
標準は0.3〜0.4 m、目標との角度差±5〜30度、制限40秒です。
最終評価は位置8 cm・角度5度・低速状態1秒を要求します。
開始位置をTの背後から45度ずつ移す5段階と、目標精度を上げる2段階を使います。
別seedの6試行を8ロールアウトごとに検証し、3/6成功かつ各段階8ロールアウト以上で進めます。
難度の変更は次のresetにだけ適用し、途中の物理状態は変えません。

GPUでは256世界・horizon 64・ミニバッチ512・4 epochs・学習率3e-4を使います。
1回に16,384チームステップを収集します。`NUM_ENVS` はロボット台数ではありません。
CPUの2世界・horizon 128はAPIの体験・接続確認向けで、本課題の実学習にはGPUを使ってください。
小さい予算を実行できたことと、運搬を獲得したことは区別します。
criticの価値正規化を使い、各軸の7速度候補から選ぶ確率をランダムな重みからMAPPOが学びます。
停止を明示的な候補に含め、評価時は平均ではなく確率最大の候補を使います。段階別の探索ノイズ調整は不要です。
T座標の指令倍率は`[0.2, 0.2, 0.6]`。変換後の機体指令は元の歩行モデルの範囲に制限します。

標準設定の新規学習はローカルRTX A6000で30.1分、未使用50配置で44件成功しました。
これは1つの学習seedで、Colab/T4の実測時間ではありません。2条件を順に学習する時間は条件ごとにCSVで確認します。

GPUでは物理計算・モーター制御・固定歩行モデル・接触計測・観測・報酬・Collectorの方策をGPUで実行します。
actor・criticの学習、GAE、ミニバッチもGPU内で完結します。CPU環境ではCPUで収集します。
最初はGPUカーネルをコンパイルし、毎回の高位ステップをCUDA graphで実行します。
報酬はCPUとGPUの共通式です。GPUの物理演算はfloat32なのでCPUの軌跡と完全には一致しません。
カリキュラム検証は学習と独立したGPU環境、最終評価・録画は通常のGym APIです。
学習中は描画せず、実際のGPU状態を確認するときだけ `envs.render(world=0)` を呼びます。
[構成・計測条件・制約](warp-training.md)に詳細があります。

両条件は同じ初期重み・seed・物理・学習量・段階移行の規則を使います。
`orientation_error` だけを0から4へ増やし、角度の改善量の報酬と成功判定は共通に残します。
補助報酬の割引率とMAPPOのgammaは0.995に揃えます。
報酬の違いによって到達段階と経験する配置が変わるため、その違いも記録します。
総報酬の大小で性能を比較せず、成功率・位置・角度・T端の誤差・接触・時間を使います。

## ルールとの比較

`forward`は前進だけ、`feedback`は位置と角度の誤差から左右の速度を変える比例制御です。
モデルとルールを同じ未使用seed・最終精度・制限時間で評価します。
ルールの行動・軌跡・重みは学習に使いません。強化学習がルールより優れるかは評価で判断します。
ルールのゲインを調整する場合も別の検証seedを使い、テストを残します。

標準評価は12試行、卒論では50試行以上・少なくとも3つの学習seedで繰り返します。
モデル選択や設定の調整には開発・検証seedを使い、最終テスト結果からモデルを選び直しません。
学習量を揃えた最後のモデルを比較し、失敗や転倒も集計します。

## 保存

`create_experiment()`は実験名の新規フォルダを作り、実行時のコード・資産・モデル・版・SHA256を保存します。
同じ名前で再実行せず、新しい実験名を付けてください。
各条件に`run.json`、`initial.pt`、`progress.csv`、`curriculum.json`、途中checkpoint、最後の`pose.pt`が入ります。
評価JSON・比較CSV・図・動画も同じ結果フォルダへ保存し、最後にZIPを作ります。

`pose.pt`はactor・critic・optimizer・価値正規化の統計・設定・段階・CPUと使用したGPUのPyTorch乱数状態を含み、別の押すモデルは必要ありません。
`run.json`には学習先・GPU名・物理計算先を記録します。GPUで保存したモデルもCPUで再生できます。
結果フォルダを移動しても読み込めます。参考モデルの課題・学習量・評価条件は `checkpoints/manifest.json` を確認してください。旧形状のモデルと新しい課題の結果は区別します。
参考動画は開発用配置の成功例として表示します。参考モデル・動画は学習へ渡さず、固定歩行モデル以外はランダムな重みから学習します。
未保存の結果はランタイム削除で失われます。本実験では途中checkpointもDrive等へ保存します。

## 学習を再開する

```python
import torch
from hexapod_transport_rl import (
    load_mappo, PosePushConfig, PoseStage, PoseCurriculum,
    TorchRLTransportEnv, MAPPOSettings, make_mappo_loss,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor, critic, saved = load_mappo("runs/goal_side_reward_trial_01/baseline/pose.pt", device=device)
config = PosePushConfig.from_checkpoint(saved["pose_config"])
settings = MAPPOSettings(**saved["training"]["settings"])
loss = make_mappo_loss(
    actor, critic, settings, value_normalizer_state=saved["value_normalizer"],
)
optimizer = torch.optim.Adam(loss.parameters(), lr=settings.learning_rate)
optimizer.load_state_dict(saved["optimizer"])
envs = TorchRLTransportEnv(
    config, num_envs=saved["training"]["num_envs"],
    backend=saved["training"].get("physics_backend", "cpu"),
)
envs.set_seed(saved["training"]["seed"])
curriculum = PoseCurriculum(
    config, envs, actor, "runs/pose_continued",
    seed=saved["training"]["seed"], settings=settings,
    horizon=saved["training"]["horizon"],
    stages=tuple(PoseStage(**stage) for stage in saved["training"]["curriculum"]),
    validate_every=saved["training"]["validate_every"],
    minimum_rollouts=saved["training"]["minimum_rollouts"],
    validation_episodes=saved["training"]["validation_episodes"],
    advance_threshold=saved["training"]["advance_threshold"], resume=saved,
)
torch.set_rng_state(saved["torch_rng_state"])
if device.type == "cuda" and saved.get("cuda_rng_state") is not None:
    torch.cuda.set_rng_state(saved["cuda_rng_state"], device)
```

この後はノートブックと同じCollector・buffer・更新ループを使います。
Collectorの`total_frames`は追加で収集する量です。行動候補とその確率は保存モデルから復元されます。
再開は重み・optimizer・段階を引き継ぎますが、
中断時の物理状態や進行中のエピソードは復元しません。連続実行と完全に同じ軌跡にはなりません。
最後に`curriculum.close()`と`collector.shutdown()`を呼びます。

## 拡張

`PosePushConfig(num_robots=4)`で機体数を増やせます。Tの横棒・縦棒は4台用では両方2.6 mです。
ネットワークは`make_mappo_networks(envs.num_robots, envs.obs_dim, action_grid=ACTION_GRID)`で機体数と観測の形に合わせます。
2台の保存モデルを4台へそのまま使いません。役割配置・学習・成功率を再検証します。
摩擦・質量・初期角度・停止精度は1つずつ変更して比較してください。
現状は平坦な床・障害物なし・シミュレータの状態を観測する条件です。
