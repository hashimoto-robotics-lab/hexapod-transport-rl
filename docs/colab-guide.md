# Colabで報酬設計を比較する

[学生用Colab](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/hexapod_transport_rl_colab.ipynb)を新しいCPUランタイムで開き、上から実行します。
GitHubの認証は不要です。既に取得したコードは学生の編集を残すため上書きしません。
古い教材を使っている場合は、新規ランタイムで始めてください。

研究の動機、学習済み歩行モデルへの指令、運搬Gym API、報酬、MAPPO学習、評価・動画の順です。
`with`構文や独自の実行クラスは使わず、普通のimportとセルで進めます。動画はmediapyで表示します。

## 課題と学習量

等長3腕のTを2台が脚・足で押し、位置・向き・停止を学びます。歩行モデルだけを固定します。
標準は0.3〜0.4 m、目標との角度差±5〜30度、制限12秒です。
最終評価は位置8 cm・角度5度・低速状態1秒を要求します。
3つの段階を別seedの6試行で検証し、50%以上かつ各段階25ロールアウト以上で進めます。
難度の変更は次のresetにだけ適用し、途中の物理状態は変えません。

学習は65,536チームステップ／条件、2世界並列、horizon 128、ミニバッチ128、4 epochs、学習率3e-4です。
学習時間は実行時のCSVを確認します。[測定記録](pose-task.md)に実験条件と結果があります。
256ステップに減らす場合は接続確認だけで、成功する学習とは区別します。
criticの価値正規化と、探索ノイズ上限を約0.37から0.05へ下げる設定を使います。
ノイズの変更はPPO更新後・次の収集前に行い、actorが学ぶ行動の平均を変更しません。
GPUによるMuJoCoの高速化は組み込んでいません。学習中は描画せず、評価時にだけ録画します。

両条件は同じ初期重み・seed・物理・学習量・段階移行の規則を使います。
角度の補助報酬だけを0にし、角度の成功判定は共通に残します。
補助報酬の割引率とMAPPOのgammaは0.99に揃えます。
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

`pose.pt`はactor・critic・optimizer・価値正規化の統計・設定・段階・PyTorchの乱数状態を含み、別の押すモデルは必要ありません。
結果フォルダを移動しても読み込めます。同梱の参考モデルは参考再生専用で、自分の学習には使いません。
未保存の結果はランタイム削除で失われます。本実験では途中checkpointもDrive等へ保存します。

## 学習を再開する

```python
import torch
from hexapod_transport_rl import (
    load_mappo, PosePushConfig, PoseStage, PoseCurriculum,
    TorchRLTransportEnv, MAPPOSettings, make_mappo_loss,
)

actor, critic, saved = load_mappo("runs/pose_reward_trial_01/baseline/pose.pt")
config = PosePushConfig(**saved["pose_config"])
settings = MAPPOSettings(**saved["training"]["settings"])
loss = make_mappo_loss(
    actor, critic, settings, value_normalizer_state=saved["value_normalizer"],
)
optimizer = torch.optim.Adam(loss.parameters(), lr=settings.learning_rate)
optimizer.load_state_dict(saved["optimizer"])
envs = TorchRLTransportEnv(config, num_envs=saved["training"]["num_envs"])
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
```

この後はノートブックと同じCollector・buffer・更新ループを使います。
Collectorの`total_frames`は追加で収集する量です。探索ノイズの進捗は、
再開前の`transitions`も含めた収集量を当初の総予算で割って渡します。
再開は重み・optimizer・段階を引き継ぎますが、
中断時の物理状態や進行中のエピソードは復元しません。連続実行と完全に同じ軌跡にはなりません。
最後に`curriculum.close()`と`collector.shutdown()`を呼びます。

## 拡張

`PosePushConfig(num_robots=4)`で機体数を増やせます。Tの3腕は4台用では各1.3 mです。
ネットワークは`make_mappo_networks(envs.num_robots, envs.obs_dim)`で機体数と観測の形に合わせます。
2台の保存モデルを4台へそのまま使いません。役割配置・学習・成功率を再検証します。
摩擦・質量・初期角度・停止精度は1つずつ変更して比較してください。
現状は平坦な床・障害物なし・シミュレータの状態を観測する条件です。
