"""Build the student lesson; infrastructure lives in colab_runtime.py."""

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/hexapod_transport_rl_colab.ipynb"
REPOSITORY = "hashimoto-robotics-lab/hexapod-transport-rl"


def main():
    cells = []

    def markdown(source):
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": textwrap.dedent(source).strip() + "\n",
            }
        )

    def code(source):
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "source": textwrap.dedent(source).strip() + "\n",
                "outputs": [],
                "execution_count": None,
            }
        )

    markdown(r"""
# 六足ロボットの協調物資運搬

## なぜこの研究をするのか

複数のロボットで荷物を押すには、荷物の周囲で移動する位置、押す向き、
相手に合わせるタイミングを決める必要があります。
この研究では、**2台の六足ロボットがT字物体を回り込み、脚で目標位置まで押す協調動作**を学習します。
接触や初期位置によって必要な指令が変わるため、試行した結果を報酬として返し、
状況に合った指令を学ぶ強化学習を使います。
まず平坦な床のシミュレーションで、どの初期配置から成功し、どんな条件で失敗するかを調べます。

研究で確かめたい問いは、次の3つです。

- 前方や側方から始めても、荷物を避けて押す位置へ回り込めるか。
- 2台が荷物の位置と向きを合わせながら、目標まで運べるか。
- 報酬や学習量、初期配置を変えると、成功率や接触・転倒はどう変わるか。

## 出発点：六足ロボットは、すでに歩行を学習している

この教材には、**速度コマンドを受け取り、18関節を動かす学習済み歩行モデル**が入っています。
前進・横移動・旋回の目標速度を送ると、ロボットが脚を動かします。
最初はこのモデルを読み込んで、指令を変えるだけで動かせることを動画で確かめましょう。

```text
速度コマンド [前後速度, 左右速度, 旋回速度]
                   ↓
         学習済み歩行モデル（固定）
                   ↓
           18関節の目標角度
                   ↓
          モーター・脚・床の接触
                   ↓
             ロボットが移動
```

その後、**荷物と相手の状態から、各機へ送る速度コマンドを決める運搬方策**を強化学習します。
歩行モデルは固定したままです。運搬方策にはチームの報酬を使い、デモを教師には使いません。

**上から順に実行**してください。
研究の動機 → 準備 → コマンドで歩行を体験 → 協調運搬の学習 → 評価・動画 → 保存、の順に進みます。
学習は最初に `quick` で一巡し、その後 `research` で本学習に進みます。
""")
    markdown(r"""
## Step 1 — GitHubから教材を取得する

公開リポジトリからコード・ロボット形状・学習済みモデルを取得します。
**GitHubへのログインやアクセストークンは不要**です。このセルはそのまま実行してください。
""")
    code(r"""
from pathlib import Path
import subprocess
import sys

PROJECT_DIR = Path("/content/hexapod_transport_rl")
if not PROJECT_DIR.exists():
    subprocess.run([
        "git", "clone", "--depth", "1", "--branch", "main",
        "https://github.com/hashimoto-robotics-lab/hexapod-transport-rl.git", str(PROJECT_DIR)
    ], check=True)
sys.path.insert(0, str(PROJECT_DIR / "tools"))
""")
    markdown(r"""
## Step 2 — 実行環境を準備する

準備用APIが、Python 3.12・MuJoCo・固定した依存ライブラリと動画の描画環境を用意します。
このセルもそのまま実行してください。以降の `lesson` は学習・評価・保存を実行する補助APIです。
ロボットの操作には、次の `WalkingSimulation` を使います。
""")
    code(r"""
from colab_runtime import ColabLesson

lesson = ColabLesson(PROJECT_DIR)
""")
    markdown(r"""
## Step 3 — 学習済みロボットにコマンドを送る

`WalkingSimulation` は、**床と指定した台数のロボットだけ**を用意する歩行用APIです。
新しい学習は行いません。まず1台にコマンドを送り、歩行を動画で確かめます。

- `set_velocity(robot_id=0, vx=0.12)`：0番のロボットの前進指令を0.12 m/sにする。
- `run_for(seconds=3.0)`：全機を、現在の指令で同時に3秒間動かす。
- `stop()`：全機にゼロ速度を指令する。実際の減速を観察するには、その後も時間を進める。

指令は、次に変更するまで続きます。`set_velocity()` で省略した速度成分はゼロになります。
`robot_id` は0から始まります。`vx`・`vy` は機体座標の速度、`yaw_rate` は旋回速度です。
指令は目標速度なので、実際の動きには姿勢や接触による差が出ます。

| 引数 | 単位 | 指定できる範囲 | 正の値の意味 |
|---|---|---|---|
| `vx` | m/s | −0.15〜0.20 | 前進 |
| `vy` | m/s | −0.10〜0.10 | 左移動 |
| `yaw_rate` | rad/s | −0.60〜0.60 | 左旋回 |

下のAPI呼び出しを1つ変えて再実行し、動きの違いを確かめてください。
`with` は最後に動画の保存と後片付けを行います。
`lesson.run_example()` の中のコードが、専用環境で実行されます。
**最初は `set_velocity()` の速度と `run_for()` の時間を変更してみましょう。**
`run_for()` の時間は、歩行制御周期の0.04秒刻みで指定します。
""")
    code(r"""
lesson.run_example(r'''
from hexapod_transport_rl import WalkingSimulation

with WalkingSimulation(num_robots=1, video_path="walking_commands.mp4") as sim:
    sim.stop()
    sim.run_for(seconds=0.8)

    sim.set_velocity(robot_id=0, vx=0.12)
    sim.run_for(seconds=3.0)
    print("前進後の位置 [m]:", sim.positions)

    sim.set_velocity(robot_id=0, vy=0.06)
    sim.run_for(seconds=2.4)
    print("左移動後の位置 [m]:", sim.positions)

    sim.set_velocity(robot_id=0, yaw_rate=0.40)
    sim.run_for(seconds=2.4)
    print("旋回後の向き [rad]:", sim.headings)

    sim.stop()
    sim.run_for(seconds=0.8)
''', name="walking_commands")
lesson.show_example_video("walking_commands", "walking_commands.mp4")
""")
    markdown(r"""
### 同じAPIで4台を動かす

`num_robots=4` とし、0〜3番へ別々の指令を設定します。
**指令の設定だけでは時間は進みません。** `run_for()` で4台が同時に動きます。
2番には前進と左移動を同時に指令します。固定歩行モデルは、停止から横移動だけを始めると速度の追従が弱い場合があります。
ここでも学習は行わず、荷物はありません。4台の協調運搬を学習する環境は、今後拡張する研究課題です。
""")
    code(r"""
lesson.run_example(r'''
from hexapod_transport_rl import WalkingSimulation

with WalkingSimulation(num_robots=4, video_path="four_robots.mp4") as sim:
    sim.run_for(seconds=0.8)
    sim.set_velocity(robot_id=0, vx=0.10)
    sim.set_velocity(robot_id=1, vx=0.05)
    sim.set_velocity(robot_id=2, vx=0.08, vy=0.04)
    sim.set_velocity(robot_id=3, yaw_rate=0.30)
    sim.run_for(seconds=3.0)
    print("4台の位置 [m]:", sim.positions)
    print("4台の向き [rad]:", sim.headings)
    sim.stop()
    sim.run_for(seconds=0.8)
''', name="four_robot_commands")
lesson.show_example_video("four_robot_commands", "four_robots.mp4")
""")
    markdown(r"""
## 歩行コマンドから協調運搬へ

今は私たちが速度コマンドを指定しました。ここからは、**荷物・目標・相手の状態を観測して、
2台それぞれのコマンドを運搬方策が決める**ようにします。

```text
荷物・目標・相手の状態 → 運搬方策（今回MAPPOで学習）
                                 ↓ 各機の速度コマンド
                        歩行モデル（学習済み・固定）
                                 ↓ 関節の動作
                          2台とT字物体の接触・移動
                                 ↓ チーム報酬
                           運搬方策を更新
```

歩行体験で指定したコマンドは、運搬学習の教師には使いません。
次の工程では、環境APIの観測・行動・報酬を確認し、参考の運搬方策を再生してから、自分の運搬方策を学習します。
""")
    markdown(r"""
## Step 4 — 実験を設定する

まず `mode="quick"` で全工程を確認し、次に `mode="research"` で本学習します。
**quickの数回の更新による成功率を、卒論の性能として扱わないでください。**
researchでは回り込み409,600、押す方策の適応153,600チームステップを目安に学習します。

- `name`：結果を保存する実験名。条件を変えるときは、新しい名前にします。
- `num_envs`：経験を集める独立した世界数。1世界には2台のロボットがいます。
- `seed`：学習の乱数seed。複数回の比較実験ではこの値を変えます。
- `start_from_scratch`：Trueなら上位の押す方策も新規学習します。歩行モデルはどちらでも固定します。

設定・教材の版・実行コードは自動で記録されます。同じ設定で再実行すると、保存済みの続きから学習します。
""")
    code(r"""
lesson.configure(
    name="trial_01",
    mode="quick",
    num_envs=2,
    seed=20261006,
    start_from_scratch=False,
)
""")
    markdown(r"""
## Step 5 — 環境APIを確認する
2台の局所観測は `(2,20)`、各機の行動は `[前後, 左右, 旋回]` の `(2,3)` です。
1 stepは0.2秒。1つのMuJoCo世界に2台とTがあり、報酬はチームで共有します。
""")
    code(r"""
lesson.run_python(r'''
import numpy as np
from hexapod_transport_rl import HexapodPushEnv, PushConfig

with HexapodPushEnv(PushConfig(shape="T"), flatten=False) as env:
    observation, info = env.reset(seed=42)
    action = np.zeros((2, 3), dtype=np.float32)
    next_observation, reward, terminated, truncated, info = env.step(action)
    print("観測の形:", observation.shape)
    print("行動の形:", action.shape)
    print("チーム報酬:", reward)
    print("終了:", terminated, "時間切れ:", truncated)
    print("報酬の内訳:", info["reward_terms"])
''', name="inspect_api.py", show_output=True)
""")
    markdown(r"""
## Step 6 — 参考の学習済みモデルを再生する
これは事前に**報酬だけで学習した参考方策**です。次の学習の教師データには使用しません。
同じseedで実際にMuJoCoを動かし、動画を生成します。モデルを読み込むだけの静止画ではありません。
640×480、5 fpsで影・反射を省いて描画し、物理更新は200 Hzを保ちます。
""")
    code(r"""
lesson.show_reference()
""")
    markdown(r"""
## Step 7 — 初期の押す方策を準備する
`start_from_scratch=False` では、報酬で学習済みの押す方策を利用します。
`True` では、押す方策をランダムな重みから学習した後、左右対称化と探索幅の調整を適用して追加学習します。
歩行方策は固定です。両方の方法でデモ行動や模倣損失は使いません。
""")
    code(r"""
pusher = lesson.prepare_pusher()
""")
    markdown(r"""
## Step 8 — 回り込みを学習する
actorをランダムな重みからMAPPOで学習します。初期配置は `near → rear → side → front`。
検証成功率が75%以上かつ一定の反復数を経過すると、次の難度へ進みます。
報酬はTを避けて後方へ向かう距離の減少、接触、時間などから計算します。距離計算は行動の教師には使いません。
両機が後方の担当位置と向きに整列した時点で、回り込みのエピソードを終了します。
""")
    code(r"""
navigator = lesson.train_navigation(pusher)
""")
    markdown(r"""
## Step 9 — 回り込み後の位置から押す動作を学習する
回り込み完了位置のばらつきに適応する追加のMAPPOです。
毎回の学習で回り込みを再生せず、その完了位置付近から始めて経験を集めます。
初期配置の25%には、元の押す学習の配置も残します。
""")
    code(r"""
adapted_pusher = lesson.train_handover(pusher)
""")
    markdown(r"""
## Step 10 — 検証用の初期配置でモデルを選ぶ
回り込みの最終モデルと、保存されていれば前方検証の最良モデルを候補にします。
押す方策と組み合わせ、**運搬全体**をseed 66000以降で評価します。
成功数、胴体接触、回り込み中の機体同士の接触、最終位置誤差の順に選びます。
このseed集合を、次のテスト集合と混ぜません。
""")
    code(r"""
model = lesson.select_model(navigator, adapted_pusher)
""")
    markdown(r"""
## Step 11 — 未使用の初期配置で評価する
前方はseed 80000以降、側方は81000以降を使います。Tの初期yaw角は目標方向に対して±30度です。
`quick` の評価は10秒で打ち切ります。`research` の100秒評価と混ぜて比較しないでください。
成功しなかった試行も、接触・転倒・誤差を含めて記録します。
""")
    code(r"""
test_results = lesson.evaluate(model)
""")
    markdown(r"""
## Step 12 — 学習曲線と論文用の評価表を作る
学習中のsuccess rateはその時点のカリキュラムの初期配置での値です。運搬全体のテスト成功率とは異なります。
完了エピソードがない時点は、成功率を描画しません。
成功率、接触、転倒、位置・yaw誤差、時間をCSVにまとめ、成功率のWilson 95%区間も表示します。
区間は初期配置の試行に対するもので、異なる学習seedのばらつきではありません。卒論では学習seedを変えて複数回実行してください。
""")
    code(r"""
lesson.show_results()
""")
    markdown(r"""
## Step 13 — 自分で学習した方策を録画する
参考モデルを使い回さず、Step 10で選んだ**自分のモデル**を実行します。
seedは82000で固定し、成功した試行だけを探す選び方はしません。
`quick` は後方近くから10秒、`research` は前方から100秒を上限にします。失敗した動画も結果として確認します。
""")
    code(r"""
lesson.show_learned_video(model)
""")
    markdown(r"""
## Step 14 — 実験結果を保存する

設定、使用したコード・資産、ライブラリの版、重み、学習ログ、評価、図、CSV、動画をZIPへまとめます。
Colabではダウンロードします。ランタイムの終了前に保存してください。
長い学習のDrive保存・中断後の復元は、[保存と再開のガイド](https://github.com/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/docs/colab-guide.md#保存と再開)を参照してください。
""")
    code(r"""
lesson.save_results()
""")
    markdown(r"""
## 卒論で次に行う比較実験

1. `research` で学習seedを変え、別の実験名で少なくとも3回実行する。
2. 同じテストseed集合で、報酬係数・カリキュラム・初期配置など1条件ずつ変えて比較する。
3. 運搬成功率だけでなく、胴体接触・機体同士の接触・転倒・位置/yaw誤差・所要時間も報告する。
4. 回り込み環境は2台専用で、障害物・観測誤差・実機条件は含まないことを論文で明示する。

編集場所は展開した `docs/code-guide.md` を参照してください。回り込みの報酬は `ApproachRewardWeights`、押す報酬は `PushRewardWeights`、学習設定は `PPOSettings` にあります。
ソースを編集した後は新しい実験名で別プロセスの学習を開始してください。変更内容も論文・記録へ残してください。

Colabの実行環境には時間・資源の制限があり、ランタイム内の未保存ファイルは失われることがあります。
[Colab公式FAQ](https://research.google.com/colaboratory/faq.html) / [uvのPython管理](https://docs.astral.sh/uv/guides/install-python/) / [MuJoCoのPython API](https://mujoco.readthedocs.io/en/stable/python.html)
""")
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python", "version": "3.12"},
            "hexapod_transport_rl": {"repository": REPOSITORY, "mode_default": "quick"},
        },
        "cells": cells,
    }
    for index, cell in enumerate(cells):
        cell["id"] = f"lesson_{index:02d}"
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(f"Built {OUTPUT}: {len(cells)} cells, {OUTPUT.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
