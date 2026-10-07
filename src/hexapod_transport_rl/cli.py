"""One entry point for reward-only T transport: train, compose, evaluate."""

import argparse
from pathlib import Path

import torch

from .approach_evaluation import evaluate
from .approach_training import train_approach
from .config import PushConfig
from .handover_training import compose, train_handover
from .mappo import load_checkpoint, train


def _train_push(resume=None, asset_root=None, **kwargs):
    """Train the initial T pusher, or continue its original rear-start task."""
    cfg = PushConfig(shape="T")
    if resume:
        _, saved = load_checkpoint(resume)
        cfg = PushConfig(**saved["config"])
    return train(cfg, asset_root, resume=resume, **kwargs)


def _training_args(parser, *, iterations, horizon, seed):
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, help="Override bundled robot assets")
    parser.add_argument("--iterations", type=int, default=iterations)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=horizon)
    parser.add_argument("--seed", type=int, default=seed)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="2台のhexapodによるT字運搬。学習・結合・評価の入口です。"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    navigation = commands.add_parser("train", help="回り込みをMAPPOで学習")
    navigation.set_defaults(function=train_approach)
    _training_args(navigation, iterations=800, horizon=32, seed=20261008)
    navigation.add_argument("--pushing-checkpoint", type=Path, required=True)
    navigation.add_argument("--epochs", type=int, default=4)
    navigation.add_argument("--minibatch", type=int, default=128)
    navigation.add_argument("--validate-every", type=int, default=50)
    navigation.add_argument("--resume", type=Path)

    handover = commands.add_parser(
        "train-handover", help="押す方策を回り込み後の配置へ追加学習"
    )
    handover.set_defaults(function=train_handover)
    _training_args(handover, iterations=150, horizon=64, seed=20261009)
    handover.add_argument("--checkpoint", type=Path, required=True)

    combine = commands.add_parser("compose", help="回り込み・押す方策を結合")
    combine.set_defaults(function=compose)
    combine.add_argument("--navigation-checkpoint", type=Path, required=True)
    combine.add_argument("--pushing-checkpoint", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)

    evaluation = commands.add_parser("eval", help="学習済み運搬方策を評価・録画")
    evaluation.set_defaults(function=evaluate)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--asset-root", type=Path)
    evaluation.add_argument("--episodes", type=int, default=24)
    evaluation.add_argument("--seed", type=int, default=70000)
    evaluation.add_argument("--workers", type=int, default=4)
    evaluation.add_argument(
        "--layout", choices=("front", "side", "rear", "near", "mixed"), default="front"
    )
    evaluation.add_argument("--max-yaw-degrees", type=float, default=30)
    evaluation.add_argument(
        "--seconds", dest="episode_seconds", type=float, default=100
    )
    evaluation.add_argument("--video-width", type=int, default=960)
    evaluation.add_argument("--video-height", type=int, default=720)
    evaluation.add_argument("--video-fps", type=int, default=25)
    evaluation.add_argument(
        "--fast-video", action="store_true", help="影と反射を省いてCPU描画を軽くする"
    )
    evaluation.add_argument(
        "--video-dir", type=Path, help="録画時は --episodes 1 を指定"
    )

    pushing = commands.add_parser(
        "train-push", help="初期の押す方策を学習（base_pusher.pt があれば省略可）"
    )
    pushing.set_defaults(function=_train_push)
    _training_args(pushing, iterations=1000, horizon=64, seed=20261006)
    pushing.add_argument("--epochs", type=int, default=4)
    pushing.add_argument("--minibatch", type=int, default=128)
    pushing.add_argument("--backend", choices=("auto", "sync", "async"), default="auto")
    pushing.add_argument("--resume", type=Path)
    pushing.add_argument("--mirror-equivariant", action="store_true")
    pushing.add_argument("--initial-log-std", type=float)
    return parser


def main():
    arguments = vars(_build_parser().parse_args())
    torch.set_num_threads(1)
    function = arguments.pop("function")
    arguments.pop("command")
    result = function(**arguments)
    if isinstance(result, Path):
        print(f"Saved: {result}")


if __name__ == "__main__":
    main()
