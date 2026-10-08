"""Controlled teacher-free MAPPO timing and reward adaptation trials."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement

from hexapod_transport_rl import (
    MAPPOSettings,
    PoseCurriculum,
    PosePushConfig,
    PoseStage,
    TorchRLTransportEnv,
    load_mappo,
    make_mappo_loss,
    make_mappo_networks,
    save_mappo,
)
from hexapod_transport_rl.experiments import create_experiment

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--checkpoint")
    p.add_argument("--steps", type=int, default=131072)
    p.add_argument("--worlds", type=int, default=256)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--minibatch", type=int, default=512)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261020)
    p.add_argument("--weights", default="{}")
    p.add_argument("--smooth-curriculum", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    a = p.parse_args()
    frames = a.worlds * a.horizon
    if (
        min(a.worlds, a.horizon, a.minibatch, a.epochs) < 1
        or a.steps < frames
        or a.steps % frames
    ):
        p.error("--steps must be a positive multiple of worlds * horizon")
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    run = create_experiment(ROOT, a.name)
    (run / "training_driver.py").write_text(Path(__file__).read_text())
    output = run / "trial"
    config = PosePushConfig(shaping_discount=0.995)
    parent = None
    if a.checkpoint:
        actor, critic, parent = load_mappo(a.checkpoint, device="cuda")
        config = PosePushConfig.from_checkpoint(parent["pose_config"])
    config = replace(
        config, reward_weights=replace(config.reward_weights, **json.loads(a.weights))
    )
    settings = MAPPOSettings(
        epochs=a.epochs,
        minibatch_size=a.minibatch,
        gamma=0.995,
        gae_lambda=0.99,
        value_normalization=True,
    )
    torch.manual_seed(a.seed)
    init_start = time.perf_counter()
    env = TorchRLTransportEnv(config, num_envs=a.worlds, backend="warp")
    env.set_seed(a.seed)
    if parent is None:
        actor, critic = make_mappo_networks(
            env.num_robots, env.obs_dim, action_grid=(-1, -0.5, -0.25, 0, 0.25, 0.5, 1)
        )
        actor.cuda()
        critic.cuda()
    loss = make_mappo_loss(
        actor,
        critic,
        settings,
        value_normalizer_state=parent["value_normalizer"] if parent else None,
    )
    optimizer = torch.optim.Adam(loss.parameters(), lr=settings.learning_rate)
    stages = None
    if parent:
        stages = (
            PoseStage(
                "reward_adaptation",
                config.goal_distance,
                config.max_yaw_degrees,
                config.position_tolerance,
                config.yaw_tolerance,
                config.success_hold_seconds,
            ),
        )
    elif a.smooth_curriculum:
        from hexapod_transport_rl import POSE_STAGES

        stages = (
            tuple(
                PoseStage(
                    f"approach_{ang}",
                    0.3,
                    15.0,
                    0.08,
                    torch.pi * 8 / 180,
                    0.6,
                    0.0,
                    float(ang),
                )
                for ang in [0, 45, 60, 75, 90, 135, 180]
            )
            + POSE_STAGES[-2:]
        )
    kwargs = {"stages": stages} if stages else {}
    # Fixed-task short adaptation does not need intermediate curriculum validation.
    curriculum = PoseCurriculum(
        config,
        env,
        actor,
        output,
        seed=a.seed,
        settings=settings,
        horizon=a.horizon,
        validate_every=None if parent else 8,
        initial_checkpoint=a.checkpoint,
        minimum_rollouts=8,
        **kwargs,
    )
    curriculum.training["benchmark_only"] = a.benchmark
    if parent is None:
        curriculum.training["initialization"] = {"mode": "random_weights"}
    (output / "run.json").write_text(json.dumps(curriculum.training, indent=2) + "\n")

    def save(path):
        return save_mappo(
            path,
            actor,
            critic,
            optimizer,
            config=config,
            provenance=env.provenance,
            training=curriculum.training,
            curriculum=curriculum.state,
            loss=loss,
        )

    save(output / "initial.pt")
    collector = Collector(
        env,
        actor,
        frames_per_batch=frames,
        total_frames=a.steps,
        env_device=env.device,
        policy_device=env.device,
        storing_device=env.device,
        auto_register_policy_transforms=True,
    )
    buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames, device=env.device),
        sampler=SamplerWithoutReplacement(),
        batch_size=settings.minibatch_size,
    )
    torch.cuda.synchronize()
    initialization = time.perf_counter() - init_start
    timings = []
    iterator = iter(collector)
    for _ in range(a.steps // frames):
        torch.cuda.synchronize()
        t = time.perf_counter()
        batch = next(iterator)
        torch.cuda.synchronize()
        collection = time.perf_counter() - t
        t = time.perf_counter()
        loss.value_estimator(batch)
        buffer.empty()
        buffer.extend(batch.reshape(-1))
        torch.cuda.synchronize()
        gae = time.perf_counter() - t
        t = time.perf_counter()
        updates = 0
        for _ in range(settings.epochs):
            for minibatch in buffer:
                metrics = loss(minibatch)
                objective = (
                    metrics["loss_objective"]
                    + metrics["loss_critic"]
                    + metrics["loss_entropy"]
                )
                optimizer.zero_grad()
                objective.backward()
                torch.nn.utils.clip_grad_norm_(
                    loss.parameters(), settings.max_grad_norm
                )
                optimizer.step()
                updates += 1
        torch.cuda.synchronize()
        update = time.perf_counter() - t
        t = time.perf_counter()
        curriculum.record(batch, metrics)
        torch.cuda.synchronize()
        validation_log = time.perf_counter() - t
        collector.update_policy_weights_()
        timings.append(
            dict(
                rollout=curriculum.rollouts,
                team_steps=curriculum.transitions,
                collection_seconds=collection,
                gae_seconds=gae,
                update_seconds=update,
                validation_and_logging_seconds=validation_log,
                updates=updates,
            )
        )
        if curriculum.rollouts % 8 == 0:
            save(output / "checkpoints" / f"step_{curriculum.transitions}.pt")
        (output / "timing.json").write_text(
            json.dumps(
                dict(
                    initialization_seconds=initialization,
                    args=vars(a),
                    rollouts=timings,
                ),
                indent=2,
            )
            + "\n"
        )
    save(output / "pose.pt")
    curriculum.close()
    collector.shutdown()
    print("COMPLETED", run, flush=True)


if __name__ == "__main__":
    main()
