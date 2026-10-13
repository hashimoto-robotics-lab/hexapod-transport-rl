"""One fresh MAPPO run: collect, GAE, update, log and save.

No transport checkpoint is loaded. The time budget is checked between rollouts;
the final batch and physics diagnostics always remain intact.
"""

import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement

from .pusht import PushTConfig
from .torchrl_env import TorchRLTransportEnv
from .torchrl_mappo import (
    MAPPOSettings,
    make_mappo_loss,
    make_pusht_networks,
    save_mappo,
)


def train_mappo(
    config: PushTConfig,
    output,
    *,
    seed=42,
    steps=4_194_304,
    max_seconds=360.0,
    num_envs=None,
    horizon=32,
    settings=None,
    backend="auto",
):
    """Train an uninitialized transport policy and return its checkpoint path.

    ``max_seconds`` includes environment construction, collection and updates.
    ``None`` disables the time limit for experiments with equal step budgets.
    Stop before starting another rollout when its estimated cost exceeds the
    remaining budget. A cold first rollout can overrun; this is not a hard timer.
    """
    if not isinstance(config, PushTConfig):
        raise TypeError("Current MAPPO experiments require PushTConfig")
    if horizon < 1 or steps < 1 or (max_seconds is not None and max_seconds <= 0):
        raise ValueError("Learning budgets must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists():
        raise FileExistsError("Use a new output directory for each fresh run")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    started = time.monotonic()
    gpu = backend == "warp" or (backend == "auto" and torch.cuda.is_available())
    num_envs = (512 if gpu else 2) if num_envs is None else num_envs
    frames = num_envs * horizon
    if steps < frames:
        raise ValueError("Step budget must include at least one complete rollout")
    settings = settings or MAPPOSettings(
        epochs=2,
        minibatch_size=1024,
        gamma=0.995 ** (config.dt / 0.2),
        gae_lambda=0.99 ** (config.dt / 0.2),
        initial_log_std=-1.8,
        value_normalization=True,
    )
    settings = replace(settings, minibatch_size=min(settings.minibatch_size, frames))
    env = TorchRLTransportEnv(
        config, num_envs=num_envs, backend=backend, asynchronous=False
    )
    env.set_seed(seed)
    actor, critic = make_pusht_networks(
        config, device=env.device, initial_log_std=settings.initial_log_std
    )
    loss = make_mappo_loss(actor, critic, settings)
    optimizer = torch.optim.Adam(loss.parameters(), lr=settings.learning_rate)
    metadata = dict(
        algorithm="TorchRL MAPPO; shared actor, central critic, per-agent PPO ratios",
        initialization="random transport weights, empty Adam and fresh value statistics",
        seed=seed,
        config=asdict(config),
        settings=asdict(settings),
        num_envs=num_envs,
        horizon=horizon,
        requested_steps=steps,
        max_seconds=max_seconds,
        backend=env.backend,
        gpu=torch.cuda.get_device_name(env.device) if gpu else None,
        demonstration_actions_used=False,
        provenance=env.provenance,
    )
    state = dict(transitions=0, rollouts=0, stage="fixed_task")

    def save(name):
        return save_mappo(
            output / name,
            actor,
            critic,
            optimizer,
            config=config,
            provenance=env.provenance,
            training=metadata,
            curriculum=state,
            loss=loss,
        )

    save("initial.pt")
    collector = Collector(
        env,
        actor,
        frames_per_batch=frames,
        total_frames=steps // frames * frames,
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
    metadata["initialization_seconds"] = time.monotonic() - started
    durations = []
    rows = []
    fieldnames = (
        "team_steps",
        "elapsed_seconds",
        "mean_reward",
        "mean_coverage",
        "mean_approach_m",
        "episodes",
        "successes",
    )
    with (output / "progress.csv").open("w", newline="") as stream:
        csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n").writeheader()
    iterator = iter(collector)
    try:
        for _ in range(steps // frames):
            elapsed = time.monotonic() - started
            estimate = max(durations[-3:], default=0.0) * 1.3
            if max_seconds is not None and elapsed + estimate >= max_seconds:
                break
            rollout_start = time.monotonic()
            # 1. Collect real contact-driven transitions. 2. Compute TorchRL GAE.
            batch = next(iterator)
            loss.value_estimator(batch)
            buffer.empty()
            buffer.extend(batch.reshape(-1))
            # 3. Optimize only the current on-policy batch with TorchRL MAPPO.
            for _ in range(settings.epochs):
                for minibatch in buffer:
                    metrics = loss(minibatch)
                    objective = sum(
                        metrics[key]
                        for key in ("loss_objective", "loss_critic", "loss_entropy")
                    )
                    optimizer.zero_grad()
                    objective.backward()
                    torch.nn.utils.clip_grad_norm_(
                        loss.parameters(), settings.max_grad_norm
                    )
                    optimizer.step()
            collector.update_policy_weights_()
            env.check_physics()
            # 4. Synchronize through diagnostic scalars and record the actual budget.
            state["transitions"] += batch.numel()
            state["rollouts"] += 1
            row = dict(
                team_steps=state["transitions"],
                elapsed_seconds=time.monotonic() - started,
                mean_reward=float(batch["next", "reward"].mean()),
                mean_coverage=float(batch["next", "coverage"].mean()),
                mean_approach_m=float(batch["next", "approach"].mean()),
                episodes=int(batch["next", "done"].sum()),
                successes=int((batch["next", "success"] & batch["next", "done"]).sum()),
            )
            rows.append(row)
            durations.append(time.monotonic() - rollout_start)
            with (output / "progress.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=row, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            if state["rollouts"] % 10 == 0:
                print(
                    f"{state['transitions']:,} steps, {row['elapsed_seconds']:.0f} s, "
                    f"coverage {row['mean_coverage']:.3f}"
                )
        metadata.update(
            team_steps=state["transitions"],
            rollouts=state["rollouts"],
            wall_seconds=time.monotonic() - started,
            stopped_by="step_budget"
            if state["transitions"] >= steps // frames * frames
            else "time_budget",
        )
        checkpoint = save("pusht.pt")
        (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved {checkpoint}: {state['transitions']:,} steps")
        return checkpoint
    finally:
        collector.shutdown()
