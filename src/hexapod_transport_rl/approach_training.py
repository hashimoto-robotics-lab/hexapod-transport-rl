"""Reward-only curriculum learning for approach, retaining the learned pusher.

No expert action, demonstration, imitation loss, or behavior cloning is used.
Physics collection is parallel; training episodes stop at the handover condition.
"""

import hashlib
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from gymnasium.vector import VectorEnv

from .approach import CURRICULUM_LAYOUTS, OBS_DIM, ApproachConfig, make_approach_vector
from .mappo import MAPPO, PPO_SETTINGS, collect_rollout, load_checkpoint, update_policy
from .types import Info

FORMAT = "hexapod-approach-mappo-v1"
LEGACY_FORMAT = "sixtrail-approach-mappo-v1"
ADVANCE_SUCCESS_RATE = 0.75
MIN_PHASE_ITERATIONS = 50
VALIDATION_SEED_START = 62000
VALIDATION_SEED_STRIDE = 1000
CHECKPOINT_INTERVAL = 25


def load_approach_checkpoint(path: str | Path) -> tuple[MAPPO, MAPPO, Info]:
    """回り込みと押す方策を読み込み、形式とSHA-256を確認する。"""
    saved = torch.load(path, weights_only=True, map_location="cpu")
    if (
        saved.get("format") not in (FORMAT, LEGACY_FORMAT)
        or saved.get("obs_dim") != OBS_DIM
    ):
        raise ValueError("Expected a hexapod approach checkpoint")
    actor = MAPPO(OBS_DIM, 2)
    actor.load_state_dict(saved["model"])
    if any(not torch.isfinite(v).all() for v in actor.state_dict().values()):
        raise ValueError("Non-finite approach weights")
    pushing_path = Path(saved["pushing_checkpoint"])
    if not pushing_path.is_absolute():
        pushing_path = Path(path).resolve().parent / pushing_path
    saved = {**saved, "pushing_checkpoint": str(pushing_path.resolve())}
    if hashlib.sha256(pushing_path.read_bytes()).hexdigest() != saved["pushing_sha256"]:
        raise ValueError("Frozen pushing checkpoint has changed")
    pusher, push_saved = load_checkpoint(pushing_path)
    if push_saved["config"]["shape"] != "T" or push_saved["config"]["num_robots"] != 2:
        raise ValueError("Expected the two-robot T pushing actor")
    return actor.eval(), pusher, saved


def save_checkpoint(
    path: Path,
    agent: MAPPO,
    optimizer: torch.optim.Adam,
    run: Info,
    iteration: int,
    transitions: int,
    level: int,
) -> None:
    """重み・optimizer・カリキュラム進捗を一時ファイル経由で保存する。"""
    saved = dict(
        format=FORMAT,
        obs_dim=OBS_DIM,
        model=agent.state_dict(),
        optimizer=optimizer.state_dict(),
        iteration=iteration,
        transitions=transitions,
        curriculum_level=level,
        pushing_checkpoint=os.path.relpath(
            run["pushing_checkpoint"], path.resolve().parent
        ),
        pushing_sha256=run["pushing_sha256"],
        approach_config=run["approach_config"],
        run=run,
    )
    temporary = path.with_suffix(".tmp")
    torch.save(saved, temporary)
    temporary.replace(path)


def validate(agent: MAPPO, envs: VectorEnv, layout: str, seed: int) -> float:
    """One independent episode per lane; validation experiences are discarded."""
    envs.call("set_layout", layout)
    obs, _ = envs.reset(seed=seed)
    finished = np.zeros(envs.num_envs, dtype=bool)
    successes = np.zeros(envs.num_envs, dtype=bool)
    while not finished.all():
        obs, _, terminated, truncated, info = envs.step(agent.act(obs))
        done = terminated | truncated
        first_done = done & ~finished
        successes[first_done] = info["is_success"][first_done]
        finished |= done
        if done.any() and not finished.all():
            obs, _ = envs.reset(options={"reset_mask": done})
    return float(successes.mean())


def _initialize_navigator(
    pushing_checkpoint: Path, resume: str | Path | None
) -> tuple[MAPPO, torch.optim.Adam, Info | None]:
    """新規actorを作成する。再開時は重みとoptimizerを復元する。"""
    navigator = MAPPO(OBS_DIM, 2)
    with torch.no_grad():
        navigator.log_std.fill_(-1.0)
    optimizer = torch.optim.Adam(navigator.parameters(), lr=PPO_SETTINGS.learning_rate)
    if not resume:
        return navigator, optimizer, None

    navigator, _, saved = load_approach_checkpoint(resume)
    if saved["pushing_checkpoint"] != str(pushing_checkpoint):
        raise ValueError("Resume requires the same frozen pusher")
    optimizer = torch.optim.Adam(navigator.parameters(), lr=PPO_SETTINGS.learning_rate)
    optimizer.load_state_dict(saved["optimizer"])
    return navigator, optimizer, saved


def train_approach(
    output: str | Path,
    pushing_checkpoint: str | Path,
    *,
    config: ApproachConfig | None = None,
    iterations: int = 700,
    num_envs: int = 8,
    horizon: int = 64,
    epochs: int = 4,
    minibatch: int = 128,
    seed: int = 20261008,
    validate_every: int = 50,
    resume: str | Path | None = None,
    asset_root: str | Path | None = None,
) -> Path:
    """Learn from this config's reward/reset settings, saving the final actor.

    Layout follows the near/rear/side/front curriculum. Passing config on resume
    requires the same reward/reset settings; omitting it restores saved settings.
    """
    # 1. 設定・乱数・学習開始位置を準備する。
    if min(iterations, num_envs, horizon, epochs, minibatch, validate_every) < 1:
        raise ValueError("Training sizes must be positive")
    if num_envs * horizon < 2:
        raise ValueError("At least two rollout transitions required")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pushing_checkpoint = Path(pushing_checkpoint).resolve()
    _, push_saved = load_checkpoint(pushing_checkpoint)
    if push_saved["config"]["shape"] != "T" or push_saved["config"]["num_robots"] != 2:
        raise ValueError("Expected the two-robot T pushing checkpoint")
    requested_config = config
    config = config or ApproachConfig(layout="near", easier_reset_fraction=0.25)
    navigator, optimizer, resumed_state = _initialize_navigator(
        pushing_checkpoint, resume
    )
    start_iteration, transitions, curriculum_level = 0, 0, 0
    if resumed_state is not None:
        start_iteration = resumed_state["iteration"]
        transitions = resumed_state["transitions"]
        curriculum_level = resumed_state["curriculum_level"]
        saved_config = ApproachConfig(**resumed_state["approach_config"])
        if (
            requested_config is not None
            and replace(requested_config, layout=saved_config.layout) != saved_config
        ):
            raise ValueError(
                "Resume reward/reset settings differ from saved experiment"
            )
        config = saved_config
    config = replace(config, layout=CURRICULUM_LAYOUTS[curriculum_level])
    started = time.perf_counter()
    # 2. 学習用と検証用の世界を別々に準備する。
    # Override the original machine's asset path when sharing the checkpoints.
    train_envs = make_approach_vector(config, num_envs, asset_root)
    validation_envs = make_approach_vector(
        replace(config, easier_reset_fraction=0),
        min(8, num_envs),
        asset_root,
    )
    try:
        run = dict(
            approach_config=asdict(config),
            seed=seed,
            iterations=iterations,
            num_envs=num_envs,
            horizon=horizon,
            epochs=epochs,
            minibatch=minibatch,
            resume=str(resume) if resume else None,
            pushing_checkpoint=str(pushing_checkpoint),
            pushing_sha256=hashlib.sha256(pushing_checkpoint.read_bytes()).hexdigest(),
            provenance=train_envs.call("provenance")[0],
            initialization="random navigation actor; reward-only PPO; frozen existing pusher",
            curriculum=list(CURRICULUM_LAYOUTS),
            advance_threshold=ADVANCE_SUCCESS_RATE,
            validation_seeds=[
                VALIDATION_SEED_START + i * VALIDATION_SEED_STRIDE
                for i in range(len(CURRICULUM_LAYOUTS))
            ],
            validation_episodes=validation_envs.num_envs,
        )
        (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
        observations, _ = train_envs.reset(seed=seed)
        completed_episodes = successes = 0
        best_front_success_rate = -1
        phase_start_iteration = start_iteration
        for iteration in range(start_iteration + 1, start_iteration + iterations + 1):
            # 3. 経験を集め、共有actorと中央criticを更新する。
            iteration_started_at = time.perf_counter()
            rollout = collect_rollout(train_envs, navigator, observations, horizon)
            collected_at = time.perf_counter()
            losses = update_policy(navigator, optimizer, rollout, epochs, minibatch)
            updated_at = time.perf_counter()
            observations = rollout.next_observations
            transitions += num_envs * horizon
            completed_episodes += rollout.completed_episodes
            successes += rollout.successes
            record = dict(
                iteration=iteration,
                transitions=transitions,
                layout=CURRICULUM_LAYOUTS[curriculum_level],
                completed_episodes=completed_episodes,
                successes=successes,
                training_success_rate=successes / max(completed_episodes, 1),
                mean_reward=float(rollout.rewards.mean()),
                **losses,
                elapsed_seconds=updated_at - started,
                rollout_seconds=collected_at - iteration_started_at,
                update_seconds=updated_at - collected_at,
                transitions_per_second=num_envs
                * horizon
                / (updated_at - iteration_started_at),
            )
            # 4. 検証結果から保存候補と次の難度を決める。
            if (
                iteration % validate_every == 0
                or iteration == start_iteration + iterations
            ):
                validation_success_rate = validate(
                    navigator,
                    validation_envs,
                    CURRICULUM_LAYOUTS[curriculum_level],
                    VALIDATION_SEED_START + curriculum_level * VALIDATION_SEED_STRIDE,
                )
                record["validation_success_rate"] = validation_success_rate
                if (
                    curriculum_level == len(CURRICULUM_LAYOUTS) - 1
                    and validation_success_rate > best_front_success_rate
                ):
                    best_front_success_rate = validation_success_rate
                    save_checkpoint(
                        output / "best_front_checkpoint.pt",
                        navigator,
                        optimizer,
                        run,
                        iteration,
                        transitions,
                        curriculum_level,
                    )
                if (
                    validation_success_rate >= ADVANCE_SUCCESS_RATE
                    and curriculum_level < len(CURRICULUM_LAYOUTS) - 1
                    and iteration - phase_start_iteration >= MIN_PHASE_ITERATIONS
                ):
                    curriculum_level += 1
                    phase_start_iteration = iteration
                    train_envs.call("set_layout", CURRICULUM_LAYOUTS[curriculum_level])
                    observations, _ = train_envs.reset()
                    completed_episodes = successes = 0
                    record["advanced_to"] = CURRICULUM_LAYOUTS[curriculum_level]
            # 5. 指標を記録し、再開用のcheckpointを定期保存する。
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if (
                iteration % CHECKPOINT_INTERVAL == 0
                or iteration == start_iteration + iterations
            ):
                save_checkpoint(
                    output / "checkpoint.pt",
                    navigator,
                    optimizer,
                    run,
                    iteration,
                    transitions,
                    curriculum_level,
                )
        (output / "duration.json").write_text(
            json.dumps({"wall_seconds": time.perf_counter() - started}) + "\n"
        )
    finally:
        train_envs.close()
        validation_envs.close()
    return output / "checkpoint.pt"
