"""Validate, change future reset difficulty and log pose learning; no PPO loop."""

import csv
import json
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torchrl

from .pose_push import POSE_STAGES, PosePushConfig, make_pose_vector
from .torchrl_mappo import MAPPOSettings, policy_action


class PoseCurriculum:
    """Advance using held-out pose successes, without feeding validation to PPO."""

    def __init__(
        self,
        config,
        env,
        actor,
        output,
        *,
        seed,
        settings,
        horizon,
        stages=POSE_STAGES,
        validate_every=25,
        minimum_rollouts=25,
        validation_episodes=6,
        advance_threshold=0.5,
        resume=None,
    ):
        if settings.gamma != config.shaping_discount:
            raise ValueError("Potential shaping discount must match MAPPO gamma")
        if min(validate_every, minimum_rollouts, validation_episodes, horizon) < 1:
            raise ValueError("Validation intervals must be positive")
        if not 0 < advance_threshold <= 1:
            raise ValueError("Advance threshold must be in (0,1]")
        self.config, self.env, self.actor = config, env, actor
        self.stages = tuple(stages)
        if not self.stages:
            raise ValueError("At least one pose stage is required")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "run.json").exists():
            raise FileExistsError("Use a new experiment directory")
        self.validate_every, self.minimum_rollouts = validate_every, minimum_rollouts
        self.validation_episodes = validation_episodes
        self.advance_threshold = advance_threshold
        self.rollouts = self.level = self.phase_start = self.transitions = 0
        self.initial_transitions = 0
        self.last_validation_success_rate = None
        self.validation_env = None
        self.started = time.monotonic()
        self.training = dict(
            algorithm="TorchRL MAPPO; shared local actor, central critic, per-agent ratios",
            torchrl=torchrl.__version__,
            torch=str(torch.__version__),
            learning_device=str(next(actor.parameters()).device),
            learning_gpu=(
                torch.cuda.get_device_name(next(actor.parameters()).device)
                if next(actor.parameters()).is_cuda
                else None
            ),
            physics_device=str(env.device),
            physics_backend=env.backend,
            seed=seed,
            num_envs=env.worlds.num_envs,
            horizon=horizon,
            settings=asdict(settings),
            pose_config=asdict(config),
            provenance=env.provenance,
            curriculum=[asdict(stage) for stage in self.stages],
            validate_every=validate_every,
            minimum_rollouts=minimum_rollouts,
            validation_episodes=validation_episodes,
            advance_threshold=advance_threshold,
            validation_seed_start=62000,
            validation_seed_stride=1000,
        )
        if env.backend == "warp":
            import mujoco_warp
            import warp

            self.training.update(
                mujoco_warp=mujoco_warp.__version__,
                warp=warp.__version__,
                physics_compile_seconds=env.worlds.physics.compile_seconds,
                contact_capacity=env.worlds.physics.data.naconmax,
                constraint_capacity_per_world=env.worlds.physics.data.njmax,
                reset_rng="torch CUDA; uniform reset ranges match native MuJoCo",
            )
        if resume:
            if (
                PosePushConfig.from_checkpoint(resume["pose_config"]) != config
                or MAPPOSettings(**resume["training"]["settings"]) != settings
                or resume["training"]["curriculum"] != self.training["curriculum"]
                or resume["training"]["advance_threshold"] != advance_threshold
            ):
                raise ValueError(
                    "Resume requires identical pose task and update settings"
                )
            state = resume["curriculum"]
            self.rollouts, self.level, self.phase_start, self.transitions = (
                state[k] for k in ("rollouts", "level", "phase_start", "transitions")
            )
            self.initial_transitions = self.transitions
            self.last_validation_success_rate = state["validation_success_rate"]
        env.set_stage(self.stages[self.level])
        (self.output / "run.json").write_text(
            json.dumps(self.training, indent=2) + "\n"
        )

    @property
    def state(self):
        return dict(
            rollouts=self.rollouts,
            level=self.level,
            phase_start=self.phase_start,
            transitions=self.transitions,
            stage=self.stages[self.level].name,
            validation_success_rate=self.last_validation_success_rate,
        )

    def _validate(self):
        # Validation advances CPU physics step by step; copy current weights once.
        actor = (
            deepcopy(self.actor).cpu()
            if next(self.actor.parameters()).is_cuda
            else self.actor
        )
        if self.validation_env is None:
            self.validation_env = make_pose_vector(
                self.config,
                min(2, self.validation_episodes),
                self.env.provenance["asset_root"],
                asynchronous=self.env.asynchronous or self.env.backend == "warp",
            )
        self.validation_env.call("set_stage", self.stages[self.level])
        successes = []
        for start in range(0, self.validation_episodes, self.validation_env.num_envs):
            obs, _ = self.validation_env.reset(seed=62000 + self.level * 1000 + start)
            finished = np.zeros(self.validation_env.num_envs, dtype=bool)
            while not finished.all():
                obs, _, terminated, truncated, info = self.validation_env.step(
                    policy_action(actor, obs)
                )
                done = terminated | truncated
                for lane in np.flatnonzero(done & ~finished):
                    if start + lane < self.validation_episodes:
                        successes.append(bool(info["is_success"][lane]))
                finished |= done
                if done.any() and not finished.all():
                    obs, _ = self.validation_env.reset(options={"reset_mask": done})
        return float(np.mean(successes))

    def record(self, batch, metrics):
        self.env.check_physics()
        self.rollouts += 1
        self.transitions += batch.numel()
        collected = self.stages[self.level].name
        if self.rollouts % self.validate_every == 0:
            self.last_validation_success_rate = self._validate()
            if (
                self.last_validation_success_rate >= self.advance_threshold
                and self.rollouts - self.phase_start >= self.minimum_rollouts
                and self.level < len(self.stages) - 1
            ):
                self.level += 1
                self.phase_start = self.rollouts
                self.env.set_stage(self.stages[self.level])
        elapsed = time.monotonic() - self.started
        row = dict(
            rollouts=self.rollouts,
            team_steps=self.transitions,
            stage=collected,
            next_stage=self.stages[self.level].name,
            validation_success_rate=self.last_validation_success_rate,
            elapsed_seconds=elapsed,
            team_steps_per_second=(self.transitions - self.initial_transitions)
            / elapsed,
            episodes=int(batch["next", "done"].sum()),
            successes=int((batch["next", "success"] & batch["next", "done"]).sum()),
            mean_step_reward=float(batch["next", "reward"].mean()),
            **{
                k: float(metrics[k].detach().mean())
                for k in (
                    "loss_objective",
                    "loss_critic",
                    "loss_entropy",
                    "kl_approx",
                    "clip_fraction",
                )
            },
        )
        logfile = self.output / "progress.csv"
        exists = logfile.exists()
        with logfile.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        (self.output / "curriculum.json").write_text(
            json.dumps(self.state, indent=2) + "\n"
        )
        print(
            f"{self.output.name}: {self.transitions} team steps, {collected}, {row['team_steps_per_second']:.1f} steps/s"
        )

    def close(self):
        if self.validation_env is not None:
            self.validation_env.close()
            self.validation_env = None
