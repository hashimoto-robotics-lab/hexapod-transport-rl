"""Navigation curriculum and experiment logging, independent of MAPPO updates.

Students run the TorchRL Collector, replay-buffer sampler and loss in Python.
This class only validates the policy, changes future reset layouts, and records
the training conditions. It never supplies actions or demonstration targets.
"""

import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torchrl

from .approach import CURRICULUM_LAYOUTS, ApproachConfig, make_approach_vector
from .torchrl_mappo import MAPPOSettings, policy_action


class ApproachCurriculum:
    """After MAPPO updates, validate and apply the next layout to later resets.

    Validation uses separate seeds and contributes no training experiences.
    Interrupted runs can restore curriculum counters with ``resume=saved``.
    Physics states are reset on resume, not restored from the old episode.
    """

    def __init__(
        self,
        config: ApproachConfig,
        env,
        actor,
        output: str | Path,
        *,
        seed: int,
        settings: MAPPOSettings,
        horizon: int,
        validate_every=50,
        minimum_rollouts=50,
        layouts=CURRICULUM_LAYOUTS,
        resume: dict | None = None,
    ):
        if min(validate_every, minimum_rollouts, horizon) < 1:
            raise ValueError("Curriculum intervals and horizon must be positive")
        self.layouts = tuple(layouts)
        if not self.layouts or any(
            layout not in CURRICULUM_LAYOUTS for layout in self.layouts
        ):
            raise ValueError("Select curriculum layouts from near/rear/side/front")
        self.config, self.env, self.actor = config, env, actor
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "run.json").exists():
            raise FileExistsError("Use a new directory for this training attempt")
        self.validate_every, self.minimum_rollouts = validate_every, minimum_rollouts
        self.rollouts = self.level = self.phase_start = self.transitions = 0
        self.validation_env = None
        self.last_validation_success_rate = None
        self.started = time.monotonic()
        self.initial_transitions = 0
        self.training = dict(
            algorithm="TorchRL MAPPO; shared local actor, central critic, per-agent ratios",
            torchrl=torchrl.__version__,
            seed=seed,
            num_envs=env.worlds.num_envs,
            horizon=horizon,
            settings=asdict(settings),
            approach_config=asdict(config),
            provenance=env.provenance,
            curriculum=list(self.layouts),
            advance_threshold=0.75,
            validate_every=validate_every,
            minimum_rollouts=minimum_rollouts,
            validation_seed_start=62000,
            validation_seed_stride=1000,
        )
        if resume is not None:
            if (
                ApproachConfig(**resume["approach_config"]) != config
                or MAPPOSettings(**resume["training"]["settings"]) != settings
                or resume["training"]["curriculum"] != list(self.layouts)
            ):
                raise ValueError("Resume requires the same reward and MAPPO settings")
            state = resume["curriculum"]
            self.rollouts, self.level = state["rollouts"], state["level"]
            self.phase_start = state["phase_start"]
            self.transitions = self.initial_transitions = state["transitions"]
            self.last_validation_success_rate = state["validation_success_rate"]
            self.training["resumed_from_transitions"] = self.transitions
        env.set_layout(self.layouts[self.level])
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
            layout=self.layouts[self.level],
            validation_success_rate=self.last_validation_success_rate,
        )

    def _validate(self) -> float:
        if self.validation_env is None:
            self.validation_env = make_approach_vector(
                replace(self.config, easier_reset_fraction=0),
                num_envs=self.env.worlds.num_envs,
                asset_root=self.env.provenance["asset_root"],
                asynchronous=self.env.asynchronous,
            )
        self.validation_env.call("set_layout", self.layouts[self.level])
        obs, _ = self.validation_env.reset(seed=62000 + self.level * 1000)
        finished = np.zeros(self.validation_env.num_envs, dtype=bool)
        successes = np.zeros_like(finished)
        while not finished.all():
            obs, _, terminated, truncated, info = self.validation_env.step(
                policy_action(self.actor, obs)
            )
            done = terminated | truncated
            first_done = done & ~finished
            successes[first_done] = info["is_success"][first_done]
            finished |= done
            if done.any() and not finished.all():
                obs, _ = self.validation_env.reset(options={"reset_mask": done})
        return float(successes.mean())

    def record(self, batch, metrics):
        """Call once after updating a rollout; metrics are TorchRL's returned losses."""
        self.rollouts += 1
        self.transitions += batch.numel()
        collected_layout = self.layouts[self.level]
        if self.rollouts % self.validate_every == 0:
            success = self._validate()
            self.last_validation_success_rate = success
            if (
                success >= 0.75
                and self.rollouts - self.phase_start >= self.minimum_rollouts
                and self.level < len(self.layouts) - 1
            ):
                self.level += 1
                self.phase_start = self.rollouts
                self.env.set_layout(self.layouts[self.level])
        elapsed = time.monotonic() - self.started
        done = batch["next", "done"]
        successes = batch["next", "success"] & done
        row = dict(
            rollouts=self.rollouts,
            team_steps=self.transitions,
            layout=collected_layout,
            next_layout=self.layouts[self.level],
            validation_success_rate=self.last_validation_success_rate,
            elapsed_seconds=elapsed,
            team_steps_per_second=(self.transitions - self.initial_transitions)
            / elapsed,
            episodes=int(done.sum()),
            successes=int(successes.sum()),
            mean_step_reward=float(batch["next", "reward"].mean()),
            **{
                key: float(metrics[key].detach().mean())
                for key in (
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
            f"{self.output.name}: {self.transitions} team steps, {collected_layout}, "
            f"{row['team_steps_per_second']:.1f} steps/s"
        )

    def close(self):
        if self.validation_env is not None:
            self.validation_env.close()
            self.validation_env = None
