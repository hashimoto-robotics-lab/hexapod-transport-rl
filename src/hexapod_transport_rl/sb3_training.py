"""Domain curriculum and transport metadata around Stable-Baselines3.

This module contains no rollout buffer, advantage calculation, optimizer step,
or PPO loss. Students call PPO.learn() and PPO.save() themselves.
"""

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from .approach import CURRICULUM_LAYOUTS, ApproachConfig
from .config import PushConfig
from .mappo import load_checkpoint
from .sb3_policy import SB3Navigator, SharedTeamPolicy

FORMAT = "hexapod-transport-sb3-ppo-v1"


class ApproachCurriculum(BaseCallback):
    """Validate on separate seeds; adjust the next reset's layout every 50 rollouts.

    Validation uses the currently trained policy before that rollout's PPO update.
    Existing episodes finish in their original layout, preserving SB3's buffer
    and final-observation/time-limit handling. Only later resets change layout.
    """

    def __init__(
        self,
        config: ApproachConfig,
        output: str | Path,
        *,
        validate_every=50,
        minimum_rollouts=50,
    ):
        super().__init__()
        if min(validate_every, minimum_rollouts) < 1:
            raise ValueError("Curriculum intervals must be positive")
        self.config = config
        self.output = Path(output)
        self.validate_every = validate_every
        self.minimum_rollouts = minimum_rollouts
        self.rollouts = self.level = self.phase_start = 0
        self.validation_env = None
        self.last_validation_success_rate = None

    def _on_training_start(self):
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "run.json").exists():
            raise FileExistsError("Use a new output directory for the training attempt")
        record = dict(
            algorithm="Stable-Baselines3 PPO; shared local actor, central team critic, joint-action ratio",
            stable_baselines3=stable_baselines3.__version__,
            approach_config=asdict(self.config),
            seed=self.model.seed,
            num_envs=self.training_env.num_envs,
            horizon=self.model.n_steps,
            batch_size=self.model.batch_size,
            epochs=self.model.n_epochs,
            learning_rate=self.model.lr_schedule(1.0),
            discount=self.model.gamma,
            gae_lambda=self.model.gae_lambda,
            clip_range=self.model.clip_range(1.0),
            entropy_coefficient=self.model.ent_coef,
            provenance=self.training_env.get_attr("provenance")[0],
            curriculum=list(CURRICULUM_LAYOUTS),
            advance_threshold=0.75,
            validation_seed_start=62000,
            validation_seed_stride=1000,
        )
        (self.output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        self.validation_env = make_vec_env(
            "hexapod_transport_rl:HexapodApproach-v0",
            n_envs=self.training_env.num_envs,
            env_kwargs={
                "config": replace(self.config, layout="near", easier_reset_fraction=0),
                "flatten": True,
            },
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={"start_method": "spawn"},
        )

    def _on_step(self):
        return True

    def _validate(self) -> float:
        self.validation_env.env_method("set_layout", CURRICULUM_LAYOUTS[self.level])
        self.validation_env.seed(62000 + self.level * 1000)
        observation = self.validation_env.reset()
        finished = [False] * self.validation_env.num_envs
        successes = 0
        while not all(finished):
            action, _ = self.model.predict(observation, deterministic=True)
            observation, _, done, infos = self.validation_env.step(action)
            for index, ended in enumerate(done):
                if ended and not finished[index]:
                    successes += int(infos[index]["is_success"])
                    finished[index] = True
        return successes / len(finished)

    def _on_rollout_end(self):
        self.rollouts += 1
        self.logger.record("curriculum/layout", CURRICULUM_LAYOUTS[self.level])
        self.logger.record("curriculum/rollouts", self.rollouts)
        if self.rollouts % self.validate_every == 0:
            success_rate = self._validate()
            self.last_validation_success_rate = success_rate
            self.logger.record("curriculum/validation_success_rate", success_rate)
            if (
                success_rate >= 0.75
                and self.rollouts - self.phase_start >= self.minimum_rollouts
                and self.level < len(CURRICULUM_LAYOUTS) - 1
            ):
                self.level += 1
                self.phase_start = self.rollouts
                self.training_env.env_method(
                    "set_layout", CURRICULUM_LAYOUTS[self.level]
                )
        self._save_state()

    def _save_state(self):
        state = dict(
            rollouts=self.rollouts,
            level=self.level,
            phase_start=self.phase_start,
            layout=CURRICULUM_LAYOUTS[self.level],
            validation_success_rate=self.last_validation_success_rate,
        )
        (self.output / "curriculum.json").write_text(json.dumps(state, indent=2) + "\n")

    def _on_training_end(self):
        self._save_state()
        if self.validation_env is not None:
            self.validation_env.close()


def bind_transport(
    navigation_checkpoint: str | Path,
    pushing_checkpoint: str | Path,
    output: str | Path,
    *,
    config: ApproachConfig,
    provenance: dict,
) -> Path:
    """Record the SB3 zip and the frozen pusher for transport replay, without training."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    navigation_checkpoint = Path(navigation_checkpoint).resolve()
    pushing_checkpoint = Path(pushing_checkpoint).resolve()
    model = PPO.load(navigation_checkpoint, device="cpu")
    if (
        not isinstance(model.policy, SharedTeamPolicy)
        or model.observation_space.shape != (20,)
        or model.action_space.shape != (6,)
    ):
        raise ValueError("Expected a two-robot SB3 SharedTeamPolicy approach model")
    actual_config = ApproachConfig(**provenance["approach_config"])
    if replace(config, layout=actual_config.layout) != actual_config:
        raise ValueError(
            "Recorded reward/reset settings differ from the training environment"
        )
    _, pushing_saved = load_checkpoint(pushing_checkpoint)
    pushing_config = PushConfig(**pushing_saved["config"])
    if pushing_config.num_robots != 2 or pushing_config.shape != "T":
        raise ValueError("Expected the frozen two-robot T pusher")
    for key in ("low_level_sha256", "robot_xml_sha256"):
        if provenance[key] != pushing_saved[key]:
            raise ValueError("Navigation and pushing assets differ")
    metadata = dict(
        format=FORMAT,
        navigation_checkpoint=os.path.relpath(navigation_checkpoint, output.parent),
        navigation_sha256=hashlib.sha256(
            navigation_checkpoint.read_bytes()
        ).hexdigest(),
        pushing_checkpoint=os.path.relpath(pushing_checkpoint, output.parent),
        pushing_sha256=hashlib.sha256(pushing_checkpoint.read_bytes()).hexdigest(),
        approach_config=asdict(config),
        num_timesteps=model.num_timesteps,
        run=dict(
            provenance=provenance, algorithm="Stable-Baselines3 PPO", seed=model.seed
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2) + "\n")
    return output


def load_sb3_transport(path: str | Path):
    """Verify a portable transport manifest and load the two learned actors."""
    path = Path(path).resolve()
    saved = json.loads(path.read_text())
    if saved.get("format") != FORMAT:
        raise ValueError("Expected an SB3 transport manifest")
    for prefix in ("navigation", "pushing"):
        checkpoint = (path.parent / saved[f"{prefix}_checkpoint"]).resolve()
        if (
            hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            != saved[f"{prefix}_sha256"]
        ):
            raise ValueError(f"{prefix} checkpoint has changed")
        saved[f"{prefix}_checkpoint"] = str(checkpoint)
    model = PPO.load(saved["navigation_checkpoint"], device="cpu")
    if not isinstance(model.policy, SharedTeamPolicy):
        raise ValueError("Expected SharedTeamPolicy")
    pusher, _ = load_checkpoint(saved["pushing_checkpoint"])
    return SB3Navigator(model), pusher, saved
