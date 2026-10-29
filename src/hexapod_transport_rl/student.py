"""Student-facing cylinder pushing: change terms, keep the original PPO/physics.

See docs/student-api.md and examples/student/ for runnable experiments.
"""

import hashlib
import importlib.util
import inspect
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
import torch

from .single_push import SinglePushConfig
from .student_terms import DEFAULT_REWARDS, OBSERVATIONS, REWARDS, PushState

__all__ = [
    "CylinderPushTask",
    "ObservationTerm",
    "RewardTerm",
    "DiscreteDirections",
    "ContinuousVelocity",
    "PushState",
    "TrainingRun",
    "train",
    "evaluate",
    "record_video",
    "load_task",
    "load_task_file",
    "compare",
    "available_observations",
    "available_rewards",
]


def _function_ref(func, artifact_dir=None):
    if isinstance(func, str):
        return {"kind": "builtin", "name": func}
    if not inspect.isfunction(func) or "<" in func.__qualname__:
        raise TypeError(
            "Custom terms must be named, module-level functions; use def instead of a lambda or nested function"
        )
    source = inspect.getsourcefile(func)
    if source is None or not Path(source).is_file():
        raise ValueError("Put custom terms in a .py file so their source can be saved")
    source = Path(source).resolve()
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    filename = f"task_sources/{digest}.py"
    if artifact_dir is not None:
        target = Path(artifact_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(content)
    return dict(
        kind="python",
        name=func.__qualname__,
        file=filename,
        original_file=str(source),
        source_sha256=digest,
        function_sha256=hashlib.sha256(inspect.getsource(func).encode()).hexdigest(),
    )


def _resolve_function(ref, artifact_dir=None):
    if ref["kind"] == "builtin":
        return ref["name"]
    candidate = (
        Path(artifact_dir) / ref["file"]
        if artifact_dir is not None
        else Path(ref["original_file"])
    )
    if not candidate.is_file():
        candidate = Path(ref["original_file"])
    if (
        not candidate.is_file()
        or hashlib.sha256(candidate.read_bytes()).hexdigest() != ref["source_sha256"]
    ):
        raise ValueError(
            "Custom task source is missing or changed; keep task_sources/ beside the checkpoint"
        )
    location = hashlib.sha256(str(candidate.resolve()).encode()).hexdigest()[:12]
    module_name = f"_hexapod_student_{ref['source_sha256']}_{location}"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, candidate)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    value = sys.modules[module_name]
    for part in ref["name"].split("."):
        value = getattr(value, part)
    return value


def _function_signature(func):
    ref = _function_ref(func)
    return {key: ref[key] for key in ("kind", "name", "function_sha256") if key in ref}


@dataclass(frozen=True)
class ObservationTerm:
    """A named physical feature or function(state) -> Tensor[B] / Tensor[B,D].

    Output is divided by scale. Scalar features become one input column.
    """

    func: str | Callable
    scale: float = 1.0
    name: str | None = None
    enabled: bool = True

    def __post_init__(self):
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Observation scale must be finite and positive")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be True or False")
        if isinstance(self.func, str) and self.func not in OBSERVATIONS:
            raise ValueError(
                f"Unknown observation {self.func!r}; available: {', '.join(OBSERVATIONS)}"
            )
        _function_ref(self.func)
        name = self.name or (
            self.func if isinstance(self.func, str) else self.func.__name__
        )
        if not name.isidentifier():
            raise ValueError("Observation names must be Python identifiers")
        object.__setattr__(self, "name", name)

    def compute(self, state):
        func = OBSERVATIONS[self.func] if isinstance(self.func, str) else self.func
        value = func(state) / self.scale
        return value if self.enabled else torch.zeros_like(value)


@dataclass(frozen=True)
class RewardTerm:
    """Function(before, after) -> Tensor[B], multiplied by weight.

    per_second=True additionally multiplies by the high-level step duration.
    Built-in terms choose the appropriate default; custom terms default False.
    """

    func: str | Callable
    weight: float = 1.0
    per_second: bool | None = None

    def __post_init__(self):
        if not math.isfinite(self.weight):
            raise ValueError("Reward weight must be finite")
        if isinstance(self.func, str) and self.func not in REWARDS:
            raise ValueError(
                f"Unknown reward {self.func!r}; available: {', '.join(REWARDS)}"
            )
        _function_ref(self.func)
        if self.per_second is None:
            object.__setattr__(
                self,
                "per_second",
                REWARDS[self.func][1] if isinstance(self.func, str) else False,
            )
        elif not isinstance(self.per_second, bool):
            raise TypeError("per_second must be True or False")

    def compute(self, before, after):
        func = REWARDS[self.func][0] if isinstance(self.func, str) else self.func
        return (
            func(before, after) * self.weight * (after.dt if self.per_second else 1.0)
        )


def _validate_action(speed, frame):
    if not math.isfinite(speed) or not 0 < speed <= 0.2:
        raise ValueError("Maximum command speed must be in (0,0.2] m/s")
    if frame not in ("world", "body"):
        raise ValueError("Action frame must be world or body")


@dataclass(frozen=True)
class DiscreteDirections:
    """Uniform direction choices starting at +X, counterclockwise; optional stop 0."""

    count: int = 8
    include_stop: bool = True
    frame: str = "world"
    speed: float = 0.15

    def __post_init__(self):
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or not 2 <= self.count <= 64
        ):
            raise ValueError("Direction count must be an integer in [2,64]")
        if not isinstance(self.include_stop, bool):
            raise TypeError("include_stop must be True or False")
        _validate_action(self.speed, self.frame)

    @property
    def action_mode(self):
        return "discrete"

    @property
    def action_count(self):
        return self.count + int(self.include_stop)

    @property
    def directions(self):
        angle = np.arange(self.count) * (2 * np.pi / self.count)
        xy = np.stack((np.cos(angle), np.sin(angle)), -1)
        if self.include_stop:
            xy = np.concatenate((np.zeros((1, 2)), xy))
        return xy.tolist()


@dataclass(frozen=True)
class ContinuousVelocity:
    """Two normalized XY components in [-1,1], projected onto the unit disk.

    The frozen walker turns toward the requested direction, as in the original
    continuous PPO. This does not expose independent joint or yaw-rate control.
    """

    frame: str = "world"
    speed: float = 0.15

    def __post_init__(self):
        _validate_action(self.speed, self.frame)

    @property
    def action_mode(self):
        return "continuous"

    @property
    def action_count(self):
        return None


def _default_config():
    return SinglePushConfig(
        start_angle_degrees=180.0,
        goal_distance=0.4,
        min_yaw_degrees=5.0,
        max_yaw_degrees=30.0,
        goal_lateral_range=0.08,
        position_jitter=0.04,
        spawn_mode="around",
        spawn_heading_mode="uniform",
        goal_position_mode="uniform",
        approach_strategy="rear_contact",
        approach_cost_weight=1.0,
        goal_distance_cost_weight=4.0,
    )


@dataclass(frozen=True)
class CylinderPushTask:
    """Select policy inputs, actions and rewards; preserve articulated physics.

    Default: original 14 inputs/9 world choices/reward, with rotation disabled.
    baseline() additionally enables the common goal-octant rotation.
    """

    observations: tuple = ("baseline",)
    actions: DiscreteDirections | ContinuousVelocity = field(
        default_factory=DiscreteDirections
    )
    rewards: Mapping = field(default_factory=lambda: dict(DEFAULT_REWARDS))
    rotation: str = "none"
    config: SinglePushConfig = field(default_factory=_default_config)

    def __post_init__(self):
        if (
            not isinstance(self.config, SinglePushConfig)
            or self.config.shape != "cylinder"
            or self.config.goal_mode != "position"
        ):
            raise ValueError(
                "Student tasks require a position-only SinglePushConfig cylinder"
            )
        if not isinstance(self.actions, (DiscreteDirections, ContinuousVelocity)):
            raise TypeError("Use DiscreteDirections or ContinuousVelocity")
        observations = tuple(
            x if isinstance(x, ObservationTerm) else ObservationTerm(x)
            for x in self.observations
        )
        if not observations or len({x.name for x in observations}) != len(observations):
            raise ValueError("Select at least one observation, with unique names")
        rewards = {}
        for name, term in self.rewards.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError("Reward names must be Python identifiers")
            rewards[name] = (
                term if isinstance(term, RewardTerm) else RewardTerm(name, float(term))
            )
        if not rewards:
            raise ValueError("Select at least one reward term")
        if self.rotation not in ("none", "goal_octant"):
            raise ValueError("rotation must be none or goal_octant")
        if self.rotation == "goal_octant" and (
            observations != (ObservationTerm("baseline"),)
            or not isinstance(self.actions, DiscreteDirections)
            or self.actions.count != 8
            or not self.actions.include_stop
            or self.actions.frame != "world"
        ):
            raise ValueError(
                "goal_octant requires baseline observations and stop + 8 WORLD directions; set rotation='none' for observation/action experiments"
            )
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "rewards", MappingProxyType(rewards))
        object.__setattr__(
            self,
            "config",
            replace(
                self.config,
                policy_rotation_mode=self.rotation,
                direction_speed=self.actions.speed,
            ),
        )

    @classmethod
    def baseline(cls):
        """Standard inputs, actions and rewards with goal-octant rotation."""
        return cls(rotation="goal_octant")

    @property
    def action_mode(self):
        return self.actions.action_mode

    def make_env(self, *, num_envs=2, backend="auto", render_mode=None):
        from .student_env import StudentPushEnv

        return StudentPushEnv(
            self, num_envs=num_envs, backend=backend, render_mode=render_mode
        )

    def policy_signature(self):
        return dict(
            observations=[
                dict(func=_function_signature(t.func), scale=t.scale, enabled=t.enabled)
                for t in self.observations
            ],
            actions=dict(kind=self.action_mode, **asdict(self.actions)),
        )

    def to_dict(self, artifact_dir=None):
        return dict(
            version=1,
            config=asdict(self.config),
            rotation=self.rotation,
            observations=[
                dict(
                    func=_function_ref(t.func, artifact_dir),
                    scale=t.scale,
                    name=t.name,
                    enabled=t.enabled,
                )
                for t in self.observations
            ],
            rewards={
                name: dict(
                    func=_function_ref(t.func, artifact_dir),
                    weight=t.weight,
                    per_second=t.per_second,
                )
                for name, t in self.rewards.items()
            },
            actions=dict(kind=self.action_mode, **asdict(self.actions)),
            policy_signature=self.policy_signature(),
        )

    @classmethod
    def from_dict(cls, data, artifact_dir=None):
        if data.get("version") != 1:
            raise ValueError("Unsupported student task version")
        actions = dict(data["actions"])
        kind = actions.pop("kind")
        if kind not in ("discrete", "continuous"):
            raise ValueError("Unsupported student action type")
        return cls(
            config=SinglePushConfig.from_checkpoint(data["config"]),
            rotation=data["rotation"],
            actions=(DiscreteDirections if kind == "discrete" else ContinuousVelocity)(
                **actions
            ),
            observations=tuple(
                ObservationTerm(
                    _resolve_function(t["func"], artifact_dir),
                    scale=t["scale"],
                    name=t["name"],
                    enabled=t.get("enabled", True),
                )
                for t in data["observations"]
            ),
            rewards={
                name: RewardTerm(
                    _resolve_function(t["func"], artifact_dir),
                    weight=t["weight"],
                    per_second=t["per_second"],
                )
                for name, t in data["rewards"].items()
            },
        )


def available_observations():
    return tuple(OBSERVATIONS)


def available_rewards():
    return tuple(REWARDS)


def load_task(checkpoint):
    """Restore a saved student task, or the exact legacy policy interface."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("student_task") is not None:
        return CylinderPushTask.from_dict(
            saved["student_task"], Path(checkpoint).parent
        )
    from .torchrl_ppo import FORMAT

    if saved.get("format") != FORMAT:
        raise ValueError("Expected a single-hexapod PPO checkpoint")
    cfg = SinglePushConfig.from_checkpoint(saved["config"])
    if cfg.shape != "cylinder" or cfg.goal_mode != "position":
        raise ValueError(
            "Only legacy cylinder position tasks can enter the student API"
        )
    actions = (
        DiscreteDirections(speed=cfg.direction_speed)
        if saved["action_mode"] == "discrete"
        else ContinuousVelocity(speed=cfg.direction_speed)
    )
    return CylinderPushTask(
        config=cfg,
        actions=actions,
        rotation=cfg.policy_rotation_mode,
        rewards={
            **DEFAULT_REWARDS,
            "goal_distance": -cfg.goal_distance_cost_weight,
            "approach_distance": -cfg.approach_cost_weight,
            "success": cfg.success_bonus,
        },
    )


def load_task_file(path):
    """Load TASK from a self-contained .py task file (main guards are skipped)."""
    path = Path(path).resolve()
    if not path.is_file() or path.suffix != ".py":
        raise ValueError("Choose an existing Python task file")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        task = _resolve_function(
            dict(
                kind="python",
                name="TASK",
                original_file=str(path),
                file=path.name,
                source_sha256=digest,
            )
        )
    except AttributeError as error:
        raise ValueError(
            "The task file must define TASK = CylinderPushTask(...)"
        ) from error
    if not isinstance(task, CylinderPushTask):
        raise TypeError("The task file must define TASK = CylinderPushTask(...)")
    return task


@dataclass(frozen=True)
class TrainingRun:
    directory: Path
    latest: Path
    best: Path
    selection: str = "best"

    @property
    def checkpoint(self):
        return self.latest if self.selection == "latest" else self.best

    def evaluate(self, *, episodes=128, seed=240000, backend="auto"):
        result = evaluate(
            self.checkpoint, episodes=episodes, seed=seed, backend=backend
        )
        filename = self.directory / f"evaluation_{result['backend']}_s{seed}.json"
        filename.write_text(json.dumps(result, indent=2) + "\n")
        return result

    def video(self, path=None, *, seed=240000):
        return record_video(
            self.checkpoint, path or self.directory / "episode.mp4", seed=seed
        )


def train(
    task,
    output,
    *,
    seed=42,
    steps=655_360,
    max_seconds=None,
    num_envs=None,
    horizon=64,
    settings=None,
    backend="auto",
    validation_envs=None,
    validate_every=20,
    initial_checkpoint=None,
    checkpoint_selection="best",
    curriculum=None,
):
    """Run the existing ClipPPOLoss/GAE loop with a student task.

    A fresh actor/critic is the default. initial_checkpoint explicitly requests
    compatible actor/critic weights and a fresh optimizer, not a resumed run.
    curriculum schedules reset distributions and learning rates in this same
    run; actor, critic and optimizer are retained across phases.
    checkpoint_selection='latest' reports the final updated policy, including
    for evaluate(), video() and compare(); 'best' selects using validation.
    """
    from .torchrl_ppo import train_ppo

    if not isinstance(task, CylinderPushTask):
        raise TypeError("Use CylinderPushTask")
    gpu = backend == "warp" or (backend == "auto" and torch.cuda.is_available())
    worlds = (512 if gpu else 2) if num_envs is None else num_envs
    if min(worlds, horizon, steps) < 1 or steps % (worlds * horizon):
        raise ValueError(
            "steps must be a positive multiple of num_envs * horizon; this keeps comparisons at equal sample counts"
        )
    latest = train_ppo(
        task.config,
        output,
        action_mode=task.action_mode,
        seed=seed,
        steps=steps,
        max_seconds=max_seconds,
        num_envs=num_envs,
        horizon=horizon,
        settings=settings,
        backend=backend,
        validation_envs=validation_envs,
        validate_every=validate_every,
        initial_checkpoint=initial_checkpoint,
        checkpoint_selection=checkpoint_selection,
        curriculum=curriculum,
        student_task=task,
    )
    return TrainingRun(
        Path(output).resolve(),
        latest.resolve(),
        (Path(output) / "best.pt").resolve(),
        checkpoint_selection,
    )


def evaluate(checkpoint, *, episodes=128, seed=240000, backend="auto"):
    from .torchrl_ppo import evaluate_ppo

    return evaluate_ppo(checkpoint, episodes=episodes, seed=seed, backend=backend)


def compare(
    tasks,
    output,
    *,
    seeds=(42, 43, 44),
    evaluation_episodes=128,
    evaluation_seed=240000,
    **training_options,
):
    """Train each condition/seed and save per-run JSON, CSV, and seed summaries.

    Physics, resets, episode limits and success criteria must be common. Each
    call to train starts fresh unless initial_checkpoint is explicitly supplied.
    """
    import csv
    import statistics

    if not tasks or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide named tasks and unique training seeds")
    if "seed" in training_options or "output" in training_options:
        raise ValueError("Use seeds= and the comparison output directory")
    if any(not isinstance(name, str) or not name.isidentifier() for name in tasks):
        raise ValueError("Condition names must be Python identifiers")
    common = None
    allowed = {
        "policy_rotation_mode",
        "direction_speed",
        "approach_strategy",
        "rear_approach_gain",
        "approach_cost_weight",
        "goal_distance_cost_weight",
        "success_bonus",
    }
    for task in tasks.values():
        cfg = {
            key: value
            for key, value in asdict(task.config).items()
            if key not in allowed
        }
        if common is not None and cfg != common:
            raise ValueError(
                "Comparison tasks must share physics, resets, duration and success criteria"
            )
        common = cfg
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    result = dict(
        evaluation_seed=evaluation_seed,
        evaluation_episodes=evaluation_episodes,
        seeds=list(seeds),
        common_environment=common,
        records=records,
        summary={},
    )
    for name, task in tasks.items():
        for seed in seeds:
            run = train(
                task, output / name / f"seed_{seed}", seed=seed, **training_options
            )
            report = run.evaluate(
                episodes=evaluation_episodes,
                seed=evaluation_seed,
                backend=training_options.get("backend", "auto"),
            )
            metadata = json.loads((run.directory / "run.json").read_text())
            records.append(
                dict(
                    condition=name,
                    seed=seed,
                    checkpoint=str(run.checkpoint),
                    trained_steps=metadata["steps"],
                    training_seconds=metadata["seconds"],
                    rotation=task.rotation,
                    policy_input_dim=report["policy_input_dim"],
                    action_mode=task.action_mode,
                    action_frame=task.actions.frame,
                    successes=report["successes"],
                    episodes=report["episode_count"],
                    success_rate=report["successes"] / report["episode_count"],
                    mean_distance_m=report["mean_distance_m"],
                    mean_episode_seconds=statistics.mean(
                        r["seconds"] for r in report["episodes"]
                    ),
                )
            )
            group = [r for r in records if r["condition"] == name]
            rates = [r["success_rate"] for r in group]
            result["summary"][name] = dict(
                seeds_completed=len(group),
                mean_success_rate=statistics.mean(rates),
                std_success_rate=statistics.stdev(rates) if len(rates) > 1 else None,
            )
            (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
            with (output / "comparison.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    return result


@torch.no_grad()
def record_video(checkpoint, path, *, seed=240000):
    """Native CPU MuJoCo frames at 25 Hz, including the outcome in a sidecar."""
    import imageio.v2 as imageio
    from torchrl.envs.utils import ExplorationType, set_exploration_type

    from .torchrl_ppo import load_ppo, make_checkpoint_env

    saved, cfg, actor, _ = load_ppo(checkpoint)
    env = make_checkpoint_env(
        checkpoint, saved, cfg, num_envs=1, backend="cpu", render_mode="rgb_array_list"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        env.set_seed(seed)
        td = env.reset()
        core = env.worlds.envs[0]
        mode = (
            ExplorationType.MODE
            if env.action_mode == "discrete"
            else ExplorationType.DETERMINISTIC
        )
        with (
            imageio.get_writer(path, fps=25, macro_block_size=1) as writer,
            set_exploration_type(mode),
        ):
            for frame in core.render():
                writer.append_data(frame)
            for _ in range(round(cfg.episode_seconds / cfg.dt)):
                actor(td)
                td = env.step(td)["next"]
                for frame in core.render():
                    writer.append_data(frame)
                if td["done"].any():
                    break
        result = dict(
            checkpoint=str(Path(checkpoint).resolve()),
            video=str(path.resolve()),
            seed=seed,
            fps=25,
            success=bool(td["success"].item()),
            distance_m=float(td["distance"].item()),
            seconds=core.core.steps * cfg.dt,
        )
        path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        env.close()


def snapshot_student_task(task, output, *, observation_layout=None):
    """Save readable configuration, custom source, package source and versions."""
    import importlib.metadata

    output = Path(output)
    data = task.to_dict(output)
    data["observation_layout"] = observation_layout
    (output / "task.json").write_text(json.dumps(data, indent=2) + "\n")
    source = Path(__file__).parent
    target = output / "source_snapshot" / "hexapod_transport_rl"
    target.mkdir(parents=True, exist_ok=True)
    for file in source.glob("*.py"):
        shutil.copy2(file, target / file.name)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "torchrl", "tensordict", "mujoco", "gymnasium", "numpy")
    }
    for name in ("mujoco-warp", "warp-lang"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    (output / "environment.json").write_text(
        json.dumps(dict(python=sys.version, packages=versions), indent=2) + "\n"
    )
    return data
