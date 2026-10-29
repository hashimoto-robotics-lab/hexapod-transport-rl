"""Ordinary single-agent TorchRL PPO with categorical or Gaussian actions."""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torch.distributions import Categorical
from torchrl.collectors import Collector
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss

from .single_push import (
    DIRECTIONS,
    SinglePushConfig,
    SinglePushEnv,
    SinglePushResetStage,
)


def _native_metadata(value):
    """Keep checkpoint metadata compatible with weights_only=True loading."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _native_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_native_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple((_native_metadata(item) for item in value))
    return value


class SharedScale(nn.Module):
    """Learn one exploration scale per action component, shared across robots."""

    def __init__(self, initial_log_std=-1.0, action_dim=3):
        super().__init__()
        if not np.isfinite(initial_log_std):
            raise ValueError("Initial log standard deviation must be finite")
        self.log_std = nn.Parameter(torch.full((action_dim,), float(initial_log_std)))

    def forward(self, loc):
        return (loc, self.log_std.exp().expand_as(loc))


FORMAT = "single-hexapod-torchrl-ppo-v1"
STUDENT_FORMAT = "single-hexapod-student-ppo-v1"


@dataclass(frozen=True)
class PPOSettings:
    epochs: int = 4
    minibatch_size: int = 2048
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    critic_coeff: float = 0.5
    max_grad_norm: float = 0.5

    def __post_init__(self):
        if self.epochs < 1 or self.minibatch_size < 2:
            raise ValueError("Positive epochs and minibatches of at least two required")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Discount and GAE lambda must be in [0,1]")
        values = torch.tensor(
            [
                self.learning_rate,
                self.clip_epsilon,
                self.entropy_coeff,
                self.critic_coeff,
                self.max_grad_norm,
            ]
        )
        if (
            not torch.isfinite(values).all()
            or (values < 0).any()
            or self.learning_rate == 0
        ):
            raise ValueError("PPO settings must be finite and nonnegative")


class GoalOctantMLP(nn.Sequential):
    """Express a state near +45 degrees, returning the original WORLD logits.

    Only a multiple of 45 degrees is used, so the nine-action set is preserved
    exactly. The MLP still chooses every action including stop. There is no
    suggested pushing action or path. Sequential keys match existing weights.
    """

    def __init__(self, *layers, actor=False):
        super().__init__(*layers)
        self.actor = actor

    def forward(self, observation):
        step = torch.pi / 4
        angle = torch.atan2(observation[..., 3], observation[..., 2])
        octant = torch.floor((angle - step) / step + 0.5).long().remainder(8)
        frame = octant.to(observation.dtype) * step
        c, s = frame.cos(), frame.sin()
        canonical = observation.clone()
        for start in (0, 2, 6, 8):
            x, y = observation[..., start], observation[..., start + 1]
            canonical[..., start] = c * x + s * y
            canonical[..., start + 1] = -s * x + c * y
        sine, cosine = observation[..., 11], observation[..., 12]
        canonical[..., 11] = sine * c - cosine * s
        canonical[..., 12] = cosine * c + sine * s
        output = super().forward(canonical)
        if not self.actor:
            return output
        indices = (torch.arange(8, device=output.device) - octant[..., None]).remainder(
            8
        ) + 1
        return torch.cat((output[..., :1], output.gather(-1, indices)), -1)


def _mlp(output, *, actor=False, rotation_mode="none", observation_dim=14):
    layers = [
        nn.Linear(observation_dim, 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh(),
        nn.Linear(128, output),
    ]
    network = (
        GoalOctantMLP(*layers, actor=actor)
        if rotation_mode == "goal_octant"
        else nn.Sequential(*layers)
    )
    for module in network.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, 2**0.5)
            nn.init.zeros_(module.bias)
    nn.init.orthogonal_(network[-1].weight, 0.01 if actor else 1.0)
    return network


def make_ppo_networks(
    action_mode="discrete",
    *,
    device="cpu",
    rotation_mode="none",
    observation_dim=14,
    action_count=9,
):
    if action_mode not in ("discrete", "continuous"):
        raise ValueError("Unknown action mode")
    if rotation_mode not in ("none", "goal_octant") or (
        rotation_mode != "none" and action_mode != "discrete"
    ):
        raise ValueError("Goal-octant rotation supports nine discrete actions only")
    if observation_dim < 1 or action_count < 2:
        raise ValueError(
            "Positive observation size and at least two discrete actions required"
        )
    if rotation_mode != "none" and (observation_dim != 14 or action_count != 9):
        raise ValueError("goal_octant requires the original 14 inputs and 9 actions")
    # Both critics start identically for the same torch seed.
    critic = TensorDictModule(
        _mlp(1, rotation_mode=rotation_mode, observation_dim=observation_dim),
        in_keys=["observation"],
        out_keys=["state_value"],
    )
    if action_mode == "discrete":
        module = TensorDictModule(
            _mlp(
                action_count,
                actor=True,
                rotation_mode=rotation_mode,
                observation_dim=observation_dim,
            ),
            in_keys=["observation"],
            out_keys=["logits"],
        )
        actor = ProbabilisticActor(
            module,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=Categorical,
            return_log_prob=True,
            default_interaction_type=ExplorationType.RANDOM,
        )
    else:
        module = TensorDictModule(
            nn.Sequential(
                _mlp(2, actor=True, observation_dim=observation_dim),
                SharedScale(-0.7, 2),
            ),
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        )
        actor = ProbabilisticActor(
            module,
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            distribution_kwargs={"low": -1.0, "high": 1.0},
            return_log_prob=True,
            default_interaction_type=ExplorationType.RANDOM,
        )
    return actor.to(device), critic.to(device)


def make_ppo_loss(actor, critic, settings):
    loss = ClipPPOLoss(
        actor,
        critic,
        functional=False,
        clip_epsilon=settings.clip_epsilon,
        entropy_coeff=settings.entropy_coeff,
        critic_coeff=settings.critic_coeff,
        normalize_advantage=True,
        loss_critic_type="smooth_l1",
    )
    loss.make_value_estimator(gamma=settings.gamma, lmbda=settings.gae_lambda)
    return loss


@torch.no_grad()
def evaluate_actor(actor, env, *, seed=140000):
    """One independent episode per lane; completed lanes never contribute again."""
    env.set_seed(seed)
    td = env.reset()
    count = env.batch_size[0]
    active = torch.ones(count, dtype=torch.bool, device=env.device)
    records = [None] * count
    evaluation_state = td.get("evaluation_state", td["observation"])
    initial_relative = evaluation_state[:, :2] * env.config.width
    initial_goal = evaluation_state[:, 2:4] * env.config.width
    initial_goal_radius = initial_goal.norm(dim=-1)
    initial_goal_azimuth = torch.atan2(
        initial_goal[:, 1], initial_goal[:, 0]
    ).remainder(2 * torch.pi) * (180 / torch.pi)
    initial_direction = initial_goal / initial_goal.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    initial_front = (initial_relative * initial_direction).sum(-1) > initial_goal.norm(
        dim=-1
    )
    initial_azimuth = torch.atan2(
        initial_direction[:, 0] * initial_relative[:, 1]
        - initial_direction[:, 1] * initial_relative[:, 0],
        -(initial_relative * initial_direction).sum(-1),
    ).remainder(2 * torch.pi) * (180 / torch.pi)
    initial_radius = initial_relative.norm(dim=-1)
    initial_yaw = torch.atan2(evaluation_state[:, 11], evaluation_state[:, 12])
    toward_cargo = torch.atan2(-initial_relative[:, 1], -initial_relative[:, 0])
    initial_heading_offset = (initial_yaw - toward_cargo).remainder(2 * torch.pi) * (
        180 / torch.pi
    )
    went_behind = torch.zeros(count, dtype=torch.bool, device=env.device)
    lateral_excursion = torch.zeros(count, device=env.device)
    reward_totals = (
        {name: torch.zeros(count, device=env.device) for name in env.task.rewards}
        if hasattr(env, "task")
        else {}
    )
    mode = (
        ExplorationType.MODE
        if env.action_mode == "discrete"
        else ExplorationType.DETERMINISTIC
    )
    with set_exploration_type(mode):
        for step in range(round(env.config.episode_seconds / env.config.dt) + 1):
            actor(td)
            transition, td = env.step_and_maybe_reset(td)
            after = transition["next"]
            measured = after.get("evaluation_state", after["observation"])
            relative = measured[:, :2] * env.config.width
            for name, total in reward_totals.items():
                total.add_(
                    torch.where(active, after["reward_terms", name].squeeze(-1), 0.0)
                )
            behind = (relative * initial_direction).sum(-1) < -env.config.cargo_radius
            went_behind |= active & behind
            lateral = abs(
                relative[:, 0] * initial_direction[:, 1]
                - relative[:, 1] * initial_direction[:, 0]
            )
            lateral_excursion = torch.where(
                active, torch.maximum(lateral_excursion, lateral), lateral_excursion
            )
            finished = active & after["done"].squeeze(-1)
            for lane in finished.nonzero().flatten().tolist():
                records[lane] = dict(
                    lane=lane,
                    success=bool(after["success"][lane]),
                    distance_m=float(after["distance"][lane]),
                    coverage=float(after["coverage"][lane]),
                    seconds=(step + 1) * env.config.dt,
                    initial_azimuth_degrees=float(initial_azimuth[lane]) % 360,
                    initial_radius_m=float(initial_radius[lane]),
                    initial_goal_radius_m=float(initial_goal_radius[lane]),
                    initial_goal_azimuth_degrees=float(initial_goal_azimuth[lane])
                    % 360,
                    initial_yaw_degrees=float(initial_yaw[lane] * (180 / torch.pi))
                    % 360,
                    initial_heading_offset_degrees=float(initial_heading_offset[lane])
                    % 360,
                    started_beyond_goal=bool(initial_front[lane]),
                    went_behind_cargo=bool(went_behind[lane]),
                    max_lateral_offset_m=float(lateral_excursion[lane]),
                )
                if reward_totals:
                    records[lane]["reward_terms"] = {
                        name: float(total[lane])
                        for name, total in reward_totals.items()
                    }
                    records[lane]["return"] = sum(
                        records[lane]["reward_terms"].values()
                    )
            active &= ~finished
            if not active.any():
                break
    env.check_physics()
    if any(record is None for record in records):
        raise RuntimeError("Evaluation did not reach an episode boundary in every lane")
    sectors = []
    for sector in range(12):
        rows = [
            r for r in records if int(r["initial_azimuth_degrees"] // 30) % 12 == sector
        ]
        sectors.append(
            dict(
                degrees=[sector * 30, (sector + 1) * 30],
                episode_count=len(rows),
                successes=sum(r["success"] for r in rows),
                success_rate=sum(r["success"] for r in rows) / len(rows)
                if rows
                else None,
            )
        )
    heading_sectors = []
    for sector in range(8):
        rows = [
            r
            for r in records
            if int(r["initial_heading_offset_degrees"] // 45) % 8 == sector
        ]
        heading_sectors.append(
            dict(
                degrees=[sector * 45, (sector + 1) * 45],
                episode_count=len(rows),
                successes=sum(r["success"] for r in rows),
                success_rate=sum(r["success"] for r in rows) / len(rows)
                if rows
                else None,
            )
        )
    pose_cells = []
    for position in range(4):
        for heading in range(4):
            rows = [
                r
                for r in records
                if int(r["initial_azimuth_degrees"] // 90) % 4 == position
                and int(r["initial_heading_offset_degrees"] // 90) % 4 == heading
            ]
            pose_cells.append(
                dict(
                    position_degrees=[position * 90, (position + 1) * 90],
                    heading_offset_degrees=[heading * 90, (heading + 1) * 90],
                    episode_count=len(rows),
                    successes=sum(r["success"] for r in rows),
                    success_rate=sum(r["success"] for r in rows) / len(rows)
                    if rows
                    else None,
                )
            )
    goal_sectors = []
    for sector in range(8):
        rows = [
            r
            for r in records
            if int(r["initial_goal_azimuth_degrees"] // 45) % 8 == sector
        ]
        goal_sectors.append(
            dict(
                degrees=[sector * 45, (sector + 1) * 45],
                episode_count=len(rows),
                successes=sum(r["success"] for r in rows),
                success_rate=sum(r["success"] for r in rows) / len(rows)
                if rows
                else None,
            )
        )
    return dict(
        seed=seed,
        backend=env.backend,
        action_mode=env.action_mode,
        policy_input_dim=env.observation_spec["observation"].shape[-1],
        goal_mode=env.config.goal_mode,
        cargo_shape=env.config.shape,
        spawn_mode=env.config.spawn_mode,
        spawn_heading_mode=env.config.spawn_heading_mode,
        goal_position_mode=env.config.goal_position_mode,
        goal_radius_range_m=[env.config.goal_radius_min, env.config.goal_radius_max]
        if env.config.goal_position_mode == "uniform"
        else None,
        goal_azimuth_convention="initial world cargo-to-goal vector: 0=+X, 90=+Y",
        goal_sectors=goal_sectors,
        heading_offset_convention="0=toward cylinder, 90=left turn, 180=away, 270=right turn",
        heading_sectors=heading_sectors,
        pose_cells=pose_cells,
        azimuth_convention="0=rear, 90=left, 180=goal side, 270=right; relative to initial cargo-to-goal axis",
        angle_sectors=sectors,
        cargo_radius_m=env.config.cargo_radius
        if env.config.shape == "cylinder"
        else None,
        success_criterion=(
            f"cargo origin distance < {env.config.position_tolerance} m and speed < "
            f"0.06 m/s for {env.config.success_hold_seconds} s, no physical failure"
            if env.config.goal_mode == "position"
            else "coverage > 0.95, no physical failure"
        ),
        episode_count=count,
        successes=sum(r["success"] for r in records),
        mean_distance_m=sum(r["distance_m"] for r in records) / count,
        mean_coverage=sum(r["coverage"] for r in records) / count,
        started_beyond_goal=sum(r["started_beyond_goal"] for r in records),
        went_behind_cargo=sum(r["went_behind_cargo"] for r in records),
        rear_diagnostic="robot projection on initial cargo-to-goal axis < -cargo_radius",
        episodes=records,
    )


def load_ppo(checkpoint, *, device="cpu"):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if saved.get("format") not in (FORMAT, STUDENT_FORMAT):
        raise ValueError("Expected a single-hexapod TorchRL PPO checkpoint")
    config = SinglePushConfig.from_checkpoint(saved["config"])
    actor, critic = make_ppo_networks(
        saved["action_mode"],
        device=device,
        rotation_mode=config.policy_rotation_mode,
        **saved.get("network", {}),
    )
    actor.load_state_dict(saved["actor"])
    critic.load_state_dict(saved["critic"])
    return saved, config, actor.eval(), critic.eval()


def make_checkpoint_env(
    checkpoint, saved, config, *, num_envs, backend="auto", render_mode=None
):
    """Restore the interface belonging to a checkpoint, including custom terms."""
    if saved.get("student_task") is not None:
        from dataclasses import replace

        from .student import CylinderPushTask

        task = CylinderPushTask.from_dict(
            saved["student_task"], Path(checkpoint).parent
        )
        task = replace(task, config=config)
        return task.make_env(
            num_envs=num_envs, backend=backend, render_mode=render_mode
        )
    return SinglePushEnv(
        config,
        action_mode=saved["action_mode"],
        num_envs=num_envs,
        backend=backend,
        render_mode=render_mode,
    )


def evaluate_ppo(checkpoint, *, episodes=128, seed=140000, backend="auto"):
    gpu = backend == "warp" or (backend == "auto" and torch.cuda.is_available())
    saved, config, actor, _ = load_ppo(checkpoint, device="cuda:0" if gpu else "cpu")
    env = make_checkpoint_env(
        checkpoint, saved, config, num_envs=episodes, backend=backend
    )
    try:
        return evaluate_actor(actor, env, seed=seed)
    finally:
        env.close()


def _validation_score(config, report):
    score = (report["successes"], -report["mean_distance_m"])
    if config.spawn_mode == "around":
        worst = min(s["success_rate"] or 0.0 for s in report["angle_sectors"])
        if config.spawn_heading_mode == "uniform":
            joint = min(s["success_rate"] or 0.0 for s in report["pose_cells"])
            heading = min(s["success_rate"] or 0.0 for s in report["heading_sectors"])
            score = (joint, heading, worst, *score)
            if config.goal_position_mode == "uniform":
                goal = min(s["success_rate"] or 0.0 for s in report["goal_sectors"])
                return (goal, *score)
            return score
        return (worst, *score)
    return score


def train_ppo(
    config,
    output,
    *,
    action_mode="discrete",
    seed=42,
    steps=655_360,
    max_seconds=600.0,
    num_envs=None,
    horizon=64,
    settings=None,
    backend="auto",
    capture_step=True,
    validate_every=20,
    bootstrap_rollouts=0,
    initial_checkpoint=None,
    validation_envs=None,
    student_task=None,
    checkpoint_selection="best",
    curriculum=None,
):
    """Train PPO, optionally fine-tuning compatible weights with a fresh Adam."""
    if not isinstance(config, SinglePushConfig):
        raise TypeError("Use SinglePushConfig")
    if min(steps, horizon, validate_every) < 1 or (
        max_seconds is not None and max_seconds <= 0
    ):
        raise ValueError("Learning budgets must be positive")
    if validation_envs is not None and validation_envs < 1:
        raise ValueError("Validation world count must be positive")
    if checkpoint_selection not in ("best", "latest"):
        raise ValueError("checkpoint_selection must be best or latest")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(
        (output / name).exists() for name in ("initial.pt", "run.json", "latest.pt")
    ):
        raise FileExistsError("Use a new directory for each fresh PPO run")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    started = time.monotonic()
    gpu = backend == "warp" or (backend == "auto" and torch.cuda.is_available())
    num_envs = (512 if gpu else 2) if num_envs is None else num_envs
    frames = num_envs * horizon
    if steps < frames:
        raise ValueError("Step budget must include a full rollout")
    if not 0 <= bootstrap_rollouts < steps // frames:
        raise ValueError(
            "Bootstrap must leave rollouts for the final reset distribution"
        )
    if bootstrap_rollouts and (
        config.start_angle_degrees != 180
        or config.rear_start_fraction != 0
        or config.robot_start != "goal_side"
        or config.spawn_mode != "fixed"
    ):
        raise ValueError("Rear bootstrap requires a pure front final task")
    settings = settings or PPOSettings()
    if curriculum is not None:
        from .curriculum import PushCurriculum

        if not isinstance(curriculum, PushCurriculum) or bootstrap_rollouts:
            raise ValueError("Use PushCurriculum without bootstrap_rollouts")
        for phase in curriculum.phases:
            phase.apply(config)
    if student_task is not None:
        if student_task.config != config or student_task.action_mode != action_mode:
            raise ValueError(
                "Student task and PPO environment configuration must agree"
            )
        env = student_task.make_env(num_envs=num_envs, backend=backend)
    else:
        env = SinglePushEnv(
            config, action_mode=action_mode, num_envs=num_envs, backend=backend
        )
    validation_env = None
    collector = None
    try:
        if capture_step and gpu:
            from .warp_step_graph import capture_world_step

            capture_world_step(env.worlds)
        env.set_seed(seed)
        current_phase = curriculum.phase_at(0, steps // frames) if curriculum else None
        if current_phase is not None:
            env.set_reset_stage(current_phase)
        if bootstrap_rollouts:
            env.set_reset_stage(SinglePushResetStage("mixed_rear_bootstrap", 0.25))
        network = dict(
            observation_dim=env.observation_spec["observation"].shape[-1],
            action_count=student_task.actions.action_count
            if student_task is not None and action_mode == "discrete"
            else 9,
        )
        actor, critic = make_ppo_networks(
            action_mode,
            device=env.device,
            rotation_mode=config.policy_rotation_mode,
            **network,
        )
        initialization = "random actor and critic, empty Adam; no demonstrations"
        if initial_checkpoint is not None:
            previous, previous_config, previous_actor, previous_critic = load_ppo(
                initial_checkpoint, device=env.device
            )
            if student_task is not None or previous.get("student_task") is not None:
                from .student import (
                    ContinuousVelocity,
                    CylinderPushTask,
                    DiscreteDirections,
                    load_task,
                )

                target_task = student_task or CylinderPushTask(
                    config=config,
                    rotation=config.policy_rotation_mode,
                    actions=DiscreteDirections(speed=config.direction_speed)
                    if action_mode == "discrete"
                    else ContinuousVelocity(speed=config.direction_speed),
                )
                if (
                    target_task.policy_signature()
                    != load_task(initial_checkpoint).policy_signature()
                ):
                    raise ValueError(
                        "Checkpoint observation/action semantics differ; start a fresh model without initial_checkpoint"
                    )
            transfer_fields = {
                "episode_seconds",
                "approach_strategy",
                "rear_approach_gain",
                "approach_cost_weight",
                "goal_distance_cost_weight",
                "success_bonus",
                "start_angle_degrees",
                "goal_distance",
                "min_yaw_degrees",
                "max_yaw_degrees",
                "goal_lateral_range",
                "position_jitter",
                "yaw_jitter",
                "robot_start",
                "rear_start_fraction",
                "start_clearance",
                "spawn_mode",
                "spawn_radius_min",
                "spawn_radius_max",
                "spawn_azimuth_degrees",
                "spawn_heading_mode",
                "spawn_heading_offset_degrees",
                "goal_position_mode",
                "goal_radius_min",
                "goal_radius_max",
                "goal_azimuth_degrees",
                "policy_rotation_mode",
            }
            if previous["action_mode"] != action_mode or any(
                value != asdict(previous_config)[key]
                for key, value in asdict(config).items()
                if key not in transfer_fields
            ):
                raise ValueError(
                    "Fine-tuning requires the same physics, observations and actions; "
                    "only reward and reset distribution may change"
                )
            actor.load_state_dict(previous_actor.state_dict())
            critic.load_state_dict(previous_critic.state_dict())
            initialization = (
                "checkpoint actor and critic, empty Adam; no demonstrations"
            )
        loss = make_ppo_loss(actor, critic, settings)
        optimizer = torch.optim.Adam(loss.parameters(), lr=settings.learning_rate)
        task_data = None
        if student_task is not None:
            from .student import snapshot_student_task

            task_data = snapshot_student_task(
                student_task, output, observation_layout=env.observation_layout
            )
        metadata = dict(
            algorithm="TorchRL ClipPPOLoss + GAE, single actor and single critic",
            config=asdict(config),
            settings=asdict(settings),
            action_mode=action_mode,
            direction_table=student_task.actions.directions
            if student_task is not None and action_mode == "discrete"
            else DIRECTIONS
            if action_mode == "discrete"
            else None,
            student_task=task_data,
            seed=seed,
            num_envs=num_envs,
            horizon=horizon,
            requested_steps=steps,
            max_seconds=max_seconds,
            backend=env.backend,
            initialization=initialization,
            initial_checkpoint=(
                dict(
                    path=str(Path(initial_checkpoint).resolve()),
                    sha256=hashlib.sha256(
                        Path(initial_checkpoint).read_bytes()
                    ).hexdigest(),
                    training=previous["training"],
                )
                if initial_checkpoint is not None
                else None
            ),
            checkpoint_selection=checkpoint_selection,
            provenance=env.provenance,
            capture_step=bool(capture_step and gpu),
            bootstrap_rollouts=bootstrap_rollouts,
            validation_envs=validation_envs
            or (
                1024
                if gpu and config.goal_position_mode == "uniform"
                else 512
                if gpu and config.spawn_heading_mode == "uniform"
                else 128
                if gpu and config.spawn_mode == "around"
                else 32
                if gpu
                else 2
            ),
            selection_criterion="worst goal-direction sector, worst position/heading cell, worst heading sector, worst position sector, total successes, mean distance"
            if config.goal_position_mode == "uniform"
            and config.spawn_heading_mode == "uniform"
            else "worst position/heading cell, worst heading sector, worst position sector, total successes, mean distance"
            if config.spawn_heading_mode == "uniform"
            else "worst 30-degree sector success rate, total successes, mean distance"
            if config.spawn_mode == "around"
            else "total successes, mean distance",
            curriculum=asdict(curriculum)
            if curriculum is not None
            else (
                dict(
                    initial_rear_fraction=0.25,
                    final_rear_fraction=0.0,
                    changes_apply_to="future resets only",
                )
                if bootstrap_rollouts
                else None
            ),
        )

        def save(name):
            path = output / name
            torch.save(
                _native_metadata(
                    dict(
                        format=STUDENT_FORMAT if student_task is not None else FORMAT,
                        network=network,
                        student_task=task_data,
                        config=asdict(config),
                        action_mode=action_mode,
                        actor=actor.state_dict(),
                        critic=critic.state_dict(),
                        optimizer=optimizer.state_dict(),
                        training=metadata,
                        torch_rng_state=torch.get_rng_state(),
                    )
                ),
                path,
            )
            return path

        save("initial.pt")
        validation_env = (
            student_task.make_env(num_envs=metadata["validation_envs"], backend=backend)
            if student_task is not None
            else SinglePushEnv(
                config,
                action_mode=action_mode,
                num_envs=metadata["validation_envs"],
                backend=backend,
            )
        )
        initial = evaluate_actor(actor, validation_env, seed=131000)
        (output / "initial_evaluation.json").write_text(
            json.dumps(initial, indent=2) + "\n"
        )
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
        iterator = iter(collector)
        rows, durations, validations = [], [], []
        best = (-1, float("-inf"))
        if config.spawn_mode == "around" or student_task is not None:
            if curriculum is None:
                best = _validation_score(config, initial)
            metadata.update(steps=0, rollouts=0, seconds=time.monotonic() - started)
            save("best.pt")

        def validate(row):
            nonlocal best
            report = evaluate_actor(actor, validation_env, seed=131000)
            validations.append(dict(rollout=row["rollout"], report=report))
            score = _validation_score(config, report)
            if config.spawn_mode == "around":
                if config.spawn_heading_mode == "uniform":
                    row["validation_worst_joint_rate"] = min(
                        s["success_rate"] or 0.0 for s in report["pose_cells"]
                    )
                    row["validation_worst_heading_rate"] = min(
                        s["success_rate"] or 0.0 for s in report["heading_sectors"]
                    )
                else:
                    row["validation_worst_sector_rate"] = score[0]
            if config.goal_position_mode == "uniform":
                row["validation_worst_goal_rate"] = min(
                    s["success_rate"] or 0.0 for s in report["goal_sectors"]
                )
            eligible = current_phase is None or current_phase == curriculum.phases[-1]
            if eligible and score > best:
                best = score
                metadata.update(
                    steps=row["steps"],
                    rollouts=row["rollout"],
                    seconds=time.monotonic() - started,
                )
                if current_phase is not None:
                    metadata["training_stage"] = current_phase.name
                save("best.pt")
            row["validation_successes"] = report["successes"]
            (output / "validations.json").write_text(
                json.dumps(validations, indent=2) + "\n"
            )

        for rollout in range(steps // frames):
            if (
                max_seconds is not None
                and time.monotonic() - started + max(durations[-3:], default=0) * 1.2
                >= max_seconds
            ):
                break
            batch_started = time.monotonic()
            if curriculum is not None:
                phase = curriculum.phase_at(rollout, steps // frames)
                if phase != current_phase:
                    env.set_reset_stage(phase)
                    current_phase = phase
                for group in optimizer.param_groups:
                    group["lr"] = settings.learning_rate * phase.learning_rate_scale
            if bootstrap_rollouts and rollout == bootstrap_rollouts:
                env.set_reset_stage(SinglePushResetStage("front_only", 0.0))
                metadata["front_only_started_at_step"] = rollout * frames
            batch = next(iterator)
            env.check_physics()
            for _ in range(settings.epochs):
                with torch.no_grad():
                    loss.value_estimator(batch)
                flat = batch.reshape(-1)
                permutation = torch.randperm(flat.numel(), device=env.device)
                for indexes in permutation.split(settings.minibatch_size):
                    if indexes.numel() < 2:
                        continue
                    metrics = loss(flat[indexes])
                    objective = sum(
                        metrics[k]
                        for k in ("loss_objective", "loss_critic", "loss_entropy")
                    )
                    if not torch.isfinite(objective):
                        raise FloatingPointError("Non-finite PPO objective")
                    optimizer.zero_grad()
                    objective.backward()
                    torch.nn.utils.clip_grad_norm_(
                        loss.parameters(), settings.max_grad_norm
                    )
                    optimizer.step()
            collector.update_policy_weights_()
            row = dict(
                rollout=rollout + 1,
                steps=(rollout + 1) * frames,
                seconds=time.monotonic() - started,
                reward=float(batch["next", "reward"].mean()),
                mean_distance_m=float(batch["next", "distance"].mean()),
                episodes=int(batch["next", "done"].sum()),
                successes=int((batch["next", "done"] & batch["next", "success"]).sum()),
            )
            if current_phase is not None:
                row["curriculum_stage"] = current_phase.name
                row["learning_rate"] = optimizer.param_groups[0]["lr"]
            if student_task is not None:
                row["reward_terms"] = {
                    name: float(batch["next", "reward_terms", name].mean())
                    for name in student_task.rewards
                }
            if (rollout + 1) % validate_every == 0:
                validate(row)
            rows.append(row)
            durations.append(time.monotonic() - batch_started)
            (output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
            if (rollout + 1) % 10 == 0:
                print(json.dumps(row), flush=True)
        if (
            student_task is not None
            and rows
            and (not validations or validations[-1]["rollout"] != rows[-1]["rollout"])
        ):
            validate(rows[-1])
            (output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
        metadata.update(
            steps=rows[-1]["steps"] if rows else 0,
            rollouts=len(rows),
            seconds=time.monotonic() - started,
            stopped_by="step_budget" if len(rows) == steps // frames else "time_budget",
            training_stage=(
                current_phase.name
                if current_phase is not None
                else "fixed_task"
                if not bootstrap_rollouts
                else "front_only"
                if len(rows) > bootstrap_rollouts
                else "bootstrap"
            ),
        )
        checkpoint = save("latest.pt")
        (output / "run.json").write_text(
            json.dumps(_native_metadata(metadata), indent=2) + "\n"
        )
        return checkpoint
    finally:
        if collector is not None:
            collector.shutdown()
        if validation_env is not None:
            validation_env.close()
        env.close()
