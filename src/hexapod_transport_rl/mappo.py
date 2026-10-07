"""Shared decentralized actor and centralized team critic, with PPO/GAE.

Each CPU world contains all interacting robots. Independent worlds can advance
in separate processes; the parent performs shared actor/critic updates.
"""

import json
import time
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from gymnasium.vector import VectorEnv
from numpy.typing import ArrayLike
from torch import nn
from torch.distributions import Normal

from .config import ENV_VERSION, LEGACY_ENV_VERSION, PushConfig
from .types import Info, Observation
from .vector import make_vector_env


@dataclass(frozen=True)
class PPOSettings:
    """MAPPOの共通設定。回り込み・押す学習の両方で使用する。"""

    learning_rate: float = 3e-4
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_weight: float = 0.5
    entropy_weight: float = 0.005
    max_gradient_norm: float = 0.5


PPO_SETTINGS = PPOSettings()
HIDDEN_UNITS = 128  # 保存済みモデルのネットワーク構造に合わせる。


def mlp(n_in: int, n_out: int) -> nn.Sequential:
    """Two hidden layers shared by the actor and critic architectures."""
    return nn.Sequential(
        nn.Linear(n_in, HIDDEN_UNITS),
        nn.Tanh(),
        nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS),
        nn.Tanh(),
        nn.Linear(HIDDEN_UNITS, n_out),
    )


class MAPPO(nn.Module):
    """Shared local actor plus team critic; leading dimensions are batch axes.

    Actor: (..., N, D) -> independent Gaussian commands (..., N, 3).
    Critic: (..., N, D) -> one team value (...,).
    Module/parameter names are kept stable for existing checkpoints.
    """

    def __init__(
        self, obs_dim: int, num_robots: int, *, mirror_equivariant: bool = False
    ) -> None:
        super().__init__()
        self.obs_dim, self.num_robots = obs_dim, num_robots
        self.mirror_equivariant = mirror_equivariant
        if mirror_equivariant and obs_dim != 16 + 4 * (num_robots - 1):
            raise ValueError(
                "Reflection requires the documented hexapod observation layout"
            )
        self.actor = mlp(obs_dim, 3)
        self.critic = mlp(obs_dim * num_robots, 1)
        self.log_std = nn.Parameter(torch.full((3,), -0.7))
        for net in (self.actor, self.critic):
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, np.sqrt(2))
                    nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1)

    def actor_mean(self, obs: torch.Tensor) -> torch.Tensor:
        """Optionally enforce left/right reflection in both learning and acting.

        Lateral positions/velocities, yaw rates, relative-angle sines and the
        assigned slot change sign. Forward commands and cosine features do not.
        This is an architectural constraint on a learned actor, not a controller.
        """
        mean = self.actor(obs)
        if not self.mirror_equivariant:
            return mean
        observation_signs = obs.new_tensor(
            [1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1]
            + [1, -1, -1, 1] * (self.num_robots - 1)
        )
        action_signs = obs.new_tensor([1, -1, -1])
        return (mean + self.actor(obs * observation_signs) * action_signs) / 2

    def distribution(self, obs: torch.Tensor) -> Normal:
        """Distribution of pre-tanh actions, with a shared learned log-std."""
        return Normal(self.actor_mean(obs), self.log_std.clamp(-5, 1).exp())

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """Concatenate the N local observations for the centralized critic."""
        return self.critic(obs.flatten(-2)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: ArrayLike) -> np.ndarray:
        """Deterministic normalized actions used during checkpoint evaluation."""
        return self.actor_mean(torch.as_tensor(obs, dtype=torch.float32)).tanh().numpy()


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float = PPO_SETTINGS.discount,
    lam: float = PPO_SETTINGS.gae_lambda,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return advantages and value targets, each shaped (time, worlds).

    Timeouts bootstrap final states but do not propagate into reset episodes.
    True terminations have no bootstrap contribution.
    """
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = (
            rewards[t] + gamma * (~terminated[t]).float() * next_values[t] - values[t]
        )
        carry = delta + gamma * lam * (~(terminated[t] | truncated[t])).float() * carry
        advantages[t] = carry
    return advantages, advantages + values


def load_checkpoint(path: str | Path) -> tuple[MAPPO, Info]:
    """Load and validate a CPU policy in the existing hexapod MAPPO format."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") not in (
        "hexapod-transport-mappo-v1",
        "hexapod-transport-mappo-v2",
        "sixtrail-push-mappo-v1",
        "sixtrail-push-mappo-v2",
    ) or saved.get("env_version") not in (ENV_VERSION, LEGACY_ENV_VERSION):
        raise ValueError(
            "Expected hexapod leg-pushing v1/v2 checkpoint; other robot policies are incompatible"
        )
    mirrored = saved.get("format") in (
        "hexapod-transport-mappo-v2",
        "sixtrail-push-mappo-v2",
    )
    if mirrored and saved.get("model_options") != {"mirror_equivariant": True}:
        raise ValueError("Missing or unsupported v2 actor options")
    model = MAPPO(
        saved["obs_dim"], saved["config"]["num_robots"], mirror_equivariant=mirrored
    )
    model.load_state_dict(saved["model"])
    if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError("Non-finite checkpoint weights")
    return model.eval(), saved


def train(
    cfg: PushConfig,
    asset_root: str | Path | None,
    output: str | Path,
    iterations: int = 100,
    num_envs: int = 2,
    horizon: int = 64,
    epochs: int = 4,
    minibatch: int = 64,
    seed: int = 42,
    resume: str | Path | None = None,
    *,
    backend: Literal["auto", "sync", "async"] = "sync",
    mirror_equivariant: bool = False,
    initial_log_std: float | None = None,
) -> Path:
    """Train on CPU worlds; save metrics, provenance and an atomic checkpoint.

    One transition contains the whole robot team. Resume restores optimizer and
    model weights but starts new episodes and RNG streams, as before.
    Python callers default to sync so notebooks do not spawn implicitly. The
    CLI uses auto; async callers must use a script with an __main__ guard.
    """
    started_at = time.perf_counter()
    if backend not in ("auto", "sync", "async"):
        raise ValueError("backend must be auto, sync, or async")
    if initial_log_std is not None and (
        not np.isfinite(initial_log_std) or not -5 <= initial_log_std <= 1
    ):
        raise ValueError("initial_log_std must be finite and between -5 and 1")
    asynchronous = backend == "async" or (backend == "auto" and num_envs > 1)
    if (
        min(iterations, num_envs, horizon, epochs, minibatch) < 1
        or num_envs * horizon < 2
    ):
        raise ValueError(
            "Positive training sizes and at least two rollout transitions required"
        )
    torch.manual_seed(seed)
    np.random.seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(
        (output / name).exists()
        for name in ("metrics.jsonl", "run.json", "checkpoint.pt")
    ):
        raise FileExistsError(
            "Use a new output directory to keep training runs separate"
        )
    with closing(
        make_vector_env(
            num_envs,
            config=cfg,
            asset_root=asset_root,
            flatten=False,
            asynchronous=asynchronous,
        )
    ) as envs:
        return train_rollouts(
            envs=envs,
            cfg=cfg,
            output=output,
            iterations=iterations,
            num_envs=num_envs,
            horizon=horizon,
            epochs=epochs,
            minibatch=minibatch,
            seed=seed,
            resume=resume,
            started_at=started_at,
            mirror_equivariant=mirror_equivariant,
            initial_log_std=initial_log_std,
        )


@dataclass
class RolloutBatch:
    """One collection window before flattening time and world axes.

    T = horizon, B = worlds, N = robots, D = local observation size.
    next_observations is the input for the next window, after any masked reset;
    next_values was evaluated on final observations BEFORE those resets.
    """

    observations: torch.Tensor  # (T,B,N,D)
    latents: torch.Tensor  # (T,B,N,3), before tanh
    old_log_probs: torch.Tensor  # (T,B,N)
    values: torch.Tensor  # (T,B), one value/reward per team
    next_values: torch.Tensor  # (T,B)
    rewards: torch.Tensor  # (T,B)
    terminated: torch.Tensor  # (T,B), bool
    truncated: torch.Tensor  # (T,B), bool
    next_observations: Observation  # (B,N,D)
    completed_episodes: int
    successes: int


def collect_rollout(
    envs: VectorEnv, agent: MAPPO, current_observations: Observation, horizon: int
) -> RolloutBatch:
    """経験を収集する。全機を同時に動かし、終了した世界だけをresetする。"""
    completed = successes = 0
    observations, latents, old_log_probs, values, next_values = [], [], [], [], []
    rewards, terminals, timeouts = [], [], []
    for _ in range(horizon):
        with torch.no_grad():
            observation_tensor = torch.as_tensor(
                current_observations, dtype=torch.float32
            )
            action_distribution = agent.distribution(observation_tensor)
            sampled_latent = action_distribution.sample()
            action_log_probability = action_distribution.log_prob(sampled_latent).sum(
                -1
            )
            value = agent.value(observation_tensor)
        final_observations, step_rewards, terminated, truncated, infos = envs.step(
            sampled_latent.tanh().numpy()
        )
        with torch.no_grad():
            next_value = agent.value(
                torch.as_tensor(final_observations, dtype=torch.float32)
            )
        observations.append(observation_tensor)
        latents.append(sampled_latent)
        old_log_probs.append(action_log_probability)
        values.append(value)
        next_values.append(next_value)
        rewards.append(step_rewards.copy())
        terminals.append(terminated.copy())
        timeouts.append(truncated.copy())
        current_observations = final_observations
        done = terminated | truncated
        if done.any():
            completed += int(done.sum())
            successes += int(infos["is_success"][done].sum())
            # Final-state value above must be evaluated before masked reset.
            current_observations, _ = envs.reset(options={"reset_mask": done})
    return RolloutBatch(
        observations=torch.stack(observations),
        latents=torch.stack(latents),
        old_log_probs=torch.stack(old_log_probs),
        values=torch.stack(values),
        next_values=torch.stack(next_values),
        rewards=torch.as_tensor(np.stack(rewards), dtype=torch.float32),
        terminated=torch.as_tensor(np.stack(terminals)),
        truncated=torch.as_tensor(np.stack(timeouts)),
        next_observations=current_observations,
        completed_episodes=completed,
        successes=successes,
    )


def update_policy(
    agent: MAPPO,
    optimizer: torch.optim.Adam,
    rollout: RolloutBatch,
    epochs: int,
    minibatch: int,
) -> dict[str, float]:
    """GAE → 正規化 → ミニバッチ → PPO損失 → 勾配更新の順で学習する。"""
    # 1. 各世界のチームadvantageとcriticの教師値を計算する。
    advantages, value_targets = compute_gae(
        rollout.rewards,
        rollout.values,
        rollout.next_values,
        rollout.terminated,
        rollout.truncated,
    )
    advantages = advantages.flatten()
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    value_targets = value_targets.flatten()
    batch_observations = rollout.observations.flatten(0, 1)
    batch_latents = rollout.latents.flatten(0, 1)
    batch_log_probs = rollout.old_log_probs.flatten(0, 1)
    batch_size = len(batch_observations)
    minibatch_metrics = []
    # 2. 時間×世界を1つのバッチにし、同じ経験で複数epochs更新する。
    for _ in range(epochs):
        permutation = torch.randperm(batch_size)
        for minibatch_indices in permutation.split(minibatch):
            action_distribution = agent.distribution(
                batch_observations[minibatch_indices]
            )
            # tanh's Jacobian cancels in the ratio at the same stored latent.
            log_prob = action_distribution.log_prob(
                batch_latents[minibatch_indices]
            ).sum(-1)
            probability_ratio = (log_prob - batch_log_probs[minibatch_indices]).exp()
            team_advantage = advantages[minibatch_indices, None]
            # 3. 方策の変化をclipで抑え、同じチームadvantageを全機へ配る。
            unclipped_objective = probability_ratio * team_advantage
            clipped_objective = (
                probability_ratio.clamp(
                    1 - PPO_SETTINGS.clip_ratio, 1 + PPO_SETTINGS.clip_ratio
                )
                * team_advantage
            )
            policy_loss = -torch.minimum(unclipped_objective, clipped_objective).mean()
            value_loss = (
                (
                    agent.value(batch_observations[minibatch_indices])
                    - value_targets[minibatch_indices]
                )
                .square()
                .mean()
            )
            # Entropy bonus on the latent Gaussian (before tanh).
            entropy = action_distribution.entropy().sum(-1).mean()
            loss = (
                policy_loss
                + PPO_SETTINGS.value_loss_weight * value_loss
                - PPO_SETTINGS.entropy_weight * entropy
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PPO loss")
            # 4. actorとcriticを同時に更新し、勾配の大きさを制限する。
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                agent.parameters(),
                PPO_SETTINGS.max_gradient_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            minibatch_metrics.append(
                [policy_loss.item(), value_loss.item(), entropy.item()]
            )
    mean_metrics = np.mean(minibatch_metrics, axis=0)
    return dict(
        policy_loss=float(mean_metrics[0]),
        value_loss=float(mean_metrics[1]),
        entropy=float(mean_metrics[2]),
    )


def _save_checkpoint(
    output: Path,
    agent: MAPPO,
    optimizer: torch.optim.Adam,
    *,
    iteration: int,
    transitions: int,
    run: Info,
) -> None:
    """Replace the checkpoint only after writing the complete temporary file."""
    saved = dict(
        format="hexapod-transport-mappo-v2"
        if agent.mirror_equivariant
        else "hexapod-transport-mappo-v1",
        model_options={"mirror_equivariant": agent.mirror_equivariant},
        env_version=ENV_VERSION,
        config=run["config"],
        obs_dim=agent.obs_dim,
        model=agent.state_dict(),
        optimizer=optimizer.state_dict(),
        iteration=iteration,
        transitions=transitions,
        low_level_sha256=run["low_level_sha256"],
        robot_xml_sha256=run["robot_xml_sha256"],
        run=run,
    )
    temporary_path = output / "checkpoint.tmp"
    torch.save(saved, temporary_path)
    temporary_path.replace(output / "checkpoint.pt")


def train_rollouts(
    envs: VectorEnv,
    cfg: PushConfig,
    output: Path,
    iterations: int,
    num_envs: int,
    horizon: int,
    epochs: int,
    minibatch: int,
    seed: int,
    resume: str | Path | None,
    started_at: float,
    mirror_equivariant: bool = False,
    initial_log_std: float | None = None,
) -> Path:
    """押す学習の共通ループ。初期化、経験収集、PPO更新、指標・重みの保存。"""
    obs, _ = envs.reset(seed=seed)
    provenance = envs.call("provenance")[0]
    agent = MAPPO(
        envs.single_observation_space.shape[-1],
        cfg.num_robots,
        mirror_equivariant=mirror_equivariant,
    )
    optimizer = torch.optim.Adam(agent.parameters(), lr=PPO_SETTINGS.learning_rate)
    start = 0
    transitions = 0
    if resume:
        agent, saved = load_checkpoint(resume)
        if (
            saved["config"] != asdict(cfg)
            or saved["low_level_sha256"] != provenance["low_level_sha256"]
            or saved["robot_xml_sha256"] != provenance["robot_xml_sha256"]
        ):
            raise ValueError("Resume environment/controller differs from saved run")
        optimizer = torch.optim.Adam(agent.parameters(), lr=PPO_SETTINGS.learning_rate)
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["iteration"]
        transitions = saved.get(
            "transitions", start * saved["run"]["horizon"] * saved["run"]["num_envs"]
        )
        # A v2 resume retains its architecture. Opting in upgrades a v1 actor.
        if mirror_equivariant:
            agent.mirror_equivariant = True
    if initial_log_std is not None:
        with torch.no_grad():
            agent.log_std.fill_(initial_log_std)
        # Moments from a different exploration scale should not undo the reset.
        optimizer.state.pop(agent.log_std, None)
    run = dict(
        env_version=ENV_VERSION,
        config=asdict(cfg),
        iterations=iterations,
        num_envs=num_envs,
        horizon=horizon,
        epochs=epochs,
        minibatch=minibatch,
        seed=seed,
        backend=type(envs).__name__,
        asset_root=provenance["asset_root"],
        low_level_sha256=provenance["low_level_sha256"],
        robot_xml_sha256=provenance["robot_xml_sha256"],
        walking_checkpoint_iteration=provenance["walking_checkpoint_iteration"],
        resume=str(resume) if resume else None,
        mirror_equivariant=agent.mirror_equivariant,
        initial_log_std=initial_log_std,
    )
    if "reset_distribution" in provenance:
        run["reset_distribution"] = provenance["reset_distribution"]
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    completed = successes = 0
    agent.train()
    for iteration in range(start + 1, start + iterations + 1):
        iteration_started = time.perf_counter()
        rollout = collect_rollout(envs, agent, obs, horizon)
        collected_at = time.perf_counter()
        obs = rollout.next_observations
        completed += rollout.completed_episodes
        successes += rollout.successes
        losses = update_policy(agent, optimizer, rollout, epochs, minibatch)
        updated_at = time.perf_counter()
        iteration_seconds = updated_at - iteration_started
        transitions += horizon * num_envs
        record = dict(
            iteration=iteration,
            transitions=transitions,
            mean_reward=rollout.rewards.mean().item(),
            **losses,
            completed_episodes=completed,
            successes=successes,
            success_rate=successes / completed if completed else None,
            elapsed_seconds=updated_at - started_at,
            iteration_seconds=iteration_seconds,
            rollout_seconds=collected_at - iteration_started,
            update_seconds=updated_at - collected_at,
            transitions_per_second=horizon * num_envs / iteration_seconds,
        )
        with (output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if iteration % 10 == 0 or iteration == start + iterations:
            _save_checkpoint(
                output,
                agent,
                optimizer,
                iteration=iteration,
                transitions=transitions,
                run=run,
            )
    return output / "checkpoint.pt"
