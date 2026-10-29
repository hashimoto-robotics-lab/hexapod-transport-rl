"""Internal adapter: student terms on the original articulated SinglePushEnv."""

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Categorical, Composite, Unbounded

from .single_push import SinglePushEnv, cylinder_approach
from .student import DiscreteDirections
from .student_terms import PushState


class StudentPushEnv(SinglePushEnv):
    def __init__(self, task, *, num_envs=2, backend="auto", render_mode=None):
        self.task = task
        super().__init__(
            task.config,
            action_mode=task.action_mode,
            num_envs=num_envs,
            backend=backend,
            render_mode=render_mode,
        )
        try:
            self._command_velocity = torch.zeros(num_envs, 2, device=self.device)
            self._invalid_terms = torch.zeros((), dtype=torch.bool, device=self.device)
            self._feature_sizes = None
            self._current_state = None
            if isinstance(task.actions, DiscreteDirections):
                self.directions = torch.tensor(
                    task.actions.directions, device=self.device
                )
                self.action_spec = Categorical(
                    task.actions.action_count, shape=(num_envs,), device=self.device
                )
            else:
                self.action_spec = Bounded(
                    -1.0, 1.0, shape=(num_envs, 2), device=self.device
                )
            raw, info = self.worlds.reset(seed=0)
            probe = self._observation(raw, info)
            self.observation_spec = Composite(
                observation=Unbounded(
                    shape=probe["observation"].shape, device=self.device
                ),
                evaluation_state=Unbounded(shape=(num_envs, 14), device=self.device),
                reward_terms=Composite(
                    {
                        name: Unbounded(shape=(num_envs, 1), device=self.device)
                        for name in task.rewards
                    },
                    shape=(num_envs,),
                    device=self.device,
                ),
                **{
                    key: Unbounded(shape=(num_envs, 1), device=self.device)
                    for key in ("distance", "coverage", "approach")
                },
                success=Categorical(
                    2, shape=(num_envs, 1), dtype=torch.bool, device=self.device
                ),
                shape=(num_envs,),
                device=self.device,
            )
            # Detect programming mistakes in custom functions before collection.
            self._reward_terms(self._current_state, self._current_state)
            self.check_physics()
        except BaseException:
            self.close()
            raise

    def _physical_state(self, raw, info, baseline):
        raw = torch.as_tensor(raw, device=self.device, dtype=torch.float32)[:, 0]
        if self.backend == "warp":
            worlds = self.worlds
            velocity = worlds.physics.qvel
            rv, cv = int(worlds.reference.core.root_v[0]), worlds.cargo_v
            own = velocity[:, rv : rv + 2]
            cargo = velocity[:, cv : cv + 2]
            own_yaw = velocity[:, rv + 5]
            cargo_yaw = velocity[:, cv + 5]
            goal = worlds.goal
            elapsed = worlds.steps * self.config.dt
        else:
            cores = [e.core for e in self.worlds.envs]
            own = np.stack([c.data.qvel[c.root_v[0] : c.root_v[0] + 2] for c in cores])
            cargo = np.stack([c.data.qvel[c.cargo_v : c.cargo_v + 2] for c in cores])
            own_yaw = [c.data.qvel[c.root_v[0] + 5] for c in cores]
            cargo_yaw = [c.data.qvel[c.cargo_v + 5] for c in cores]
            goal = np.stack([c.goal for c in cores])
            elapsed = [c.steps * self.config.dt for c in cores]

        def tensor(value, dtype=torch.float32):
            return torch.as_tensor(value, device=self.device, dtype=dtype).clone()

        return PushState(
            robot_position=raw[:, :2].clone(),
            cargo_position=raw[:, 2:4].clone(),
            goal_position=tensor(goal),
            robot_yaw=raw[:, 5].clone(),
            cargo_yaw=raw[:, 4].clone(),
            robot_velocity=tensor(own),
            cargo_velocity=tensor(cargo),
            robot_yaw_rate=tensor(own_yaw),
            cargo_yaw_rate=tensor(cargo_yaw),
            goal_distance=(tensor(goal) - raw[:, 2:4]).norm(dim=-1),
            approach_distance=cylinder_approach(
                self.config, raw[:, :2] - raw[:, 2:4], tensor(goal) - raw[:, 2:4], torch
            ).clone(),
            success=baseline["success"].squeeze(-1).clone(),
            failed=tensor(
                info.get("failed", np.zeros(self.batch_size[0], dtype=bool)), torch.bool
            ),
            elapsed_seconds=tensor(elapsed),
            command_velocity=self._command_velocity.clone(),
            baseline_observation=baseline["observation"].clone(),
            cargo_radius=self.config.cargo_radius,
            dt=self.config.dt,
        )

    def _validate_tensor(self, value, name, *, observation=False):
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"{name} must return a torch.Tensor with one row per environment"
            )
        if value.device != self.device:
            raise ValueError(
                f"{name} returned {value.device}; use the state's device {self.device}"
            )
        if observation and value.ndim == 1:
            value = value.unsqueeze(-1)
        expected = (
            value.ndim == (2 if observation else 1)
            and value.shape[0] == self.batch_size[0]
        )
        if not expected or (observation and value.shape[1] == 0):
            shape = "(B,D) or (B,)" if observation else "(B,)"
            raise ValueError(
                f"{name} must return {shape}, B={self.batch_size[0]}; got {tuple(value.shape)}"
            )
        value = value.detach().float()
        finite = torch.isfinite(value).all()
        self._invalid_terms.logical_or_(~finite)
        if self._feature_sizes is None and not bool(finite):
            raise ValueError(f"{name} returned NaN or infinity")
        return value

    def _features(self, state):
        values = [
            self._validate_tensor(
                t.compute(state), f"Observation {t.name}", observation=True
            )
            for t in self.task.observations
        ]
        sizes = tuple(v.shape[-1] for v in values)
        if self._feature_sizes is None:
            self._feature_sizes = sizes
        elif self._feature_sizes != sizes:
            raise ValueError(
                f"Observation sizes changed after reset: {self._feature_sizes} -> {sizes}"
            )
        return torch.cat(values, -1).clone()

    def _reward_terms(self, before, after):
        return {
            name: self._validate_tensor(term.compute(before, after), f"Reward {name}")
            .unsqueeze(-1)
            .clone()
            for name, term in self.task.rewards.items()
        }

    def _observation(self, raw, info):
        result = super()._observation(raw, info)
        state = self._physical_state(raw, info, result)
        result["distance"] = state.goal_distance.unsqueeze(-1).clone()
        result["approach"] = state.approach_distance.unsqueeze(-1).clone()
        result["evaluation_state"] = result["observation"].clone()
        result["observation"] = self._features(state)
        result["reward_terms"] = TensorDict(
            {
                name: torch.zeros((*self.batch_size, 1), device=self.device)
                for name in self.task.rewards
            },
            self.batch_size,
            self.device,
        )
        self._current_state = state
        return result

    @property
    def state(self):
        """Current named physical state; custom terms must not mutate tensors."""
        return self._current_state

    def render(self):
        """Render the first world as an RGB image."""
        if self.backend == "warp":
            return self.worlds.render()
        return self.worlds.envs[0].render()

    @property
    def observation_layout(self):
        offset, result = 0, []
        for term, size in zip(self.task.observations, self._feature_sizes, strict=True):
            result.append(
                dict(
                    name=term.name,
                    start=offset,
                    stop=offset + size,
                    size=size,
                    scale=term.scale,
                    enabled=term.enabled,
                )
            )
            offset += size
        return result

    def _reset(self, tensordict=None, **kwargs):
        mask = tensordict.get("_reset") if tensordict is not None else None
        if mask is None:
            self._command_velocity.zero_()
        else:
            self._command_velocity.masked_fill_(mask.reshape(-1, 1), 0.0)
        return super()._reset(tensordict, **kwargs)

    def _step(self, tensordict):
        before = self._current_state
        action = tensordict["action"].detach()
        force = (
            self.directions[action]
            if self.action_mode == "discrete"
            else action.clamp(-1, 1)
        )
        if self.task.actions.frame == "body":
            c, s = before.robot_yaw.cos(), before.robot_yaw.sin()
            x, y = force.unbind(-1)
            force = torch.stack((c * x - s * y, s * x + c * y), -1)
        force = force / force.norm(dim=-1, keepdim=True).clamp_min(1.0)
        self._command_velocity.copy_(force * self.task.actions.speed)
        physical_action = force.unsqueeze(1)
        if self.backend == "cpu":
            physical_action = physical_action.numpy()
        raw, _, terminated, truncated, info = self.worlds.step(physical_action)
        result = self._observation(raw, info)
        terms = self._reward_terms(before, self._current_state)
        result["reward_terms"] = TensorDict(terms, self.batch_size, self.device)
        result["reward"] = sum(terms.values()).clone()
        for key, value in (("terminated", terminated), ("truncated", truncated)):
            result[key] = (
                torch.as_tensor(value, device=self.device).reshape(-1, 1).clone()
            )
        result["done"] = result["terminated"] | result["truncated"]
        return result

    def check_physics(self):
        super().check_physics()
        if bool(self._invalid_terms):
            raise FloatingPointError(
                "Student observation/reward contains NaN or infinity; check custom terms"
            )
