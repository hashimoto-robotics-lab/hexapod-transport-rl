"""Reset-only lessons for a pushing policy trained from random weights."""

from dataclasses import dataclass, field, replace
from math import pi


@dataclass(frozen=True)
class PushPhase:
    name: str
    until: float
    resets: dict = field(default_factory=dict)
    learning_rate_scale: float = 1.0

    def apply(self, config):
        return replace(config, **self.resets)


@dataclass(frozen=True)
class PushCurriculum:
    """Teach contact, pushing, walking around, and arbitrary initial heading.

    Only future reset distributions and the learning rate change. Physics,
    observations, actions, rewards, success precision and episode time do not.
    The last phase uses the task's full reset distribution without overrides.
    """

    phases: tuple[PushPhase, ...] = (
        PushPhase(
            "rear_contact",
            0.20,
            dict(
                spawn_azimuth_degrees=0.0,
                spawn_heading_mode="toward_cargo",
                spawn_radius_min=0.60,
                spawn_radius_max=0.75,
                goal_radius_min=0.15,
                goal_radius_max=0.25,
                goal_azimuth_degrees=45.0,
            ),
        ),
        PushPhase(
            "rear_push",
            0.35,
            dict(spawn_azimuth_degrees=0.0, spawn_heading_mode="toward_cargo"),
        ),
        PushPhase(
            "walk_around",
            0.65,
            dict(spawn_heading_mode="toward_cargo"),
        ),
        PushPhase(
            "turn_and_push",
            0.80,
            dict(spawn_heading_mode="toward_cargo", yaw_jitter=pi / 2),
        ),
        PushPhase("full_task", 1.0, {}, 0.10),
    )

    def __post_init__(self):
        allowed = {
            "spawn_azimuth_degrees",
            "spawn_heading_mode",
            "spawn_radius_min",
            "spawn_radius_max",
            "goal_radius_min",
            "goal_radius_max",
            "goal_azimuth_degrees",
            "yaw_jitter",
        }
        previous = 0.0
        for phase in self.phases:
            if (
                not previous < phase.until <= 1.0
                or not 0 < phase.learning_rate_scale <= 1
            ):
                raise ValueError(
                    "Curriculum fractions must increase to 1; learning-rate scales must be in (0,1]"
                )
            if not set(phase.resets) <= allowed:
                raise ValueError(
                    "Curriculum phases may change only reset distributions"
                )
            previous = phase.until
        if not self.phases or previous != 1.0 or self.phases[-1].resets:
            raise ValueError("The last curriculum phase must use the full task")

    def phase_at(self, rollout, total_rollouts):
        for phase in self.phases:
            if rollout / total_rollouts < phase.until:
                return phase
        return self.phases[-1]
