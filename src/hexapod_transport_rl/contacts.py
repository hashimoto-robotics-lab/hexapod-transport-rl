"""Measure physical contacts over one high-level step; never apply forces.

Contact fractions count physics ticks with contact, not individual contact
points. Normal impulses integrate force over time and are measured in N·s.
"""

import mujoco
import numpy as np

from .config import PHYSICS_DT
from .types import Info

WORLD_OWNER = -1
CARGO_OWNER = -2
BODY, LEG_LINK, FOOT = range(3)
PART_NAMES = ("body", "leg_link", "foot")
MIN_CONTACT_FORCE_N = 1e-6


class ContactTracker:
    """Map geoms to robots/parts once, then accumulate each physics tick."""

    def __init__(self, model: mujoco.MjModel, num_robots: int) -> None:
        self.num_robots = num_robots
        self.geom_owner = np.full(model.ngeom, WORLD_OWNER)
        self.geom_kind = np.zeros(model.ngeom, dtype=int)
        for geom_id in range(model.ngeom):
            name = model.geom(geom_id).name or ""
            if name.startswith("cargo_"):
                self.geom_owner[geom_id] = CARGO_OWNER
            elif name.startswith("r") and "/" in name:
                robot_name, local_name = name.split("/", 1)
                self.geom_owner[geom_id] = int(robot_name[1:])
                if local_name.endswith("_foot_collision"):
                    self.geom_kind[geom_id] = FOOT
                elif any(
                    f"_{part}_collision" in local_name
                    for part in ("coxa", "femur", "tibia")
                ):
                    self.geom_kind[geom_id] = LEG_LINK
        self.reset()

    def reset(self) -> None:
        """Start a fresh measurement window for one high-level action."""
        self.contact_counts = np.zeros(self.num_robots)
        self.part_contact_counts = np.zeros((len(PART_NAMES), self.num_robots))
        self.normal_impulses = np.zeros_like(self.part_contact_counts)
        self.robot_collision_steps = 0
        self._contact_force = np.zeros(6)

    def record(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Accumulate one tick immediately after mj_step, preserving its contacts."""
        touching = np.zeros(self.num_robots, dtype=bool)
        part_touching = np.zeros_like(self.part_contact_counts, dtype=bool)
        # Most contacts are feet against the floor. Filter contiguous MuJoCo
        # arrays first, avoiding a Python object/loop iteration for each one.
        geoms = data.contact.geom
        owners = self.geom_owner[geoms]
        owner_a, owner_b = owners[:, 0], owners[:, 1]
        active = data.contact.dist <= 0
        robots_collided = bool(
            np.any(active & (owner_a >= 0) & (owner_b >= 0) & (owner_a != owner_b))
        )
        robot_is_a = (owner_a >= 0) & (owner_b == CARGO_OWNER)
        robot_is_b = (owner_b >= 0) & (owner_a == CARGO_OWNER)
        cargo_contacts = np.flatnonzero(active & (robot_is_a | robot_is_b))
        # Keep contact order unchanged so accumulated impulses remain identical.
        for contact_id in cargo_contacts:
            robot_side = 0 if robot_is_a[contact_id] else 1
            robot_index = owners[contact_id, robot_side]
            robot_geom = geoms[contact_id, robot_side]
            mujoco.mj_contactForce(model, data, contact_id, self._contact_force)
            normal_force = self._contact_force[0]
            if normal_force > MIN_CONTACT_FORCE_N:
                part = self.geom_kind[robot_geom]
                touching[robot_index] = True
                part_touching[part, robot_index] = True
                self.normal_impulses[part, robot_index] += normal_force * PHYSICS_DT
        self.contact_counts += touching
        self.part_contact_counts += part_touching
        self.robot_collision_steps += robots_collided

    def as_info(self, physics_steps: int) -> Info:
        """Return owned lists with the existing public info keys and robot order."""
        info = {
            "contact_fraction": (self.contact_counts / physics_steps).tolist(),
            "robot_collision_fraction": self.robot_collision_steps / physics_steps,
        }
        for part_index, part_name in enumerate(PART_NAMES):
            info[f"{part_name}_contact_fraction"] = (
                self.part_contact_counts[part_index] / physics_steps
            ).tolist()
            info[f"{part_name}_normal_impulse_ns"] = self.normal_impulses[
                part_index
            ].tolist()
        return info
