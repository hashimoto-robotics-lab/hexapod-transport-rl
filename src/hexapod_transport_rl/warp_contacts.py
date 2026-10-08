"""Batched contact counts and impulses on GPU, using MuJoCo Warp forces.

Counts measure physics ticks with contact, rather than contact-point counts.
No force is applied here. The same geom ownership as ContactTracker is used.
"""

import mujoco_warp as mjw
import warp as wp

from .config import PHYSICS_DT


@wp.kernel
def _mark_contacts(
    total: wp.array[int],
    world: wp.array[int],
    geoms: wp.array[wp.vec2i],
    distance: wp.array[float],
    forces: wp.array[wp.spatial_vector],
    owner: wp.array[int],
    kind: wp.array[int],
    touching: wp.array2d[int],
    parts: wp.array3d[int],
    collisions: wp.array[int],
    impulses: wp.array3d[float],
):
    contact = wp.tid()
    if contact >= total[0] or distance[contact] > 0.0:
        return
    lane = world[contact]
    a, b = geoms[contact][0], geoms[contact][1]
    ra, rb = owner[a], owner[b]
    if ra >= 0 and rb >= 0 and ra != rb:
        wp.atomic_max(collisions, lane, 1)
    robot, geom = int(-1), int(-1)
    if ra >= 0 and rb == -2:
        robot, geom = int(ra), int(a)
    elif rb >= 0 and ra == -2:
        robot, geom = int(rb), int(b)
    if robot >= 0 and forces[contact][0] > 1.0e-6:
        part = kind[geom]
        wp.atomic_max(touching, lane, robot, 1)
        wp.atomic_max(parts, lane, part, robot, 1)
        wp.atomic_add(impulses, lane, part, robot, forces[contact][0] * PHYSICS_DT)


class WarpContactTracker:
    """Keep native contact semantics while never copying contacts to CPU."""

    def __init__(self, model, data, reference, num_worlds, num_robots):
        self.model, self.data = model, data
        self.owner = wp.array(reference.geom_owner, dtype=int)
        self.kind = wp.array(reference.geom_kind, dtype=int)
        self.ids = wp.array(list(range(data.naconmax)), dtype=int)
        self.forces = wp.zeros(data.naconmax, dtype=wp.spatial_vector)
        self.touching = wp.zeros((num_worlds, num_robots), dtype=int)
        self.parts = wp.zeros((num_worlds, 3, num_robots), dtype=int)
        self.collisions = wp.zeros(num_worlds, dtype=int)
        self.impulses = wp.zeros((num_worlds, 3, num_robots), dtype=float)
        self.tick_touching = wp.to_torch(self.touching)
        self.tick_parts = wp.to_torch(self.parts)
        self.tick_collisions = wp.to_torch(self.collisions)
        self.normal_impulses = wp.to_torch(self.impulses)
        self.contact_counts = self.tick_touching.clone()
        self.part_contact_counts = self.tick_parts.clone()
        self.robot_collision_steps = self.tick_collisions.clone()

    def reset(self):
        self.contact_counts.zero_()
        self.part_contact_counts.zero_()
        self.robot_collision_steps.zero_()
        self.normal_impulses.zero_()

    def record(self):
        self.tick_touching.zero_()
        self.tick_parts.zero_()
        self.tick_collisions.zero_()
        mjw.contact_force(self.model, self.data, self.ids, False, self.forces)
        wp.launch(
            _mark_contacts,
            dim=self.data.naconmax,
            inputs=[
                self.data.nacon,
                self.data.contact.worldid,
                self.data.contact.geom,
                self.data.contact.dist,
                self.forces,
                self.owner,
                self.kind,
                self.touching,
                self.parts,
                self.collisions,
                self.impulses,
            ],
        )
        self.contact_counts.add_(self.tick_touching)
        self.part_contact_counts.add_(self.tick_parts)
        self.robot_collision_steps.add_(self.tick_collisions)
