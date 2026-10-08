"""State cost for approaching a pushing position around the cargo.

This is a state potential for rewards, not a controller. It returns only a
scalar cost for each robot; no waypoint or action is given to the policy.
Front/side clearance allows room for the legs; rear clearance permits pushing.
"""

import numpy as np
import torch


def _amin(value, math):
    return math.amin(value, axis=-1) if math is np else value.amin(-1)


def _amax(value, math):
    return math.amax(value, axis=-1) if math is np else value.amax(-1)


def _length(vector, math):
    return math.sqrt((vector * vector).sum(axis=-1))


def _inside(point, bounds):
    low, high = bounds
    return ((point > low) & (point < high)).all(-1)


def _blocked(start, end, rectangles, math):
    """Whether a segment crosses the interior of either inflated T piece."""
    delta = end - start
    flat = abs(delta) < 1e-8
    divisor = math.where(flat, 1.0, delta)
    blocked = (
        (delta[..., 0] * 0).astype(bool)
        if math is np
        else torch.zeros_like(delta[..., 0], dtype=torch.bool)
    )
    for low, high in rectangles:
        # Edges and vertices themselves remain available to a shortest path.
        low, high = low + 1e-5, high - 1e-5
        a, b = (low - start) / divisor, (high - start) / divisor
        entering, leaving = math.minimum(a, b), math.maximum(a, b)
        parallel_inside = (start >= low) & (start <= high)
        entering = math.where(
            flat, math.where(parallel_inside, -math.inf, math.inf), entering
        )
        leaving = math.where(
            flat, math.where(parallel_inside, math.inf, -math.inf), leaving
        )
        begin = math.maximum(_amax(entering, math), delta[..., 0] * 0)
        finish = math.minimum(_amin(leaving, math), delta[..., 0] * 0 + 1)
        blocked |= begin < finish - 1e-5
    return blocked


class PushApproachDistance:
    """Shortest outside path plus a cost for entering the clearance region.

    The visibility graph is built once. At each physics step, only distances
    and visibility from the current positions are computed, in NumPy or CUDA.
    Inside points use the cheapest boundary exit, with penetration weighted
    more than outside travel. This keeps the cost continuous and discourages
    shortcuts. A weight of one preserves earlier experimental checkpoints.
    """

    def __init__(self, config, device=None):
        self.penetration_cost = config.approach_penetration_cost
        clearance = config.approach_clearance
        rear_x = 0.1 + min(clearance, config.push_gap * 0.8)
        bar_x, bar_y = 0.1 + clearance, config.width / 2 + clearance
        stem_x, stem_y = config.t_stem_length + clearance, 0.1 + clearance
        polygon = np.array(
            [
                [-rear_x, -bar_y],
                [bar_x, -bar_y],
                [bar_x, -stem_y],
                [stem_x, -stem_y],
                [stem_x, stem_y],
                [bar_x, stem_y],
                [bar_x, bar_y],
                [-rear_x, bar_y],
            ]
        )
        rectangles = np.array(
            [
                [[-rear_x, -bar_y], [bar_x, bar_y]],
                [[max(0.1 - clearance, -rear_x), -stem_y], [stem_x, stem_y]],
            ]
        )
        targets = np.column_stack(
            (
                np.full(config.num_robots, config.rear_face - config.push_gap),
                config.slots,
            )
        )
        points = np.concatenate((polygon, targets))
        start, end = points[:, None], points[None, :]
        costs = np.where(
            _blocked(start, end, rectangles, np), np.inf, _length(end - start, np)
        )
        for k in range(len(points)):
            costs = np.minimum(costs, costs[:, k, None] + costs[None, k, :])
        assert np.isfinite(costs[: len(polygon), len(polygon) :]).all()
        self.math = np if device is None else torch

        def array(value):
            return (
                np.asarray(value)
                if device is None
                else torch.as_tensor(value, dtype=torch.float32, device=device)
            )

        self.vertices = array(polygon)
        self.edge_vectors = array(np.roll(polygon, -1, axis=0) - polygon)
        self.edge_lengths_squared = (self.edge_vectors * self.edge_vectors).sum(axis=-1)
        self.edge_ids = array(np.arange(len(polygon)))
        self.rectangles = array(rectangles)
        self.targets = array(targets)
        self.cost_to_target = array(costs[: len(polygon), len(polygon) :].T)

    def __call__(self, positions):
        math = self.math
        inside = _inside(positions, self.rectangles[0]) | _inside(
            positions, self.rectangles[1]
        )
        relative = positions[..., None, :] - self.vertices
        fraction = (
            (relative * self.edge_vectors).sum(axis=-1) / self.edge_lengths_squared
        ).clip(0, 1)
        projections = self.vertices + fraction[..., None] * self.edge_vectors
        errors = _length(projections - positions[..., None, :], math)
        if self.penetration_cost > 1:
            # Penalize entering the leg-clearance region. Taking the minimum
            # over all boundary exits gives a continuous extension; choosing
            # only the nearest boundary can reward shortcuts through the T.
            escaped = self._outside_distance(projections, extra_axis=True)
            inside_cost = _amin(self.penetration_cost * errors + escaped, math)
            return math.where(inside, inside_cost, self._outside_distance(positions))
        nearest = errors.argmin(-1)
        boundary = (projections * (nearest[..., None] == self.edge_ids)[..., None]).sum(
            axis=-2
        )
        origin = math.where(inside[..., None], boundary, positions)
        escape = math.where(inside, _length(positions - boundary, math), 0.0)
        return escape + self._outside_distance(origin)

    def _outside_distance(self, origin, *, extra_axis=False):
        math = self.math
        targets = self.targets[:, None] if extra_axis else self.targets
        costs = self.cost_to_target[:, None] if extra_axis else self.cost_to_target
        visible = ~_blocked(origin[..., None, :], self.vertices, self.rectangles, math)
        via = _length(origin[..., None, :] - self.vertices, math) + costs
        via = math.where(visible, via, math.inf)
        direct = _length(origin - targets, math)
        direct = math.where(
            _blocked(origin, targets, self.rectangles, math), math.inf, direct
        )
        return math.minimum(direct, _amin(via, math))
