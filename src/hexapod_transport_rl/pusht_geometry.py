"""Exact planar footprint overlap for the two non-overlapping T rectangles.

Push-T coverage is intersection area / goal area, not IoU. NumPy uses polygon
clipping; Torch uses the same clipping on padded tensors, entirely on device.
"""

import numpy as np
import torch


def local_rectangles(config):
    """CCW vertices matching cargo_bar and cargo_stem in model.py, in meters."""
    arm, tip = config.t_arm_length, config.t_stem_length
    return np.array(
        [
            [[-0.1, -arm], [0.1, -arm], [0.1, arm], [-0.1, arm]],
            [[0.1, -0.1], [tip, -0.1], [tip, 0.1], [0.1, 0.1]],
        ]
    )


def transform_rectangles(rectangles, xy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return rectangles @ np.array([[c, s], [-s, c]]) + xy


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _polygon_area(vertices):
    if len(vertices) < 3:
        return 0.0
    return abs(_cross(vertices, np.roll(vertices, -1, axis=0)).sum()) / 2


def _intersection_area(subject, clip):
    vertices = subject
    for start, end in zip(clip, np.roll(clip, -1, axis=0), strict=True):
        if not len(vertices):
            return 0.0
        result = []
        previous = vertices[-1]
        previous_distance = _cross(end - start, previous - start)
        if abs(previous_distance) < 1e-12:
            previous_distance = 0.0
        for current in vertices:
            distance = _cross(end - start, current - start)
            if abs(distance) < 1e-12:
                distance = 0.0
            if (
                (distance >= 0) != (previous_distance >= 0)
                and abs(distance) > 1e-12
                and abs(previous_distance) > 1e-12
            ):
                fraction = previous_distance / (previous_distance - distance)
                result.append(previous + fraction * (current - previous))
            if distance >= 0:
                result.append(current)
            previous, previous_distance = current, distance
        vertices = np.asarray(result)
    return _polygon_area(vertices)


def coverage_numpy(rectangles, cargo_xy, cargo_yaw, goal_xy, goal_yaw):
    actual = transform_rectangles(rectangles, cargo_xy, cargo_yaw)
    target = transform_rectangles(rectangles, goal_xy, goal_yaw)
    area = sum(_intersection_area(a, b) for a in actual for b in target)
    return float(np.clip(area / sum(_polygon_area(b) for b in target), 0, 1))


def transform_rectangles_torch(rectangles, xy, yaw):
    c, s = yaw.cos()[:, None, None], yaw.sin()[:, None, None]
    x, y = rectangles[..., 0], rectangles[..., 1]
    return torch.stack((c * x - s * y, s * x + c * y), -1) + xy[:, None, None]


def coverage_torch(rectangles, cargo_xy, cargo_yaw, goal_xy, goal_yaw):
    """Exact convex intersections, at most eight vertices per rectangle pair."""
    actual = transform_rectangles_torch(rectangles, cargo_xy, cargo_yaw)
    target = transform_rectangles_torch(rectangles, goal_xy, goal_yaw)
    batch = len(cargo_xy)
    subject = actual[:, :, None].expand(-1, -1, 2, -1, -1).reshape(batch, 4, 4, 2)
    clip = target[:, None].expand(-1, 2, -1, -1, -1).reshape(batch, 4, 4, 2)
    vertices = torch.cat((subject, torch.zeros_like(subject)), -2)
    counts = torch.full((batch, 4), 4, dtype=torch.long, device=vertices.device)
    indices = torch.arange(8, device=vertices.device).expand(batch, 4, -1)
    for edge in range(4):
        start, end = clip[:, :, edge], clip[:, :, (edge + 1) % 4]
        previous_indices = (indices - 1).remainder(counts.clamp_min(1)[..., None])
        previous = vertices.gather(
            -2, previous_indices[..., None].expand(-1, -1, -1, 2)
        )
        distance = _cross((end - start)[..., None, :], vertices - start[..., None, :])
        previous_distance = _cross(
            (end - start)[..., None, :], previous - start[..., None, :]
        )
        epsilon = 1e-12 if vertices.dtype == torch.float64 else 1e-6
        distance = torch.where(distance.abs() < epsilon, 0.0, distance)
        previous_distance = torch.where(
            previous_distance.abs() < epsilon, 0.0, previous_distance
        )
        valid = indices < counts[..., None]
        inside, previous_inside = distance >= 0, previous_distance >= 0
        crossing = valid & (inside != previous_inside)
        crossing &= (distance.abs() > 1e-12) & (previous_distance.abs() > 1e-12)
        denominator = previous_distance - distance
        denominator = torch.where(denominator.abs() > 1e-12, denominator, 1.0)
        fraction = (previous_distance / denominator).clamp(0, 1)
        intersection = previous + fraction[..., None] * (vertices - previous)
        candidates = torch.stack((intersection, vertices), -2).flatten(-3, -2)
        keep = torch.stack((crossing, valid & inside), -1).flatten(-2)
        output_indices = (keep.long().cumsum(-1) - 1).clamp(0, 7)
        vertices = torch.zeros_like(vertices).scatter_add(
            -2,
            output_indices[..., None].expand(-1, -1, -1, 2),
            candidates * keep[..., None],
        )
        counts = keep.sum(-1)
    following = vertices.gather(
        -2,
        ((indices + 1).remainder(counts.clamp_min(1)[..., None]))[..., None].expand(
            -1, -1, -1, 2
        ),
    )
    area = (_cross(vertices, following) * (indices < counts[..., None])).sum(
        -1
    ).abs() / 2
    goal_area = _cross(rectangles, rectangles.roll(-1, -2)).sum(-1).abs().sum() / 2
    return (area.sum(-1) / goal_area).clamp(0, 1)
