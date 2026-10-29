"""Capture physics, observations and rewards in one graph for either robot model."""

import torch


def capture_world_step(worlds):
    """Optimize execution without changing equations, actions or reset behavior.

    A single external Warp capture is necessary: nested replay of the existing
    physics CUDA graph is avoided. Returned tensors are copied, so subsequent
    graph replays cannot overwrite transitions retained by a collector.
    """
    import warp as wp

    if hasattr(worlds, "step_graph"):
        return
    physics = worlds.physics
    raw_step, physical_step = worlds.step, physics.step
    targets = torch.zeros(
        worlds.num_envs, *worlds.single_action_space.shape, device=worlds.device
    )
    for _ in range(2):
        raw_step(targets)
    torch.cuda.synchronize(worlds.device)

    def eager_physics(commands):
        physics.commands.copy_(commands)
        physics._advance()
        physics.invalid_seen.logical_or_(
            ~(
                torch.isfinite(physics.qpos).all(-1)
                & torch.isfinite(physics.qvel).all(-1)
            )
        )

    graph = torch.cuda.CUDAGraph()
    physics.stream.wait_stream(torch.cuda.current_stream(worlds.device))
    physics.step = eager_physics
    try:
        with (
            torch.cuda.graph(graph, stream=physics.stream),
            wp.ScopedStream(physics.warp_stream, sync_enter=False),
            wp.ScopedCapture(
                stream=physics.warp_stream,
                external=True,
                capture_mode=wp.CaptureMode.GLOBAL,
            ) as capture,
        ):
            result = raw_step(targets)
    finally:
        physics.step = physical_step
    torch.cuda.current_stream(worlds.device).wait_stream(physics.stream)
    worlds.step_graph, worlds.step_warp_graph = graph, capture.graph
    worlds.reset(seed=0)

    def copied(value):
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, dict):
            return {key: copied(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(copied(item) for item in value)
        return value

    def replay(action):
        targets.copy_(action)
        graph.replay()
        return copied(result)

    worlds.step = replay
    close_world = worlds.close

    def close():
        close_world()
        worlds.step = raw_step
        worlds.close = close_world
        worlds.step_graph = worlds.step_warp_graph = None

    worlds.close = close
