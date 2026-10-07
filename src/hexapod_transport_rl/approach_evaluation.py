"""Evaluate learned approach followed by the frozen learned pushing actor."""

import json
import multiprocessing as mp
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch

from .approach import (
    HANDOVER_HOLD_STEPS,
    ApproachConfig,
    approach_ready,
    observe_approach,
    physical_actions,
    reset_approach,
)
from .approach_training import load_approach_checkpoint
from .config import CONTROL_DT, PushConfig
from .env import PushEnv
from .types import AgentPolicy, FloatArray, Info


class LearnedTransport:
    """Two learned actors and an explicit, one-way geometric handover condition.

    Neither the navigation reward nor a route/controller supplies actions here.
    Call reset for each episode. The learned actors retain no episode state.
    """

    def __init__(self, navigator: AgentPolicy, pusher: AgentPolicy) -> None:
        self.navigator, self.pusher = navigator, pusher
        self.reset()

    def reset(self) -> None:
        self.ready_steps = 0
        self.pushing = False
        self.handover_time = None

    def __call__(self, env: PushEnv) -> FloatArray:
        if not self.pushing:
            self.ready_steps = self.ready_steps + 1 if approach_ready(env).all() else 0
            if self.ready_steps >= HANDOVER_HOLD_STEPS:
                self.pushing = True
                self.handover_time = env.steps * env.cfg.dt
        if self.pushing:
            return self.pusher.act(env.observe())
        return physical_actions(self.navigator.act(observe_approach(env)))


def _load_evaluation_task(
    checkpoint: str | Path,
    layout: str,
    max_yaw_degrees: float,
    asset_root: str | Path | None,
    episode_seconds: float,
) -> tuple[PushEnv, ApproachConfig, LearnedTransport]:
    """保存された物理設定と重みを読み込み、学習時の資産と照合する。"""
    torch.set_num_threads(1)
    navigator, pusher, saved = load_approach_checkpoint(checkpoint)
    pushing_saved = torch.load(
        saved["pushing_checkpoint"], weights_only=True, map_location="cpu"
    )
    core = PushEnv(
        replace(PushConfig(**pushing_saved["config"]), episode_seconds=episode_seconds),
        asset_root,
    )
    provenance = saved["run"]["provenance"]
    if (
        core.policy_sha256 != provenance["low_level_sha256"]
        or core.model_sha256 != provenance["robot_xml_sha256"]
    ):
        raise ValueError("Evaluation robot or walking model differs from training")
    config = replace(
        ApproachConfig(**saved["approach_config"]),
        layout=layout,
        max_yaw_degrees=max_yaw_degrees,
        easier_reset_fraction=0,
    )
    return core, config, LearnedTransport(navigator, pusher)


def _run_episode(
    core: PushEnv,
    controller: LearnedTransport,
    seed: int,
    on_control_step: Callable[[], None] | None = None,
    trace: list[Info] | None = None,
) -> Info:
    """reset済みの世界を運搬終了まで進め、接触・報酬・最終状態を集計する。"""
    initial = core.info()
    body_impulse = np.zeros(2)
    leg_impulse = np.zeros(2)
    foot_impulse = np.zeros(2)
    robot_contact = navigation_contact = False
    total_reward = 0.0

    while not core.done:
        action = controller(core)
        _, reward, _, _, info = core.step(action, on_control_step=on_control_step)
        total_reward += reward
        if trace is not None:
            trace.append(
                dict(
                    time=info["elapsed_seconds"],
                    phase="push" if controller.pushing else "approach",
                    robot_xy=core.robot_xy.tolist(),
                    cargo_xy=info["cargo_xy"],
                    distance=info["distance"],
                    action=action.tolist(),
                )
            )
        body_impulse += info["body_normal_impulse_ns"]
        leg_impulse += info["leg_link_normal_impulse_ns"]
        foot_impulse += info["foot_normal_impulse_ns"]
        robot_contact |= info["robot_collision_fraction"] > 0
        navigation_contact |= (
            not controller.pushing and info["robot_collision_fraction"] > 0
        )

    return dict(
        seed=seed,
        initial=initial,
        **info,
        handover=controller.pushing,
        handover_time=controller.handover_time,
        total_reward=total_reward,
        body_contact=bool(body_impulse.sum() > 1e-6),
        robot_contact=robot_contact,
        navigation_robot_contact=navigation_contact,
        total_cargo_normal_impulse_ns=dict(
            body=body_impulse.tolist(),
            leg_link=leg_impulse.tolist(),
            foot=foot_impulse.tolist(),
        ),
    )


def run_batch(
    checkpoint: str | Path,
    seeds: Sequence[int],
    layout: str,
    max_yaw_degrees: float,
    video_dir: str | Path | None = None,
    asset_root: str | Path | None = None,
    *,
    episode_seconds: float = 100,
    video_width: int = 960,
    video_height: int = 720,
    video_fps: int = 25,
    fast_video: bool = False,
) -> list[Info]:
    """指定seedを順番に評価する。録画は物理更新後のコールバックで行う。"""
    core, config, controller = _load_evaluation_task(
        checkpoint, layout, max_yaw_degrees, asset_root, episode_seconds
    )
    renderer = None
    video_path = Path(video_dir) if video_dir else None
    if video_path is not None:
        import imageio.v2 as imageio

        video_path.mkdir(parents=True, exist_ok=False)
        renderer = mujoco.Renderer(core.model, height=video_height, width=video_width)
        if fast_video:
            # Software rendering benefits from fewer drawing passes; physics is unchanged.
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 6, 135, -50
        scene_options = mujoco.MjvOption()
        scene_options.geomgroup[3] = 0

    results = []
    try:
        for seed in seeds:
            reset_approach(core, config, seed)
            controller.reset()
            trace = []
            movie = None
            frame_index = 0
            if renderer is not None:
                # Include the starting robots and the destination in one fixed view.
                visible_xy = np.vstack((core.robot_xy, core.cargo_xy, core.goal))
                lower, upper = visible_xy.min(axis=0), visible_xy.max(axis=0)
                camera.lookat[:] = [*((lower + upper) / 2), 0.15]
                camera.distance = max(6.0, 1.8 * float((upper - lower).max()))
                movie = imageio.get_writer(
                    video_path / f"seed_{seed}.mp4", fps=video_fps
                )

            def capture_frame(movie=movie):
                nonlocal frame_index
                frame_stride = round(1 / CONTROL_DT) // video_fps
                if frame_index % frame_stride == 0:
                    renderer.update_scene(
                        core.data, camera=camera, scene_option=scene_options
                    )
                    movie.append_data(renderer.render())
                frame_index += 1

            try:
                results.append(
                    _run_episode(
                        core,
                        controller,
                        seed,
                        on_control_step=capture_frame if renderer is not None else None,
                        trace=trace if video_path is not None else None,
                    )
                )
            finally:
                if movie is not None:
                    movie.close()
                    (video_path / f"seed_{seed}_trace.json").write_text(
                        json.dumps(trace, indent=2) + "\n"
                    )
    finally:
        if renderer is not None:
            renderer.close()
    return results


def evaluate(
    checkpoint: str | Path,
    output: str | Path,
    episodes: int = 24,
    seed: int = 70000,
    workers: int = 4,
    layout: str = "front",
    max_yaw_degrees: float = 30,
    video_dir: str | Path | None = None,
    asset_root: str | Path | None = None,
    episode_seconds: float = 100,
    video_width: int = 960,
    video_height: int = 720,
    video_fps: int = 25,
    fast_video: bool = False,
) -> Info:
    """未使用seedで運搬全体を評価し、成功率・接触・誤差をJSONに保存する。"""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if episodes < 1 or workers < 1:
        raise ValueError("Positive episodes and workers required")
    if min(video_width, video_height) < 1 or video_fps not in (1, 5, 25):
        raise ValueError("Positive video dimensions and fps in (1, 5, 25) required")
    if video_dir and episodes != 1:
        raise ValueError("Record one explicit seed per video command")
    if video_dir:
        results = run_batch(
            checkpoint,
            [seed],
            layout,
            max_yaw_degrees,
            video_dir,
            asset_root,
            episode_seconds=episode_seconds,
            video_width=video_width,
            video_height=video_height,
            video_fps=video_fps,
            fast_video=fast_video,
        )
    else:
        workers = min(workers, episodes)
        seeds = list(range(seed, seed + episodes))
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(
                    run_batch,
                    checkpoint,
                    seeds[i::workers],
                    layout,
                    max_yaw_degrees,
                    asset_root=asset_root,
                    episode_seconds=episode_seconds,
                )
                for i in range(workers)
            ]
            results = sorted(
                [row for future in futures for row in future.result()],
                key=lambda row: row["seed"],
            )
    successes = sum(row["success"] for row in results)
    report = dict(
        checkpoint=str(Path(checkpoint).resolve()),
        controller="learned_approach_then_frozen_learned_push",
        demonstration_actions_used=False,
        layout=layout,
        max_yaw_degrees=max_yaw_degrees,
        seed=seed,
        episodes=results,
        success_rate=successes / episodes,
        successes=successes,
        handovers=sum(row["handover"] for row in results),
        body_contact_episodes=sum(row["body_contact"] for row in results),
        robot_contact_episodes=sum(row["robot_contact"] for row in results),
        navigation_robot_contact_episodes=sum(
            row["navigation_robot_contact"] for row in results
        ),
        falls=sum(row["robot_fall"] or row["cargo_fall"] for row in results),
        mean_final_distance_m=float(np.mean([row["distance"] for row in results])),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}), flush=True)
    return report
