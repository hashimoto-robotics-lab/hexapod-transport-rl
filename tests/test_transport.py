"""The student checkpoint bundle must still transport T using physical contacts."""

import hashlib
import json
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
import torch

from hexapod_transport_rl.approach_evaluation import evaluate
from hexapod_transport_rl.approach_training import load_approach_checkpoint
from hexapod_transport_rl.config import find_asset_root
from hexapod_transport_rl.handover_training import compose
from hexapod_transport_rl.torchrl_mappo import load_mappo

CHECKPOINTS = Path(__file__).resolve().parents[1] / "checkpoints"


def test_short_torchrl_bundle_replays_transport_after_moving(tmp_path):
    """The published 32k-step lesson must run using only its relocated bundle."""
    for name in ("lesson_transport.pt", "pusher.pt"):
        shutil.copy2(CHECKPOINTS / name, tmp_path / name)
    manifest = json.loads((CHECKPOINTS / "manifest.json").read_text())
    checkpoint = tmp_path / "lesson_transport.pt"
    assert (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        == manifest["lesson_transport.pt"]["sha256"]
    )
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["pushing_checkpoint"] == "pusher.pt"
    assert saved["curriculum"]["transitions"] == 32768
    original, _, _ = load_mappo(CHECKPOINTS / "lesson_transport.pt")
    moved, _, _ = load_mappo(checkpoint)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, moved.state_dict()[name], rtol=0, atol=0)
    report = evaluate(
        checkpoint,
        tmp_path / "result.json",
        episodes=1,
        seed=83000,
        workers=1,
        layout="rear",
        max_yaw_degrees=15,
        episode_seconds=30,
        asset_root=find_asset_root(),
    )
    assert report["successes"] == report["handovers"] == 1
    assert report["body_contact_episodes"] == report["falls"] == 0
    assert report["navigation_robot_contact_episodes"] == 0
    impulse = report["episodes"][0]["total_cargo_normal_impulse_ns"]
    assert np.all(np.array(impulse["leg_link"]) > 1)
    assert np.all(np.array(impulse["foot"]) > 1)
    assert np.all(np.array(impulse["body"]) == 0)


def test_shared_bundle_replays_successful_transport_after_moving(tmp_path):
    """Exercise relative model paths, an asset override, and full learned control."""
    bundle = tmp_path / "student"
    bundle.mkdir()
    shutil.copy2(CHECKPOINTS / "pusher.pt", bundle / "pusher.pt")
    compose(CHECKPOINTS / "transport.pt", bundle / "pusher.pt", bundle / "transport.pt")
    saved = torch.load(bundle / "transport.pt", weights_only=True)
    assert saved["pushing_checkpoint"] == "pusher.pt"
    # The original training machine is unavailable on a student's computer.
    saved["run"]["provenance"]["asset_root"] = str(tmp_path / "absent_training_machine")
    torch.save(saved, bundle / "transport.pt")
    moved = tmp_path / "moved"
    bundle.rename(moved)
    original_actor, _, _ = load_approach_checkpoint(CHECKPOINTS / "transport.pt")
    moved_actor, _, _ = load_approach_checkpoint(moved / "transport.pt")
    for name, value in original_actor.state_dict().items():
        torch.testing.assert_close(
            value, moved_actor.state_dict()[name], rtol=0, atol=0
        )

    result_path = tmp_path / "result.json"
    result = evaluate(
        moved / "transport.pt",
        result_path,
        episodes=1,
        seed=70000,
        workers=1,
        asset_root=find_asset_root(),
    )
    assert json.loads(result_path.read_text()) == result
    assert result["successes"] == result["handovers"] == 1
    assert result["body_contact_episodes"] == result["falls"] == 0
    assert result["navigation_robot_contact_episodes"] == 0
    episode = result["episodes"][0]
    # The pre-cleanup selected model's held-out episode, recorded on 2026-10-06.
    assert episode["distance"] == pytest.approx(0.11483067604983968, abs=1e-5)
    assert episode["yaw_error"] == pytest.approx(0.24061641787838184, abs=1e-5)
    assert episode["handover_time"] == pytest.approx(16.2)
    assert episode["elapsed_seconds"] == pytest.approx(39.8)
    impulse = episode["total_cargo_normal_impulse_ns"]
    assert np.all(np.array(impulse["leg_link"]) + impulse["foot"] > 1)
    assert np.all(np.array(impulse["body"]) == 0)


def test_fast_recording_preserves_the_learned_transport_result(tmp_path):
    """Drawing fewer frames/passes must retain the physical success and contacts."""
    report = evaluate(
        CHECKPOINTS / "transport.pt",
        tmp_path / "result.json",
        episodes=1,
        seed=70000,
        video_dir=tmp_path / "video",
        video_width=320,
        video_height=240,
        video_fps=5,
        fast_video=True,
    )
    episode = report["episodes"][0]
    assert report["successes"] == report["handovers"] == 1
    assert report["body_contact_episodes"] == report["falls"] == 0
    assert episode["elapsed_seconds"] == pytest.approx(39.8)
    assert episode["distance"] == pytest.approx(0.11483067604983968, abs=1e-5)
    assert episode["yaw_error"] == pytest.approx(0.24061641787838184, abs=1e-5)
    with imageio.get_reader(tmp_path / "video/seed_70000.mp4") as video:
        assert video.get_meta_data()["fps"] == 5
        assert video.count_frames() == 199
