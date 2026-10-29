"""Reset scheduling must train fresh PPO without changing the final task."""

import json
from dataclasses import asdict, replace

import pytest
import torch

from hexapod_transport_rl import (
    CylinderPushTask,
    PPOSettings,
    PushCurriculum,
    PushPhase,
    train,
)


def test_curriculum_changes_only_resets_and_finishes_with_the_full_task():
    task = CylinderPushTask.baseline()
    plan = PushCurriculum()
    assert plan.phase_at(0, 200).name == "rear_contact"
    assert plan.phase_at(199, 200).name == "full_task"
    assert plan.phases[-1].apply(task.config) == task.config
    for phase in plan.phases:
        changed = asdict(phase.apply(task.config))
        original = asdict(task.config)
        assert {k for k in changed if changed[k] != original[k]} <= set(phase.resets)
        assert changed["position_tolerance"] == 0.05
        assert changed["success_hold_seconds"] == 0.4
        assert changed["episode_seconds"] == 40
    with pytest.raises(ValueError, match="reset"):
        PushCurriculum(
            (PushPhase("bad", 0.5, {"position_tolerance": 1}), PushPhase("full", 1))
        )


@pytest.mark.parametrize("backend", ["cpu", "warp"])
def test_scheduled_reset_applies_to_real_physics_on_both_backends(backend):
    if backend == "warp" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    task = CylinderPushTask.baseline()
    plan = PushCurriculum()
    env = task.make_env(num_envs=8, backend=backend)
    try:
        env.set_reset_stage(plan.phases[0])
        env.set_seed(7041)
        env.reset()
        assert torch.all(
            (env.state.goal_distance >= 0.15) & (env.state.goal_distance <= 0.25)
        )
        radii = env.state.robot_to_cargo.norm(dim=-1)
        assert torch.all((radii >= 0.60 - 1e-5) & (radii <= 0.75 + 1e-5))
        env.set_reset_stage(plan.phases[-1])
        env.set_seed(7041)
        env.reset()
        assert torch.all(
            (env.state.goal_distance >= 0.30) & (env.state.goal_distance <= 0.60)
        )
        assert (env.state.goal_distance > 0.25).all()
    finally:
        env.close()


def test_one_continuous_ppo_run_starts_random_and_visits_every_phase(tmp_path):
    task = CylinderPushTask.baseline()
    task = replace(task, config=replace(task.config, episode_seconds=0.4))
    run = train(
        task,
        tmp_path / "fresh",
        seed=201,
        steps=80,
        num_envs=2,
        horizon=2,
        backend="cpu",
        validation_envs=2,
        validate_every=10,
        curriculum=PushCurriculum(),
        checkpoint_selection="latest",
        settings=PPOSettings(epochs=1, minibatch_size=4),
    )
    metadata = json.loads((run.directory / "run.json").read_text())
    rows = json.loads((run.directory / "progress.json").read_text())
    assert metadata["initial_checkpoint"] is None
    assert metadata["initialization"].startswith("random actor and critic")
    assert metadata["training_stage"] == "full_task"
    assert {r["curriculum_stage"] for r in rows} == {
        p.name for p in PushCurriculum().phases
    }
    assert rows[0]["learning_rate"] == 3e-4
    assert rows[-1]["learning_rate"] == pytest.approx(3e-5)
    initial = torch.load(
        run.directory / "initial.pt", map_location="cpu", weights_only=True
    )
    final = torch.load(run.checkpoint, map_location="cpu", weights_only=True)
    selected = torch.load(run.best, map_location="cpu", weights_only=True)
    assert selected["training"]["training_stage"] == "full_task"
    assert selected["training"]["steps"] >= 68
    assert not initial["optimizer"]["state"]
    assert final["optimizer"]["state"]
    assert {int(value["step"]) for value in final["optimizer"]["state"].values()} == {
        20
    }
    for model in ("actor", "critic"):
        assert any(
            not torch.equal(initial[model][k], final[model][k]) for k in initial[model]
        )
