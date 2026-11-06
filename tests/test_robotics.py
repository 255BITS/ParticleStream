"""Simulator contracts, destructive controls and actual public-trainer execution."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from particlegan_forge import CandidateSpec, NamedStreams

from particlestream import robotics_tasks as data
from particlestream.robotics_adapter import RoboticsSession, _evaluate
from particlestream.minimal_adapter import _build
from particlestream.minimal_online import isolated_session
from particlestream.suite import PRIMARY_VIEW, ROBOTICS_TASKS, bootstrap_workspace

pytest.importorskip("gymnasium_robotics")
pytest.importorskip("mujoco")


def _streams():
    return NamedStreams(seed=0)


def _workspace(path):
    workspace = bootstrap_workspace(path)
    source = Path(__file__).resolve().parents[1] / "configs/forge/ideas/online-feature-prior-bce.json"
    workspace.register_candidate(CandidateSpec.from_dict(json.loads(source.read_text())))
    return workspace


def _cases(task):
    streams = _streams()
    rng = streams.generator("eval", component="robotics", purpose="heldout_episodes", device="cpu")
    return data.heldout_transitions(task.host_definition["mode"], rng)


def test_roles_budget_and_physical_encoding():
    assignments = {assignment.task: assignment for assignment in PRIMARY_VIEW.assignments}
    assert len(ROBOTICS_TASKS) == 5
    assert sum(task.steps for task in ROBOTICS_TASKS) == 2400
    assert sum(assignment.importance == "required" for assignment in assignments.values()) == 10
    for task in ROBOTICS_TASKS:
        assert assignments[task.id].importance == "stretch"
        assert assignments[task.id].qualification_tier == 2
        assert task.batch_size == 1 and len(task.observation_steps()) == 24
    delta = np.array([.0123, -.0045, .0007, .0067, -.0008, .0033])
    decoded = data.decode_delta(data.encode_delta(delta), 6)
    assert np.allclose(decoded[0].numpy(), delta, atol=2e-8)
    with pytest.raises(ValueError, match="no clipping"):
        data.encode_delta(np.array([.101]))


@pytest.mark.parametrize("task", ROBOTICS_TASKS, ids=lambda task: task.id)
def test_real_simulator_controls_reproducibility_and_input_completeness(task):
    cases, seeds = _cases(task)
    again, again_seeds = _cases(task)
    assert seeds == again_seeds and len(set(seeds)) == 4
    assert len(cases) == 16
    for case, other in zip(cases, again):
        assert np.array_equal(case.delta, other.delta)
        assert np.array_equal(case.opposite_delta, other.opposite_delta)
    results = data.validate_controls(task, cases)
    assert results["oracle"]["verdict"] == "PASS"
    assert all(value["verdict"] == "FAIL" for name, value in results.items() if name != "oracle")
    assert results["persistence"]["metrics"]["moving_coordinate_fraction"] > .1
    corrupted = torch.cat([data.encode_delta(case.delta) for case in cases])[:, None]
    corrupted[0, 0, 0, 0, 0] = float("nan")
    assert data.score_predictions(corrupted, cases)["finite_fraction"] < 1
    mode = task.host_definition["mode"]
    env = data.make_env(mode)
    try:
        obs, _ = env.reset(seed=0)
        action = np.linspace(-1, 1, env.action_space.shape[0], dtype=np.float32)
        state_frame, action_frame = data.encode_source(obs, action)
        n = len(obs["observation"])
        assert torch.allclose(state_frame.flatten()[:n], torch.tensor(.5 + .5 * np.tanh(obs["observation"]), dtype=torch.float32))
        assert torch.allclose(action_frame.flatten()[:len(action)], torch.tensor(.5 + .5 * action))
    finally:
        env.close()


def test_prediction_precedes_physics_and_evaluation_restores_training_state(tmp_path):
    workspace = _workspace(tmp_path)
    task = ROBOTICS_TASKS[0]
    candidate = workspace.load_candidates()["online-feature-prior-bce"]
    streams = _streams()
    trainer, g, d, memory, _ = _build(task, candidate, streams, 2, "cpu")
    session = RoboticsSession(trainer, g, d, memory,
        serving_generator=streams.generator("eval", component="serving", purpose="prediction", device="cpu"))
    env = data.make_env("fetch_reach")
    try:
        obs, _ = env.reset(seed=0)
        action = np.array([.8, -.3, .4, 0.], dtype=np.float32)
        time_before = env.unwrapped.data.time
        session.source_pair(*data.encode_source(obs, action))
        with pytest.raises(RuntimeError, match="prediction"):
            session.observe_target(torch.full((1, 1, 8, 8), .5))
        with streams.isolated_evaluation():
            session.predict()
        assert env.unwrapped.data.time == time_before
        following, *_ = env.step(action)
        assert env.unwrapped.data.time > time_before
        session.observe_target(data.encode_delta(data.tracked_positions("fetch_reach", following) - data.tracked_positions("fetch_reach", obs)))
        cases, _ = _cases(task)
        roots = [g, d, memory, trainer.prior, trainer.transport]
        before = [deepcopy(module.state_dict()) for module in roots]
        diagnostics = session.diagnostics()
        modes = [[module.training for module in root.modules()] for root in roots]
        context = session.context
        with streams.isolated_evaluation(), isolated_session(session):
            metrics, samples = _evaluate(session, cases, session.serving_generator, "cpu")
        assert streams.last_eval_audit["unintended_rng_deviations"] == 0
        assert samples.shape == (16, 4, 1, 8, 8)
        assert metrics["finite_fraction"] == 1
        for root, saved, flags in zip(roots, before, modes):
            for key, value in root.state_dict().items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, saved[key])
                else:
                    assert value == saved[key]
            assert [module.training for module in root.modules()] == flags
        assert torch.equal(session.context, context)
        assert session.diagnostics() == diagnostics
    finally:
        env.close()


def test_opposite_action_branch_preserves_the_original_simulator_trajectory():
    task = ROBOTICS_TASKS[1]
    cases, seeds = _cases(task)
    streams = _streams()
    rng = streams.generator("eval", component="robotics", purpose="heldout_episodes", device="cpu")
    env = data.make_env("fetch_push")
    try:
        for episode in range(4):
            observation, seed = data.reset_env(env, rng)
            assert seed == seeds[episode]
            for step in range(data.EPISODE_LENGTH):
                action = data.policy_action("fetch_push", observation, step, rng, 4)
                following, *_ = env.step(action)
                if step in (7, 17, 27, 37):
                    case = next(case for case in cases if case.episode == episode and case.step == step)
                    delta = data.tracked_positions("fetch_push", following) - data.tracked_positions("fetch_push", observation)
                    assert np.allclose(delta, case.delta, atol=1e-12, rtol=0)
                observation = following
    finally:
        env.close()


@pytest.mark.parametrize("task", ROBOTICS_TASKS, ids=lambda task: task.id)
def test_public_forge_short_run_has_real_training_and_causal_receipt(tmp_path, task):
    workspace = _workspace(tmp_path)
    receipt = workspace.run_task("online-feature-prior-bce", task.id, device="cpu", steps_override=2)
    assert receipt["verdict"] == "INCOMPLETE", receipt["error"]
    assert receipt["all_finite"] and receipt["unintended_rng_deviations"] == 0
    artifacts = receipt["artifacts"]
    execution = artifacts["online_execution"]
    assert execution["updates"] == execution["live_predictions"] == 2
    assert execution["source_only_arrivals"] == 4 and execution["target_arrivals"] == 2
    assert execution["arrivals"] == 6 and execution["target_memory_writes"] == 0
    assert execution["max_real_batch"] == execution["max_fake_batch"] == 1
    assert execution["replayed_examples"] == 0
    assert artifacts["recipe_delta"] == {}
    assert not set(artifacts["training_reset_seeds"]) & set(artifacts["eval_reset_seeds"])
    assert artifacts["compute"]["optimizer_updates"] == 2
    assert artifacts["transport"]["target_retained"] is False
    assert (workspace.queue_root / "online-feature-prior-bce" / task.id / "goal.gif").is_file()
    assert not (workspace.reports_dir / "receipts/cpu/online-feature-prior-bce" / f"{task.id}.json").exists()
