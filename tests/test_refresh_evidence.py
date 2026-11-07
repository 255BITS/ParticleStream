"""The published full comparison must remain current, complete and reproducible."""
import json
from pathlib import Path

from PIL import Image
import pytest
from particlegan_forge import file_hash, grade_sustained_observations

from particlestream.forge import StreamWorkspace
from particlestream.online import validate_execution_policy
from particlestream.suite import REGISTRY


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def refreshed():
    workspace = StreamWorkspace(ROOT, registry=REGISTRY)
    audit = json.loads((workspace.reports_dir / "refresh-audit.json").read_text())
    assert audit["completed_at"] is not None, "full recipe refresh is unfinished"
    candidates = workspace.load_candidates()
    executable, historical = set(), set()
    for cid, candidate in candidates.items():
        try:
            validate_execution_policy(candidate)
        except ValueError:
            historical.add(cid)
        else:
            executable.add(cid)
    view_id = "single_stream_stability"
    view = workspace.load_views()[view_id]
    tasks = workspace.load_tasks()
    task_ids = {a.task for a in view.assignments}
    assert set(audit["candidates"]) == executable
    assert set(audit["historical_read_only"]) == historical
    assert set(audit["tasks"]) == task_ids
    pairs = {f"{cid}/{tid}" for cid in executable for tid in task_ids}
    assert set(audit["runs"]) == pairs
    assert audit["planned_runs"] == len(pairs)
    assert set(audit["controls"]) == task_ids
    for controls in audit["controls"].values():
        assert controls["oracle"]["verdict"] == "PASS"
        assert len(controls) >= 2
        assert all(v["verdict"] == "FAIL" for k, v in controls.items() if k != "oracle")
    qualifications = {cid: workspace.qualify(cid, view_id, device=audit["device"])
                      for cid in executable}
    return workspace, audit, candidates, tasks, qualifications


def test_full_refresh_receipts_gates_training_artifacts_and_shared_recipes(refreshed):
    workspace, audit, candidates, tasks, qualifications = refreshed
    for cid in audit["candidates"]:
        outcomes = {tid: outcome for tier in qualifications[cid]["tiers"].values()
                    for tid, outcome in tier["tasks"].items()}
        global_recipe = None
        for tid in audit["tasks"]:
            task = tasks[tid]
            entry = audit["runs"][f"{cid}/{tid}"]
            path = workspace.reports_dir / f"receipts/{audit['device']}/{cid}/{tid}.json"
            assert file_hash(path) == entry["receipt_sha256"], path
            receipt = json.loads(path.read_text())
            assert outcomes[tid]["status"] == receipt["verdict"] == entry["verdict"]
            assert receipt["verdict"] in {"PASS", "FAIL"}
            assert receipt["seed"] == audit["seed"] == 0
            assert receipt["error"] is None
            assert isinstance(receipt["all_finite"], bool)
            if receipt["verdict"] == "PASS":
                assert receipt["all_finite"] is True
            assert receipt["unintended_rng_deviations"] == 0
            assert receipt["completed_steps"] == task.steps == entry["completed_steps"]
            assert audit["started_at"] <= receipt["completed_at"] <= audit["completed_at"]
            assert len(receipt["observations"]) == 24
            assert [o["step"] for o in receipt["observations"]] == task.observation_steps()
            grade = grade_sustained_observations(task, receipt["observations"],
                completed_steps=task.steps, all_finite=receipt["all_finite"], unintended_rng_deviations=0)
            assert grade["status"] == receipt["verdict"]
            assert grade["terminal_passing"] == receipt["terminal_passing"]
            artifacts = receipt["artifacts"]
            assert artifacts["implementation"] == task.host_definition["implementation"]
            assert artifacts["particlegan_version"] == "0.9.7"
            assert artifacts["memory_policy"] == candidates[cid].memory_policy
            # Budgets belong to tasks. All optimizer/loss/prior choices belong
            # to one unchanged recipe across the complete task suite.
            recipe = {k: v for k, v in artifacts["resolved_recipe"].items() if k != "total_steps"}
            if global_recipe is None:
                global_recipe = recipe
            assert recipe == global_recipe, (cid, tid)
            sessions = [artifacts["online_execution"]]
            if "cyclegan" in artifacts:
                sessions.append(artifacts["cyclegan"]["reverse_online_execution"])
            for session in sessions:
                assert session["max_real_batch"] == session["max_fake_batch"] == 1
                assert session["updates"] == task.steps
                assert session["replayed_examples"] == 0
                assert session["gradient_accumulation_steps"] == 1
            if "transport" in artifacts:
                assert artifacts["transport"]["target_retained"] is False
            gif = workspace.reports_dir / artifacts["goal_gif"]
            assert file_hash(gif) == artifacts["gif_sha256"] == entry["gif_sha256"]
            with Image.open(gif) as animation:
                assert animation.n_frames == 24
            if "simulator_trace" in artifacts:
                trace = workspace.reports_dir / artifacts["simulator_trace"]
                assert file_hash(trace) == artifacts["simulator_trace_sha256"]
