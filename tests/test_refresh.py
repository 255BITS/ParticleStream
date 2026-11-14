"""Full-suite orchestration and safe resume, without launching any training."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from particlegan_forge import atomic_json, file_hash, stable_hash
from particlegan_forge.tiers import adapter_source_digest
from particlestream import refresh
from particlestream.forge import StreamWorkspace
from particlestream.suite import PRIMARY_VIEW, REGISTRY


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "online-bcap-adam"


@pytest.fixture
def mocked_workspace(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "configs/forge", tmp_path / "configs/forge")
    workspace = StreamWorkspace(tmp_path, registry=REGISTRY)
    calls, compile_calls, outcomes = [], [], {}

    def fake_controls(tasks):
        return {task.id: {"oracle": {"verdict": "PASS"},
                          "destructive": {"verdict": "FAIL"}} for task in tasks}

    def run_task(cid, tid, *, device, seed):
        assert seed == 0
        calls.append((cid, tid))
        task = workspace.load_tasks()[tid]
        candidate = workspace.load_candidates()[cid]
        binding = workspace._binding(task, candidate, device, seed,
            adapter_source_digest(workspace.registry.get_adapter(task.adapter)))
        goal_gif = f"gifs/{cid}_{tid}.gif"
        gif = workspace.reports_dir / goal_gif
        gif.parent.mkdir(parents=True, exist_ok=True)
        gif.write_bytes(b"mock training artifact, no training launched")
        verdict = outcomes.get((cid, tid), "FAIL")
        receipt = {
            "binding": binding, "cohort_key": stable_hash(binding), "verdict": verdict,
            "completed_at": refresh.now(), "completed_steps": task.steps,
            "error": "mock execution failure" if verdict in {"INVALID", "INCOMPLETE"} else None,
            "cost": {"wall_seconds": 0.001},
            "artifacts": {"goal_gif": goal_gif, "gif_sha256": file_hash(gif)},
            "grade": {"terminal_passing": 0, "required_terminal_checks": 5,
                      "failed_bounds": ["mock numerical gate"], "live": {}},
        }
        atomic_json(workspace._receipt_path(cid, tid, device=device), receipt)
        return receipt

    monkeypatch.setattr(refresh, "validate_controls", fake_controls)
    monkeypatch.setattr(workspace, "run_task", run_task)
    monkeypatch.setattr(workspace, "compile_reports", lambda **kwargs: compile_calls.append(kwargs))
    return workspace, calls, compile_calls, outcomes


def load_audit(workspace):
    return json.loads((workspace.reports_dir / "refresh-audit.json").read_text())


def test_default_refresh_measures_all_executable_recipes_and_20_tasks(mocked_workspace):
    workspace, calls, _, _ = mocked_workspace
    result = refresh.refresh_all(workspace, PRIMARY_VIEW.id, device="cpu")
    audit = load_audit(workspace)
    assert result["completed_runs"] == audit["planned_runs"] == 880
    assert len(audit["candidates"]) == 44
    assert len(audit["historical_read_only"]) == 14
    assert all(cid.startswith("online-") for cid in audit["candidates"])
    assert len(set(calls)) == len(calls) == 880
    assert set(audit["tasks"]) == set(workspace.load_tasks())
    assert audit["completed_at"] is not None
    assert audit["sessions"][0]["device_name"] == "cpu"
    assert audit["sessions"][0]["particlegan_version"] == "0.9.7"


def test_numerical_failures_measure_higher_tiers_without_changing_qualification(mocked_workspace):
    workspace, calls, _, _ = mocked_workspace
    before = workspace.load_views()[PRIMARY_VIEW.id].to_dict()
    result = refresh.refresh_all(workspace, PRIMARY_VIEW.id, device="cpu", candidates=[CANDIDATE])
    assert result["completed_runs"] == len(calls) == 20
    assert workspace.load_views()[PRIMARY_VIEW.id].to_dict() == before
    qualification = workspace.qualify(CANDIDATE, PRIMARY_VIEW.id, device="cpu")
    assert qualification["status"] == "FAIL"
    assert qualification["qualified_tier"] == 0
    assert qualification["eligible"] is False
    assert qualification["tiers"][2]["gated_by_lower_tier"] is True
    assert qualification["tiers"][3]["gated_by_lower_tier"] is True
    assert all(outcome["status"] == "FAIL" for tier in qualification["tiers"].values()
               for outcome in tier["tasks"].values())


def test_repeated_candidate_selection_runs_each_pair_once(mocked_workspace):
    workspace, calls, _, _ = mocked_workspace
    result = refresh.refresh_all(workspace, PRIMARY_VIEW.id, device="cpu",
                                 candidates=[CANDIDATE, CANDIDATE])
    assert result["completed_runs"] == len(calls) == 20
    assert load_audit(workspace)["planned_runs"] == 20


def test_explicit_historical_recipe_rejected_before_any_execution(mocked_workspace):
    workspace, calls, _, _ = mocked_workspace
    with pytest.raises(ValueError, match="historical replay candidates are read-only"):
        refresh.refresh_all(workspace, PRIMARY_VIEW.id, device="cpu", candidates=["stream-bcap"])
    assert not calls


@pytest.mark.parametrize("verdict", ["INVALID", "INCOMPLETE"])
def test_execution_failure_stops_without_claiming_completion(mocked_workspace, verdict):
    workspace, calls, compile_calls, outcomes = mocked_workspace
    view = workspace.load_views()[PRIMARY_VIEW.id]
    assignments = sorted(view.assignments, key=lambda a: (a.qualification_tier, a.order))
    outcomes[(CANDIDATE, assignments[2].task)] = verdict
    with pytest.raises(RuntimeError, match=f"{verdict}: mock execution failure"):
        refresh.refresh_all(workspace, PRIMARY_VIEW.id, device="cpu", candidates=[CANDIDATE])
    audit = load_audit(workspace)
    assert len(calls) == 3
    assert len(audit["runs"]) == 2
    assert audit["completed_at"] is None
    assert compile_calls


def test_resume_skips_only_unchanged_receipts_and_gifs_with_current_bindings(mocked_workspace):
    workspace, calls, _, _ = mocked_workspace
    refresh.refresh_all(workspace, PRIMARY_VIEW.id, device="cpu", candidates=[CANDIDATE])
    calls.clear()
    refresh.refresh_all(workspace, PRIMARY_VIEW.id, device="cpu", candidates=[CANDIDATE], resume=True)
    assert not calls
    audit = load_audit(workspace)
    assert len(audit["sessions"]) == 2
    task_ids = audit["tasks"]

    # Semantically identical receipt text still has a different artifact hash.
    receipt_path = workspace._receipt_path(CANDIDATE, task_ids[0], device="cpu")
    receipt_path.write_text(receipt_path.read_text() + "\n")
    # A changed training GIF must also be regenerated.
    gif = workspace.reports_dir / audit["runs"][f"{CANDIDATE}/{task_ids[1]}"]["goal_gif"]
    gif.write_bytes(b"changed artifact")
    # Even an audit whose hash agrees cannot make an obsolete adapter current.
    stale_path = workspace._receipt_path(CANDIDATE, task_ids[2], device="cpu")
    stale = json.loads(stale_path.read_text())
    stale["binding"]["adapter_source_sha256"] = "obsolete-adapter"
    stale["cohort_key"] = stable_hash(stale["binding"])
    atomic_json(stale_path, stale)
    audit["runs"][f"{CANDIDATE}/{task_ids[2]}"]["receipt_sha256"] = file_hash(stale_path)
    atomic_json(workspace.reports_dir / "refresh-audit.json", audit)

    refresh.refresh_all(workspace, PRIMARY_VIEW.id, device="cpu", candidates=[CANDIDATE], resume=True)
    assert calls == [(CANDIDATE, tid) for tid in task_ids[:3]]
    assert len(load_audit(workspace)["runs"]) == 20


def test_resume_clears_completed_timestamp_before_a_failed_rerun(mocked_workspace):
    workspace, calls, _, outcomes = mocked_workspace
    refresh.refresh_all(workspace, PRIMARY_VIEW.id, device="cpu", candidates=[CANDIDATE])
    audit = load_audit(workspace)
    task_id = audit["tasks"][0]
    receipt_path = workspace._receipt_path(CANDIDATE, task_id, device="cpu")
    receipt_path.write_text(receipt_path.read_text() + "\n")
    outcomes[(CANDIDATE, task_id)] = "INVALID"
    calls.clear()
    with pytest.raises(RuntimeError, match="INVALID"):
        refresh.refresh_all(workspace, PRIMARY_VIEW.id, device="cpu", candidates=[CANDIDATE], resume=True)
    assert calls == [(CANDIDATE, task_id)]
    audit = load_audit(workspace)
    assert audit["completed_at"] is None
    assert len(audit["runs"]) == 19


def test_resume_rejects_changed_declarations_before_any_execution(mocked_workspace):
    workspace, calls, _, _ = mocked_workspace
    refresh.refresh_all(workspace, PRIMARY_VIEW.id, device="cpu", candidates=[CANDIDATE])
    calls.clear()
    candidate = workspace.load_candidates()[CANDIDATE]
    candidate.recipe_overrides["lr"] = 0.0001
    workspace.register_candidate(candidate)
    with pytest.raises(ValueError, match="refresh declarations differ"):
        refresh.refresh_all(workspace, PRIMARY_VIEW.id, device="cpu", candidates=[CANDIDATE], resume=True)
    assert not calls
