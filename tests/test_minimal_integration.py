"""Exercise the real public trainer and Forge adapter in a software-only cohort."""
import json
from pathlib import Path

import pytest
from particlegan_forge import CandidateSpec

from particlestream.minimal_adapter import validate_controls
from particlestream.suite import MINIMAL_TASKS, PRIMARY_VIEW, bootstrap_workspace

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "online-feature-prior-bce"


def test_core_budget_roles_and_production_scorer_controls():
    assert [task.steps for task in MINIMAL_TASKS] == [96, 96, 192, 192, 384]
    assert sum(task.steps for task in MINIMAL_TASKS) == 960
    assignments = {a.task: a for a in PRIMARY_VIEW.assignments}
    assert sum(a.importance == "required" for a in assignments.values()) == 10
    for task in MINIMAL_TASKS:
        assert assignments[task.id].importance == "stretch"
        assert assignments[task.id].qualification_tier == task.host_definition["qualification_tier"]
    controls = validate_controls(MINIMAL_TASKS)
    for outcomes in controls.values():
        assert outcomes["oracle"]["verdict"] == "PASS"
        assert all(result["verdict"] == "FAIL" for name, result in outcomes.items() if name != "oracle")


@pytest.mark.parametrize("task", MINIMAL_TASKS, ids=lambda task: task.id)
def test_real_adapter_short_cohort_is_causal_finite_and_not_qualification(tmp_path, task):
    workspace = bootstrap_workspace(tmp_path)
    workspace.register_candidate(CandidateSpec.from_dict(json.loads(
        (ROOT / f"configs/forge/ideas/{CANDIDATE}.json").read_text())))
    receipt = workspace.run_task(CANDIDATE, task.id, device="cpu", steps_override=3)
    assert receipt["verdict"] == "INCOMPLETE", receipt["error"]
    assert receipt["all_finite"] and receipt["unintended_rng_deviations"] == 0
    execution = receipt["artifacts"]["online_execution"]
    assert execution["updates"] == execution["live_predictions"] == 3
    assert execution["max_real_batch"] == execution["max_fake_batch"] == 1
    assert execution["replayed_examples"] == 0
    mode = task.host_definition["mode"]
    assert execution["arrivals"] == (3 if mode in {"dense4", "tokens4"} else 6)
    assert receipt["artifacts"]["compute"]["optimizer_updates"] == 3
    assert receipt["artifacts"]["compute"]["evaluation_g_forwards"] > 0
    assert receipt["artifacts"]["transport"]["target_retained"] is False
    assert (workspace.queue_root / CANDIDATE / task.id / "goal.gif").is_file()
    assert not (workspace.reports_dir / "receipts/cpu" / CANDIDATE / f"{task.id}.json").exists()
