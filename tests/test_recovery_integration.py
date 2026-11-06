"""The opt-in recovery mechanisms execute through the same public online path."""
import hashlib
import json
from pathlib import Path

import pytest
from particlegan_forge import CandidateSpec

from particlestream.tier1 import TIER1_TASKS, bootstrap_workspace


ROOT = Path(__file__).resolve().parents[1]


def test_pre_recovery_evidence_is_preserved_byte_for_byte():
    manifest = json.loads((ROOT / "reports/forge/cohorts/pre-recovery/manifest.json").read_text())
    assert len(manifest) == 256
    for relative, expected in manifest.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative


def test_recovery_preserves_task_architectures_budgets_and_numerical_gates():
    archived = json.loads((ROOT / "reports/forge/cohorts/pre-recovery/tasks.json").read_text())
    for task in TIER1_TASKS:
        current = task.to_dict()
        previous = archived[task.id]
        current["execution"]["host_definition"].pop("implementation")
        previous["execution"]["host_definition"].pop("implementation")
        assert current == previous


@pytest.mark.parametrize("task", TIER1_TASKS, ids=lambda task: task.id)
def test_public_recovery_execution_uses_causal_reader_and_single_image_transport(tmp_path, task):
    candidate = CandidateSpec.from_dict(json.loads(
        (ROOT / "configs/forge/ideas/online-bcap-transport-causal.json").read_text()))
    workspace = bootstrap_workspace(tmp_path)
    workspace.register_candidate(candidate)
    receipt = workspace.run_task(candidate.id, task.id, steps_override=4)
    assert receipt["error"] is None
    assert receipt["all_finite"]
    assert receipt["unintended_rng_deviations"] == 0
    assert receipt["grade"]["status"] == "INCOMPLETE"
    artifacts = receipt["artifacts"]
    assert artifacts["memory_policy"]["reader"] == "causal_state"
    assert {"memory.py", "transport.py"} <= artifacts["implementation"].keys()
    transport = artifacts["transport"]
    assert transport["assignment_updates"] == 4
    assert transport["probe_forwards"] == 4 * 32
    assert transport["target_retained"] is False
    assert sum(transport["assignment_mass"]) == 4
    assert transport["critic_sampling"] == "uniform"
    assert transport["manipulator"]["config"]["penalty_weight"] == 10
    execution = artifacts["online_execution"]
    assert execution["max_real_batch"] == execution["max_fake_batch"] == 1
    assert execution["replayed_examples"] == 0
    assert execution["gradient_accumulation_steps"] == 1
