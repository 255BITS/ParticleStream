"""A recorded numeric success must also qualify against the current full view."""
import hashlib
import json
from pathlib import Path

from particlegan_forge import NamedStreams, grade_sustained_observations
from particlegan_forge.tiers import adapter_source_digest
from particlestream.forge import StreamWorkspace
from particlestream.tier1 import IMPLEMENTATION, REGISTRY, run_seq_image_task


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "online-feature-prior-bce"


def test_feature_recipe_qualifies_all_eight_current_required_tasks():
    workspace = StreamWorkspace(ROOT, registry=REGISTRY)
    workspace.validate()
    view = workspace.load_views()["single_stream_stability"]
    assert view.revision >= 6
    assert len(view.assignments) == 9
    assert all(assignment.importance == "required" for assignment in view.assignments)
    result = workspace.qualify(CANDIDATE, view.id, device="cuda")
    assert result["status"] == "PASS" and result["eligible"] is True
    assert result["adapter_source_verified"] is True
    expected_source = adapter_source_digest(run_seq_image_task)
    shared = []
    for task_id, task in workspace.load_tasks().items():
        receipt = json.loads((workspace.reports_dir / "receipts/cuda" / CANDIDATE / f"{task_id}.json").read_text())
        assert receipt["binding"]["adapter_source_sha256"] == expected_source
        assert receipt["artifacts"]["implementation"] == IMPLEMENTATION
        assert receipt["artifacts"]["particlegan_version"] == "0.9.7"
        assert receipt["seed"] == 0 and receipt["completed_steps"] == 480
        assert receipt["all_finite"] is True and receipt["unintended_rng_deviations"] == 0
        grade = grade_sustained_observations(
            task, receipt["observations"], all_finite=True,
            unintended_rng_deviations=0, completed_steps=480,
        )
        assert grade["status"] == receipt["verdict"] == "PASS"
        assert grade["terminal_passing"] == receipt["terminal_passing"] == 5
        execution = receipt["artifacts"]["online_execution"]
        assert execution["max_real_batch"] == execution["max_fake_batch"] == 1
        assert execution["replayed_examples"] == 0 and execution["gradient_accumulation_steps"] == 1
        assert execution["updates"] == execution["live_predictions"] == 480
        assert execution["arrivals"] == 480 + task.host_definition["online_contract"]["warmup_arrivals"]
        transport = receipt["artifacts"]["transport"]
        assert transport["target_retained"] is False and transport["observation_state_retained"] is False
        assert transport["manipulator"]["config"]["encoder_seed"] == NamedStreams(seed=0).seed_for(
            "init", component="feature_encoder", purpose="projection")
        shared.append(tuple(receipt["artifacts"][key] for key in ("resolved_recipe", "memory_policy"))
                      + (transport["manipulator"],))
        gif = workspace.reports_dir / receipt["artifacts"]["goal_gif"]
        assert hashlib.sha256(gif.read_bytes()).hexdigest() == receipt["artifacts"]["gif_sha256"]
    assert all(configuration == shared[0] for configuration in shared)
