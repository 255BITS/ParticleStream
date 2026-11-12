"""A recorded numeric success must also qualify against the current full view."""
import hashlib
import json
from pathlib import Path

from particlegan_forge import NamedStreams, grade_sustained_observations
from particlegan_forge.tiers import adapter_source_digest
from particlestream.forge import StreamWorkspace
from particlestream.tier1 import IMPLEMENTATION, run_seq_image_task
from particlestream.suite import MINIMAL_TASKS, REGISTRY, ROBOTICS_TASKS


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "online-feature-prior-bce"


def test_feature_recipe_qualifies_all_ten_current_required_tasks():
    workspace = StreamWorkspace(ROOT, registry=REGISTRY)
    workspace.validate()
    view = workspace.load_views()["single_stream_stability"]
    assert view.revision >= 7
    required = {assignment.task for assignment in view.assignments if assignment.importance == "required"}
    assert len(required) == 10
    stretch = {a.task for a in view.assignments if a.importance == "stretch"}
    assert stretch == {task.id for task in [*MINIMAL_TASKS, *ROBOTICS_TASKS]}
    assert len(stretch) == 10
    result = workspace.qualify(CANDIDATE, view.id, device="cuda")
    assert result["status"] == "PASS" and result["eligible"] is True
    assert result["adapter_source_verified"] is True
    expected_source = adapter_source_digest(run_seq_image_task)
    shared = []
    for task_id, task in workspace.load_tasks().items():
        if task_id not in required:
            continue
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
        if task_id == "seq_img_circle_orbit":
            readme_gif = ROOT / "docs/media/readme/circle-orbit.gif"
            assert readme_gif.is_file()
            assert hashlib.sha256(readme_gif.read_bytes()).hexdigest() == receipt["artifacts"]["orbit_gif_sha256"]
        if task_id == "seq_img_square_circle_cycle":
            readme_cycle_gif = ROOT / "docs/media/readme/square-circle-cycle.gif"
            assert readme_cycle_gif.is_file()
            assert hashlib.sha256(readme_cycle_gif.read_bytes()).hexdigest() == receipt["artifacts"]["cycle_gif_sha256"]
    assert all(configuration == shared[0] for configuration in shared)


def test_gail_bcap_recipes_qualify_all_ten_current_required_tasks():
    workspace = StreamWorkspace(ROOT, registry=REGISTRY)
    workspace.validate()
    view = workspace.load_views()["single_stream_stability"]
    required = {assignment.task for assignment in view.assignments if assignment.importance == "required"}
    expected_source = adapter_source_digest(run_seq_image_task)
    for candidate_id in (
        "online-gail-bcap-hinge",
        "online-gail-bcap-bce",
        "online-gail-bcap-momentum",
        "online-gail-bcap-spike-slab",
        "online-gail-bcap-snr",
    ):
        result = workspace.qualify(candidate_id, view.id, device="cuda")
        assert result["status"] == "PASS" and result["eligible"] is True, candidate_id
        assert result["adapter_source_verified"] is True
        shared = []
        for task_id, task in workspace.load_tasks().items():
            if task_id not in required:
                continue
            receipt = json.loads((workspace.reports_dir / "receipts/cuda" / candidate_id / f"{task_id}.json").read_text())
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
            transport = receipt["artifacts"]["transport"]
            assert transport["target_retained"] is False and transport["observation_state_retained"] is False
            assert transport["manipulator"]["class"] == "particlestream.gail_prior.GAILConditionalReconstruction"
            assert transport["manipulator"]["config"]["encoder_seed"] == NamedStreams(seed=0).seed_for(
                "init", component="feature_encoder", purpose="projection")
            shared.append(tuple(receipt["artifacts"][key] for key in ("resolved_recipe", "memory_policy"))
                          + (transport["manipulator"],))
            gif = workspace.reports_dir / receipt["artifacts"]["goal_gif"]
            assert hashlib.sha256(gif.read_bytes()).hexdigest() == receipt["artifacts"]["gif_sha256"]
        assert all(configuration == shared[0] for configuration in shared)

    snr_result = workspace.qualify("online-gail-bcap-snr", view.id, device="cuda")
    stretch_outcomes = {
        tid: outcome
        for tier in snr_result["tiers"].values()
        for tid, outcome in tier["tasks"].items()
        if outcome["importance"] == "stretch"
    }
    for stretch_pass_id in (
        "seq_minimal_tokens4",
        "seq_minimal_addition2bit",
        "seq_robotics_fetch_reach",
    ):
        assert stretch_outcomes[stretch_pass_id]["status"] == "PASS", stretch_pass_id
    assert stretch_outcomes["seq_robotics_fetch_pick_and_place"]["terminal_passing"] >= 4
