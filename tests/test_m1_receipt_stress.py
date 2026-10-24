"""Empirical stress test harness for Milestone M1: receipt structure, ForgeWorkspace execution,
non-finite value handling, RNG stream isolation, failure modes, and backward compatibility.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch
import pytest
import torch

from particlegan_forge import (
    CandidateSpec,
    ForgeRegistry,
    ForgeWorkspace,
    MetricBound,
    NamedStreams,
    TaskSpec,
    TierView,
    render_training_gif,
)
import particlegan_forge.__main__ as forge_cli
from particlestream.tier1 import (
    CANDIDATES,
    PRIMARY_VIEW,
    REGISTRY,
    TIER1_TASKS,
    bootstrap_workspace,
    run_seq_image_task,
)


REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version",
    "candidate_id",
    "technique",
    "task_id",
    "device",
    "seed",
    "completed_steps",
    "all_finite",
    "unintended_rng_deviations",
    "terminal_passing",
    "task_fingerprint",
    "candidate_fingerprint",
    "rng_manifest",
    "verdict",
    "grade",
    "live",
    "observations",
    "artifacts",
    "cost",
    "timestamp",
    "completed_at",
}


def _assert_valid_receipt_contract(rec: dict, *, expected_task: str, expected_cand: str, expected_steps: int):
    # Top-level required keys existence
    missing = REQUIRED_TOP_LEVEL_KEYS - set(rec.keys())
    assert not missing, f"Receipt missing top-level keys: {missing}"

    # Verify types of new top-level keys
    assert isinstance(rec["completed_steps"], int), f"completed_steps must be int, got {type(rec['completed_steps'])}"
    assert isinstance(rec["all_finite"], bool), f"all_finite must be bool, got {type(rec['all_finite'])}"
    assert isinstance(rec["unintended_rng_deviations"], int), f"unintended_rng_deviations must be int, got {type(rec['unintended_rng_deviations'])}"
    assert isinstance(rec["terminal_passing"], int), f"terminal_passing must be int, got {type(rec['terminal_passing'])}"

    # Verify values match expectations
    assert rec["task_id"] == expected_task
    assert rec["candidate_id"] == expected_cand
    assert rec["completed_steps"] == expected_steps

    # Backward compatibility: verify existing nested keys remain intact
    assert isinstance(rec["cost"], dict)
    assert "wall_seconds" in rec["cost"]
    assert "completed_steps" in rec["cost"]
    assert rec["cost"]["completed_steps"] == rec["completed_steps"]
    assert isinstance(rec["cost"]["wall_seconds"], float)

    assert isinstance(rec["grade"], dict)
    assert "status" in rec["grade"]
    assert "terminal_passing" in rec["grade"]
    assert "failed_bounds" in rec["grade"]
    assert rec["grade"]["terminal_passing"] == rec["terminal_passing"]
    assert rec["verdict"] == rec["grade"]["status"]

    assert isinstance(rec["live"], dict)
    assert rec["live"] == rec["grade"].get("live", {})

    assert isinstance(rec["observations"], list)
    assert isinstance(rec["artifacts"], dict)
    assert isinstance(rec["rng_manifest"], dict)


# ---------------------------------------------------------------------------
# Test 1: Parameter configuration matrix on real ParticleStream tasks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("task_id", [
    "seq_img_round_robin_bars4",
    "seq_img_dwell_bars4",
    "seq_img_translating_bar_next",
    "seq_img_fork_rollout",
])
@pytest.mark.parametrize("candidate_id", ["fast-weight-bcap", "bcap", "ka2"])
def test_receipt_top_level_keys_across_tasks_and_candidates(tmp_path: Path, task_id: str, candidate_id: str):
    ws = bootstrap_workspace(tmp_path)
    steps_override = 4
    seed = 0
    device = "cpu"

    rec = ws.run_task(candidate_id, task_id, seed=seed, device=device, steps_override=steps_override)

    _assert_valid_receipt_contract(rec, expected_task=task_id, expected_cand=candidate_id, expected_steps=steps_override)
    assert rec["all_finite"] is True
    assert rec["unintended_rng_deviations"] == 0
    assert rec["seed"] == seed
    assert rec["device"] == device

    # Verify attempt receipt persisted to disk
    attempt_receipt_path = tmp_path / "runs/forge" / candidate_id / task_id / "receipt.json"
    assert attempt_receipt_path.is_file(), f"Attempt receipt missing at {attempt_receipt_path}"
    with attempt_receipt_path.open() as f:
        disk_rec = json.load(f)
    _assert_valid_receipt_contract(disk_rec, expected_task=task_id, expected_cand=candidate_id, expected_steps=steps_override)
    assert disk_rec["completed_steps"] == steps_override
    assert disk_rec["all_finite"] is True
    assert disk_rec["unintended_rng_deviations"] == 0


def test_receipt_durability_and_full_budget(tmp_path: Path):
    """Verify that full-budget execution writes durable receipts while steps_override < task.steps does not."""
    custom_reg = ForgeRegistry()

    @custom_reg.adapter("mock_short_task")
    def mock_adapter(*, task, candidate, streams, attempt_dir, workspace_root=None, device="cpu", steps_override=None, log_event=None):
        steps = int(steps_override or task.steps)
        obs = [{"step": i, "metrics": {"loss": 0.05, "hq": 1.0}} for i in range(1, steps + 1)]
        return {
            "observations": obs,
            "all_finite": True,
            "unintended_rng_deviations": 0,
            "completed_steps": steps,
            "artifacts": {"test_artifact": "ok.txt"},
        }

    ws = ForgeWorkspace(tmp_path, registry=custom_reg)
    task = TaskSpec(
        id="mock_t",
        description="Mock short task",
        adapter="mock_short_task",
        steps=10,
        observations=5,
        minimum_stable_checks=2,
        thresholds=[MetricBound(metric="hq", op=">=", bound=0.9)],
        host_definition={},
    )
    cand = CandidateSpec(id="mock_c", technique="test", changed_factors=["baseline"])
    ws.register_task(task)
    ws.register_candidate(cand)

    durable_path = tmp_path / "reports/forge/receipts/cpu/mock_c/mock_t.json"

    # Step override < steps: durable receipt NOT written
    rec_short = ws.run_task("mock_c", "mock_t", device="cpu", steps_override=5)
    assert rec_short["completed_steps"] == 5
    assert not durable_path.exists(), "Short run should not write durable report receipt"

    # Full budget (steps_override is None): durable receipt IS written
    rec_full = ws.run_task("mock_c", "mock_t", device="cpu")
    assert rec_full["completed_steps"] == 10
    assert durable_path.is_file(), "Full budget run must write durable report receipt"
    with durable_path.open() as f:
        disk_rec = json.load(f)
    _assert_valid_receipt_contract(disk_rec, expected_task="mock_t", expected_cand="mock_c", expected_steps=10)
    assert disk_rec["all_finite"] is True
    assert disk_rec["unintended_rng_deviations"] == 0
    assert disk_rec["terminal_passing"] == 2
    assert disk_rec["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# Test 2: Simulated Non-Finite Values (Clean propagation vs Serialization crash)
# ---------------------------------------------------------------------------
def test_simulated_non_finite_values_via_adapter_flag(tmp_path: Path):
    """When all_finite=False is set and metrics are finite numbers, verify receipt top-level all_finite=False and verdict=FAIL."""
    reg = ForgeRegistry()

    @reg.adapter("nan_adapter")
    def nan_adapter(*, task, candidate, streams, attempt_dir, workspace_root=None, device="cpu", steps_override=None, log_event=None):
        steps = int(steps_override or task.steps)
        # Metrics are numbers, but all_finite is flagged False
        obs = [{"step": i, "metrics": {"loss": 999.0}} for i in range(1, steps + 1)]
        return {
            "observations": obs,
            "all_finite": False,  # Simulated non-finite
            "unintended_rng_deviations": 0,
            "completed_steps": steps,
        }

    ws = ForgeWorkspace(tmp_path, registry=reg)
    task = TaskSpec(
        id="task_nan",
        description="Task with nonfinite simulation",
        adapter="nan_adapter",
        steps=10,
        observations=5,
        minimum_stable_checks=2,
        thresholds=[],
    )
    cand = CandidateSpec(id="cand_nan", technique="test", changed_factors=["baseline"])
    ws.register_task(task)
    ws.register_candidate(cand)

    rec = ws.run_task("cand_nan", "task_nan", device="cpu")

    _assert_valid_receipt_contract(rec, expected_task="task_nan", expected_cand="cand_nan", expected_steps=10)
    assert rec["all_finite"] is False
    assert rec["verdict"] == "FAIL"
    assert rec["grade"]["status"] == "FAIL"
    assert rec["grade"]["reason"] == "nonfinite parameters, gradients, or metrics observed"
    assert "all_finite == 1" in rec["grade"]["failed_bounds"]
    assert rec["terminal_passing"] == 0


def test_non_finite_metric_floats_trigger_atomic_json_crash(tmp_path: Path):
    """EMPIRICAL CHALLENGE FINDING (RESOLVED): When observation metrics contain literal float('nan'),
    json_safe converts NaN to None, allowing receipt to be written with all_finite=False and verdict=FAIL.
    """
    reg = ForgeRegistry()

    @reg.adapter("literal_nan_adapter")
    def literal_nan_adapter(*, task, candidate, streams, attempt_dir, workspace_root=None, device="cpu", steps_override=None, log_event=None):
        steps = int(steps_override or task.steps)
        obs = [{"step": i, "metrics": {"loss": float("nan")}} for i in range(1, steps + 1)]
        return {
            "observations": obs,
            "all_finite": False,
            "unintended_rng_deviations": 0,
            "completed_steps": steps,
        }

    ws = ForgeWorkspace(tmp_path, registry=reg)
    task = TaskSpec(
        id="task_literal_nan",
        description="Task producing literal NaN in metrics",
        adapter="literal_nan_adapter",
        steps=10,
        observations=5,
        minimum_stable_checks=2,
        thresholds=[],
    )
    cand = CandidateSpec(id="cand_literal_nan", technique="test", changed_factors=["baseline"])
    ws.register_task(task)
    ws.register_candidate(cand)

    rec = ws.run_task("cand_literal_nan", "task_literal_nan", device="cpu")
    assert rec["all_finite"] is False
    assert rec["verdict"] == "FAIL"
    assert rec["terminal_passing"] == 0
    assert rec["observations"][0]["metrics"]["loss"] is None

    # Disk receipt must parse cleanly as standard JSON with null metric
    disk_path = tmp_path / "runs/forge/cand_literal_nan/task_literal_nan/receipt.json"
    assert disk_path.is_file()
    disk_rec = json.loads(disk_path.read_text(encoding="utf-8"))
    assert disk_rec["all_finite"] is False
    assert disk_rec["verdict"] == "FAIL"
    assert disk_rec["observations"][0]["metrics"]["loss"] is None


def test_injected_nan_in_real_training_loop_reveals_nan_metric_crash(tmp_path: Path):
    """EMPIRICAL CHALLENGE FINDING (RESOLVED): Injecting NaN into model weights during real run_seq_image_task
    causes score_multimodal_bars to produce mean_rmse=nan, which is sanitized to None without crashing.
    """
    ws = bootstrap_workspace(tmp_path)

    from particlegan.training import GANTrainer
    original_step = GANTrainer.step

    def corrupting_step(self, *args, **kwargs):
        stats = original_step(self, *args, **kwargs)
        # Inject NaN into discriminator parameter
        for p in self.D.parameters():
            p.data.fill_(float("nan"))
            break
        return stats

    with patch.object(GANTrainer, "step", corrupting_step):
        rec = ws.run_task("fast-weight-bcap", "seq_img_round_robin_bars4", device="cpu", steps_override=4)
        assert rec["all_finite"] is False
        assert rec["verdict"] == "FAIL"
        assert rec["terminal_passing"] == 0
        assert any(o["metrics"].get("mean_rmse") is None for o in rec["observations"])
        assert rec["observations"][-1]["metrics"]["mean_rmse"] is None

        disk_path = tmp_path / "runs/forge/fast-weight-bcap/seq_img_round_robin_bars4/receipt.json"
        assert disk_path.is_file()
        disk_rec = json.loads(disk_path.read_text(encoding="utf-8"))
        assert disk_rec["all_finite"] is False
        assert disk_rec["verdict"] == "FAIL"
        assert any(o["metrics"].get("mean_rmse") is None for o in disk_rec["observations"])
        assert disk_rec["observations"][-1]["metrics"]["mean_rmse"] is None


def test_render_training_gif_handles_non_finite_and_none_metrics(tmp_path: Path):
    """Milestone M1 Iteration 3 unit test: render_training_gif safely handles all-NaN,
    all-Inf, None, and mixed non-finite metrics without raising ValueError or TypeError.
    """
    # 1. All-NaN metric values
    gif_nan = tmp_path / "nan.gif"
    frames_nan = [
        {"step": i, "passed": False, "metrics": {"loss": float("nan")}}
        for i in range(1, 4)
    ]
    res_nan = render_training_gif(
        frames_nan, gif_nan, task_id="task_nan", candidate_id="cand_nan", verdict="FAIL", primary_metric="loss"
    )
    assert Path(res_nan["path"]).is_file()
    assert res_nan["frames"] == 3
    assert res_nan["sha256"] is not None
    assert gif_nan.stat().st_size > 0

    # 2. All-Inf metric values (including positive and negative infinity)
    gif_inf = tmp_path / "inf.gif"
    frames_inf = [
        {"step": 1, "passed": False, "metrics": {"loss": float("inf")}},
        {"step": 2, "passed": False, "metrics": {"loss": float("-inf")}},
        {"step": 3, "passed": False, "metrics": {"loss": float("inf")}},
    ]
    res_inf = render_training_gif(
        frames_inf, gif_inf, task_id="task_inf", candidate_id="cand_inf", verdict="FAIL", primary_metric="loss"
    )
    assert Path(res_inf["path"]).is_file()
    assert res_inf["frames"] == 3
    assert res_inf["sha256"] is not None
    assert gif_inf.stat().st_size > 0

    # 3. None, non-numeric strings, and omitted metric values
    gif_none = tmp_path / "none.gif"
    frames_none = [
        {"step": 1, "passed": False, "metrics": {"loss": None}},
        {"step": 2, "passed": False, "metrics": {"loss": "corrupted_str"}},
        {"step": 3, "passed": False, "metrics": {}},
    ]
    res_none = render_training_gif(
        frames_none, gif_none, task_id="task_none", candidate_id="cand_none", verdict="FAIL", primary_metric="loss"
    )
    assert Path(res_none["path"]).is_file()
    assert res_none["frames"] == 3
    assert res_none["sha256"] is not None
    assert gif_none.stat().st_size > 0

    # 4. Mixed finite and non-finite values
    gif_mixed = tmp_path / "mixed.gif"
    frames_mixed = [
        {"step": 1, "passed": True, "metrics": {"loss": 0.42}},
        {"step": 2, "passed": False, "metrics": {"loss": float("nan")}},
        {"step": 3, "passed": False, "metrics": {"loss": None}},
        {"step": 4, "passed": True, "metrics": {"loss": 0.15}},
    ]
    res_mixed = render_training_gif(
        frames_mixed, gif_mixed, task_id="task_mixed", candidate_id="cand_mixed", verdict="FAIL", primary_metric="loss"
    )
    assert Path(res_mixed["path"]).is_file()
    assert res_mixed["frames"] == 4
    assert res_mixed["sha256"] is not None
    assert gif_mixed.stat().st_size > 0


def test_diverging_generator_in_real_training_loop_seq_img_translating_bar_next(tmp_path: Path):
    """Milestone M1 Iteration 3 regression test: Injecting NaN into Generator weights during real
    run_seq_image_task on seq_img_translating_bar_next causes next_frame_rmse to become NaN.
    Verifies that ws.run_task completes cleanly without exception, renders the GIF artifact,
    and writes receipt.json with all_finite=False, verdict='FAIL', and status='FAIL'.
    """
    ws = bootstrap_workspace(tmp_path)

    from particlegan.training import GANTrainer
    original_step = GANTrainer.step

    def corrupt_g(self, *args, **kwargs):
        stats = original_step(self, *args, **kwargs)
        for p in self.G.parameters():
            p.data.fill_(float("nan"))
            break
        return stats

    with patch.object(GANTrainer, "step", corrupt_g):
        rec = ws.run_task("fast-weight-bcap", "seq_img_translating_bar_next", device="cpu", steps_override=4)

    assert rec["all_finite"] is False
    assert rec["verdict"] == "FAIL"
    assert rec["grade"]["status"] == "FAIL"
    assert rec["terminal_passing"] == 0

    # Verify attempt receipt persisted to disk
    receipt_path = tmp_path / "runs/forge/fast-weight-bcap/seq_img_translating_bar_next/receipt.json"
    assert receipt_path.is_file(), f"Receipt missing at {receipt_path}"
    disk_rec = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert disk_rec["all_finite"] is False
    assert disk_rec["verdict"] == "FAIL"
    assert disk_rec["grade"]["status"] == "FAIL"
    assert disk_rec["terminal_passing"] == 0

    # Verify training GIF artifact exists and is non-empty
    assert "goal_gif" in disk_rec["artifacts"]
    gif_path = tmp_path / "reports/forge" / disk_rec["artifacts"]["goal_gif"]
    assert gif_path.is_file(), f"GIF artifact missing at {gif_path}"
    assert gif_path.stat().st_size > 0
    assert disk_rec["artifacts"]["gif_sha256"] is not None


# ---------------------------------------------------------------------------
# Test 3: Simulated RNG Deviations
# ---------------------------------------------------------------------------
def test_simulated_rng_deviation_via_adapter_flag(tmp_path: Path):
    """Adapter returns unintended_rng_deviations > 0 -> receipt has top-level unintended_rng_deviations > 0, verdict=INVALID."""
    reg = ForgeRegistry()

    @reg.adapter("rng_leak_adapter")
    def rng_leak_adapter(*, task, candidate, streams, attempt_dir, workspace_root=None, device="cpu", steps_override=None, log_event=None):
        steps = int(steps_override or task.steps)
        obs = [{"step": i, "metrics": {"loss": 0.01}} for i in range(1, steps + 1)]
        return {
            "observations": obs,
            "all_finite": True,
            "unintended_rng_deviations": 5,  # Simulated 5 mutated streams
            "completed_steps": steps,
        }

    ws = ForgeWorkspace(tmp_path, registry=reg)
    task = TaskSpec(
        id="task_rng",
        description="Task with RNG leakage simulation",
        adapter="rng_leak_adapter",
        steps=10,
        observations=5,
        minimum_stable_checks=2,
        thresholds=[],
    )
    cand = CandidateSpec(id="cand_rng", technique="test", changed_factors=["baseline"])
    ws.register_task(task)
    ws.register_candidate(cand)

    rec = ws.run_task("cand_rng", "task_rng", device="cpu")

    _assert_valid_receipt_contract(rec, expected_task="task_rng", expected_cand="cand_rng", expected_steps=10)
    assert rec["unintended_rng_deviations"] == 5
    assert rec["verdict"] == "INVALID"
    assert rec["grade"]["status"] == "INVALID"
    assert "evaluation mutated 5 non-eval RNG stream(s)" in rec["grade"]["reason"]
    assert "unintended_rng_deviations == 0" in rec["grade"]["failed_bounds"]
    assert rec["terminal_passing"] == 0


def test_empirical_rng_leakage_during_evaluation(tmp_path: Path):
    """Empirically advance a non-eval stream ('data') inside streams.isolated_evaluation() during run_seq_image_task."""
    ws = bootstrap_workspace(tmp_path)

    from particlegan_forge.rng import NamedStreams
    original_isolated_eval = NamedStreams.isolated_evaluation

    from contextlib import contextmanager
    @contextmanager
    def leaky_isolated_evaluation(self):
        with original_isolated_eval(self):
            # Illegally advance a non-eval stream ('data')
            data_gen = self.generator("data", component="stream", purpose="training", device=self.device)
            torch.randn(5, generator=data_gen)
            yield self

    with patch.object(NamedStreams, "isolated_evaluation", leaky_isolated_evaluation):
        rec = ws.run_task("fast-weight-bcap", "seq_img_round_robin_bars4", device="cpu", steps_override=4)

    assert rec["unintended_rng_deviations"] > 0
    assert rec["verdict"] == "INVALID"
    assert rec["grade"]["status"] == "INVALID"
    assert rec["terminal_passing"] == 0
    assert "unintended_rng_deviations == 0" in rec["grade"]["failed_bounds"]


# ---------------------------------------------------------------------------
# Test 4: Simultaneous Non-Finite and RNG Deviation
# ---------------------------------------------------------------------------
def test_simultaneous_non_finite_and_rng_deviation(tmp_path: Path):
    reg = ForgeRegistry()

    @reg.adapter("compound_failure")
    def compound_failure(*, task, candidate, streams, attempt_dir, workspace_root=None, device="cpu", steps_override=None, log_event=None):
        steps = int(steps_override or task.steps)
        obs = [{"step": i, "metrics": {"loss": 1.0}} for i in range(1, steps + 1)]
        return {
            "observations": obs,
            "all_finite": False,
            "unintended_rng_deviations": 2,
            "completed_steps": steps,
        }

    ws = ForgeWorkspace(tmp_path, registry=reg)
    task = TaskSpec(
        id="task_compound",
        description="Task with compound failure simulation",
        adapter="compound_failure",
        steps=10,
        observations=5,
        minimum_stable_checks=2,
        thresholds=[],
    )
    cand = CandidateSpec(id="cand_compound", technique="test", changed_factors=["baseline"])
    ws.register_task(task)
    ws.register_candidate(cand)

    rec = ws.run_task("cand_compound", "task_compound", device="cpu")

    _assert_valid_receipt_contract(rec, expected_task="task_compound", expected_cand="cand_compound", expected_steps=10)
    assert rec["all_finite"] is False
    assert rec["unintended_rng_deviations"] == 2
    # RNG deviation is priority check in grade_sustained_observations
    assert rec["verdict"] == "INVALID"
    assert rec["grade"]["status"] == "INVALID"
    assert rec["terminal_passing"] == 0


# ---------------------------------------------------------------------------
# Test 5: Type Coercion & Missing Field Graceful Defaults
# ---------------------------------------------------------------------------
def test_type_coercion_and_missing_field_defaults(tmp_path: Path):
    reg = ForgeRegistry()

    @reg.adapter("minimal_adapter")
    def minimal_adapter(*, task, candidate, streams, attempt_dir, workspace_root=None, device="cpu", steps_override=None, log_event=None):
        # Omits all_finite, unintended_rng_deviations, completed_steps
        return {
            "observations": [{"step": 1, "metrics": {"score": 1.0}}],
        }

    ws = ForgeWorkspace(tmp_path, registry=reg)
    task = TaskSpec(
        id="task_min",
        description="Task with minimal adapter return",
        adapter="minimal_adapter",
        steps=10,
        observations=5,
        minimum_stable_checks=2,
        thresholds=[],
    )
    cand = CandidateSpec(id="cand_min", technique="test", changed_factors=["baseline"])
    ws.register_task(task)
    ws.register_candidate(cand)

    rec = ws.run_task("cand_min", "task_min", device="cpu")

    assert rec["all_finite"] is True
    assert rec["unintended_rng_deviations"] == 0
    assert rec["completed_steps"] == 10
    assert isinstance(rec["all_finite"], bool)
    assert isinstance(rec["unintended_rng_deviations"], int)
    assert isinstance(rec["completed_steps"], int)


# ---------------------------------------------------------------------------
# Test 6: Downstream Workspace Board & Report Compilation Integration
# ---------------------------------------------------------------------------
def test_downstream_board_and_compile_compatibility(tmp_path: Path, capsys):
    ws = bootstrap_workspace(tmp_path)
    # Run a short task
    ws.run_task("fast-weight-bcap", "seq_img_round_robin_bars4", device="cpu", steps_override=4)

    # Qualify method on ForgeWorkspace
    q = ws.qualify("fast-weight-bcap", "single_stream_stability", device="cpu")
    assert "candidate_id" in q
    assert q["candidate_id"] == "fast-weight-bcap"
    assert "view_id" in q
    assert q["view_id"] == "single_stream_stability"
    assert "tiers" in q

    # Compile reports should function cleanly
    reports = ws.compile_reports(view_id="single_stream_stability")
    assert "leaderboard" in reports
    assert Path(reports["leaderboard"]).is_file()

    # CLI board command execution
    ret = forge_cli.main(["--root", str(tmp_path), "--provider", "particlestream.tier1", "board", "--view", "single_stream_stability"])
    assert ret == 0
    captured = capsys.readouterr()
    board_data = json.loads(captured.out)
    assert "fast-weight-bcap" in board_data
    assert "tiers" in board_data["fast-weight-bcap"]


def test_cli_error_handling_and_retier_safety(tmp_path: Path, capsys):
    ws = bootstrap_workspace(tmp_path)

    # 1. Unknown view in board command exits with error code 2 and prints to stderr without traceback
    ret_board = forge_cli.main(["--root", str(tmp_path), "--provider", "particlestream.tier1", "board", "--view", "non_existent_view"])
    assert ret_board == 2
    captured_board = capsys.readouterr()
    assert "Error:" in captured_board.err
    assert "Traceback (most recent call last):" not in captured_board.err

    # 2. Unknown task in retier exits with code 2 and does not create archive
    ret_task = forge_cli.main([
        "--root", str(tmp_path), "--provider", "particlestream.tier1",
        "retier", "single_stream_stability", "--task", "nonexistent_task", "--tier", "2", "--reason", "test"
    ])
    assert ret_task == 2
    captured_task = capsys.readouterr()
    assert "Error:" in captured_task.err
    assert "Traceback (most recent call last):" not in captured_task.err
    archive_dir = tmp_path / "configs/forge/view-history"
    assert not archive_dir.exists()

    # 3. Invalid tier (0) in retier exits with code 2 and does not create archive
    ret_tier = forge_cli.main([
        "--root", str(tmp_path), "--provider", "particlestream.tier1",
        "retier", "single_stream_stability", "--task", "seq_img_round_robin_bars4", "--tier", "0", "--reason", "test"
    ])
    assert ret_tier == 2
    captured_tier = capsys.readouterr()
    assert "Error:" in captured_tier.err
    assert "Traceback (most recent call last):" not in captured_tier.err
    assert not archive_dir.exists()

