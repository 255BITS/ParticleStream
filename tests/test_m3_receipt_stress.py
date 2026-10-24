"""Milestone M3 Empirical Stress & Verification Test Suite.

Audits fast-weight-bcap-ns qualification receipts, gate bounds, terminal window
calculations, GIF artifacts, and candidate qualification matrix.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from PIL import Image, ImageSequence

WORKSPACE_ROOT = Path("/nvme/dev/particlegan/ParticleStream")
RECEIPTS_DIR = WORKSPACE_ROOT / "reports/forge/receipts/cuda/fast-weight-bcap-ns"
GIFS_DIR = WORKSPACE_ROOT / "reports/forge/gifs"

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

TASK_GATE_BOUNDS = {
    "seq_img_round_robin_bars4": [
        ("hq", ">=", 0.90),
        ("modes", "==", 4.0),
        ("finite_template_tv", "<=", 0.15),
    ],
    "seq_img_dwell_bars4": [
        ("hq", ">=", 0.85),
        ("modes", "==", 4.0),
        ("finite_template_tv", "<=", 0.20),
    ],
    "seq_img_translating_bar_next": [
        ("next_frame_rmse", "<=", 0.12),
        ("direction_acc", ">=", 0.90),
    ],
    "seq_img_fork_rollout": [
        ("hq", ">=", 0.85),
        ("within_state_modes", "==", 2.0),
        ("branch_tv", "<=", 0.20),
    ],
}


def _eval_bound(val: float, op: str, target: float) -> bool:
    if op == ">=":
        return val >= target
    if op == "<=":
        return val <= target
    if op == "==":
        return math.isclose(val, target, rel_tol=1e-5, abs_tol=1e-5)
    raise ValueError(f"Unknown operator: {op}")


@pytest.mark.parametrize("task_id", [
    "seq_img_round_robin_bars4",
    "seq_img_dwell_bars4",
    "seq_img_translating_bar_next",
    "seq_img_fork_rollout",
])
def test_m3_receipt_schema_and_durability(task_id: str):
    """Verify that all 4 receipts exist, match schema, and satisfy invariant criteria."""
    receipt_path = RECEIPTS_DIR / f"{task_id}.json"
    assert receipt_path.exists(), f"Missing receipt: {receipt_path}"

    with open(receipt_path) as f:
        data = json.load(f)

    # Schema keys completeness
    missing = REQUIRED_TOP_LEVEL_KEYS - set(data.keys())
    assert not missing, f"Receipt missing keys: {missing}"

    # Invariant contracts
    assert data["candidate_id"] == "fast-weight-bcap-ns"
    assert data["task_id"] == task_id
    assert data["device"] == "cuda"
    assert data["seed"] == 0
    assert data["completed_steps"] == 480
    assert data["unintended_rng_deviations"] == 0
    assert data["all_finite"] is True
    assert isinstance(data["terminal_passing"], int)


@pytest.mark.parametrize("task_id", [
    "seq_img_round_robin_bars4",
    "seq_img_dwell_bars4",
    "seq_img_translating_bar_next",
    "seq_img_fork_rollout",
])
def test_m3_observation_cadence_and_finiteness(task_id: str):
    """Verify exactly 24 observations, 20-step cadence, and all finite metrics."""
    receipt_path = RECEIPTS_DIR / f"{task_id}.json"
    with open(receipt_path) as f:
        data = json.load(f)

    obs = data.get("observations", [])
    assert len(obs) == 24, f"Expected 24 observations, got {len(obs)}"

    for idx, o in enumerate(obs):
        expected_step = (idx + 1) * 20
        assert o["step"] == expected_step, f"Obs {idx} expected step {expected_step}, got {o['step']}"
        for metric_name, val in o["metrics"].items():
            assert isinstance(val, (int, float)), f"Metric {metric_name} must be numeric, got {type(val)}"
            assert math.isfinite(val), f"Metric {metric_name} is non-finite: {val}"


@pytest.mark.parametrize("task_id,expected_passes,expected_status", [
    ("seq_img_round_robin_bars4", 5, "PASS"),
    ("seq_img_dwell_bars4", 5, "PASS"),
    ("seq_img_translating_bar_next", 5, "PASS"),
    ("seq_img_fork_rollout", 5, "PASS"),
])
def test_m3_terminal_window_gate_evaluation(task_id: str, expected_passes: int, expected_status: str):
    """Independently re-evaluates gate bounds over terminal window (last 5 checks)."""
    receipt_path = RECEIPTS_DIR / f"{task_id}.json"
    with open(receipt_path) as f:
        data = json.load(f)

    bounds = TASK_GATE_BOUNDS[task_id]
    terminal_obs = data["observations"][-5:]
    assert len(terminal_obs) == 5

    recomputed_passes = 0
    for o in terminal_obs:
        passed = all(_eval_bound(o["metrics"][m], op, tgt) for m, op, tgt in bounds)
        if passed:
            recomputed_passes += 1

    assert recomputed_passes == expected_passes, (
        f"Recomputed terminal passes for {task_id} was {recomputed_passes}, expected {expected_passes}"
    )
    assert data["terminal_passing"] == expected_passes
    assert data["grade"]["terminal_passing"] == expected_passes
    assert data["grade"]["status"] == expected_status
    assert data["verdict"] == expected_status


def test_m3_candidate_qualification_matrix_summary():
    """Verify qualification matrix: candidate passes 4/4 tasks with 5/5 terminal passes."""
    passed_tasks = []
    failed_tasks = []

    for task_id in TASK_GATE_BOUNDS:
        receipt_path = RECEIPTS_DIR / f"{task_id}.json"
        with open(receipt_path) as f:
            data = json.load(f)
        if data["terminal_passing"] == 5 and data["grade"]["status"] == "PASS":
            passed_tasks.append(task_id)
        else:
            failed_tasks.append(task_id)

    assert set(passed_tasks) == {
        "seq_img_round_robin_bars4",
        "seq_img_dwell_bars4",
        "seq_img_translating_bar_next",
        "seq_img_fork_rollout",
    }
    assert len(passed_tasks) == 4
    assert len(failed_tasks) == 0


@pytest.mark.parametrize("task_id", [
    "seq_img_round_robin_bars4",
    "seq_img_dwell_bars4",
    "seq_img_translating_bar_next",
    "seq_img_fork_rollout",
])
def test_m3_actual_training_gifs_integrity(task_id: str):
    """Verify GIF artifacts exist, are non-empty, and contain exactly 24 frames."""
    gif_path = GIFS_DIR / f"fast-weight-bcap-ns_{task_id}.gif"
    assert gif_path.exists(), f"Missing GIF artifact: {gif_path}"
    assert gif_path.stat().st_size > 0, f"Empty GIF file: {gif_path}"

    with Image.open(gif_path) as im:
        assert im.format == "GIF"
        frames = [f.copy() for f in ImageSequence.Iterator(im)]
        assert len(frames) == 24, f"Expected 24 frames, got {len(frames)}"
        assert frames[0].size == (900, 300)


def test_m3_terminal_step480_pass_all_tasks():
    """Verify that all 4 tasks pass all gate bounds at terminal step 480."""
    for task_id in TASK_GATE_BOUNDS:
        receipt_path = RECEIPTS_DIR / f"{task_id}.json"
        with open(receipt_path) as f:
            data = json.load(f)

        assert data["terminal_passing"] == 5
        assert data["grade"]["status"] == "PASS"

        obs_480 = data["observations"][-1]
        assert obs_480["step"] == 480
        m = obs_480["metrics"]

        bounds = TASK_GATE_BOUNDS[task_id]
        for metric_name, op, tgt in bounds:
            assert _eval_bound(m[metric_name], op, tgt), (
                f"Task {task_id} failed bound {metric_name} {op} {tgt} at step 480 with value {m[metric_name]}"
            )


def test_all_cuda_receipt_hashes_match_disk():
    """Verify that every CUDA receipt on disk has a matching goal_gif whose SHA-256 equals receipt['artifacts']['gif_sha256']."""
    import hashlib
    receipts = list(WORKSPACE_ROOT.glob("reports/forge/receipts/cuda/*/*.json"))
    assert len(receipts) == 16, f"Expected 16 CUDA receipts, found {len(receipts)}"
    for p in sorted(receipts):
        data = json.loads(p.read_text())
        assert "artifacts" in data, f"Receipt {p} missing artifacts"
        assert "goal_gif" in data["artifacts"], f"Receipt {p} missing goal_gif"
        assert "gif_sha256" in data["artifacts"], f"Receipt {p} missing gif_sha256"
        gif_file = WORKSPACE_ROOT / "reports/forge" / data["artifacts"]["goal_gif"]
        assert gif_file.exists(), f"Missing artifact {gif_file} for receipt {p}"
        disk_sha = hashlib.sha256(gif_file.read_bytes()).hexdigest()
        assert disk_sha == data["artifacts"]["gif_sha256"], (
            f"Receipt hash mismatch in {p.parent.name}/{p.name}: recorded {data['artifacts']['gif_sha256']} "
            f"vs disk {disk_sha}"
        )

