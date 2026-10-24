"""Milestone M4 Adversarial Challenger Verification Suite.

Validates:
1. Actual-training GIFs exist, are non-empty, have 24 frames, (900, 300) resolution, and uncorrupted buffers.
2. Bit-for-bit SHA-256 match between on-disk GIFs and receipts.
3. Integrity of all 16 CUDA qualification receipts (JSON schema, step count, RNG isolation, finiteness).
4. Single current leaderboard compliance.
5. Repository hygiene (untracked bulk execution logs).
"""

import glob
import hashlib
import json
import os
import subprocess
from PIL import Image
import pytest

STREAM_ROOT = "/nvme/dev/particlegan/ParticleStream"
GIF_DIR = os.path.join(STREAM_ROOT, "reports/forge/gifs")
RECEIPT_DIR = os.path.join(STREAM_ROOT, "reports/forge/receipts/cuda")

WINNING_CANDIDATE = "fast-weight-bcap-ns"
TIER1_TASKS = [
    "seq_img_round_robin_bars4",
    "seq_img_dwell_bars4",
    "seq_img_translating_bar_next",
    "seq_img_fork_rollout",
]
ALL_CANDIDATES = [
    "fast-weight-bcap-ns",
    "bcap",
    "fast-weight-bcap",
    "ka2",
]


@pytest.mark.parametrize("task_id", TIER1_TASKS)
def test_winning_candidate_gif_and_sha256_match(task_id: str):
    gif_path = os.path.join(GIF_DIR, f"{WINNING_CANDIDATE}_{task_id}.gif")
    receipt_path = os.path.join(RECEIPT_DIR, WINNING_CANDIDATE, f"{task_id}.json")

    assert os.path.isfile(gif_path), f"Missing GIF at {gif_path}"
    assert os.path.isfile(receipt_path), f"Missing receipt at {receipt_path}"

    # Verify non-zero size
    file_size = os.path.getsize(gif_path)
    assert file_size > 0, f"Empty GIF file: {gif_path}"

    # Calculate SHA256
    with open(gif_path, "rb") as f:
        computed_sha = hashlib.sha256(f.read()).hexdigest()

    with open(receipt_path, "r", encoding="utf-8") as f:
        receipt_data = json.load(f)

    receipt_sha = receipt_data.get("artifacts", {}).get("gif_sha256")
    assert receipt_sha is not None, f"Receipt missing artifacts.gif_sha256: {receipt_path}"
    assert computed_sha == receipt_sha, (
        f"Bit-for-bit SHA-256 mismatch for {task_id}: computed={computed_sha} != receipt={receipt_sha}"
    )

    # Verify GIF animation and frames
    im = Image.open(gif_path)
    frame_count = 0
    dimensions = []
    try:
        while True:
            frame_count += 1
            dimensions.append(im.size)
            # Ensure buffer is non-corrupt
            raw = im.tobytes()
            assert len(raw) > 0
            im.seek(im.tell() + 1)
    except EOFError:
        pass

    assert frame_count == 24, f"Expected 24 frames in {gif_path}, got {frame_count}"
    assert set(dimensions) == {(900, 300)}, f"Expected (900, 300) dimensions, got {set(dimensions)}"


def test_all_16_cuda_receipts_integrity():
    receipt_files = glob.glob(os.path.join(RECEIPT_DIR, "*", "*.json"))
    assert len(receipt_files) == 16, f"Expected exactly 16 CUDA receipts, got {len(receipt_files)}"

    for r_path in receipt_files:
        with open(r_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cand = data.get("candidate_id")
        task = data.get("task_id")
        seed = data.get("seed")
        device = data.get("device")
        steps = data.get("completed_steps")
        all_finite = data.get("all_finite")
        rng_dev = data.get("unintended_rng_deviations")
        terminal_passing = data.get("terminal_passing")
        status = data.get("grade", {}).get("status")

        assert cand in ALL_CANDIDATES, f"Unexpected candidate {cand} in {r_path}"
        assert task in TIER1_TASKS, f"Unexpected task {task} in {r_path}"
        assert seed == 0, f"Expected seed 0 in {r_path}, got {seed}"
        assert device == "cuda", f"Expected device 'cuda' in {r_path}, got {device}"
        assert steps == 480, f"Expected completed_steps 480 in {r_path}, got {steps}"
        assert all_finite is True, f"Expected all_finite True in {r_path}, got {all_finite}"
        assert rng_dev == 0, f"Expected 0 RNG deviations in {r_path}, got {rng_dev}"
        assert isinstance(terminal_passing, int) and 0 <= terminal_passing <= 5
        assert status in ("PASS", "FAIL"), f"Unexpected status {status} in {r_path}"

        if cand == WINNING_CANDIDATE:
            assert status == "PASS", f"Winning candidate {WINNING_CANDIDATE} must PASS {task}"
            assert terminal_passing == 5, f"Winning candidate must have 5/5 terminal passes, got {terminal_passing}"


def test_single_current_leaderboard_rule():
    leaderboard_files = [
        f for f in glob.glob(os.path.join(STREAM_ROOT, "reports/forge/**"), recursive=True)
        if os.path.isfile(f) and any(kw in os.path.basename(f).lower() for kw in ["leaderboard", "inventory", "technique"])
    ]
    expected = [os.path.join(STREAM_ROOT, "reports/forge/technique-inventory.md")]
    assert leaderboard_files == expected, f"Multiple or unexpected leaderboards found: {leaderboard_files}"


def test_repository_hygiene_untracked_runs():
    res = subprocess.run(
        ["git", "ls-files", "runs/"],
        cwd=STREAM_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.strip() == "", f"Found tracked files under runs/: {res.stdout.strip()}"
