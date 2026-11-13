"""Smoke tests for the standalone MuJoCo visual world-model demo."""
from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("gymnasium_robotics")


def test_mujoco_demo_strict_batch_one_smoke(tmp_path: Path):
    from demos.mujoco_pick_and_place_demo import CANVAS_H, CANVAS_W, run_demo

    out_gif = tmp_path / "demo_smoke.gif"
    summary = run_demo(
        output_gif=out_gif,
        output_mp4=None,
        recipe_id="online-gail-bcap-schedule-free",
        device="cpu",
        steps=16,
        fps=8,
        showcase_episodes=(0,),
        frame_stride=8,
    )

    assert summary["max_real_batch"] == 1
    assert summary["max_fake_batch"] == 1
    assert summary["replayed_examples"] == 0
    assert summary["rendered_frames"] >= 2
    assert out_gif.exists()

    with Image.open(out_gif) as im:
        assert im.size == (CANVAS_W, CANVAS_H)
        assert im.n_frames == summary["rendered_frames"]
