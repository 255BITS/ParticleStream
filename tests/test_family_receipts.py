"""Verify current qualification evidence independently of historical snapshots."""
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from particlegan import Recipe
from particlegan_forge import grade_sustained_observations
from particlestream.forge import StreamWorkspace
from particlestream.tier1 import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVES = {
    "stream-bispf-early": "bispf",
    "stream-bcap-adam": "bcap",
    "stream-ka2-ready": "ka2",
}


@pytest.mark.parametrize("candidate_id,family", REPRESENTATIVES.items())
def test_qualified_family_has_current_complete_evidence(candidate_id, family):
    workspace = StreamWorkspace(ROOT, registry=REGISTRY)
    qualification = workspace.qualify(candidate_id, "single_stream_stability", device="cuda")
    assert qualification["status"] == "PASS"
    assert qualification["eligible"] is True
    assert qualification["adapter_source_verified"] is True
    tasks = workspace.load_tasks()
    recipes = []
    for task in tasks.values():
        path = workspace.reports_dir / "receipts/cuda" / candidate_id / f"{task.id}.json"
        receipt = json.loads(path.read_text())
        grade = grade_sustained_observations(
            task, receipt["observations"], all_finite=receipt["all_finite"],
            unintended_rng_deviations=receipt["unintended_rng_deviations"],
            completed_steps=receipt["completed_steps"],
        )
        assert grade["status"] == "PASS"
        assert grade["terminal_passing"] == 5
        assert receipt["seed"] == 0
        assert receipt["completed_steps"] == task.steps == 480
        assert [row["step"] for row in receipt["observations"]] == task.observation_steps()
        assert receipt["artifacts"]["particlegan_version"] == "0.9.6"
        recipe = receipt["artifacts"]["resolved_recipe"]
        assert Recipe.from_dict(recipe).effective_critic_formulation == family
        recipes.append(recipe)
        gif = workspace.reports_dir / receipt["artifacts"]["goal_gif"]
        assert hashlib.sha256(gif.read_bytes()).hexdigest() == receipt["artifacts"]["gif_sha256"]
        with Image.open(gif) as animation:
            assert animation.n_frames == 24
            assert animation.size == (900, 300)
    assert all(recipe == recipes[0] for recipe in recipes), "per-task recipe selection"


def test_all_current_trials_have_complete_numerically_consistent_receipts():
    workspace = StreamWorkspace(ROOT, registry=REGISTRY)
    tasks = workspace.load_tasks()
    for candidate_id in workspace.load_candidates():
        if not candidate_id.startswith("stream-"):
            continue
        qualification = workspace.qualify(candidate_id, "single_stream_stability", device="cuda")
        outcomes = qualification["tiers"][1]["tasks"]
        for task in tasks.values():
            path = workspace.reports_dir / "receipts/cuda" / candidate_id / f"{task.id}.json"
            receipt = json.loads(path.read_text())
            assert receipt["all_finite"] is True
            assert receipt["unintended_rng_deviations"] == 0
            assert len(receipt["observations"]) == 24
            assert receipt["completed_steps"] == 480
            grade = grade_sustained_observations(
                task, receipt["observations"], all_finite=True,
                unintended_rng_deviations=0, completed_steps=480,
            )
            assert grade["status"] == receipt["verdict"] == outcomes[task.id]["status"]
            assert grade["terminal_passing"] == receipt["terminal_passing"]
