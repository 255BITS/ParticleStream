"""Audit current evidence without requiring an experiment to succeed."""
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
    "online-bispf-early": "bispf",
    "online-bcap-adam": "bcap",
    "online-ka2-ready": "ka2",
}


def test_pre_online_receipts_and_gifs_remain_byte_identical():
    manifest = json.loads((ROOT / "tests/fixtures/pre_online_artifacts.json").read_text())
    assert len(manifest) == 112
    for relative, digest in manifest.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative


@pytest.mark.parametrize("historical_id", [name.replace("online-", "stream-", 1) for name in REPRESENTATIVES])
def test_historical_winners_cannot_qualify_against_online_protocol(historical_id):
    qualification = StreamWorkspace(ROOT, registry=REGISTRY).qualify(
        historical_id, "single_stream_stability", device="cuda",
    )
    assert qualification["eligible"] is False
    assert all(outcome["status"] == "NOT_RUN"
               for tier in qualification["tiers"].values() for outcome in tier["tasks"].values())


@pytest.mark.parametrize("candidate_id,family", REPRESENTATIVES.items())
def test_current_family_has_complete_truthfully_graded_evidence(candidate_id, family):
    workspace = StreamWorkspace(ROOT, registry=REGISTRY)
    qualification = workspace.qualify(candidate_id, "single_stream_stability", device="cuda")
    assert qualification["adapter_source_verified"] is True
    tasks = workspace.load_tasks()
    outcomes = qualification["tiers"][1]["tasks"]
    recipes = []
    for task in tasks.values():
        receipt = json.loads((workspace.reports_dir / "receipts/cuda" / candidate_id / f"{task.id}.json").read_text())
        assert receipt["error"] is None
        assert receipt["all_finite"] is True
        assert receipt["unintended_rng_deviations"] == 0
        grade = grade_sustained_observations(
            task, receipt["observations"], all_finite=receipt["all_finite"],
            unintended_rng_deviations=receipt["unintended_rng_deviations"],
            completed_steps=receipt["completed_steps"],
        )
        assert grade["status"] == receipt["verdict"] == outcomes[task.id]["status"]
        assert grade["terminal_passing"] == receipt["terminal_passing"]
        assert receipt["seed"] == 0
        assert receipt["completed_steps"] == task.steps == 480
        assert [row["step"] for row in receipt["observations"]] == task.observation_steps()
        assert receipt["artifacts"]["particlegan_version"] == "0.9.6"
        recipe = receipt["artifacts"]["resolved_recipe"]
        assert Recipe.from_dict(recipe).effective_critic_formulation == family
        assert recipe["batch_size"] == 1
        recipes.append(recipe)
        online = receipt["artifacts"]["online_execution"]
        warmup = task.host_definition["online_contract"]["warmup_arrivals"]
        assert online["arrivals"] == 480 + warmup
        assert online["updates"] == online["live_predictions"] == 480
        assert online["max_real_batch"] == online["max_fake_batch"] == 1
        assert online["replayed_examples"] == 0
        assert online["gradient_accumulation_steps"] == 1
        gif = workspace.reports_dir / receipt["artifacts"]["goal_gif"]
        assert hashlib.sha256(gif.read_bytes()).hexdigest() == receipt["artifacts"]["gif_sha256"]
        with Image.open(gif) as animation:
            assert animation.n_frames == 24
            assert animation.size == (900, 300)
    assert all(recipe == recipes[0] for recipe in recipes), "per-task recipe selection"
