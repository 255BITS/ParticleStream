"""Strict execution and real public-trainer regression checks."""
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from particlestream.online import OnlineSession, validate_execution_policy
from particlestream.streams import ForkStream, TranslatingBarStream
from particlestream.tier1 import (
    ONLINE_PARENTS, TIER1_CANDIDATES, TIER1_TASKS, _build_recipe,
    bootstrap_workspace,
)


ONLINE_IDS = [parent.replace("stream-", "online-", 1) for parent in ONLINE_PARENTS]


def test_execution_requires_explicit_replay_prohibition():
    candidate = TIER1_CANDIDATES["online-bcap-adam"]
    validate_execution_policy(candidate)
    for policy in ({"kind": "none"}, {"replay_policy": "none", "rehearsal_batch": 7},
                   {"replay_policy": "fifo", "rehearsal_batch": 0}):
        with pytest.raises(ValueError, match="explicit replay_policy"):
            validate_execution_policy(replace(candidate, memory_policy=policy))
    with pytest.raises(ValueError, match="historical replay"):
        validate_execution_policy(TIER1_CANDIDATES["fast-weight-bcap"])


@pytest.mark.parametrize("parent_id", ONLINE_PARENTS)
def test_online_recipe_changes_only_replay_declaration(parent_id):
    from particlegan_forge import CandidateSpec
    root = Path(__file__).resolve().parents[1]
    parent = CandidateSpec.from_dict(json.loads(
        (root / "configs/forge/ideas" / f"{parent_id}.json").read_text()))
    online = TIER1_CANDIDATES[parent_id.replace("stream-", "online-", 1)]
    assert online.parent == parent_id
    assert online.recipe_preset == parent.recipe_preset
    assert online.recipe_overrides == parent.recipe_overrides
    assert online.memory_policy["replay_policy"] == "none"
    assert online.memory_policy["rehearsal_batch"] == 0
    recipes = [_build_recipe(task, online, task.steps).to_dict() for task in TIER1_TASKS]
    assert all(recipe == recipes[0] for recipe in recipes)


@pytest.mark.parametrize("stream_type,expected", [
    (TranslatingBarStream, [0, 1, 2, 3, 2, 1, 0, 1]),
])
def test_temporal_stream_reveals_one_chronological_frame(stream_type, expected):
    stream = stream_type(noise_std=0)
    rng = torch.Generator().manual_seed(0)
    for index, target_id in enumerate(expected):
        frame, mode = stream.next_frame(rng)
        assert stream.cursor == index + 1
        assert mode == target_id
        assert frame.shape == (1, 1, 8, 8)
        assert not frame.requires_grad


def test_fork_draws_independent_branches_only_on_branch_arrival():
    from particlegan_forge import NamedStreams
    streams = [ForkStream(noise_std=0.01), ForkStream(noise_std=0.01)]
    generators = [NamedStreams(seed=0).generator(
        "data", component="stream", purpose="training") for _ in streams]
    branches = []
    real_randint = torch.randint
    with patch("particlestream.streams.torch.randint", wraps=real_randint) as draws:
        for arrival in range(128):
            before = draws.call_count
            results = [stream.next_frame(rng) for stream, rng in zip(streams, generators)]
            assert draws.call_count - before == (0 if arrival % 2 == 0 else 2)
            frame, mode = results[0]
            assert frame.shape == (1, 1, 8, 8)
            assert not frame.requires_grad
            assert torch.equal(frame, results[1][0])
            assert mode == results[1][1]
            if arrival % 2 == 0:
                assert mode == 0
            else:
                assert mode in (1, 2)
                branches.append(mode)
    assert set(branches) == {1, 2}
    assert any(left == right for left, right in zip(branches, branches[1:])), "branch law must not alternate"
    assert all(stream.cursor == 128 for stream in streams)


@pytest.mark.parametrize("candidate_id", ONLINE_IDS)
@pytest.mark.parametrize("task", TIER1_TASKS, ids=lambda task: task.id)
def test_real_public_trainers_remain_strict_while_serving(tmp_path, candidate_id, task):
    workspace = bootstrap_workspace(tmp_path)
    receipt = workspace.run_task(candidate_id, task.id, device="cpu", steps_override=4)
    assert receipt["grade"]["status"] == "INCOMPLETE", receipt.get("error")
    assert receipt["all_finite"]
    assert receipt["unintended_rng_deviations"] == 0
    execution = receipt["artifacts"]["online_execution"]
    warmup = task.host_definition["online_contract"]["warmup_arrivals"]
    assert execution == {
        "arrivals": 4 + warmup, "updates": 4, "warmup_arrivals": warmup,
        "live_predictions": 4, "max_real_batch": 1, "max_fake_batch": 1,
        "replayed_examples": 0, "gradient_accumulation_steps": 1,
    }
    assert all("live_prequential_rmse" in row["metrics"] for row in receipt["observations"])
    assert (tmp_path / "reports/forge" / receipt["artifacts"]["goal_gif"]).is_file()


def test_historical_replay_adapter_cannot_start_training(tmp_path):
    workspace = bootstrap_workspace(tmp_path)
    with patch("particlestream.tier1.GANTrainer") as trainer:
        with pytest.raises(ValueError, match="replay_policy"):
            workspace.run_task("stream-bcap-adam", TIER1_TASKS[0].id,
                               device="cpu", steps_override=4)
    trainer.assert_not_called()
    assert not (workspace.reports_dir / "receipts/cpu/stream-bcap-adam").exists()
