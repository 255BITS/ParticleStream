"""Oracle/destructive controls for four chronological batch-size-one challenges."""
from collections import Counter

import pytest
import torch

from particlestream.challenges import (
    DelayedCueStream,
    PersistentForkStream,
    RareModeBarsStream,
    VariableDwellBarsStream,
    score_delayed_cue,
    score_persistent_fork,
    score_rare_mode_bars,
)
from particlestream.streams import score_multimodal_bars


@pytest.mark.parametrize("stream_cls", [VariableDwellBarsStream, RareModeBarsStream,
                                        DelayedCueStream, PersistentForkStream])
def test_single_arrival_detached_reproducible_stream(stream_cls):
    first, second = stream_cls(), stream_cls()
    first_rng = torch.Generator().manual_seed(0)
    second_rng = torch.Generator().manual_seed(0)
    for _ in range(48):
        frame, metadata = first.next_frame(first_rng)
        repeated, repeated_metadata = second.next_frame(second_rng)
        assert frame.shape == (1, 1, 8, 8)
        assert frame.requires_grad is False and frame.grad_fn is None
        assert torch.isfinite(frame).all()
        assert isinstance(metadata, dict)
        assert torch.equal(frame, repeated)
        assert metadata == repeated_metadata
    assert first.cursor == second.cursor == 48


def test_variable_dwell_balanced_and_scorer_controls():
    stream = VariableDwellBarsStream(noise_std=0)
    generator = torch.Generator().manual_seed(0)
    observed = Counter()
    blocks = []
    for _ in range(stream.period):
        frame, metadata = stream.next_frame(generator)
        observed[metadata["mode"]] += 1
        blocks.append(metadata["block"])
        assert torch.equal(frame, stream.templates[metadata["mode"]:metadata["mode"] + 1])
    assert stream.period == 240
    assert observed == {0: 60, 1: 60, 2: 60, 3: 60}
    assert sorted(Counter(blocks).values()) == [4] * 4 + [8] * 4 + [16] * 4 + [32] * 4
    oracle = score_multimodal_bars(stream.templates.repeat(8, 1, 1, 1), stream.templates)
    collapsed = score_multimodal_bars(stream.templates[3:4].repeat(32, 1, 1, 1), stream.templates)
    assert oracle["hq"] >= 0.85 and oracle["modes"] == 4 and oracle["finite_template_tv"] <= 0.2
    assert collapsed["modes"] == 1 and collapsed["finite_template_tv"] > 0.2


def test_rare_mode_exposure_and_oracle():
    stream = RareModeBarsStream(noise_std=0)
    generator = torch.Generator().manual_seed(0)
    observed = Counter(stream.next_frame(generator)[1]["mode"] for _ in range(480))
    assert observed == {0: 150, 1: 150, 2: 150, 3: 30}
    oracle = stream.templates.repeat_interleave(torch.tensor([10, 10, 10, 2]), dim=0)
    score = score_rare_mode_bars(oracle, stream.templates)
    assert score == {"hq": 1.0, "support_modes": 4.0, "rare_mass": 0.0625,
                     "finite_template_tv": 0.0, "mean_rmse": 0.0}


def test_rare_mode_drop_control_fails_even_when_tv_alone_would_pass():
    stream = RareModeBarsStream()
    dropped = stream.templates[:3].repeat_interleave(torch.tensor([11, 11, 10]), dim=0)
    score = score_rare_mode_bars(dropped, stream.templates)
    assert score["hq"] == 1 and score["finite_template_tv"] <= 0.15
    assert score["support_modes"] == 3 and score["rare_mass"] < 1 / 32
    uniform = score_rare_mode_bars(stream.templates.repeat(8, 1, 1, 1), stream.templates)
    assert uniform["rare_mass"] > 3 / 32 and uniform["finite_template_tv"] > 0.15


def test_rare_mode_invalid_probability_mass_is_not_renormalized_away():
    stream = RareModeBarsStream()
    samples = stream.templates.repeat_interleave(torch.tensor([10, 10, 10, 2]), dim=0)
    samples[:8] = float("nan")
    score = score_rare_mode_bars(samples, stream.templates)
    assert score["hq"] == 0.75
    assert score["finite_template_tv"] == 0.25
    assert score["rare_mass"] == 2 / 32


def test_delayed_cue_histories_differ_only_at_cue_and_exclude_outcome():
    stream = DelayedCueStream()
    left, right = stream.clean_history(0), stream.clean_history(1)
    assert left.shape == right.shape == (5, 1, 8, 8)
    assert not torch.equal(left[0], right[0])
    assert torch.equal(left[1:], right[1:])
    assert torch.equal(left[1:], stream.distractor.repeat(4, 1, 1, 1))
    assert not torch.equal(left[-1], stream.outcomes[0])
    assert not torch.equal(right[-1], stream.outcomes[1])
    with pytest.raises(ValueError):
        stream.clean_history(2)


def test_delayed_cue_arrivals_have_four_identical_delays_and_balanced_cues():
    stream = DelayedCueStream(noise_std=0.01)
    generator = torch.Generator().manual_seed(0)
    observed_cues = []
    for _ in range(12):
        frames, metadata = zip(*(stream.next_frame(generator) for _ in range(6)))
        assert [item["phase"] for item in metadata] == ["cue", "delay", "delay", "delay", "delay", "outcome"]
        observed_cues.append(metadata[0]["cue"])
        assert all(torch.equal(frame, stream.distractor) for frame in frames[1:5])
        assert len({item["cue"] for item in metadata}) == 1
    assert all(set(observed_cues[i:i + 2]) == {0, 1} for i in range(0, 12, 2))
    assert len(set(observed_cues[::2])) == 2, "pair order must not be a fixed step-index cue"


def test_delayed_cue_oracle_and_forgotten_cue_control():
    stream = DelayedCueStream()
    left = stream.outcomes[0:1].repeat(32, 1, 1, 1)
    right = stream.outcomes[1:2].repeat(32, 1, 1, 1)
    oracle = score_delayed_cue(left, right, stream.outcomes)
    assert oracle == {"prediction_rmse": 0.0, "outcome_accuracy": 1.0}
    forgotten = score_delayed_cue(left, left, stream.outcomes)
    assert forgotten["outcome_accuracy"] == 0.5
    assert forgotten["prediction_rmse"] > 0.12
    blurred = (left + right) / 2
    assert score_delayed_cue(blurred, blurred, stream.outcomes)["outcome_accuracy"] == 0.0


def test_persistent_fork_single_frame_paths_and_balanced_branches():
    stream = PersistentForkStream(noise_std=0)
    generator = torch.Generator().manual_seed(0)
    branches = []
    for _ in range(8):
        frames, metadata = zip(*(stream.next_frame(generator) for _ in range(4)))
        branch = metadata[0]["branch"]
        branches.append(branch)
        assert torch.equal(frames[0], stream.stem)
        assert all(torch.equal(frames[depth + 1], stream.branch_paths[branch, depth:depth + 1])
                   for depth in range(3))
        assert all(item["branch"] == branch for item in metadata)
        assert [item["depth"] for item in metadata] == [0, 1, 2, 3]
    assert Counter(branches) == {0: 4, 1: 4}
    flattened = stream.branch_paths.reshape(6, 64)
    assert torch.unique(flattened, dim=0).shape[0] == 6


def test_persistent_fork_oracle_and_collapsed_control():
    stream = PersistentForkStream()
    oracle = stream.branch_paths.repeat_interleave(16, dim=0)
    score = score_persistent_fork(oracle, stream.branch_paths)
    assert score["hq"] == 1 and score["within_state_modes"] == 2
    assert score["branch_tv"] == 0 and score["trajectory_consistency"] == 1
    collapsed = score_persistent_fork(oracle[:16].repeat(2, 1, 1, 1, 1), stream.branch_paths)
    assert collapsed["within_state_modes"] == 1 and collapsed["branch_tv"] == 0.5
    assert collapsed["trajectory_consistency"] == 1


def test_fork_marginally_perfect_branch_switching_fails_consistency():
    stream = PersistentForkStream()
    switched = stream.branch_paths.repeat_interleave(16, dim=0)
    switched[:16, 1] = stream.branch_paths[1, 1]
    switched[16:, 1] = stream.branch_paths[0, 1]
    score = score_persistent_fork(switched, stream.branch_paths)
    assert score["hq"] == 1 and score["within_state_modes"] == 2 and score["branch_tv"] == 0
    assert score["trajectory_consistency"] == 0
    assert all(score[f"depth_{depth}_hq"] == 1 for depth in (1, 2, 3))


def test_fork_invalid_frame_counts_against_marginals_and_complete_trajectories():
    stream = PersistentForkStream()
    samples = stream.branch_paths.repeat_interleave(16, dim=0)
    samples[:2, 1] = float("inf")
    score = score_persistent_fork(samples, stream.branch_paths)
    assert score["hq"] == 30 / 32
    assert score["trajectory_consistency"] == 30 / 32
    assert score["branch_tv"] == 2 / 32


@pytest.mark.parametrize("bad_samples", [torch.empty(0, 1, 8, 8), torch.zeros(32, 8, 8), torch.zeros(32, 1, 8, 8, dtype=torch.long),
                                         torch.full((32, 1, 8, 8), float("nan"))])
def test_bad_rare_and_cue_outputs_safely_fail(bad_samples):
    rare = RareModeBarsStream()
    score = score_rare_mode_bars(bad_samples, rare.templates)
    assert score["hq"] == 0 and score["support_modes"] == 0
    assert score["finite_template_tv"] == 1
    cue = DelayedCueStream()
    assert score_delayed_cue(bad_samples, bad_samples, cue.outcomes)["outcome_accuracy"] == 0


@pytest.mark.parametrize("bad_trajectories", [torch.empty(0, 3, 1, 8, 8), torch.zeros(32, 1, 8, 8),
                                              torch.full((32, 3, 1, 8, 8), float("nan"))])
def test_bad_fork_outputs_safely_fail(bad_trajectories):
    stream = PersistentForkStream()
    score = score_persistent_fork(bad_trajectories, stream.branch_paths)
    assert score["hq"] == 0 and score["trajectory_consistency"] == 0
    assert score["branch_tv"] == 1
