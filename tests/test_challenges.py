"""Oracle/destructive controls for four chronological batch-size-one challenges."""
from collections import Counter
from unittest.mock import patch

import pytest
import torch

from particlestream.challenges import (
    CIRCLE_WAYPOINTS,
    SQUARE_WAYPOINTS,
    CircleOrbitStream,
    DelayedCueStream,
    PersistentForkStream,
    RareModeBarsStream,
    SquareCircleCycleStream,
    VariableDwellBarsStream,
    circle_orbit_coords,
    circle_orbit_templates,
    extract_circle_coords,
    render_circle_orbit_gif,
    render_square_circle_cycle_gif,
    score_circle_orbit,
    score_delayed_cue,
    score_persistent_fork,
    score_rare_mode_bars,
    score_square_circle_cycle,
    square_orbit_coords,
    square_orbit_templates,
)
from particlestream.streams import score_multimodal_bars


@pytest.mark.parametrize("stream_cls", [VariableDwellBarsStream, RareModeBarsStream,
                                        DelayedCueStream, PersistentForkStream,
                                        CircleOrbitStream, SquareCircleCycleStream])
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


def test_persistent_fork_single_frame_paths_and_independent_branches():
    stream = PersistentForkStream(noise_std=0)
    generator = torch.Generator().manual_seed(0)
    branches = []
    for _ in range(32):
        frames, metadata = zip(*(stream.next_frame(generator) for _ in range(4)))
        assert metadata[0]["branch"] is None
        branch = metadata[1]["branch"]
        branches.append(branch)
        assert torch.equal(frames[0], stream.stem)
        assert all(torch.equal(frames[depth + 1], stream.branch_paths[branch, depth:depth + 1])
                   for depth in range(3))
        assert all(item["branch"] == branch for item in metadata[1:])
        assert [item["depth"] for item in metadata] == [0, 1, 2, 3]
    assert set(branches) == {0, 1}
    assert any(branches[i] == branches[i + 1] for i in range(0, 32, 2)), "paired episodes must not force opposite branches"
    flattened = stream.branch_paths.reshape(6, 64)
    assert torch.unique(flattened, dim=0).shape[0] == 6


def test_fork_branch_draw_happens_at_first_branch_arrival_only():
    stream = PersistentForkStream(noise_std=0)
    generator = torch.Generator().manual_seed(0)
    with patch("particlestream.challenges.torch.randint", wraps=torch.randint) as branch_draw:
        _, stem_metadata = stream.next_frame(generator)
        assert stem_metadata["branch"] is None
        branch_draw.assert_not_called()
        stream.next_frame(generator)
        branch_draw.assert_called_once_with(2, (), device=torch.device("cpu"), generator=generator)
        stream.next_frame(generator)
        stream.next_frame(generator)
        assert branch_draw.call_count == 1
        _, stem_metadata = stream.next_frame(generator)
        assert stem_metadata["branch"] is None
        assert branch_draw.call_count == 1
        stream.next_frame(generator)
        assert branch_draw.call_count == 2


def test_fork_future_draw_ignores_previous_branch_state():
    outcomes = []
    for previous_branch in (0, 1):
        stream = PersistentForkStream(noise_std=0)
        stream.cursor = 4  # The same next episode following different prior branches.
        stream._branch = previous_branch
        generator = torch.Generator().manual_seed(0)
        stem, metadata = stream.next_frame(generator)
        assert metadata["branch"] is None
        branch, metadata = stream.next_frame(generator)
        outcomes.append((stem, branch, metadata["branch"], generator.get_state()))
    assert outcomes[0][2] == outcomes[1][2]
    for index in (0, 1, 3):
        assert torch.equal(outcomes[0][index], outcomes[1][index])


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


def test_circle_orbit_geometry_and_chronological_cycle():
    stream = CircleOrbitStream(noise_std=0.0)
    generator = torch.Generator().manual_seed(0)
    assert stream.period == 8
    assert len(CIRCLE_WAYPOINTS) == 8
    coords = circle_orbit_coords()
    assert coords.shape == (8, 2)
    radii = coords.square().sum(dim=-1).sqrt()
    assert torch.all((radii >= 0.90) & (radii <= 1.01))
    warmup = stream.clean_warmup(0)
    assert warmup.shape == (2, 1, 8, 8)
    assert torch.equal(warmup[0], stream.templates[6])
    assert torch.equal(warmup[1], stream.templates[7])
    for step in range(16):
        frame, meta = stream.next_frame(generator)
        phase = step % 8
        assert meta["phase"] == phase
        assert meta["orbit"] == step // 8
        assert torch.equal(frame, stream.templates[phase:phase + 1])
        extracted = extract_circle_coords(frame)[0]
        assert torch.allclose(extracted, coords[phase], atol=1e-5)


def test_circle_orbit_oracle_and_destructive_controls():
    templates = circle_orbit_templates()
    oracle = templates.unsqueeze(0).repeat(32, 1, 1, 1, 1)
    oracle_score = score_circle_orbit(oracle, templates)
    assert oracle_score == {
        "orbit_rmse": 0.0,
        "xy_rmse": 0.0,
        "step_accuracy": 1.0,
        "orbit_completion": 1.0,
        "waypoint_modes": 8.0,
    }

    # Stuck particle at waypoint 0
    stuck = templates[0:1].unsqueeze(0).repeat(32, 8, 1, 1, 1)
    stuck_score = score_circle_orbit(stuck, templates)
    assert stuck_score["step_accuracy"] == 1 / 8
    assert stuck_score["orbit_completion"] == 0.0
    assert stuck_score["waypoint_modes"] == 1.0
    assert stuck_score["orbit_rmse"] > 0.12
    assert stuck_score["xy_rmse"] > 0.15

    # Reversed (counter-clockwise) orbit
    reversed_orbit = templates.flip(0).unsqueeze(0).repeat(32, 1, 1, 1, 1)
    rev_score = score_circle_orbit(reversed_orbit, templates)
    assert rev_score["orbit_completion"] == 0.0
    assert rev_score["step_accuracy"] < 0.90

    # Phase-lagged orbit (shifted by 1 step)
    lagged = templates.roll(1, dims=0).unsqueeze(0).repeat(32, 1, 1, 1, 1)
    lag_score = score_circle_orbit(lagged, templates)
    assert lag_score["step_accuracy"] == 0.0
    assert lag_score["orbit_completion"] == 0.0

    # Incomplete 7/8 arc (last waypoint repeats waypoint 6)
    incomplete = oracle.clone()
    incomplete[:, 7] = templates[6]
    inc_score = score_circle_orbit(incomplete, templates)
    assert inc_score["step_accuracy"] == 7 / 8
    assert inc_score["orbit_completion"] == 0.0
    assert inc_score["waypoint_modes"] == 7.0

    # Blurred average of all 8 waypoints
    blurred = templates.mean(dim=0, keepdim=True).unsqueeze(0).repeat(32, 8, 1, 1, 1)
    blur_score = score_circle_orbit(blurred, templates)
    assert blur_score["step_accuracy"] == 0.0
    assert blur_score["orbit_completion"] == 0.0
    assert blur_score["xy_rmse"] > 0.90

    # Blank / sub-background frames map to canvas center (0, 0) in [-1, 1]^2
    blank = torch.full((32, 8, 1, 8, 8), 0.1)
    blank_coords = extract_circle_coords(blank[0, 0])
    assert torch.allclose(blank_coords, torch.zeros(2), atol=1e-6)
    blank_score = score_circle_orbit(blank, templates)
    assert blank_score["step_accuracy"] == 0.0
    assert blank_score["orbit_completion"] == 0.0
    assert blank_score["waypoint_modes"] == 0.0
    assert blank_score["orbit_rmse"] > 0.12
    assert blank_score["xy_rmse"] > 0.90
    # Even with a relaxed quality_rmse >= 0.20, tied RMSE across all 8 waypoints must not credit index 0
    assert score_circle_orbit(blank, templates, quality_rmse=0.25, quality_xy=2.0)["step_accuracy"] == 0.0

    # Bimodal symmetric two-patch frames (opposite and adjacent waypoints both active at 0.9)
    bimodal_opp = torch.maximum(templates, templates.roll(4, dims=0)).unsqueeze(0).repeat(32, 1, 1, 1, 1)
    opp_score = score_circle_orbit(bimodal_opp, templates)
    assert opp_score["orbit_rmse"] > 0.12 and opp_score["xy_rmse"] > 0.90
    assert opp_score["step_accuracy"] == 0.0 and opp_score["orbit_completion"] == 0.0

    bimodal_adj = torch.maximum(templates, templates.roll(1, dims=0)).unsqueeze(0).repeat(32, 1, 1, 1, 1)
    adj_score = score_circle_orbit(bimodal_adj, templates)
    assert adj_score["orbit_rmse"] > 0.12 and adj_score["xy_rmse"] > 0.30
    assert adj_score["step_accuracy"] == 0.0 and adj_score["orbit_completion"] == 0.0

    # Asymmetric bimodal perturbation: target waypoint at 0.85, opposite waypoint at 0.50
    # Pixel RMSE is ~0.1008 <= 0.12, but 2D centroid error is ~0.4302 > 0.15 -> must fail step_valid!
    bimodal_asym = torch.full((8, 1, 8, 8), 0.1)
    for i in range(8):
        bimodal_asym[i] = torch.where(templates[i] > 0.5, 0.85, bimodal_asym[i])
        bimodal_asym[i] = torch.where(templates[(i + 4) % 8] > 0.5, 0.50, bimodal_asym[i])
    asym_score = score_circle_orbit(bimodal_asym.unsqueeze(0).repeat(32, 1, 1, 1, 1), templates)
    assert asym_score["orbit_rmse"] <= 0.12
    assert asym_score["xy_rmse"] > 0.15
    assert asym_score["step_accuracy"] == 0.0
    assert asym_score["orbit_completion"] == 0.0
    assert asym_score["waypoint_modes"] == 0.0

    # Mixed rollout where 4/32 (12.5%) of particles are bimodal_asym: mean xy_rmse <= 0.15,
    # but only 28/32 (87.5% < 90%) complete a valid orbit within quality_xy <= 0.15.
    mixed = oracle.clone()
    mixed[:4] = bimodal_asym
    mixed_score = score_circle_orbit(mixed, templates)
    assert mixed_score["orbit_completion"] == 28 / 32 < 0.90
    assert mixed_score["step_accuracy"] == 28 / 32 < 0.90


@pytest.mark.parametrize(
    "bad_trajectories",
    [
        torch.empty(0, 8, 1, 8, 8),
        torch.zeros(32, 3, 1, 8, 8),
        torch.zeros(32, 8, 1, 8, 8, dtype=torch.long),
        torch.full((32, 8, 1, 8, 8), float("nan")),
    ],
)
def test_bad_circle_orbit_outputs_safely_fail(bad_trajectories):
    templates = circle_orbit_templates()
    score = score_circle_orbit(bad_trajectories, templates)
    assert score["step_accuracy"] == 0.0
    assert score["orbit_completion"] == 0.0
    assert score["waypoint_modes"] == 0.0
    assert score["orbit_rmse"] == 1e6
    assert score["xy_rmse"] == 1e6


def test_render_circle_orbit_gif_produces_valid_animation(tmp_path):
    from PIL import Image

    templates = circle_orbit_templates()
    samples = templates.repeat(4, 1, 1, 1).numpy()
    frames = [
        {
            "step": step,
            "passed": step >= 120,
            "metrics": {
                "orbit_rmse": 0.02 if step >= 120 else 0.22,
                "xy_rmse": 0.01 if step >= 120 else 0.45,
                "step_accuracy": 1.0 if step >= 120 else 0.375,
                "orbit_completion": 1.0 if step >= 120 else 0.0,
                "waypoint_modes": 8.0 if step >= 120 else 3.0,
            },
            "current_stream_image": templates[0:1].numpy(),
            "target_images": templates.numpy(),
            "sample_images": samples,
        }
        for step in (40, 120, 240, 480)
    ]
    out_gif = tmp_path / "circle_orbit.gif"
    meta = render_circle_orbit_gif(frames, out_gif, substeps_per_checkpoint=1)
    assert out_gif.is_file()
    assert len(meta["sha256"]) == 64
    with Image.open(out_gif) as img:
        assert img.size == (920, 400)
        assert img.n_frames >= len(frames)
    with pytest.raises(ValueError, match="at least one frame"):
        render_circle_orbit_gif([], tmp_path / "empty.gif")


def test_square_circle_cycle_geometry_and_chronological_pairs():
    stream = SquareCircleCycleStream(noise_std=0.0)
    generator = torch.Generator().manual_seed(0)
    assert stream.period == 8
    assert len(SQUARE_WAYPOINTS) == 8
    sq_coords = square_orbit_coords()
    ci_coords = circle_orbit_coords()
    assert sq_coords.shape == (8, 2)
    assert ci_coords.shape == (8, 2)
    # Every square waypoint lies on the unit L_inf perimeter max(|x|, |y|) == 1.0
    linf_radii = sq_coords.abs().amax(dim=-1)
    assert torch.allclose(linf_radii, torch.ones(8), atol=1e-5)
    # Diagonal corners (1, 3, 5, 7) have L2 radius sqrt(2) on the square vs <1.0 on the circle
    for corner in (1, 3, 5, 7):
        assert sq_coords[corner].norm().item() > 1.35
        assert ci_coords[corner].norm().item() < 1.0
    sq_warm = stream.clean_square_warmup(0)
    ci_warm = stream.clean_circle_warmup(0)
    assert sq_warm.shape == ci_warm.shape == (2, 1, 8, 8)
    assert torch.equal(sq_warm[0], stream.square_templates[6])
    assert torch.equal(sq_warm[1], stream.square_templates[7])
    assert torch.equal(ci_warm[0], stream.circle_templates[6])
    assert torch.equal(ci_warm[1], stream.circle_templates[7])
    for step in range(16):
        s_frame, c_frame, meta = stream.next_pair(generator)
        phase = step % 8
        assert meta["phase"] == phase
        assert meta["orbit"] == step // 8
        assert s_frame.shape == c_frame.shape == (1, 1, 8, 8)
        assert not s_frame.requires_grad and not c_frame.requires_grad
        assert torch.equal(s_frame, stream.square_templates[phase:phase + 1])
        assert torch.equal(c_frame, stream.circle_templates[phase:phase + 1])


def test_square_circle_cycle_oracle_and_destructive_controls():
    sq_tmpl = square_orbit_templates()
    ci_tmpl = circle_orbit_templates()
    oracle_s2c = ci_tmpl.unsqueeze(0).repeat(32, 1, 1, 1, 1)
    oracle_c2s = sq_tmpl.unsqueeze(0).repeat(32, 1, 1, 1, 1)

    oracle_score = score_square_circle_cycle(oracle_s2c, oracle_c2s, sq_tmpl, ci_tmpl)
    assert oracle_score == {
        "cycle_rmse": 0.0,
        "s2c_rmse": 0.0,
        "c2s_rmse": 0.0,
        "xy_rmse": 0.0,
        "step_accuracy": 1.0,
        "cycle_completion": 1.0,
        "waypoint_modes": 8.0,
    }

    # Identity destructive control: G_S2C outputs Square, G_C2S outputs Circle (no shape translation!)
    identity_score = score_square_circle_cycle(oracle_c2s, oracle_s2c, sq_tmpl, ci_tmpl)
    assert identity_score["cycle_completion"] == 0.0
    assert identity_score["step_accuracy"] == 0.5
    assert identity_score["waypoint_modes"] == 4.0
    assert identity_score["cycle_rmse"] > 0.12
    assert identity_score["s2c_rmse"] > 0.12
    assert identity_score["c2s_rmse"] > 0.12
    assert identity_score["xy_rmse"] > 0.15

    # One-way collapse: S->C works, but C->S also outputs Circle instead of Square
    one_way_score = score_square_circle_cycle(oracle_s2c, oracle_s2c, sq_tmpl, ci_tmpl)
    assert one_way_score["s2c_rmse"] == 0.0
    assert one_way_score["c2s_rmse"] > 0.12
    assert one_way_score["cycle_completion"] == 0.0
    assert one_way_score["step_accuracy"] == 0.5
    assert one_way_score["waypoint_modes"] == 4.0

    # Phase-lagged by 1 step
    lagged_s2c = ci_tmpl.roll(1, dims=0).unsqueeze(0).repeat(32, 1, 1, 1, 1)
    lag_score = score_square_circle_cycle(lagged_s2c, oracle_c2s, sq_tmpl, ci_tmpl)
    assert lag_score["step_accuracy"] == 0.0
    assert lag_score["cycle_completion"] == 0.0

    # Incomplete 7/8 cycle (last waypoint repeats waypoint 6 in C->S)
    incomplete_c2s = oracle_c2s.clone()
    incomplete_c2s[:, 7] = sq_tmpl[6]
    inc_score = score_square_circle_cycle(oracle_s2c, incomplete_c2s, sq_tmpl, ci_tmpl)
    assert inc_score["step_accuracy"] == 7 / 8
    assert inc_score["cycle_completion"] == 0.0
    assert inc_score["waypoint_modes"] == 7.0

    # Midpoint interpolation between Square and Circle corners fails translation discrimination
    midpoint_s2c = 0.5 * (oracle_s2c + oracle_c2s)
    midpoint_c2s = 0.5 * (oracle_s2c + oracle_c2s)
    mid_score = score_square_circle_cycle(midpoint_s2c, midpoint_c2s, sq_tmpl, ci_tmpl, quality_rmse=0.25, quality_xy=0.50)
    assert mid_score["cycle_completion"] == 0.0
    assert mid_score["step_accuracy"] == 0.5


@pytest.mark.parametrize(
    "bad_trajectories",
    [
        torch.empty(0, 8, 1, 8, 8),
        torch.zeros(32, 3, 1, 8, 8),
        torch.zeros(32, 8, 1, 8, 8, dtype=torch.long),
        torch.full((32, 8, 1, 8, 8), float("nan")),
    ],
)
def test_bad_square_circle_cycle_outputs_safely_fail(bad_trajectories):
    sq_tmpl = square_orbit_templates()
    ci_tmpl = circle_orbit_templates()
    score = score_square_circle_cycle(bad_trajectories, bad_trajectories, sq_tmpl, ci_tmpl)
    assert score["step_accuracy"] == 0.0
    assert score["cycle_completion"] == 0.0
    assert score["waypoint_modes"] == 0.0
    assert score["s2c_rmse"] == 1e6
    assert score["c2s_rmse"] == 1e6
    assert score["cycle_rmse"] == 1e6
    assert score["xy_rmse"] == 1e6

    oracle_c2s = sq_tmpl.unsqueeze(0).repeat(32, 1, 1, 1, 1)
    one_sided = score_square_circle_cycle(bad_trajectories, oracle_c2s, sq_tmpl, ci_tmpl)
    assert one_sided["s2c_rmse"] == 1e6
    assert one_sided["step_accuracy"] == 0.0
    assert one_sided["cycle_completion"] == 0.0
    assert one_sided["waypoint_modes"] == 0.0


def test_render_square_circle_cycle_gif_produces_valid_animation(tmp_path):
    from PIL import Image

    sq_tmpl = square_orbit_templates()
    ci_tmpl = circle_orbit_templates()
    samples = torch.cat([ci_tmpl.repeat(2, 1, 1, 1), sq_tmpl.repeat(2, 1, 1, 1)], dim=0).numpy()
    frames = [
        {
            "step": step,
            "passed": step >= 120,
            "metrics": {
                "cycle_rmse": 0.02 if step >= 120 else 0.22,
                "s2c_rmse": 0.02 if step >= 120 else 0.22,
                "c2s_rmse": 0.02 if step >= 120 else 0.22,
                "xy_rmse": 0.01 if step >= 120 else 0.45,
                "step_accuracy": 1.0 if step >= 120 else 0.375,
                "cycle_completion": 1.0 if step >= 120 else 0.0,
                "waypoint_modes": 8.0 if step >= 120 else 3.0,
            },
            "current_stream_image": ci_tmpl[0:1].numpy(),
            "target_images": torch.cat([ci_tmpl, sq_tmpl], dim=0).numpy(),
            "sample_images": samples,
        }
        for step in (40, 120, 240, 480)
    ]
    out_gif = tmp_path / "square_circle_cycle.gif"
    meta = render_square_circle_cycle_gif(frames, out_gif, substeps_per_checkpoint=1)
    assert out_gif.is_file()
    assert len(meta["sha256"]) == 64
    with Image.open(out_gif) as img:
        assert img.size == (920, 400)
        assert img.n_frames >= len(frames)
    with pytest.raises(ValueError, match="at least one frame"):
        render_square_circle_cycle_gif([], tmp_path / "empty.gif")


