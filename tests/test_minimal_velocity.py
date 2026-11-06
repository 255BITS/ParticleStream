"""Controls establish the small velocity probe's finite scientific claim."""
from collections import Counter
import math

import pytest
import torch

from particlestream.minimal_velocity import (
    EXTRAPOLATION_SPEEDS, INTERPOLATION_SPEEDS, TRAIN_SPEEDS, VELOCITY_BOUNDS,
    render_point, score_velocity, velocity_cases, velocity_control_predictions,
    velocity_controls, velocity_manifest, velocity_training_episodes,
)


def _passes(metrics):
    return all(metrics[name] <= value if op == "<=" else metrics[name] == value
               for name, op, value in VELOCITY_BOUNDS)


def test_renderer_preserves_subpixel_position_and_unit_mass():
    image = render_point(.45)
    assert image.shape == (1, 1, 8, 8)
    assert not image.requires_grad
    assert float(image.sum()) == pytest.approx(1.)
    axis = torch.linspace(0, 1, 8)
    assert float((image * axis.view(1, 1, 1, 8)).sum()) == pytest.approx(.45)
    assert float((image * axis.view(1, 1, 8, 1)).sum()) == pytest.approx(.5)
    assert int((image > 0).sum()) == 4
    for x in (-.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            render_point(x)


def test_eight_heldout_clips_are_interior_and_separate_from_training():
    cases = velocity_cases()
    assert len(cases) == 8
    assert Counter(case["group"] for case in cases) == {"interpolation": 4, "extrapolation": 4}
    assert not set(TRAIN_SPEEDS) & set(INTERPOLATION_SPEEDS + EXTRAPOLATION_SPEEDS)
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert case["frames"].shape == (6, 1, 8, 8)
        assert case["prefix"].shape == case["targets"].shape == (3, 1, 8, 8)
        assert torch.allclose(case["frames"].sum((-3, -2, -1)), torch.ones(6))
        assert all(0 < case["start_x"] + step * case["speed"] < 1 for step in range(6))


def test_training_episode_accounting_and_balanced_speeds():
    episodes = list(velocity_training_episodes())
    assert len(episodes) == 128
    assert sum(len(case["targets"]) for case in episodes) == 384
    assert sum(len(case["prefix"]) for case in episodes) == 384
    assert Counter(case["speed"] for case in episodes) == {speed: 32 for speed in TRAIN_SPEEDS}
    assert all(case["group"] == "train" for case in episodes)
    assert not {case["speed"] for case in episodes} & set(EXTRAPOLATION_SPEEDS)
    manifest = velocity_manifest()
    assert manifest["training_updates"] == manifest["source_only_warmup_arrivals"] == 384
    assert manifest["total_training_arrivals"] == 768
    with pytest.raises(ValueError):
        list(velocity_training_episodes(updates=10))


def test_analytic_oracle_passes_and_every_destructive_control_fails():
    controls = velocity_controls()
    assert _passes(controls["oracle"])
    assert controls["oracle"]["point_validity"] == 1.
    for name, metrics in controls.items():
        if name != "oracle":
            assert not _passes(metrics), name
    assert controls["nearest_training_speed"]["extrapolation_speed_error"] > .01
    assert controls["nearest_training_speed"]["extrapolation_centroid_rmse"] > .01


def test_blank_nonfinite_and_diffuse_outputs_cannot_pass_centroid_gates():
    oracle = velocity_control_predictions()["oracle"]
    for replacement in (0., float("nan"), float("inf"), -.1):
        bad = torch.full_like(oracle, replacement)
        metrics = score_velocity(bad)
        assert metrics["point_validity"] == 0.
        assert not _passes(metrics)
        assert all(math.isfinite(value) for value in metrics.values())
    diffuse = torch.full_like(oracle, 1 / 64)
    assert score_velocity(diffuse)["point_validity"] == 0.
    # Preserve a correct centroid by symmetric spreading, while leaving support.
    spread = .5 * oracle + .25 * torch.roll(oracle, 2, dims=-2) + .25 * torch.roll(oracle, -2, dims=-2)
    metrics = score_velocity(spread)
    assert metrics["centroid_rmse"] < 1e-6
    assert metrics["point_validity"] == 0.


def test_bad_particle_or_one_step_cannot_hide_in_average():
    oracle = velocity_control_predictions()["oracle"]
    particles = oracle.unsqueeze(1).repeat(1, 4, 1, 1, 1, 1)
    assert _passes(score_velocity(particles))
    particles[0, -1] = 0.
    metrics = score_velocity(particles)
    assert not _passes(metrics)
    assert metrics["interpolation_point_validity"] < 1.
    assert metrics["interpolation_centroid_rmse"] > .01
    assert metrics["extrapolation_point_validity"] == 1.


def test_scorer_rejects_ambiguous_shapes_and_case_mismatch():
    with pytest.raises(ValueError):
        score_velocity(torch.zeros(8, 1, 8, 8))
    with pytest.raises(ValueError):
        score_velocity(torch.zeros(7, 3, 1, 8, 8))
