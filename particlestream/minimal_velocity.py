"""Finite, causal velocity probes for the minimal capability screen.

Task-owned laws and scorers only: this module neither trains a model nor alters a
recipe. Coordinates use image width/height, with pixel centres spanning [0, 1].
"""
from __future__ import annotations

import math
from typing import Iterable

import torch

TRAIN_SPEEDS = (-0.050, -0.025, 0.025, 0.050)
INTERPOLATION_SPEEDS = (-0.0375, 0.0375)
EXTRAPOLATION_SPEEDS = (-0.075, 0.075)
START_X = (0.45, 0.55)
POINT_Y = 0.5
PREFIX_LENGTH = 3
ROLLOUT_LENGTH = 3
TRAIN_UPDATES = 384
VELOCITY_SPEC = {
    "id": "seq_minimal_velocity_extrapolation",
    "mode": "velocity_extrapolation",
    "steps": TRAIN_UPDATES,
    "tier": 3,
    "description": "Predict three unseen-speed point frames from three observed frames, separating interpolation from extrapolation.",
}

VELOCITY_BOUNDS = [
    (f"{group}_{metric}", comparison, value)
    for group in ("interpolation", "extrapolation")
    for metric, comparison, value in (
        ("centroid_rmse", "<=", 0.010),
        ("speed_error", "<=", 0.010),
        ("raw_rmse", "<=", 0.12),
        ("point_validity", "==", 1.0),
        ("direction_accuracy", "==", 1.0),
    )
]


def render_point(x: float, y: float = POINT_Y, *, device="cpu") -> torch.Tensor:
    """Rasterize one bilinear point as a detached (1,1,8,8) unit-mass frame.

    Reject out-of-frame coordinates instead of clipping or wrapping them. All
    declared clips are interior, so the data law contains no boundary dynamics.
    """
    x, y = float(x), float(y)
    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError("point coordinates must be finite and inside [0,1]")
    px, py = x * 7, y * 7
    left, top = int(math.floor(px)), int(math.floor(py))
    right, bottom = min(7, left + 1), min(7, top + 1)
    wx, wy = px - left, py - top
    frame = torch.zeros((1, 1, 8, 8), device=device, dtype=torch.float32)
    # add_ handles the exact right/bottom edge without duplicating mass.
    frame[0, 0, top, left] += (1 - wx) * (1 - wy)
    frame[0, 0, top, right] += wx * (1 - wy)
    frame[0, 0, bottom, left] += (1 - wx) * wy
    frame[0, 0, bottom, right] += wx * wy
    return frame


def velocity_cases(split: str = "all", *, device="cpu") -> list[dict]:
    """Return fresh finite clips; tensors are [time,channel,height,width].

    ``all`` is the eight held-out evaluation clips, never the training set.
    Every case has exactly three observed prefix frames and three future targets.
    """
    groups = {
        "train": TRAIN_SPEEDS,
        "interpolation": INTERPOLATION_SPEEDS,
        "extrapolation": EXTRAPOLATION_SPEEDS,
    }
    if split == "all":
        selected = ("interpolation", "extrapolation")
    elif split in groups:
        selected = (split,)
    else:
        raise ValueError(f"unknown velocity split: {split}")
    cases = []
    for group in selected:
        for speed in groups[group]:
            for start in START_X:
                positions = [start + speed * step for step in range(PREFIX_LENGTH + ROLLOUT_LENGTH)]
                if not all(0 < x < 1 for x in positions):
                    raise ValueError("declared velocity clip leaves the image")
                frames = torch.cat([render_point(x, device=device) for x in positions], dim=0)
                cases.append({
                    "id": f"{group}_v{speed:+.4f}_x{start:.2f}",
                    "group": group,
                    "speed": speed,
                    "start_x": start,
                    "frames": frames,
                    "prefix": frames[:PREFIX_LENGTH],
                    "targets": frames[PREFIX_LENGTH:],
                })
    return cases


def velocity_training_episodes(*, updates: int = TRAIN_UPDATES, device="cpu") -> Iterable[dict]:
    """Chronological 3-prefix/3-target episodes with balanced training speeds.

    The declared screen has 128 episodes, hence 384 optimiser updates plus 384
    source-only warm-up arrivals. Reset detached temporal memory at each episode;
    do not reset trained weights, optimisers, or learned prior summaries.
    """
    if updates < 0 or updates % ROLLOUT_LENGTH:
        raise ValueError("velocity update budget must be divisible by three")
    cases = velocity_cases("train", device=device)
    for episode in range(updates // ROLLOUT_LENGTH):
        original = cases[episode % len(cases)]
        frames = original["frames"].clone()
        yield {**original, "frames": frames, "prefix": frames[:PREFIX_LENGTH],
               "targets": frames[PREFIX_LENGTH:]}


def velocity_manifest() -> dict:
    """JSON-safe law, finite splits, and explicit arrival accounting."""
    cases = velocity_cases()
    return {
        "law": "unit_mass_bilinear_point_constant_velocity_v1",
        "coordinates": "pixel_centres_0_to_1",
        "train_speeds": list(TRAIN_SPEEDS),
        "interpolation_speeds": list(INTERPOLATION_SPEEDS),
        "extrapolation_speeds": list(EXTRAPOLATION_SPEEDS),
        "start_x": list(START_X),
        "y": POINT_Y,
        "prefix_frames": PREFIX_LENGTH,
        "generated_frames": ROLLOUT_LENGTH,
        "train_episodes": TRAIN_UPDATES // ROLLOUT_LENGTH,
        "training_updates": TRAIN_UPDATES,
        "source_only_warmup_arrivals": TRAIN_UPDATES,
        "total_training_arrivals": TRAIN_UPDATES * 2,
        "episode_reset": "detached_temporal_memory_only",
        "eval_case_ids": [case["id"] for case in cases],
        "point_mass_bounds": [0.90, 1.10],
        "minimum_target_support_mass_fraction": 0.90,
    }


def _centroids(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    masses = images.sum(dim=(-3, -2, -1))
    axis = torch.linspace(0, 1, 8, device=images.device, dtype=images.dtype)
    x = (images * axis.view(1, 1, 8)).sum(dim=(-3, -2, -1))
    y = (images * axis.view(1, 8, 1)).sum(dim=(-3, -2, -1))
    return torch.stack((x, y), dim=-1) / masses.clamp_min(1e-12).unsqueeze(-1), masses


def score_velocity(predictions: torch.Tensor, cases: list[dict] | None = None) -> dict[str, float]:
    """Score every case/particle, separated into interpolation and extrapolation.

    Shapes are [case,particle,3,1,8,8] or [case,3,1,8,8]. Error metrics take the
    worst case/particle, never the best latent or an average across easier clips.
    Invalid frames retain their denominator and cannot pass centroid-only gates.
    """
    if not isinstance(predictions, torch.Tensor):
        raise TypeError("velocity predictions must be a tensor")
    if predictions.ndim == 5:
        predictions = predictions.unsqueeze(1)
    if predictions.ndim != 6 or tuple(predictions.shape[2:]) != (3, 1, 8, 8):
        raise ValueError("velocity predictions require [case,particle,3,1,8,8]")
    if predictions.shape[1] < 1:
        raise ValueError("velocity predictions require at least one particle")
    cases = velocity_cases(device=predictions.device) if cases is None else cases
    if len(cases) != len(predictions):
        raise ValueError("velocity predictions and cases differ in length")
    targets = torch.stack([case["targets"] for case in cases]).to(predictions)
    finite = torch.isfinite(predictions).all(dim=(-3, -2, -1))
    in_range = ((predictions >= 0) & (predictions <= 1)).all(dim=(-3, -2, -1))
    clean = torch.nan_to_num(predictions.detach(), nan=0., posinf=0., neginf=0.).clamp(0, 1)
    xy, mass = _centroids(clean)
    target_xy, _ = _centroids(targets)
    expected = target_xy.unsqueeze(1)
    support = (targets > 0).unsqueeze(1)
    support_mass = (clean * support).sum(dim=(-3, -2, -1))
    valid = finite & in_range & (mass >= .90) & (mass <= 1.10) & (support_mass >= .90 * mass)
    errors = (xy - expected).square().sum(dim=-1)
    # Keep malformed output finite for a numerical FAIL, without hiding it.
    errors = torch.where(finite & in_range, errors, torch.full_like(errors, 1e6))
    raw = (clean - targets.unsqueeze(1)).square().mean(dim=(-3, -2, -1))
    raw = torch.where(finite & in_range, raw, torch.full_like(raw, 1e6))
    preceding = torch.tensor(
        [case["start_x"] + 2 * case["speed"] for case in cases],
        device=clean.device, dtype=clean.dtype,
    ).view(-1, 1, 1).expand(-1, clean.shape[1], 1)
    x = xy[..., 0]
    increments = x - torch.cat((preceding, x[..., :-1]), dim=-1)
    speeds = torch.tensor([case["speed"] for case in cases], device=clean.device,
                          dtype=clean.dtype).view(-1, 1, 1)
    speed_error = (increments - speeds).abs()
    speed_error = torch.where(finite & in_range, speed_error, torch.full_like(speed_error, 1e6))
    directions = (increments * speeds > 0) & finite & in_range
    metrics = {}
    for group in ("interpolation", "extrapolation"):
        indices = [i for i, case in enumerate(cases) if case["group"] == group]
        if not indices:
            continue
        metrics.update({
            f"{group}_centroid_rmse": float(errors[indices].mean(dim=-1).sqrt().max()),
            f"{group}_speed_error": float(speed_error[indices].max()),
            f"{group}_raw_rmse": float(raw[indices].mean(dim=-1).sqrt().max()),
            f"{group}_point_validity": float(valid[indices].float().mean()),
            f"{group}_direction_accuracy": float(directions[indices].float().mean()),
        })
    metrics.update({
        "centroid_rmse": float(errors.mean(dim=-1).sqrt().max()),
        "speed_error": float(speed_error.max()),
        "raw_rmse": float(raw.mean(dim=-1).sqrt().max()),
        "point_validity": float(valid.float().mean()),
        "direction_accuracy": float(directions.float().mean()),
    })
    return metrics


def velocity_control_predictions(*, device="cpu") -> dict[str, torch.Tensor]:
    """Analytic oracle and destructive outputs, all fed through score_velocity."""
    cases = velocity_cases(device=device)
    outputs = {"oracle": [], "zero_velocity": [], "nearest_training_speed": []}
    for case in cases:
        outputs["oracle"].append(case["targets"])
        last_x = case["start_x"] + 2 * case["speed"]
        nearest = min(TRAIN_SPEEDS, key=lambda speed: (abs(speed - case["speed"]), speed))
        for name, speed in (("zero_velocity", 0.), ("nearest_training_speed", nearest)):
            outputs[name].append(torch.cat([
                render_point(last_x + speed * (step + 1), device=device)
                for step in range(ROLLOUT_LENGTH)
            ]))
    result = {name: torch.stack(clips) for name, clips in outputs.items()}
    result["blank"] = torch.zeros_like(result["oracle"])
    result["diffuse"] = torch.full_like(result["oracle"], 1 / 64)
    return result


def velocity_controls(*, device="cpu") -> dict[str, dict[str, float]]:
    """Numerical scorer calibration; no model training or rendered evidence."""
    return {name: score_velocity(predictions)
            for name, predictions in velocity_control_predictions(device=device).items()}
