"""Small Tier 1 streams that emit one chronological observation at a time.

Metadata is for the evaluator and must never be passed into the learner. Clean
histories/templates are evaluation fixtures, not training inputs or replay.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .streams import bars4_templates


def _noisy_frame(base: torch.Tensor, generator: torch.Generator, noise_std: float) -> torch.Tensor:
    with torch.no_grad():
        noise = torch.randn(base.shape, device=base.device, generator=generator)
        return (base + float(noise_std) * noise).clamp(0.0, 1.0).detach()


class VariableDwellBarsStream:
    """Rotate dwell lengths 4, 8, 16, 32 across modes in a balanced 240-frame cycle."""

    def __init__(self, noise_std: float = 0.01, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.noise_std = float(noise_std)
        self.templates = bars4_templates(self.device)
        self.cursor = 0
        dwells = (4, 8, 16, 32)
        self.blocks = tuple((mode, dwells[(mode + rotation) % 4])
                            for rotation in range(4) for mode in range(4))
        self.period = sum(dwell for _, dwell in self.blocks)

    def next_frame(self, generator: torch.Generator) -> tuple[torch.Tensor, dict]:
        offset = self.cursor % self.period
        for block, (mode, dwell) in enumerate(self.blocks):
            if offset < dwell:
                break
            offset -= dwell
        self.cursor += 1
        frame = _noisy_frame(self.templates[mode:mode + 1], generator, self.noise_std)
        return frame, {"mode": mode, "dwell": dwell, "block": block}


class RareModeBarsStream:
    """Three common modes and one rare mode, with exact 5:5:5:1 exposure per 16 frames."""

    target_probs = (5 / 16, 5 / 16, 5 / 16, 1 / 16)

    def __init__(self, noise_std: float = 0.01, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.noise_std = float(noise_std)
        self.templates = bars4_templates(self.device)
        self.cycle = (0, 1, 2) * 5 + (3,)
        self.period = len(self.cycle)
        self.cursor = 0

    def next_frame(self, generator: torch.Generator) -> tuple[torch.Tensor, dict]:
        mode = self.cycle[self.cursor % self.period]
        self.cursor += 1
        return _noisy_frame(self.templates[mode:mode + 1], generator, self.noise_std), {"mode": mode}


def delayed_cue_templates(device: str | torch.device = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return left/right cues, their shared distractor, and left/right outcomes."""
    cues = torch.full((2, 1, 8, 8), 0.1, device=device, dtype=torch.float32)
    cues[0, 0, 0:2, 0:2] = 0.9
    cues[1, 0, 0:2, 6:8] = 0.9
    distractor = torch.full((1, 1, 8, 8), 0.1, device=device, dtype=torch.float32)
    outcomes = torch.full((2, 1, 8, 8), 0.1, device=device, dtype=torch.float32)
    outcomes[0, 0, 5:8, 0:3] = 0.9
    outcomes[1, 0, 5:8, 5:8] = 0.9
    return cues.detach(), distractor.detach(), outcomes.detach()


class DelayedCueStream:
    """Cue -> four identical distractors -> outcome, emitting one frame per call.

    Both cues appear once in each pair of episodes, in an independently shuffled
    order drawn from the supplied data RNG. Distractors are exactly identical,
    including across cue values: noise cannot carry a hidden cue signal.
    """

    delay = 4
    episode_length = 6

    def __init__(self, noise_std: float = 0.01, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.noise_std = float(noise_std)
        self.cues, self.distractor, self.outcomes = delayed_cue_templates(self.device)
        self.cursor = 0
        self._cue_order: list[int] = []
        self._cue = 0

    def clean_history(self, cue: int) -> torch.Tensor:
        """Five observed frames, ending immediately before the outcome arrives."""
        if cue not in (0, 1):
            raise ValueError("cue must be 0 (left) or 1 (right)")
        return torch.cat((self.cues[cue:cue + 1], self.distractor.repeat(self.delay, 1, 1, 1))).detach()

    def next_frame(self, generator: torch.Generator) -> tuple[torch.Tensor, dict]:
        phase = self.cursor % self.episode_length
        episode = self.cursor // self.episode_length
        if phase == 0:
            if episode % 2 == 0:
                self._cue_order = torch.randperm(2, device=self.device, generator=generator).tolist()
            self._cue = self._cue_order[episode % 2]
            frame = _noisy_frame(self.cues[self._cue:self._cue + 1], generator, self.noise_std)
            phase_name = "cue"
        elif phase <= self.delay:
            frame = self.distractor.clone()
            phase_name = "delay"
        else:
            frame = _noisy_frame(self.outcomes[self._cue:self._cue + 1], generator, self.noise_std)
            phase_name = "outcome"
        self.cursor += 1
        return frame, {"phase": phase_name, "cue": self._cue, "episode": episode}


def persistent_fork_templates(device: str | torch.device = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Return a shared stem and two distinct three-step paths, (2,3,1,8,8)."""
    stem = torch.full((1, 1, 8, 8), 0.1, device=device, dtype=torch.float32)
    stem[0, 0, 0:2, 3:5] = 0.9
    paths = torch.full((2, 3, 1, 8, 8), 0.1, device=device, dtype=torch.float32)
    for depth in range(3):
        row = 2 + 2 * depth
        paths[0, depth, 0, row:row + 2, 1:3] = 0.9
        paths[1, depth, 0, row:row + 2, 5:7] = 0.9
    return stem.detach(), paths.detach()


class PersistentForkStream:
    """Shared stem -> three frames on an independent fair branch per episode.

    The branch is sampled only when its first frame arrives. Stem metadata has
    no future branch, and earlier episodes do not constrain the next draw.
    """

    horizon = 3
    episode_length = 4

    def __init__(self, noise_std: float = 0.01, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.noise_std = float(noise_std)
        self.stem, self.branch_paths = persistent_fork_templates(self.device)
        self.cursor = 0
        self._branch: int | None = None

    def next_frame(self, generator: torch.Generator) -> tuple[torch.Tensor, dict]:
        phase = self.cursor % self.episode_length
        episode = self.cursor // self.episode_length
        if phase == 0:
            self._branch = None
            base = self.stem
        else:
            if phase == 1:
                self._branch = int(torch.randint(2, (), device=self.device, generator=generator).item())
            base = self.branch_paths[self._branch, phase - 1:phase]
        self.cursor += 1
        return _noisy_frame(base, generator, self.noise_std), {
            "phase": "stem" if phase == 0 else "branch",
            "depth": phase,
            "branch": self._branch,
            "episode": episode,
        }


def _template_counts(samples: torch.Tensor, templates: torch.Tensor, quality_rmse: float):
    """Keep nonfinite samples in the denominator as invalid probability mass."""
    if (not isinstance(samples, torch.Tensor) or not isinstance(templates, torch.Tensor)
            or samples.ndim != 4 or templates.ndim != 4 or len(samples) == 0
            or len(templates) == 0 or not samples.is_floating_point()
            or not templates.is_floating_point() or tuple(samples.shape[1:]) != (1, 8, 8)
            or tuple(templates.shape[1:]) != (1, 8, 8)
            or not bool(torch.isfinite(templates).all())):
        return None
    with torch.no_grad():
        templates = templates.to(device=samples.device, dtype=samples.dtype)
        rmses = (samples[:, None] - templates[None]).square().mean(dim=(2, 3, 4)).sqrt()
        finite = torch.isfinite(rmses).all(dim=1)
        safe_rmses = torch.nan_to_num(rmses, nan=1e6, posinf=1e6, neginf=1e6)
        best, nearest = safe_rmses.min(dim=1)
        valid = finite & (best <= quality_rmse)
        counts = torch.stack([((nearest == i) & valid).sum() for i in range(len(templates))])
        return counts, nearest, valid, best


def score_rare_mode_bars(samples: torch.Tensor, templates: torch.Tensor, *, quality_rmse: float = 0.12) -> dict[str, float]:
    """Distribution TV includes an invalid bin; support means >=1 valid sample per mode."""
    scored = _template_counts(samples, templates, quality_rmse)
    if scored is None or len(templates) != 4:
        return {"hq": 0.0, "support_modes": 0.0, "rare_mass": 0.0,
                "finite_template_tv": 1.0, "mean_rmse": 1e6}
    counts, _, valid, best = scored
    probs = counts.float() / len(samples)
    target = torch.tensor(RareModeBarsStream.target_probs, device=samples.device)
    invalid_mass = (~valid).float().mean()
    tv = 0.5 * ((probs - target).abs().sum() + invalid_mass)
    return {
        "hq": round(float(valid.float().mean()), 6),
        "support_modes": float((counts > 0).sum()),
        "rare_mass": round(float(probs[3]), 6),
        "finite_template_tv": round(float(tv), 6),
        "mean_rmse": round(float(best.mean()), 6),
    }


def score_delayed_cue(predicted_left: torch.Tensor, predicted_right: torch.Tensor,
                      outcomes: torch.Tensor) -> dict[str, float]:
    """Balance both cue histories, so an outcome independent of cue scores at most 0.5."""
    if (not isinstance(outcomes, torch.Tensor) or tuple(outcomes.shape) != (2, 1, 8, 8)
            or not bool(torch.isfinite(outcomes).all())):
        return {"prediction_rmse": 1e6, "outcome_accuracy": 0.0}
    errors, accuracies = [], []
    for cue, samples in enumerate((predicted_left, predicted_right)):
        scored = _template_counts(samples, outcomes, quality_rmse=float("inf"))
        if scored is None:
            return {"prediction_rmse": 1e6, "outcome_accuracy": 0.0}
        _, _, finite, _ = scored
        with torch.no_grad():
            targets = outcomes.to(samples.device)
            correct = (samples - targets[cue:cue + 1]).square().mean((1, 2, 3)).sqrt()
            wrong = (samples - targets[1 - cue:2 - cue]).square().mean((1, 2, 3)).sqrt()
            errors.append(torch.nan_to_num(correct, nan=1e6, posinf=1e6, neginf=1e6).mean())
            accuracies.append(((correct + 1e-6 < wrong) & finite).float().mean())
    return {"prediction_rmse": round(float(torch.stack(errors).mean()), 6),
            "outcome_accuracy": round(float(torch.stack(accuracies).mean()), 6)}


def score_persistent_fork(trajectories: torch.Tensor, branch_paths: torch.Tensor,
                          *, quality_rmse: float = 0.12) -> dict[str, float]:
    """Require both correct marginals and an all-valid, unchanging branch per particle.

    Inputs are generated rollouts (N,3,1,8,8) and clean branches (2,3,1,8,8).
    Aggregate hq/modes are the minimum across depths and TV is the maximum.
    Invalid images count against both marginal quality and trajectory consistency.
    """
    failure = {"hq": 0.0, "within_state_modes": 0.0, "branch_tv": 1.0,
               "trajectory_consistency": 0.0, "mean_rmse": 1e6}
    if (not isinstance(trajectories, torch.Tensor) or not isinstance(branch_paths, torch.Tensor)
            or trajectories.ndim != 5 or tuple(trajectories.shape[1:]) != (3, 1, 8, 8)
            or len(trajectories) == 0 or tuple(branch_paths.shape) != (2, 3, 1, 8, 8)):
        return failure
    metrics, assignments, validity, errors = {}, [], [], []
    for depth in range(3):
        scored = _template_counts(trajectories[:, depth], branch_paths[:, depth], quality_rmse)
        if scored is None:
            return failure
        counts, nearest, valid, best = scored
        valid_count = counts.sum()
        probs = counts.float() / len(trajectories)
        invalid_mass = (~valid).float().mean()
        modes = ((counts.float() / valid_count.clamp_min(1)) >= 0.25).sum()
        metrics[f"depth_{depth + 1}_hq"] = round(float(valid.float().mean()), 6)
        metrics[f"depth_{depth + 1}_modes"] = float(modes)
        metrics[f"depth_{depth + 1}_tv"] = round(float(0.5 * ((probs - 0.5).abs().sum() + invalid_mass)), 6)
        assignments.append(nearest)
        validity.append(valid)
        errors.append(best.mean())
    branch_ids = torch.stack(assignments, dim=1)
    all_valid = torch.stack(validity, dim=1).all(dim=1)
    consistent = all_valid & (branch_ids == branch_ids[:, :1]).all(dim=1)
    metrics.update(
        hq=min(metrics[f"depth_{d}_hq"] for d in (1, 2, 3)),
        within_state_modes=min(metrics[f"depth_{d}_modes"] for d in (1, 2, 3)),
        branch_tv=max(metrics[f"depth_{d}_tv"] for d in (1, 2, 3)),
        trajectory_consistency=round(float(consistent.float().mean()), 6),
        mean_rmse=round(float(torch.stack(errors).mean()), 6),
    )
    return metrics


CIRCLE_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (0, 3),  # 0: N  (12:00)
    (1, 5),  # 1: NE (01:30)
    (3, 6),  # 2: E  (03:00)
    (5, 5),  # 3: SE (04:30)
    (6, 3),  # 4: S  (06:00)
    (5, 1),  # 5: SW (07:30)
    (3, 0),  # 6: W  (09:00)
    (1, 1),  # 7: NW (10:30)
)


def circle_orbit_templates(device: str | torch.device = "cpu") -> torch.Tensor:
    """Return 8 clockwise 8x8 circular-orbit point templates of shape (8, 1, 8, 8).

    Each waypoint is a disjoint 2x2 bright patch (0.9) on a 0.1 background at 45-degree
    intervals around the center (3.5, 3.5) with radius ~2.5 pixels.
    """
    templates = torch.full((len(CIRCLE_WAYPOINTS), 1, 8, 8), 0.1, device=device, dtype=torch.float32)
    for idx, (row, col) in enumerate(CIRCLE_WAYPOINTS):
        templates[idx, 0, row:row + 2, col:col + 2] = 0.9
    return templates.detach()


def extract_circle_coords(images: torch.Tensor, *, background: float = 0.1) -> torch.Tensor:
    """Extract normalized 2D (x, y) point coordinates in [-1, 1]^2 from 8x8 frames.

    Uses squared excess intensity above `background` so soft sigmoid background
    residuals do not pull the extracted point toward the canvas center. Blank or
    sub-background frames fall back to uniform weights, yielding the canvas center (0, 0).
    """
    if not isinstance(images, torch.Tensor) or images.ndim < 2 or tuple(images.shape[-2:]) != (8, 8):
        raise ValueError("images must be a Tensor ending with shape (8, 8)")
    frames = images[..., 0, :, :] if images.ndim >= 3 and images.shape[-3] == 1 else images
    if not frames.is_floating_point():
        frames = frames.float()
    weights = (frames - float(background)).clamp_min(0.0).square()
    total = weights.sum(dim=(-2, -1), keepdim=True)
    probs = torch.where(
        total > 1e-6,
        weights / total.clamp_min(1e-6),
        torch.full_like(weights, 1.0 / 64.0),
    )
    coords = torch.arange(8, device=frames.device, dtype=frames.dtype)
    col_exp = (probs.sum(dim=-2) * coords).sum(dim=-1)
    row_exp = (probs.sum(dim=-1) * coords).sum(dim=-1)
    x = (col_exp - 3.5) / 3.0
    y = (3.5 - row_exp) / 3.0
    return torch.stack((x, y), dim=-1)


def circle_orbit_coords(device: str | torch.device = "cpu") -> torch.Tensor:
    """Return the 8 ground-truth normalized (x, y) circle coordinates of shape (8, 2)."""
    return extract_circle_coords(circle_orbit_templates(device)).detach()


class CircleOrbitStream:
    """Point particle orbiting 8 circular waypoints in clockwise order, one frame per step."""

    period = len(CIRCLE_WAYPOINTS)

    def __init__(self, noise_std: float = 0.01, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.noise_std = float(noise_std)
        self.templates = circle_orbit_templates(self.device)
        self.coords = extract_circle_coords(self.templates)
        self.cursor = 0

    def clean_warmup(self, start_phase: int = 0) -> torch.Tensor:
        """Two preceding waypoints before `start_phase` to establish orbital position and velocity."""
        p0 = (int(start_phase) - 2) % self.period
        p1 = (int(start_phase) - 1) % self.period
        return torch.stack((self.templates[p0], self.templates[p1])).detach()

    def next_frame(self, generator: torch.Generator) -> tuple[torch.Tensor, dict]:
        phase = self.cursor % self.period
        orbit = self.cursor // self.period
        self.cursor += 1
        frame = _noisy_frame(self.templates[phase:phase + 1], generator, self.noise_std)
        xy = self.coords[phase]
        return frame, {
            "phase": phase,
            "orbit": orbit,
            "x": round(float(xy[0].item()), 6),
            "y": round(float(xy[1].item()), 6),
        }


def score_circle_orbit(
    trajectories: torch.Tensor,
    templates: torch.Tensor,
    *,
    quality_rmse: float = 0.12,
    quality_xy: float = 0.15,
) -> dict[str, float]:
    """Score closed-loop 8-step circular-orbit rollouts (N, 8, 1, 8, 8) against (8, 1, 8, 8).

    Every particle must step through all 8 clockwise circular waypoints (0..7) without
    teacher forcing. `orbit_completion` is the fraction of particles whose entire 8-step
    closed-loop rollout hits every waypoint in order with RMSE <= `quality_rmse` and
    normalized coordinate error <= `quality_xy`.
    """
    failure = {
        "orbit_rmse": 1e6,
        "xy_rmse": 1e6,
        "step_accuracy": 0.0,
        "orbit_completion": 0.0,
        "waypoint_modes": 0.0,
    }
    if (
        not isinstance(trajectories, torch.Tensor)
        or not isinstance(templates, torch.Tensor)
        or trajectories.ndim != 5
        or tuple(trajectories.shape[1:]) != (8, 1, 8, 8)
        or len(trajectories) == 0
        or tuple(templates.shape) != (8, 1, 8, 8)
        or not trajectories.is_floating_point()
        or not templates.is_floating_point()
        or not bool(torch.isfinite(templates).all())
    ):
        return failure
    with torch.no_grad():
        templates = templates.to(device=trajectories.device, dtype=trajectories.dtype)
        rmses = (trajectories[:, :, None] - templates[None, None]).square().mean(dim=(3, 4, 5)).sqrt()
        finite = torch.isfinite(rmses).all(dim=2)
        safe_rmses = torch.nan_to_num(rmses, nan=1e6, posinf=1e6, neginf=1e6)
        step_idx = torch.arange(8, device=trajectories.device)
        target_rmses = safe_rmses[:, step_idx, step_idx]
        other_rmses = safe_rmses.clone()
        other_rmses[:, step_idx, step_idx] = float("inf")
        strictly_nearest = (target_rmses + 1e-6) < other_rmses.min(dim=2).values

        pred_xy = extract_circle_coords(trajectories)
        target_xy = extract_circle_coords(templates)[None, :, :]
        xy_err = (pred_xy - target_xy).square().sum(dim=-1).sqrt()
        xy_err = torch.where(
            finite,
            torch.nan_to_num(xy_err, nan=1e6, posinf=1e6, neginf=1e6),
            torch.full_like(xy_err, 1e6),
        )
        step_valid = finite & strictly_nearest & (target_rmses <= quality_rmse) & (xy_err <= quality_xy)
        full_orbit = step_valid.all(dim=1)
        waypoint_modes = float((step_valid.sum(dim=0) > 0).sum().item())
        return {
            "orbit_rmse": round(float(target_rmses.mean().item()), 6),
            "xy_rmse": round(float(xy_err.mean().item()), 6),
            "step_accuracy": round(float(step_valid.float().mean().item()), 6),
            "orbit_completion": round(float(full_orbit.float().mean().item()), 6),
            "waypoint_modes": waypoint_modes,
        }


def _draw_pixel_patch(
    draw: ImageDraw.ImageDraw,
    image_2d: np.ndarray,
    x0: int,
    y0: int,
    cell: int,
    *,
    border_color: tuple[int, int, int] = (55, 65, 81),
    accent_xy: tuple[float, float] | None = None,
    accent_color: tuple[int, int, int] = (56, 189, 248),
) -> None:
    arr = np.nan_to_num(np.asarray(image_2d, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    h, w = arr.shape[-2], arr.shape[-1]
    for r in range(h):
        for c in range(w):
            v = float(arr[r, c])
            # Deep navy background to glowing cyan-gold highlight
            red = int(14 + 235 * (v ** 1.35))
            green = int(20 + 215 * (v ** 1.10))
            blue = int(34 + 190 * (v ** 0.85))
            rx0 = x0 + c * cell
            ry0 = y0 + r * cell
            draw.rectangle([rx0, ry0, rx0 + cell - 1, ry0 + cell - 1], fill=(red, green, blue))
    draw.rectangle([x0 - 1, y0 - 1, x0 + w * cell, y0 + h * cell], outline=border_color, width=1)
    if accent_xy is not None:
        col_f = 3.5 + 3.0 * float(accent_xy[0])
        row_f = 3.5 - 3.0 * float(accent_xy[1])
        cx = int(round(x0 + (col_f + 0.5) * cell))
        cy = int(round(y0 + (row_f + 0.5) * cell))
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], outline=accent_color, width=2)


def render_circle_orbit_gif(
    frames: list[dict],
    output_path: Path | str,
    *,
    task_id: str = "seq_img_circle_orbit",
    candidate_id: str = "online-feature-prior-bce",
    verdict: str = "PASS",
    substeps_per_checkpoint: int = 2,
    duration_ms: int = 110,
) -> dict:
    """Render a high-clarity circular-orbit training animation from actual checkpoint frames."""
    if not frames:
        raise ValueError("render_circle_orbit_gif requires at least one frame")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    target_templates = circle_orbit_templates("cpu")
    target_xy = extract_circle_coords(target_templates).numpy()  # (8, 2)
    waypoint_labels = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

    all_steps = [int(f["step"]) for f in frames]
    max_step = max(all_steps) if all_steps else 480
    substeps = max(1, int(substeps_per_checkpoint))

    pil_frames: list[Image.Image] = []
    width, height = 920, 400

    for idx, frame in enumerate(frames):
        step = int(frame["step"])
        passed = bool(frame.get("passed", False))
        metrics = frame.get("metrics", {})
        sample_arr = np.asarray(frame["sample_images"], dtype=np.float32)
        if sample_arr.ndim == 3:
            sample_arr = sample_arr[:, None, :, :]
        # Reconstruct 8-step closed-loop trajectories from sample_images (32, 1, 8, 8) -> (4, 8, 1, 8, 8)
        if len(sample_arr) >= 8 and len(sample_arr) % 8 == 0:
            rollouts = torch.from_numpy(sample_arr).reshape(-1, 8, 1, 8, 8)
            mean_rollout = rollouts.mean(dim=0)  # (8, 1, 8, 8)
            particle_xy = extract_circle_coords(rollouts).numpy()  # (K, 8, 2)
            pred_xy = particle_xy.mean(axis=0)  # (8, 2)
        else:
            t_tensor = torch.from_numpy(sample_arr[:8])
            if len(t_tensor) < 8:
                repeats = (8 + len(t_tensor) - 1) // max(1, len(t_tensor))
                t_tensor = t_tensor.repeat(repeats, 1, 1, 1)[:8]
            mean_rollout = t_tensor
            pred_xy = extract_circle_coords(mean_rollout).numpy()
            particle_xy = pred_xy[None, :, :]

        with torch.no_grad():
            step_rmses = (
                (mean_rollout - target_templates).square().mean(dim=(1, 2, 3)).sqrt().numpy()
            )
            step_xy_err = np.linalg.norm(pred_xy - target_xy, axis=-1)
            _, nearest_idx = (
                (mean_rollout[:, None] - target_templates[None])
                .square()
                .mean(dim=(2, 3, 4))
                .sqrt()
                .min(dim=1)
            )
            step_hits = (nearest_idx.numpy() == np.arange(8)) & (step_rmses <= 0.12) & (step_xy_err <= 0.15)

        for sub in range(substeps):
            active_phase = (idx * substeps + sub) % 8
            canvas = Image.new("RGB", (width, height), (13, 17, 23))
            draw = ImageDraw.Draw(canvas)

            # Top header bar
            draw.rectangle([0, 0, width, 40], fill=(22, 27, 34))
            draw.line([0, 40, width, 40], fill=(48, 54, 61), width=1)
            draw.text(
                (16, 13),
                f"ParticleStream  |  {task_id}  |  candidate: {candidate_id}",
                fill=(230, 237, 243),
                font=font,
            )
            verdict_bg = (35, 134, 54) if verdict == "PASS" else (218, 54, 51)
            step_bg = (31, 111, 235) if passed else (110, 118, 129)
            draw.rectangle([width - 250, 9, width - 126, 31], fill=step_bg)
            draw.text((width - 240, 14), f"step {step:3d}/{max_step}", fill=(255, 255, 255), font=font)
            draw.rectangle([width - 118, 9, width - 16, 31], fill=verdict_bg)
            draw.text((width - 106, 14), f"verdict: {verdict}", fill=(255, 255, 255), font=font)

            # Panel 1: 2D Circular Orbit Plane (left)
            p1_x0, p1_y0, p1_x1, p1_y1 = 14, 52, 314, 386
            draw.rectangle([p1_x0, p1_y0, p1_x1, p1_y1], fill=(18, 23, 31), outline=(48, 54, 61))
            draw.text((p1_x0 + 12, p1_y0 + 8), "2D Circular Orbit Plane (x, y)", fill=(201, 209, 217), font=font)
            draw.text((p1_x0 + 12, p1_y0 + 22), "Closed-loop 8-step generator trajectory", fill=(125, 133, 144), font=font)

            cx, cy = 164, 222
            radius = 90
            ring_r = int(round(radius * 0.972))

            # Subtle coordinate axes & ideal orbit ring
            draw.line([cx - radius - 16, cy, cx + radius + 16, cy], fill=(33, 38, 45), width=1)
            draw.line([cx, cy - radius - 16, cx, cy + radius + 16], fill=(33, 38, 45), width=1)
            draw.ellipse(
                [cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r],
                outline=(56, 70, 92),
                width=2,
            )

            def _to_screen(pt_xy: np.ndarray | tuple[float, float]) -> tuple[int, int]:
                sx = int(round(cx + float(pt_xy[0]) * radius))
                sy = int(round(cy - float(pt_xy[1]) * radius))
                return sx, sy

            # Draw target waypoints on the circle
            for w_idx in range(8):
                tx, ty = _to_screen(target_xy[w_idx])
                is_active = w_idx == active_phase
                r_node = 6 if is_active else 4
                fill_col = (245, 158, 11) if is_active else (48, 58, 74)
                out_col = (253, 224, 71) if is_active else (110, 125, 148)
                draw.ellipse([tx - r_node, ty - r_node, tx + r_node, ty + r_node], fill=fill_col, outline=out_col, width=2)
                lx = int(round(cx + float(target_xy[w_idx, 0]) * (radius + 17))) - 5
                ly = int(round(cy - float(target_xy[w_idx, 1]) * (radius + 17))) - 5
                draw.text((lx, ly), waypoint_labels[w_idx], fill=(139, 148, 158), font=font)

            # Draw individual particle trajectories lightly for ensemble spread
            for p_idx in range(min(4, len(particle_xy))):
                pts = [_to_screen(particle_xy[p_idx, w]) for w in range(8)]
                pts.append(pts[0])
                draw.line(pts, fill=(38, 86, 115), width=1)

            # Draw mean predicted closed-loop polygon
            pred_screen = [_to_screen(pred_xy[w]) for w in range(8)]
            closed_poly = pred_screen + [pred_screen[0]]
            poly_color = (52, 211, 153) if passed else (56, 189, 248)
            draw.line(closed_poly, fill=poly_color, width=2)
            for w_idx, (px, py) in enumerate(pred_screen):
                hit = bool(step_hits[w_idx])
                node_fill = (16, 185, 129) if hit else (248, 113, 113)
                draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=node_fill, outline=(230, 237, 243), width=1)

            # Draw active orbiting particle & comet tail along the predicted trajectory
            for tail in (3, 2, 1):
                t_prev = (active_phase - tail) % 8
                t_curr = (active_phase - tail + 1) % 8
                c_alpha = 90 + (3 - tail) * 50
                draw.line(
                    [pred_screen[t_prev], pred_screen[t_curr]],
                    fill=(c_alpha, 230, 255),
                    width=4 - tail,
                )
            ax, ay = pred_screen[active_phase]
            draw.ellipse([ax - 8, ay - 8, ax + 8, ay + 8], fill=(56, 189, 248), outline=(255, 255, 255), width=2)

            # Legend at bottom of Panel 1
            draw.ellipse([p1_x0 + 14, p1_y1 - 22, p1_x0 + 22, p1_y1 - 14], fill=(245, 158, 11), outline=(253, 224, 71))
            draw.text((p1_x0 + 28, p1_y1 - 24), "Target waypoint", fill=(139, 148, 158), font=font)
            draw.ellipse([p1_x0 + 146, p1_y1 - 22, p1_x0 + 154, p1_y1 - 14], fill=(56, 189, 248), outline=(255, 255, 255))
            draw.text((p1_x0 + 160, p1_y1 - 24), "Generator particle", fill=(139, 148, 158), font=font)

            # Panel 2: 8x8 Canvas & 8-Step Closed-Loop Rollout (center)
            p2_x0, p2_y0, p2_x1, p2_y1 = 326, 52, 616, 386
            draw.rectangle([p2_x0, p2_y0, p2_x1, p2_y1], fill=(18, 23, 31), outline=(48, 54, 61))
            draw.text((p2_x0 + 12, p2_y0 + 8), "8x8 Point Stream & Closed-Loop Rollout", fill=(201, 209, 217), font=font)
            draw.text(
                (p2_x0 + 12, p2_y0 + 22),
                f"Active waypoint {active_phase + 1}/8 ({waypoint_labels[active_phase]})",
                fill=(125, 133, 144),
                font=font,
            )

            # Top side-by-side 8x8 zoomed patches
            draw.text((p2_x0 + 20, p2_y0 + 44), "Target (x_t, y_t)", fill=(245, 158, 11), font=font)
            _draw_pixel_patch(
                draw,
                target_templates[active_phase, 0].numpy(),
                p2_x0 + 20,
                p2_y0 + 60,
                cell=13,
                border_color=(245, 158, 11),
                accent_xy=tuple(target_xy[active_phase]),
                accent_color=(253, 224, 71),
            )

            draw.text((p2_x0 + 162, p2_y0 + 44), "Predicted G(z, c_{t-1})", fill=(56, 189, 248), font=font)
            _draw_pixel_patch(
                draw,
                mean_rollout[active_phase, 0].numpy(),
                p2_x0 + 162,
                p2_y0 + 60,
                cell=13,
                border_color=(52, 211, 153) if step_hits[active_phase] else (248, 113, 113),
                accent_xy=tuple(pred_xy[active_phase]),
                accent_color=(56, 189, 248),
            )

            # 8-Step Closed-Loop Rollout Strip (t = 1..8)
            strip_y0 = p2_y0 + 184
            draw.text(
                (p2_x0 + 14, strip_y0),
                "8-Step Self-Fed Rollout (no teacher forcing):",
                fill=(201, 209, 217),
                font=font,
            )
            draw.text((p2_x0 + 14, strip_y0 + 22), "Target", fill=(139, 148, 158), font=font)
            draw.text((p2_x0 + 14, strip_y0 + 66), "Pred", fill=(139, 148, 158), font=font)

            for w_idx in range(8):
                sx = p2_x0 + 58 + w_idx * 28
                draw.text((sx + 4, strip_y0 + 14), waypoint_labels[w_idx], fill=(125, 133, 144), font=font)
                _draw_pixel_patch(
                    draw,
                    target_templates[w_idx, 0].numpy(),
                    sx,
                    strip_y0 + 28,
                    cell=3,
                    border_color=(245, 158, 11) if w_idx == active_phase else (55, 65, 81),
                )
                hit = bool(step_hits[w_idx])
                _draw_pixel_patch(
                    draw,
                    mean_rollout[w_idx, 0].numpy(),
                    sx,
                    strip_y0 + 66,
                    cell=3,
                    border_color=(52, 211, 153) if hit else (248, 113, 113),
                )

            hits_count = int(step_hits.sum())
            draw.text(
                (p2_x0 + 14, p2_y1 - 26),
                f"Closed-loop waypoints locked: {hits_count}/8   |   modes: {int(metrics.get('waypoint_modes', 0))}/8",
                fill=(52, 211, 153) if hits_count == 8 else (201, 209, 217),
                font=font,
            )

            # Panel 3: Live Online Convergence Curves (right)
            p3_x0, p3_y0, p3_x1, p3_y1 = 628, 52, 906, 386
            draw.rectangle([p3_x0, p3_y0, p3_x1, p3_y1], fill=(18, 23, 31), outline=(48, 54, 61))
            draw.text((p3_x0 + 12, p3_y0 + 8), "Online Convergence (batch_size=1)", fill=(201, 209, 217), font=font)

            def _draw_chart(
                box: tuple[int, int, int, int],
                title: str,
                series_list: list[tuple[str, list[float], tuple[int, int, int]]],
                *,
                y_min: float,
                y_max: float,
                gate_val: float,
                gate_label: str,
            ) -> None:
                cx0, cy0, cx1, cy1 = box
                draw.text((cx0, cy0 - 14), title, fill=(139, 148, 158), font=font)
                draw.rectangle([cx0, cy0, cx1, cy1], fill=(13, 17, 23), outline=(48, 54, 61))
                # Shaded terminal 5-check stability window
                if len(frames) >= 5:
                    win_start_step = int(frames[-5]["step"])
                    wx0 = int(round(cx0 + (win_start_step / max(1, max_step)) * (cx1 - cx0)))
                    draw.rectangle([wx0, cy0 + 1, cx1 - 1, cy1 - 1], fill=(19, 35, 29))
                # Gate threshold line
                gy = int(round(cy1 - ((gate_val - y_min) / max(1e-6, y_max - y_min)) * (cy1 - cy0)))
                gy = max(cy0, min(cy1, gy))
                for gx in range(cx0 + 2, cx1 - 2, 6):
                    draw.line([gx, gy, min(cx1 - 2, gx + 3), gy], fill=(110, 118, 129), width=1)
                draw.text((cx1 - 68, max(cy0 + 2, gy - 11)), gate_label, fill=(139, 148, 158), font=font)

                for _, vals, col in series_list:
                    pts = []
                    for s_i, val in zip(all_steps[: idx + 1], vals[: idx + 1]):
                        v_clamped = max(y_min, min(y_max, float(val)))
                        px = int(round(cx0 + (s_i / max(1, max_step)) * (cx1 - cx0)))
                        py = int(round(cy1 - ((v_clamped - y_min) / max(1e-6, y_max - y_min)) * (cy1 - cy0)))
                        pts.append((px, py))
                    if len(pts) >= 2:
                        draw.line(pts, fill=col, width=2)
                    if pts:
                        lx, ly = pts[-1]
                        draw.ellipse([lx - 3, ly - 3, lx + 3, ly + 3], fill=col)

            comp_vals = [float(f.get("metrics", {}).get("orbit_completion", 0.0)) for f in frames]
            acc_vals = [float(f.get("metrics", {}).get("step_accuracy", 0.0)) for f in frames]
            xy_vals = [float(f.get("metrics", {}).get("xy_rmse", 0.6)) for f in frames]
            img_vals = [float(f.get("metrics", {}).get("orbit_rmse", 0.4)) for f in frames]

            _draw_chart(
                (p3_x0 + 14, p3_y0 + 42, p3_x1 - 14, p3_y0 + 138),
                "Orbit Completion (green) & Step Acc (cyan)",
                [
                    ("step_acc", acc_vals, (56, 189, 248)),
                    ("orbit_comp", comp_vals, (52, 211, 153)),
                ],
                y_min=0.0,
                y_max=1.05,
                gate_val=0.90,
                gate_label="gate >=0.90",
            )

            _draw_chart(
                (p3_x0 + 14, p3_y0 + 164, p3_x1 - 14, p3_y0 + 260),
                "Coordinate Error xy_rmse (amber) & orbit_rmse",
                [
                    ("orbit_rmse", img_vals, (168, 85, 247)),
                    ("xy_rmse", xy_vals, (245, 158, 11)),
                ],
                y_min=0.0,
                y_max=0.65,
                gate_val=0.15,
                gate_label="gate <=0.15",
            )

            # Live metric readout cards at bottom of Panel 3
            cur_comp = float(metrics.get("orbit_completion", 0.0))
            cur_acc = float(metrics.get("step_accuracy", 0.0))
            cur_xy = float(metrics.get("xy_rmse", 0.0))
            cur_rmse = float(metrics.get("orbit_rmse", 0.0))
            draw.rectangle([p3_x0 + 14, p3_y0 + 272, p3_x1 - 14, p3_y1 - 12], fill=(13, 17, 23), outline=(48, 54, 61))
            draw.text(
                (p3_x0 + 22, p3_y0 + 279),
                f"orbit_completion: {cur_comp * 100:5.1f}%   (gate >= 90%)",
                fill=(52, 211, 153) if cur_comp >= 0.90 else (230, 237, 243),
                font=font,
            )
            draw.text(
                (p3_x0 + 22, p3_y0 + 295),
                f"step_accuracy:    {cur_acc * 100:5.1f}%   (gate >= 90%)",
                fill=(56, 189, 248) if cur_acc >= 0.90 else (230, 237, 243),
                font=font,
            )
            draw.text(
                (p3_x0 + 22, p3_y0 + 311),
                f"xy_rmse: {cur_xy:.4f} (<=0.15) | orbit_rmse: {cur_rmse:.4f}",
                fill=(201, 209, 217),
                font=font,
            )

            pil_frames.append(canvas)

    # Hold the final converged orbit for a full 8-step revolution so the loop finishes cleanly
    if len(pil_frames) >= 8:
        pil_frames.extend(pil_frames[-8:])
    elif pil_frames:
        pil_frames.extend([pil_frames[-1]] * 4)

    tmp_path = out.with_suffix(".gif.tmp")
    pil_frames[0].save(
        tmp_path,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(duration_ms),
        loop=0,
    )
    shutil.move(str(tmp_path), str(out))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    return {"path": str(out), "sha256": digest, "frames": len(pil_frames)}

