"""Small Tier 1 streams that emit one chronological observation at a time.

Metadata is for the evaluator and must never be passed into the learner. Clean
histories/templates are evaluation fixtures, not training inputs or replay.
"""
from __future__ import annotations

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
