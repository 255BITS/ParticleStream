"""Sequential batch_size=1 image streams, scorers, and oracle/destructive controls."""
from __future__ import annotations

import torch


def bars4_templates(device: str | torch.device = "cpu") -> torch.Tensor:
    """Return 4 canonical 1x8x8 bar templates (2 horizontal, 2 vertical) in [0.1, 0.9]."""
    templates = torch.full((4, 1, 8, 8), 0.1, dtype=torch.float32, device=device)
    templates[0, 0, 1:3, :] = 0.9
    templates[1, 0, 5:7, :] = 0.9
    templates[2, 0, :, 1:3] = 0.9
    templates[3, 0, :, 5:7] = 0.9
    return templates.detach()


def translating_bar_cycle(device: str | torch.device = "cpu") -> tuple[torch.Tensor, list[int]]:
    """Return 4 distinct vertical bar templates (cols 0..1, 2..3, 4..5, 6..7) and 6-step bounce cycle.

    Cycle: [0, 1, 2, 3, 2, 1]. Interior positions 1 and 2 are ambiguous from a single frame
    (moving right in 0->1->2->3 vs. moving left in 3->2->1->0) and require memory to disambiguate.
    """
    positions = torch.full((4, 1, 8, 8), 0.1, dtype=torch.float32, device=device)
    for idx, col in enumerate((0, 2, 4, 6)):
        positions[idx, 0, :, col : col + 2] = 0.9
    cycle = [0, 1, 2, 3, 2, 1]
    return positions.detach(), cycle


def fork_templates(device: str | torch.device = "cpu") -> torch.Tensor:
    """Return 3 1x8x8 templates for a bifurcating trajectory:
    0 = top-center stem, 1 = bottom-left branch, 2 = bottom-right branch.
    """
    templates = torch.full((3, 1, 8, 8), 0.1, dtype=torch.float32, device=device)
    templates[0, 0, 0:4, 3:5] = 0.9  # top-center stem
    templates[1, 0, 4:8, 0:3] = 0.9  # bottom-left branch
    templates[2, 0, 4:8, 5:8] = 0.9  # bottom-right branch
    return templates.detach()


class SequentialBarsStream:
    """Emits 1x1x8x8 frames sequentially (batch_size=1) with configurable dwell steps per mode."""

    def __init__(self, dwell: int = 1, noise_std: float = 0.01, device: str | torch.device = "cpu"):
        self.dwell = max(1, int(dwell))
        self.noise_std = float(noise_std)
        self.device = torch.device(device)
        self.templates = bars4_templates(self.device)
        self.cursor = 0

    def next_frame(self, generator: torch.Generator) -> tuple[torch.Tensor, int]:
        with torch.no_grad():
            mode = (self.cursor // self.dwell) % len(self.templates)
            self.cursor += 1
            base = self.templates[mode : mode + 1]
            noise = self.noise_std * torch.randn(base.shape, device=self.device, generator=generator)
            frame = (base + noise).clamp(0.0, 1.0).detach()
            frame.requires_grad_(False)
            return frame, mode


class TranslatingBarStream:
    """Emits 1x1x8x8 frames along the 6-step bouncing cycle [0, 1, 2, 3, 2, 1]."""

    def __init__(self, noise_std: float = 0.01, device: str | torch.device = "cpu"):
        self.noise_std = float(noise_std)
        self.device = torch.device(device)
        self.positions, self.cycle = translating_bar_cycle(self.device)
        self.cursor = 0

    def next_pair(self, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Return (x_t, x_{t+1}, pos_t, pos_{t+1}) with batch_size=1 and requires_grad=False."""
        with torch.no_grad():
            idx_t = self.cursor % len(self.cycle)
            idx_next = (self.cursor + 1) % len(self.cycle)
            self.cursor += 1
            pos_t = self.cycle[idx_t]
            pos_next = self.cycle[idx_next]
            x_t = (self.positions[pos_t : pos_t + 1] + self.noise_std * torch.randn((1, 1, 8, 8), device=self.device, generator=generator)).clamp(0.0, 1.0).detach()
            x_next = (self.positions[pos_next : pos_next + 1] + self.noise_std * torch.randn((1, 1, 8, 8), device=self.device, generator=generator)).clamp(0.0, 1.0).detach()
            x_t.requires_grad_(False)
            x_next.requires_grad_(False)
            return x_t, x_next, pos_t, pos_next


class ForkStream:
    """Alternates 2-step episodes: stem (0) -> Left branch (1) vs. stem (0) -> Right branch (2)."""

    def __init__(self, noise_std: float = 0.01, device: str | torch.device = "cpu"):
        self.noise_std = float(noise_std)
        self.device = torch.device(device)
        self.templates = fork_templates(self.device)
        self.cursor = 0

    def next_transition(self, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return (stem_frame, branch_frame, branch_id in {1, 2}) with batch_size=1."""
        with torch.no_grad():
            branch_id = 1 if (self.cursor % 2 == 0) else 2
            self.cursor += 1
            stem = (self.templates[0:1] + self.noise_std * torch.randn((1, 1, 8, 8), device=self.device, generator=generator)).clamp(0.0, 1.0).detach()
            branch = (self.templates[branch_id : branch_id + 1] + self.noise_std * torch.randn((1, 1, 8, 8), device=self.device, generator=generator)).clamp(0.0, 1.0).detach()
            stem.requires_grad_(False)
            branch.requires_grad_(False)
            return stem, branch, branch_id


def score_multimodal_bars(samples: torch.Tensor, templates: torch.Tensor, *, quality_rmse: float = 0.12) -> dict[str, float]:
    """Score generated (N, 1, 8, 8) samples against the 4 canonical bar templates."""
    with torch.no_grad():
        diffs = samples[:, None] - templates[None, :]
        rmses = diffs.pow(2).mean(dim=(2, 3, 4)).sqrt()  # (N, M)
        min_rmse, nearest = rmses.min(dim=1)
        valid = min_rmse <= quality_rmse
        hq = float(valid.float().mean().item())
        m = len(templates)
        counts = torch.zeros(m + 1, dtype=torch.float32, device=samples.device)
        for idx in range(m):
            counts[idx] = ((nearest == idx) & valid).float().sum()
        counts[m] = (~valid).float().sum()
        probs = counts / max(1, len(samples))
        target_probs = torch.tensor([1.0 / m] * m + [0.0], device=samples.device)
        finite_tv = float(0.5 * (probs - target_probs).abs().sum().item())
        valid_counts = counts[:m]
        valid_total = valid_counts.sum().clamp_min(1e-8)
        modes = int((valid_counts / valid_total >= (0.5 / m)).sum().item()) if valid.any() else 0
        return {
            "hq": round(hq, 6),
            "modes": float(modes),
            "mean_rmse": round(float(min_rmse.mean().item()), 6),
            "finite_template_tv": round(finite_tv, 6),
        }


def score_translating_next(
    predicted_right: torch.Tensor,
    predicted_left: torch.Tensor,
    positions: torch.Tensor,
) -> dict[str, float]:
    """Score 1-step predictions from interior position 1 when moving right (target=2) vs. moving left (target=0)."""
    with torch.no_grad():
        target_right = positions[2:3]
        target_left = positions[0:1]
        rmse_r = (predicted_right - target_right).pow(2).mean(dim=(1, 2, 3)).sqrt()
        rmse_l = (predicted_left - target_left).pow(2).mean(dim=(1, 2, 3)).sqrt()
        next_rmse = 0.5 * (rmse_r.mean() + rmse_l.mean())

        # Check whether each prediction is closer to the true direction target than the opposite direction target
        wrong_r = (predicted_right - target_left).pow(2).mean(dim=(1, 2, 3)).sqrt()
        wrong_l = (predicted_left - target_right).pow(2).mean(dim=(1, 2, 3)).sqrt()
        acc_r = (rmse_r < wrong_r).float().mean()
        acc_l = (rmse_l < wrong_l).float().mean()
        direction_acc = 0.5 * (acc_r + acc_l)
        return {
            "next_frame_rmse": round(float(next_rmse.item()), 6),
            "direction_acc": round(float(direction_acc.item()), 6),
        }


def score_fork_rollout(samples: torch.Tensor, templates: torch.Tensor, *, quality_rmse: float = 0.12) -> dict[str, float]:
    """Score within-state particle predictions against the 2 fork branches (templates[1:3])."""
    branches = templates[1:3]
    base = score_multimodal_bars(samples, branches, quality_rmse=quality_rmse)
    return {
        "hq": base["hq"],
        "within_state_modes": base["modes"],
        "branch_tv": base["finite_template_tv"],
        "mean_rmse": base["mean_rmse"],
    }
