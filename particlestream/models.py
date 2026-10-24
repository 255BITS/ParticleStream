"""8x8 residual image networks and detached Writer/Reader FastWeightMemory for batch_size=1."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FastWeightMemory(nn.Module):
    """Detached Hebbian / fast-weight memory matrix M in R^{d x d} + rolling frame buffer.

    Enforces the Writer/Reader contract:
      - Only the Writer mutates M on real observations (`update_memory=True`), always detached.
      - Fake evaluations (`update_memory=False`) never mutate M.
      - The Reader queries `M.detach()` without backpropagating through the Writer's history.
    """

    def __init__(self, dim: int = 16, decay: float = 0.90, write_lr: float = 0.35, reservoir_size: int = 64):
        super().__init__()
        self.dim = int(dim)
        self.decay = float(decay)
        self.write_lr = float(write_lr)
        self.reservoir_size = int(reservoir_size)
        self.register_buffer("matrix", torch.zeros(self.dim, self.dim))
        self.register_buffer("prev_key", torch.zeros(1, self.dim))
        self.register_buffer("reservoir", torch.zeros(self.reservoir_size, 1, 8, 8))
        self.register_buffer("reservoir_count", torch.tensor(0, dtype=torch.long))
        self.key_proj = nn.Linear(64, self.dim, bias=False)
        self.val_proj = nn.Linear(64, self.dim, bias=False)

    def reset_state(self) -> None:
        self.matrix.zero_()
        self.prev_key.zero_()
        self.reservoir.zero_()
        self.reservoir_count.zero_()

    def reset(self) -> None:
        self.reset_state()

    def write(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode 1x1x8x8 real frame, return post-write temporal direction context, and update detached memory."""
        if len(frame) == 0:
            return torch.zeros(0, self.dim, device=self.matrix.device)
        if not torch.isfinite(frame).all():
            return torch.zeros(frame.shape[0], self.dim, device=self.matrix.device)
        flat = frame.reshape(frame.shape[0], -1).detach()
        with torch.no_grad():
            k = F.normalize(self.key_proj(flat), dim=-1)
            v = F.normalize(self.val_proj(flat), dim=-1)
            assoc = v.t() @ self.prev_key
            self.matrix.mul_(self.decay).add_(assoc, alpha=self.write_lr)

            diff = k - self.prev_key
            diff_norm = torch.linalg.norm(diff, dim=-1, keepdim=True)
            mask = (self.prev_key.norm() > 0) & (diff_norm > 0.35)
            context = torch.where(mask, F.normalize(diff, dim=-1) * 3.0, torch.zeros_like(diff))

            self.prev_key.copy_(k[-1:])
            slot = int(self.reservoir_count.item()) % self.reservoir_size
            self.reservoir[slot].copy_(frame[-1].detach())
            self.reservoir_count.add_(1)
        return context.detach()

    def read_context(self, frame: torch.Tensor | None = None) -> torch.Tensor:
        """Read detached temporal context vector without mutating memory."""
        with torch.no_grad():
            if frame is not None:
                if len(frame) == 0:
                    return torch.zeros(0, self.dim, device=self.matrix.device)
                flat = frame.reshape(frame.shape[0], -1).detach()
                q = F.normalize(self.key_proj(flat), dim=-1)
                diff = q - self.prev_key
                diff_norm = torch.linalg.norm(diff, dim=-1, keepdim=True)
                mask = (self.prev_key.norm() > 0) & (diff_norm > 0.35)
                context = torch.where(mask, F.normalize(diff, dim=-1) * 3.0, torch.zeros_like(diff))
                return context.detach()
            return torch.zeros(1, self.dim, device=self.matrix.device)

    def sample_reservoir(self, generator: torch.Generator) -> torch.Tensor | None:
        """Draw 1 historical frame from the bounded FIFO reservoir if populated."""
        filled = min(int(self.reservoir_count.item()), self.reservoir_size)
        if filled <= 0:
            return None
        idx = int(torch.randint(filled, (1,), device=self.reservoir.device, generator=generator).item())
        return self.reservoir[idx : idx + 1].detach().clone()


class StreamImageGenerator(nn.Module):
    """Sample-independent 1x8x8 residual-upsample generator (no BatchNorm2d)."""

    def __init__(self, z_dim: int = 8, width: int = 16, context_dim: int = 0):
        super().__init__()
        self.z_dim = int(z_dim)
        self.width = int(width)
        self.context_dim = int(context_dim)
        self.fc = nn.Linear(self.z_dim + self.context_dim, self.width * 4)
        self.conv1 = nn.Conv2d(self.width, self.width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.width, self.width, kernel_size=3, padding=1)
        self.head = nn.Conv2d(self.width, 1, kernel_size=3, padding=1)
        self.register_buffer("active_context", torch.zeros(1, max(1, self.context_dim)))

    def set_context(self, context: torch.Tensor) -> None:
        if self.context_dim > 0:
            self.active_context.copy_(context.detach()[:1, : self.context_dim])

    def forward(self, z: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if self.context_dim > 0:
            ctx = context if context is not None else self.active_context.expand(len(z), -1)
            inp = torch.cat([z, ctx.detach()], dim=-1)
        else:
            inp = z
        h = F.leaky_relu(self.fc(inp), 0.2).reshape(-1, self.width, 2, 2)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = h + F.leaky_relu(self.conv1(h), 0.2)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = h + F.leaky_relu(self.conv2(h), 0.2)
        return torch.sigmoid(self.head(h))


class StreamImageCritic(nn.Module):
    """Sample-independent 1x8x8 convolutional critic (no BatchNorm2d)."""

    def __init__(self, width: int = 16, context_dim: int = 0):
        super().__init__()
        w = max(12, int(width))
        self.context_dim = int(context_dim)
        self.conv1 = nn.Conv2d(1, w, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(w, 2 * w, kernel_size=3, stride=2, padding=1)
        if self.context_dim > 0:
            self.fc1 = nn.Linear(8 * w + self.context_dim, 2 * w)
            self.fc2 = nn.Linear(2 * w, 1)
        else:
            self.fc = nn.Linear(8 * w, 1)
        self.register_buffer("active_context", torch.zeros(1, max(1, self.context_dim)))

    def set_context(self, context: torch.Tensor) -> None:
        if self.context_dim > 0:
            self.active_context.copy_(context.detach()[:1, : self.context_dim])

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if len(x) == 0:
            return torch.zeros(0, 1, device=x.device, dtype=x.dtype)
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = torch.flatten(F.leaky_relu(self.conv2(h), 0.2), start_dim=1)
        if self.context_dim > 0:
            if context is not None:
                ctx = context.expand(len(x), -1) if (context.shape[0] == 1 and len(x) > 1) else context
            else:
                ctx = self.active_context.expand(len(x), -1)
            h = F.leaky_relu(self.fc1(torch.cat([h, ctx.detach()], dim=-1)), 0.2)
            return self.fc2(h)
        return self.fc(h)
