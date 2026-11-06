"""Opt-in detached causal memory readers; neither reader replays observations."""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from .models import FastWeightMemory


CAUSAL_STATE_DEFAULTS = {
    "contrast_threshold": 0.03,
    "change_threshold": 0.04,
    "context_scale": 3.0,
    "association_weight": 0.125,
    "age_horizon": 8,
}


class CausalStateMemory(FastWeightMemory):
    """Current appearance, velocity, retained content and repetition age in dim entries.

    Centering pixels removes a uniform image's DC level. A low-contrast frame
    does not erase the most recent contrast-bearing observation. This is a
    generic image statistic, not a cue/template classifier. An associative
    matrix read contributes learned transition information in the same fixed
    projection coordinates. The final context entry reports how many consecutive
    nearly identical arrivals have occurred, capped by ``age_horizon``.

    All writes are detached and causal. Parameters and output dimension match
    FastWeightMemory; additional history consists only of bounded buffers. The
    inherited frame ring remains observational state and is never sampled here.
    """

    def __init__(self, dim: int = 16, decay: float = 0.90, write_lr: float = 0.35,
                 reservoir_size: int = 64, *, contrast_threshold: float = 0.03,
                 change_threshold: float = 0.04, context_scale: float = 3.0,
                 association_weight: float = 0.125, age_horizon: int = 8):
        if int(dim) < 2 or int(reservoir_size) < 1 or int(age_horizon) < 1:
            raise ValueError("causal state memory requires dim>=2, reservoir_size>=1, age_horizon>=1")
        settings = (decay, write_lr, contrast_threshold, change_threshold, context_scale, association_weight)
        if not all(math.isfinite(float(value)) for value in settings):
            raise ValueError("memory settings must be finite")
        if not 0 <= decay < 1 or write_lr < 0 or contrast_threshold < 0 or change_threshold < 0:
            raise ValueError("memory decay must be in [0,1), with nonnegative write and image thresholds")
        if context_scale <= 0 or association_weight < 0:
            raise ValueError("context scale must be positive and association weight nonnegative")
        super().__init__(dim, decay, write_lr, reservoir_size)
        self.contrast_threshold = float(contrast_threshold)
        self.change_threshold = float(change_threshold)
        self.context_scale = float(context_scale)
        self.association_weight = float(association_weight)
        self.age_horizon = int(age_horizon)
        self.register_buffer("retained_key", torch.zeros(1, self.dim))
        self.register_buffer("repetition_age", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_frame", torch.zeros(1, 1, 8, 8))
        self.register_buffer("active_context", torch.zeros(1, self.dim))

    def reset_state(self) -> None:
        super().reset_state()
        self.retained_key.zero_()
        self.repetition_age.zero_()
        self.last_frame.zero_()
        self.active_context.zero_()

    def _check_frame(self, frame: torch.Tensor) -> torch.Tensor:
        if (not isinstance(frame, torch.Tensor) or tuple(frame.shape) != (1, 1, 8, 8)
                or not frame.is_floating_point() or not bool(torch.isfinite(frame).all())):
            raise ValueError("causal memory requires one finite floating frame shaped (1,1,8,8)")
        return frame.detach()

    def _next_state(self, frame: torch.Tensor):
        """Compute a detached post-arrival state without changing a live buffer."""
        flat = frame.reshape(1, -1)
        centered = flat - flat.mean(dim=-1, keepdim=True)
        salient = bool(centered.square().mean().sqrt() >= self.contrast_threshold)
        key = F.normalize(self.key_proj(centered), dim=-1) if salient else torch.zeros_like(self.prev_key)
        value = F.normalize(self.val_proj(centered), dim=-1) if salient else torch.zeros_like(self.prev_key)
        initialized = int(self.reservoir_count.item()) > 0
        difference = key - self.prev_key if initialized else torch.zeros_like(key)
        retained = key if salient else self.retained_key
        repeated = initialized and bool((frame - self.last_frame).square().mean().sqrt() <= self.change_threshold)
        age = min(int(self.repetition_age.item()) + 1, self.age_horizon) if repeated else 0
        matrix = self.matrix * self.decay + self.write_lr * (value.t() @ self.prev_key)
        query = key if salient else retained
        recalled = torch.tanh(query @ matrix.t())
        fallback = torch.zeros_like(retained) if salient else retained
        features = key + 0.5 * difference + fallback + self.association_weight * recalled
        context = torch.zeros_like(self.active_context)
        context[:, :-1] = self.context_scale * features[:, :-1].clamp(-1.0, 1.0)
        context[:, -1] = self.context_scale * age / self.age_horizon
        return key, retained, matrix, age, context

    def write(self, frame: torch.Tensor) -> torch.Tensor:
        """Consume one observed image and return an independent detached context."""
        frame = self._check_frame(frame)
        with torch.no_grad():
            key, retained, matrix, age, context = self._next_state(frame)
            self.matrix.copy_(matrix)
            self.prev_key.copy_(key)
            self.retained_key.copy_(retained)
            self.repetition_age.fill_(age)
            self.last_frame.copy_(frame)
            self.active_context.copy_(context)
            slot = int(self.reservoir_count.item()) % self.reservoir_size
            self.reservoir[slot].copy_(frame[0])
            self.reservoir_count.add_(1)
            return self.active_context.detach().clone()

    def read_context(self, frame: torch.Tensor | None = None) -> torch.Tensor:
        """Read current state, or preview one observed query without writing history."""
        with torch.no_grad():
            if frame is None:
                return self.active_context.detach().clone()
            return self._next_state(self._check_frame(frame))[-1].detach().clone()


def make_stream_memory(policy: Mapping) -> FastWeightMemory | None:
    """Build the recipe-selected reader; unchanged historical policies use velocity."""
    kind = policy.get("kind", "none")
    if kind == "none":
        return None
    if kind != "fast_weight":
        raise ValueError(f"unknown memory kind: {kind}")
    common = {name: policy[name] for name in ("dim", "decay", "write_lr", "reservoir_size") if name in policy}
    reader = policy.get("reader", "velocity")
    if reader == "velocity":
        return FastWeightMemory(**common)
    if reader == "causal_state":
        settings = {name: policy.get(name, default) for name, default in CAUSAL_STATE_DEFAULTS.items()}
        return CausalStateMemory(**common, **settings)
    raise ValueError(f"unknown memory reader: {reader}")
