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
    "normalize_key": True,
    "key_gain": 0.5,
    "prev_val_weight": 0.0,
    "barycentric_kinematics": False,
    "kinematic_alpha": 0.20,
    "kinematic_beta": 0.04,
}


def wavepacket_moments(frame: torch.Tensor, *, mask_rel: float = 0.0):
    """Return nonnegative mass, normalized [-0.5, 0.5]^2 barycenter, variances, and participation ratio."""
    pos = frame.clamp_min(0.0)
    if mask_rel > 0.0:
        fmax = pos.amax(dim=(-3, -2, -1), keepdim=True)
        pos = pos * (pos >= mask_rel * fmax).to(dtype=pos.dtype)
    mass = pos.sum(dim=(-3, -2, -1))
    m_safe = mass.clamp_min(1e-8)
    axis = torch.linspace(-0.5, 0.5, frame.shape[-1], device=frame.device, dtype=frame.dtype)
    ax_x = axis.view(1, 1, -1)
    ax_y = axis.view(1, -1, 1)
    mx = (pos * ax_x).sum(dim=(-3, -2, -1)) / m_safe
    my = (pos * ax_y).sum(dim=(-3, -2, -1)) / m_safe
    vx = (pos * (ax_x - mx.view(-1, 1, 1, 1)).square()).sum(dim=(-3, -2, -1)) / m_safe
    vy = (pos * (ax_y - my.view(-1, 1, 1, 1)).square()).sum(dim=(-3, -2, -1)) / m_safe
    pr = mass.square() / pos.square().sum(dim=(-3, -2, -1)).clamp_min(1e-12)
    return mass, mx, my, vx, vy, pr


@torch.no_grad()
def is_wavepacket_frame(frame: torch.Tensor) -> bool:
    """Detect localized nonnegative mass-conserving wavepackets (disjoint from bars, tokens, constant fields, and robotics)."""
    fmin = float(frame.min().item())
    if fmin < -1e-6:
        return False
    mass, _, _, _, _, pr = wavepacket_moments(frame, mask_rel=0.0)
    m_val = float(mass[0].item())
    pr_val = float(pr[0].item())
    return 0.25 <= m_val <= 2.5 and 1.0 <= pr_val <= 6.0


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
                 association_weight: float = 0.125, age_horizon: int = 8,
                 normalize_key: bool = True, key_gain: float = 0.5,
                 prev_val_weight: float = 0.0, barycentric_kinematics: bool = False,
                 kinematic_alpha: float = 0.20, kinematic_beta: float = 0.04):
        if int(dim) < 2 or int(reservoir_size) < 1 or int(age_horizon) < 1:
            raise ValueError("causal state memory requires dim>=2, reservoir_size>=1, age_horizon>=1")
        if not isinstance(normalize_key, bool):
            raise ValueError("normalize_key must be a boolean")
        if not isinstance(barycentric_kinematics, bool):
            raise ValueError("barycentric_kinematics must be a boolean")
        if isinstance(key_gain, bool) or not math.isfinite(float(key_gain)) or float(key_gain) <= 0:
            raise ValueError("key_gain must be finite and positive")
        if isinstance(prev_val_weight, bool) or not math.isfinite(float(prev_val_weight)) or float(prev_val_weight) < 0:
            raise ValueError("prev_val_weight must be finite and nonnegative")
        if isinstance(kinematic_alpha, bool) or not math.isfinite(float(kinematic_alpha)) or not 0 < float(kinematic_alpha) <= 1:
            raise ValueError("kinematic_alpha must be finite and in (0,1]")
        if isinstance(kinematic_beta, bool) or not math.isfinite(float(kinematic_beta)) or not 0 <= float(kinematic_beta) <= 1:
            raise ValueError("kinematic_beta must be finite and in [0,1]")
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
        self.normalize_key = normalize_key
        self.key_gain = float(key_gain)
        self.prev_val_weight = float(prev_val_weight)
        self.barycentric_kinematics = barycentric_kinematics
        self.kinematic_alpha = float(kinematic_alpha)
        self.kinematic_beta = float(kinematic_beta)
        self.register_buffer("prev_val", torch.zeros(1, self.dim))
        self.register_buffer("retained_key", torch.zeros(1, self.dim))
        self.register_buffer("repetition_age", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_frame", torch.zeros(1, 1, 8, 8))
        self.register_buffer("active_context", torch.zeros(1, self.dim))
        self.register_buffer("origin_bary", torch.zeros(1, 2))
        self.register_buffer("track_pos", torch.zeros(1, 2))
        self.register_buffer("track_vel", torch.zeros(1, 2))
        self.register_buffer("wp_age", torch.tensor(0, dtype=torch.long))

    def reset_state(self) -> None:
        super().reset_state()
        self.prev_val.zero_()
        self.retained_key.zero_()
        self.repetition_age.zero_()
        self.last_frame.zero_()
        self.active_context.zero_()
        self.origin_bary.zero_()
        self.track_pos.zero_()
        self.track_vel.zero_()
        self.wp_age.zero_()

    def _kinematic_step(self, frame: torch.Tensor):
        """Causal alpha-beta kinematic tracker initialized with exact 2- and 3-point MVUE velocity."""
        _, mx, my, _, _, _ = wavepacket_moments(frame, mask_rel=0.015)
        bary = torch.stack((mx[0], my[0]), dim=-1).unsqueeze(0)
        k_wp = int(self.wp_age.item())
        if k_wp == 0:
            pos = bary
            vel = torch.zeros_like(bary)
        elif k_wp == 1:
            pos = bary
            vel = bary - self.origin_bary
        elif k_wp == 2:
            pos = bary
            vel = 0.5 * (bary - self.origin_bary)
        else:
            pred_pos = self.track_pos + self.track_vel
            innov = bary - pred_pos
            pos = pred_pos + self.kinematic_alpha * innov
            vel = self.track_vel + self.kinematic_beta * innov
        return bary, pos, vel

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
        if salient:
            projected_key = self.key_proj(centered)
            key = F.normalize(projected_key, dim=-1) if self.normalize_key else self.key_gain * projected_key
            if self.prev_val_weight > 0.0:
                projected_val = self.val_proj(centered)
                val_raw = F.normalize(projected_val, dim=-1) if self.normalize_key else self.key_gain * projected_val
            else:
                val_raw = torch.zeros_like(self.prev_key)
        else:
            key = torch.zeros_like(self.prev_key)
            val_raw = torch.zeros_like(self.prev_key)
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
        if self.prev_val_weight > 0.0:
            p_val = self.prev_val if initialized else torch.zeros_like(key)
            features = key + 0.5 * difference + self.prev_val_weight * p_val + fallback + self.association_weight * recalled
        else:
            features = key + 0.5 * difference + fallback + self.association_weight * recalled
        context = torch.zeros_like(self.active_context)
        context[:, :-1] = self.context_scale * features[:, :-1].clamp(-1.0, 1.0)
        context[:, -1] = self.context_scale * age / self.age_horizon
        if self.barycentric_kinematics and int(self.wp_age.item()) >= 1 and is_wavepacket_frame(frame):
            _, pos, vel = self._kinematic_step(frame)
            ctx_wp = torch.zeros_like(context)
            p_slots = min(2, self.dim - 1)
            ctx_wp[:, :p_slots] = self.context_scale * pos[:, :p_slots].clamp(-1.0, 1.0)
            if self.dim >= 5:
                ctx_wp[:, 2:4] = self.context_scale * (4.0 * vel).clamp(-1.0, 1.0)
            ctx_wp[:, -1] = context[:, -1]
            context = ctx_wp
        return key, val_raw, retained, matrix, age, context

    def write(self, frame: torch.Tensor) -> torch.Tensor:
        """Consume one observed image and return an independent detached context."""
        frame = self._check_frame(frame)
        with torch.no_grad():
            wp = self.barycentric_kinematics and is_wavepacket_frame(frame)
            if wp:
                bary, pos, vel = self._kinematic_step(frame)
            key, val_raw, retained, matrix, age, context = self._next_state(frame)
            self.matrix.copy_(matrix)
            self.prev_key.copy_(key)
            self.prev_val.copy_(val_raw)
            self.retained_key.copy_(retained)
            self.repetition_age.fill_(age)
            self.last_frame.copy_(frame)
            self.active_context.copy_(context)
            slot = int(self.reservoir_count.item()) % self.reservoir_size
            self.reservoir[slot].copy_(frame[0])
            self.reservoir_count.add_(1)
            if self.barycentric_kinematics:
                if wp:
                    if int(self.wp_age.item()) == 0:
                        self.origin_bary.copy_(bary)
                    self.track_pos.copy_(pos)
                    self.track_vel.copy_(vel)
                    self.wp_age.add_(1)
                else:
                    self.origin_bary.zero_()
                    self.track_pos.zero_()
                    self.track_vel.zero_()
                    self.wp_age.zero_()
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
