"""Context-local balanced routing of the one observation currently being learned."""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch

from .transport import CompetitiveReconstruction


class ContextBalancedReconstruction(CompetitiveReconstruction):
    """Use online transport duals to share equal-mass rows within each context.

    The routing cost is current-image MSE plus a context-local row dual. After
    a successful single-observation update, each dual receives
    ``dual_rate * (one_hot(selected_row) - 1 / num_particles)``. Positive duals
    discourage repeatedly selecting the same row. Contexts are bounded centroid
    statistics from the generator's already-observed conditioning vector; no
    output image, target history, gradient history or replay target is retained.

    Context lookup is Euclidean, with a declared matching tolerance. A new
    unmatched context uses a free slot or replaces the least recently updated
    slot. Allocation, centroid changes and dual updates commit only after the
    public training update succeeds. Mean centering and uniform rescaling bound
    duals while preserving their zero-sum gauge.
    """

    config_fields = CompetitiveReconstruction.config_fields + (
        "context_capacity", "context_tolerance", "context_clip", "dual_rate",
        "dual_clip", "context_eviction_law",
    )

    def __init__(self, generator, *, context_capacity=64, context_tolerance=0.25,
                 context_clip=8.0, dual_rate=0.01, dual_clip=0.25, **kwargs):
        if (isinstance(context_capacity, bool) or not isinstance(context_capacity, (int, float))
                or not math.isfinite(context_capacity) or int(context_capacity) != context_capacity or context_capacity < 1):
            raise ValueError("context_capacity must be a positive integer")
        for name, value in (("context_tolerance", context_tolerance), ("context_clip", context_clip),
                            ("dual_rate", dual_rate), ("dual_clip", dual_clip)):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if context_clip == 0 or dual_clip == 0:
            raise ValueError("context_clip and dual_clip must be positive")
        super().__init__(generator, **kwargs)
        self.context_capacity = int(context_capacity)
        self.context_tolerance = float(context_tolerance)
        self.context_clip = float(context_clip)
        self.dual_rate = float(dual_rate)
        self.dual_clip = float(dual_clip)
        self.context_eviction_law = "least_recently_updated_v1"
        self.register_buffer("context_keys", torch.empty(0, 0))
        self.register_buffer("context_duals", torch.empty(0, 0))
        self.register_buffer("context_assignment_mass", torch.empty(0, 0))
        self.register_buffer("context_visits", torch.empty(0, dtype=torch.long))
        self.register_buffer("context_last_update", torch.empty(0, dtype=torch.long))
        self.register_buffer("context_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("context_evictions", torch.zeros((), dtype=torch.long))
        self._pending_key = self._pending_bucket = None
        self._pending_new = False

    def _current_context(self):
        model = self._generator()
        if model is None:
            raise RuntimeError("the routing generator model no longer exists")
        active = getattr(model, "active_context", None)
        dimension = int(getattr(model, "context_dim", 0 if active is None else active.shape[-1]))
        if dimension == 0:
            return self.assignment_mass.new_empty(0)
        if (active is None or tuple(active.shape) != (1, dimension)
                or not bool(torch.isfinite(active).all())):
            raise ValueError("routing requires one finite generator conditioning vector")
        return active[0].detach().to(self.assignment_mass).clamp(-self.context_clip, self.context_clip).clone()

    def bind(self, base_prior):
        super().bind(base_prior)
        dimension = self._current_context().numel()
        expected = (self.context_capacity, base_prior.num_particles)
        if self.context_duals.numel():
            if tuple(self.context_duals.shape) != expected or tuple(self.context_keys.shape) != (self.context_capacity, dimension):
                raise ValueError("router context or particle capacity differs from its bound state")
            return
        self.context_keys = base_prior.z.new_zeros(self.context_capacity, dimension)
        self.context_duals = base_prior.z.new_zeros(expected)
        self.context_assignment_mass = base_prior.z.new_zeros(expected)
        self.context_visits = torch.zeros(self.context_capacity, device=base_prior.z.device, dtype=torch.long)
        self.context_last_update = torch.zeros_like(self.context_visits)
        self.context_count = self.context_count.to(base_prior.z.device)
        self.context_evictions = self.context_evictions.to(base_prior.z.device)

    @torch.no_grad()
    def _lookup_context(self, key):
        count = int(self.context_count.item())
        if count:
            distances = torch.linalg.vector_norm(self.context_keys[:count] - key, dim=1)
            nearest = int(distances.argmin().item())
            if float(distances[nearest]) <= self.context_tolerance:
                return nearest, False
        if count < self.context_capacity:
            return count, True
        return int(self.context_last_update.argmin().item()), True

    @contextmanager
    def training_target(self, frame):
        if self._target is not None:
            raise RuntimeError("a transport update is already active")
        committed = None
        try:
            with super().training_target(frame):
                self._pending_key = self._current_context()
                yield
                if self._phase == 2 and self._selected is not None and self._pending_bucket is not None:
                    committed = (self._pending_bucket, self._pending_new, self._pending_key, self._selected)
            if committed is not None:
                self._commit_context(*committed)
        finally:
            self._pending_key = self._pending_bucket = None
            self._pending_new = False

    @torch.no_grad()
    def _costs(self, base_prior):
        costs = super()._costs(base_prior)
        bucket, new = self._lookup_context(self._pending_key)
        self._pending_bucket, self._pending_new = bucket, new
        if new:
            return costs
        return costs + self.context_duals[bucket]

    @torch.no_grad()
    def _commit_context(self, bucket, new, key, selected):
        if new:
            if int(self.context_count.item()) == self.context_capacity:
                self.context_evictions.add_(1)
            else:
                self.context_count.add_(1)
            self.context_keys[bucket].copy_(key)
            self.context_duals[bucket].zero_()
            self.context_assignment_mass[bucket].zero_()
            self.context_visits[bucket].zero_()
        visits = int(self.context_visits[bucket].item()) + 1
        self.context_keys[bucket].add_((key - self.context_keys[bucket]) / visits)
        dual = self.context_duals[bucket]
        dual.sub_(self.dual_rate / dual.numel())
        dual[selected].add_(self.dual_rate)
        dual.sub_(dual.mean())
        maximum = float(dual.abs().max())
        if maximum > self.dual_clip:
            dual.mul_(self.dual_clip / maximum)
        self.context_assignment_mass[bucket, selected].add_(1)
        self.context_visits[bucket].fill_(visits)
        self.context_last_update[bucket].copy_(self.assignment_updates)

    @torch.no_grad()
    def reset_state(self):
        if self._target is not None:
            raise RuntimeError("cannot reset a router during a training update")
        for buffer in self.buffers():
            buffer.zero_()
        self._pending_key = self._pending_bucket = None
        self._pending_new = False

    def diagnostics(self):
        result = super().diagnostics()
        count = int(self.context_count.item())
        result.update(
            generator_sampling="current_target_context_balanced_competition",
            context_count=count, context_evictions=int(self.context_evictions.item()),
            context_visits=self.context_visits[:count].detach().cpu().tolist(),
            context_assignment_mass=self.context_assignment_mass[:count].detach().cpu().tolist(),
            context_dual_max_abs=float(self.context_duals.abs().max()) if self.context_duals.numel() else 0.0,
            context_state_retained=self._pending_key is not None,
        )
        return result
