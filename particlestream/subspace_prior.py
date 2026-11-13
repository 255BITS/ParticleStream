"""Precision-whitened Gram-Schmidt subspace and semiparametric NW residual prior.

Extends ``ResidualFeatureConditionalReconstruction`` along two orthogonal axes:
1. Precision-whitened anchored Gram-Schmidt innovation subspace + JL tail:
   replaces blind fixed random ``8x64`` projection on the leading ``subspace_dims``
   coordinates with causally anchored orthonormal two-pass Gram-Schmidt innovation
   vectors scaled once at discovery by ``(r_norm^2 + nu_0^2)^(-1/2)``. This
   eliminates random-projection nullspace loss and 1:1 MoG Wiener attenuation on
   millimetre physical micro-displacements while preserving sweet-spot latent norms
   on high-contrast visual tasks without retroactive coordinate drift.
2. Semiparametric global RLS + unquantized Nadaraya-Watson kernel-ridge residual:
   records exact unquantized latent feature means per context bucket and adds a
   smooth continuous local residual gated by ``(1 - g_disc)`` alongside discrete
   empirical quantiles gated by ``g_disc``.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch

from .residual_prior import ResidualFeatureConditionalReconstruction
from .transport import CompetitiveReconstruction


class SubspaceResidualConditionalReconstruction(ResidualFeatureConditionalReconstruction):
    """Precision-whitened online innovation subspace plus semiparametric RLS/NW prior."""

    additive_transform = True
    config_fields = ResidualFeatureConditionalReconstruction.config_fields + (
        "subspace_dims",
        "subspace_abs_tol",
        "subspace_rel_tol",
        "subspace_nu0",
        "nw_weight",
        "nw_bandwidth",
        "nw_ridge",
    )

    def __init__(
        self,
        generator,
        *,
        subspace_dims: int = 6,
        subspace_abs_tol: float = 0.008,
        subspace_rel_tol: float = 0.12,
        subspace_nu0: float = 0.10,
        nw_weight: float = 1.0,
        nw_bandwidth: float = 0.65,
        nw_ridge: float = 0.06,
        **kwargs,
    ):
        if type(subspace_dims) is not int or not 0 <= subspace_dims <= 64:
            raise ValueError("subspace_dims must be an integer in [0, 64]")
        for name, value in (
            ("subspace_abs_tol", subspace_abs_tol),
            ("subspace_rel_tol", subspace_rel_tol),
            ("subspace_nu0", subspace_nu0),
            ("nw_bandwidth", nw_bandwidth),
            ("nw_ridge", nw_ridge),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(nw_weight, bool) or not isinstance(nw_weight, (int, float)) or not math.isfinite(nw_weight) or nw_weight < 0:
            raise ValueError("nw_weight must be finite and nonnegative")
        super().__init__(generator, **kwargs)
        self.subspace_dims = subspace_dims
        self.subspace_abs_tol = float(subspace_abs_tol)
        self.subspace_rel_tol = float(subspace_rel_tol)
        self.subspace_nu0 = float(subspace_nu0)
        self.nw_weight = float(nw_weight)
        self.nw_bandwidth = float(nw_bandwidth)
        self.nw_ridge = float(nw_ridge)
        self.encoder_law = "precision_whitened_gram_schmidt_subspace_plus_jl_v1"
        self.predictor_law = "semiparametric_rls_nw_spike_slab_v1"
        self.residual_gate_law = "dual_regime_nw_and_empirical_quantiles_v1"
        self.conditional_sampling_law = "subspace_semiparametric_quantiles_v1"
        self.serving_context_law = "semiparametric_subspace_context_v1"
        self.register_buffer("subspace_basis", torch.empty(0, 0))
        self.register_buffer("subspace_gains", torch.empty(0))
        self.register_buffer("subspace_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("context_features", torch.empty(0, 0, dtype=torch.float64))
        self._pending_basis_row = None
        self._pending_basis_gain = None

    def bind(self, base_prior):
        super().bind(base_prior)
        dimension = base_prior.z_dim
        device = base_prior.z.device
        if self.subspace_dims > dimension:
            raise ValueError("subspace_dims cannot exceed z_dim")
        if self.context_features.numel():
            if (
                tuple(self.subspace_basis.shape) != (self.subspace_dims, 64)
                or tuple(self.subspace_gains.shape) != (self.subspace_dims,)
                or tuple(self.context_features.shape) != (self.context_capacity, dimension)
            ):
                raise ValueError("subspace residual prior capacity differs from the task prior")
            return
        self.subspace_basis = base_prior.z.new_zeros(self.subspace_dims, 64)
        self.subspace_gains = base_prior.z.new_ones(self.subspace_dims)
        self.subspace_count = self.subspace_count.to(device)
        self.context_features = torch.zeros(
            self.context_capacity, dimension, device=device, dtype=torch.float64,
        )

    @torch.no_grad()
    def _candidate_innovation(self, frame: torch.Tensor) -> tuple[torch.Tensor, float] | None:
        k = int(self.subspace_count.item())
        if k >= self.subspace_dims:
            return None
        v = frame.detach().reshape(64).to(device=self.encoder.device, dtype=torch.float64) - self.image_center
        v_norm = float(torch.linalg.vector_norm(v).item())
        r = v.clone()
        if k > 0:
            basis64 = self.subspace_basis[:k].to(dtype=torch.float64)
            proj1 = basis64 @ r
            r = r - basis64.t() @ proj1
            proj2 = basis64 @ r
            r = r - basis64.t() @ proj2
        r_norm = float(torch.linalg.vector_norm(r).item())
        if r_norm > self.subspace_abs_tol and (r_norm / (v_norm + 1e-3)) > self.subspace_rel_tol:
            unit = (r / r_norm).to(self.subspace_basis.dtype)
            gain = 1.0 / math.sqrt(r_norm * r_norm + self.subspace_nu0 * self.subspace_nu0)
            return unit, gain
        return None

    @torch.no_grad()
    def encode(self, frame):
        if (
            not isinstance(frame, torch.Tensor)
            or tuple(frame.shape) != (1, 1, 8, 8)
            or not frame.is_floating_point()
            or not bool(torch.isfinite(frame).all())
        ):
            raise ValueError("feature encoder requires one finite image")
        v = frame.detach().reshape(1, 64).to(self.encoder) - self.image_center
        k = int(self.subspace_count.item())
        rows = self.encoder.clone()
        if self.subspace_dims > 0:
            rows[:self.subspace_dims].zero_()
            if k > 0:
                rows[:k].copy_(self.subspace_basis[:k] * self.subspace_gains[:k].unsqueeze(1))
            if self._pending_basis_row is not None and k < self.subspace_dims:
                rows[k].copy_(self._pending_basis_row * float(self._pending_basis_gain))
        return self.feature_scale * (v @ rows.t())

    @contextmanager
    def training_target(self, frame):
        if self._target is not None:
            raise RuntimeError("a feature-prior update is already active")
        try:
            with CompetitiveReconstruction.training_target(self, frame):
                self._pending_context = self._current_context()
                cand = self._candidate_innovation(frame)
                if cand is not None:
                    self._pending_basis_row, self._pending_basis_gain = cand
                else:
                    self._pending_basis_row = self._pending_basis_gain = None
                self._pending_feature = self.encode(frame)[0]
                yield
            self._commit_observation(self._pending_context, self._pending_feature)
        finally:
            self._pending_context = self._pending_feature = None
            self._pending_basis_row = self._pending_basis_gain = None

    def transform(self, codes, idx=None, *, base_prior, generator=None, training=True):
        if idx is None:
            raise ValueError("subspace residual prior requires the task's particle row indices")
        centers = base_prior.means() if hasattr(base_prior, "means") else base_prior.z
        perturbation = codes - centers[idx]
        if training and self._target is not None and self._phase == 2:
            return self._pending_feature.unsqueeze(0) + perturbation
        context = self._pending_context if self._target is not None else self._current_context()
        if context.numel() == 0:
            return super().transform(codes, idx, base_prior=base_prior, generator=generator, training=training)
        updates = int(self.predictor_updates.item())
        b_count = int(self.context_count.item())
        if updates == 0 or b_count == 0:
            return codes
        with torch.no_grad():
            z_ctx = self._prior_mean(context)
            g_disc = self._disc_gate()
            z_base = z_ctx
            if self.nw_weight > 0.0 and updates > context.numel() and (1.0 - g_disc) > 1e-6:
                keys64 = self.context_keys[:b_count].to(dtype=torch.float64)
                c64 = context.detach().to(device=keys64.device, dtype=torch.float64)
                dist_sq_all = (keys64 - c64.unsqueeze(0)).square().sum(dim=-1)
                visit_cap = self.context_visits[:b_count].to(dtype=torch.float64).clamp(min=1.0, max=4.0)
                weights = visit_cap * torch.exp(
                    -(dist_sq_all / (2.0 * self.nw_bandwidth * self.nw_bandwidth)).clamp_max(80.0)
                )
                w_sum = float(weights.sum().item())
                if w_sum > 1e-12:
                    k_aug = torch.cat((keys64, keys64.new_ones(b_count, 1)), dim=-1)
                    z_lin_keys = k_aug @ self.rls_weights.t()
                    res_keys = self.context_features[:b_count] - z_lin_keys
                    r_nw = (weights.unsqueeze(-1) * res_keys).sum(dim=0) / (self.nw_ridge + w_sum)
                    z_base = z_ctx + self.nw_weight * (1.0 - g_disc) * r_nw

            bucket = self._lookup_context(context, require_match=False)
            if bucket is None:
                return z_base.to(codes).unsqueeze(0) + perturbation
            masses = self.outcome_masses[bucket]
            total = masses.sum()
            if float(total.item()) <= 0.0:
                return z_base.to(codes).unsqueeze(0) + perturbation
            cdf = masses.cumsum(0) / total
            quantiles = (idx.double() + 0.5) / base_prior.num_particles
            features = torch.searchsorted(cdf, quantiles, right=True).clamp_max(base_prior.num_particles - 1)
            empirical = self.feature_centroids[features].to(dtype=torch.float64)
            key = self.context_keys[bucket]
            dist_sq = float((context.double() - key.double()).square().sum().item())
            kernel = math.exp(-min(dist_sq / (self.kernel_bandwidth * self.kernel_bandwidth), 80.0))
            excess_visits = max(int(self.context_visits[bucket].item()) - 1, 0)
            visit_gate = (
                1.0
                if self.visit_shrinkage == 0.0 and excess_visits > 0
                else excess_visits / (excess_visits + self.visit_shrinkage)
                if excess_visits > 0
                else 0.0
            )
            gamma = visit_gate * g_disc * kernel
            if gamma <= 1e-8:
                return z_base.to(codes).unsqueeze(0) + perturbation
            z_key = self._prior_mean(key)
            combined = z_base.unsqueeze(0) + gamma * (empirical - z_key.unsqueeze(0))
            return combined.to(codes) + perturbation

    @torch.no_grad()
    def _commit_observation(self, context, feature):
        k = int(self.subspace_count.item())
        if self._pending_basis_row is not None and k < self.subspace_dims:
            self.subspace_basis[k].copy_(self._pending_basis_row)
            self.subspace_gains[k].fill_(float(self._pending_basis_gain))
            self.subspace_count.add_(1)
        super()._commit_observation(context, feature)
        b_count = int(self.context_count.item())
        if context.numel() == 0 or b_count == 0:
            return
        bucket = int(self.context_last_update[:b_count].argmax().item())
        c_visits = int(self.context_visits[bucket].item())
        z64 = feature.detach().to(device=self.context_features.device, dtype=torch.float64)
        if c_visits <= 1:
            self.context_features[bucket].copy_(z64)
        else:
            self.context_features[bucket].add_((z64 - self.context_features[bucket]) / c_visits)

    def diagnostics(self):
        result = super().diagnostics()
        k = int(self.subspace_count.item())
        result.update(
            critic_sampling="subspace_residual_context_feature_prior",
            serving_sampling=self.conditional_sampling_law,
            subspace_count=k,
            subspace_gains=self.subspace_gains[:k].detach().cpu().tolist(),
            observation_state_retained=(
                result["observation_state_retained"]
                or self._pending_basis_row is not None
                or self._pending_basis_gain is not None
            ),
        )
        return result
