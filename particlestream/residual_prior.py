"""Parametric RLS + Mahalanobis spike-and-slab residual conditional prior.

Extends ``FeatureConditionalReconstruction`` by replacing nearest-neighbor table
lookup on unseen contexts with an online Sherman-Morrison affine predictor, a
precision-weighted spike-and-slab prototype posterior, and an empirical-Bayes
kernel gate that recovers exact multimodal quantiles at recurrent contexts while
extrapolating smoothly across novel continuous or compositional states.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .feature_prior import FeatureConditionalReconstruction


class ResidualFeatureConditionalReconstruction(FeatureConditionalReconstruction):
    """Online affine RLS predictor plus kernel-gated live residual quantiles."""

    additive_transform = True
    config_fields = FeatureConditionalReconstruction.config_fields + (
        "rls_ridge", "covariance_ema", "covariance_eps", "slab_scale",
        "variance_ratio_scale", "visit_shrinkage", "kernel_bandwidth",
        "outcome_decay", "context_bypass_weight",
        "predictor_law", "residual_gate_law",
    )

    def __init__(self, generator, *, rls_ridge=0.1, covariance_ema=0.05,
                 covariance_eps=1e-4, slab_scale=1.0, variance_ratio_scale=0.15,
                 visit_shrinkage=0.25, kernel_bandwidth=0.35, outcome_decay=1.0,
                 context_bypass_weight=5.0, **kwargs):
        for name, value in (
            ("rls_ridge", rls_ridge),
            ("covariance_eps", covariance_eps),
            ("slab_scale", slab_scale),
            ("variance_ratio_scale", variance_ratio_scale),
            ("kernel_bandwidth", kernel_bandwidth),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("visit_shrinkage", visit_shrinkage),
            ("context_bypass_weight", context_bypass_weight),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (("covariance_ema", covariance_ema), ("outcome_decay", outcome_decay)):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        super().__init__(generator, **kwargs)
        self.rls_ridge = float(rls_ridge)
        self.covariance_ema = float(covariance_ema)
        self.covariance_eps = float(covariance_eps)
        self.slab_scale = float(slab_scale)
        self.variance_ratio_scale = float(variance_ratio_scale)
        self.visit_shrinkage = float(visit_shrinkage)
        self.kernel_bandwidth = float(kernel_bandwidth)
        self.outcome_decay = float(outcome_decay)
        self.context_bypass_weight = float(context_bypass_weight)
        self.predictor_law = "sherman_morrison_rls_mahalanobis_spike_slab_v1"
        self.residual_gate_law = "empirical_bayes_kernel_residual_quantiles_v1"
        self.conditional_sampling_law = "residual_prior_empirical_quantiles_v1"
        self.serving_context_law = "parametric_residual_context_v1"
        self.register_buffer("rls_weights", torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer("rls_cov", torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer("residual_cov", torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer("quantization_ema", torch.zeros((), dtype=torch.float64))
        self.register_buffer("outcome_masses", torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer("predictor_updates", torch.zeros((), dtype=torch.long))

    def bind(self, base_prior):
        super().bind(base_prior)
        count, dimension = base_prior.num_particles, base_prior.z_dim
        context_dimension = self._current_context().numel()
        aug_dim = context_dimension + 1
        device = base_prior.z.device
        if self.rls_weights.numel():
            if (tuple(self.rls_weights.shape) != (dimension, aug_dim)
                    or tuple(self.outcome_masses.shape) != (self.context_capacity, count)):
                raise ValueError("residual prior capacity differs from the task prior")
            return
        self.rls_weights = torch.zeros(dimension, aug_dim, device=device, dtype=torch.float64)
        self.rls_cov = torch.eye(aug_dim, device=device, dtype=torch.float64) / self.rls_ridge
        self.residual_cov = 1e-2 * torch.eye(dimension, device=device, dtype=torch.float64)
        self.quantization_ema = torch.zeros((), device=device, dtype=torch.float64)
        self.outcome_masses = torch.zeros(self.context_capacity, count, device=device, dtype=torch.float64)
        self.predictor_updates = torch.zeros((), device=device, dtype=torch.long)

    @torch.no_grad()
    def _disc_gate(self) -> float:
        """Dimensionless variance-ratio gate separating discrete vs continuous manifolds."""
        tr_s = float(torch.diagonal(self.residual_cov).sum().clamp_min(0.0).item())
        q_val = float(self.quantization_ema.clamp_min(0.0).item())
        ratio = q_val / (self.variance_ratio_scale * tr_s + 1e-5)
        return math.exp(-min(ratio * ratio, 80.0))

    @torch.no_grad()
    def _prior_mean(self, context: torch.Tensor) -> torch.Tensor:
        """Compute precision-gated spike-and-slab prior mean for one 1D context."""
        c64 = context.detach().to(device=self.rls_weights.device, dtype=torch.float64)
        cbar = torch.cat((c64, c64.new_ones(1)), dim=0)
        z_lin = self.rls_weights @ cbar
        m_count = int(self.feature_count.item())
        updates = int(self.predictor_updates.item())
        if m_count == 0 or updates <= c64.numel():
            return z_lin
        d_z = z_lin.numel()
        eye = torch.eye(d_z, device=z_lin.device, dtype=torch.float64)
        s_reg = 0.5 * (self.residual_cov + self.residual_cov.t()) + self.covariance_eps * eye
        protos = self.feature_centroids[:m_count].to(dtype=torch.float64)
        diff = protos - z_lin.unsqueeze(0)
        solved = torch.linalg.solve(s_reg, diff.t()).t()
        maha = (diff * solved).sum(dim=-1).clamp_min(0.0)
        best = int(maha.argmin().item())
        disc_gate = self._disc_gate()
        slab_var = self.slab_scale * self.slab_scale
        log_bf = 0.5 * float((torch.linalg.slogdet(s_reg + slab_var * eye)[1]
                              - torch.linalg.slogdet(s_reg)[1]).item()) - 0.5 * float(maha[best].item())
        slab_gate = 1.0 / (1.0 + math.exp(-max(min(log_bf, 60.0), -60.0)))
        beta = disc_gate * slab_gate
        return (1.0 - beta) * z_lin + beta * protos[best]

    def transform(self, codes, idx=None, *, base_prior, generator=None, training=True):
        if idx is None:
            raise ValueError("residual feature prior requires the task's particle row indices")
        centers = base_prior.means() if hasattr(base_prior, "means") else base_prior.z
        perturbation = codes - centers[idx]
        if training and self._target is not None and self._phase == 2:
            return self._pending_feature.unsqueeze(0) + perturbation
        context = self._pending_context if self._target is not None else self._current_context()
        if context.numel() == 0:
            return super().transform(codes, idx, base_prior=base_prior, generator=generator, training=training)
        if int(self.predictor_updates.item()) == 0 or int(self.context_count.item()) == 0:
            return codes
        with torch.no_grad():
            z_ctx = self._prior_mean(context)
            bucket = self._lookup_context(context, require_match=False)
            if bucket is None:
                return z_ctx.to(codes).unsqueeze(0) + perturbation
            masses = self.outcome_masses[bucket]
            total = masses.sum()
            if float(total.item()) <= 0.0:
                return z_ctx.to(codes).unsqueeze(0) + perturbation
            cdf = masses.cumsum(0) / total
            quantiles = (idx.double() + 0.5) / base_prior.num_particles
            features = torch.searchsorted(cdf, quantiles, right=True).clamp_max(base_prior.num_particles - 1)
            empirical = self.feature_centroids[features].to(dtype=torch.float64)
            key = self.context_keys[bucket]
            dist_sq = float((context.double() - key.double()).square().sum().item())
            kernel = math.exp(-min(dist_sq / (self.kernel_bandwidth * self.kernel_bandwidth), 80.0))
            excess_visits = max(int(self.context_visits[bucket].item()) - 1, 0)
            visit_gate = (1.0 if self.visit_shrinkage == 0.0 and excess_visits > 0
                          else excess_visits / (excess_visits + self.visit_shrinkage) if excess_visits > 0 else 0.0)
            gamma = visit_gate * self._disc_gate() * kernel
            if gamma <= 1e-8:
                return z_ctx.to(codes).unsqueeze(0) + perturbation
            z_key = self._prior_mean(key)
            combined = z_ctx.unsqueeze(0) + gamma * (empirical - z_key.unsqueeze(0))
            return combined.to(codes) + perturbation

    def regularize(self, raw_rows, *, generated=None, **kwargs):
        loss = super().regularize(raw_rows, generated=generated, **kwargs)
        model = self._generator()
        fc = getattr(model, "fc", None)
        z_dim = int(getattr(model, "z_dim", 0))
        c_dim = int(getattr(model, "context_dim", 0))
        if (self.context_bypass_weight > 0.0 and self.penalty_weight > 0.0 and c_dim > 0
                and isinstance(fc, nn.Linear) and fc.weight.shape[1] == z_dim + c_dim):
            w_ctx = fc.weight[:, z_dim:]
            loss = loss + (self.context_bypass_weight / self.penalty_weight) * w_ctx.square().sum()
        return loss

    @torch.no_grad()
    def _commit_observation(self, context, feature):
        count = int(self.feature_count.item())
        if count:
            distances = torch.linalg.vector_norm(self.feature_centroids[:count] - feature, dim=1)
            nearest_feature = int(distances.argmin().item())
        else:
            distances, nearest_feature = None, 0
        if count < len(self.feature_centroids) and (count == 0 or float(distances[nearest_feature]) > self.feature_tolerance):
            nearest_feature = count
            self.feature_count.add_(1)
        f_visits = int(self.feature_visits[nearest_feature].item()) + 1
        self.feature_centroids[nearest_feature].add_((feature - self.feature_centroids[nearest_feature]) / f_visits)
        self.feature_visits[nearest_feature].fill_(f_visits)

        bucket = self._lookup_context(context)
        if bucket is None:
            occupied = int(self.context_count.item())
            if occupied < self.context_capacity:
                bucket = occupied
                self.context_count.add_(1)
            else:
                bucket = int(self.context_last_update.argmin().item())
                self.context_evictions.add_(1)
            self.context_keys[bucket].copy_(context)
            self.context_outcomes[bucket].zero_()
            self.outcome_masses[bucket].zero_()
            self.context_visits[bucket].zero_()
        c_visits = int(self.context_visits[bucket].item()) + 1
        self.context_keys[bucket].add_((context - self.context_keys[bucket]) / c_visits)
        self.context_outcomes[bucket, nearest_feature].add_(1)
        if self.outcome_decay < 1.0:
            self.outcome_masses[bucket].mul_(self.outcome_decay)
        self.outcome_masses[bucket, nearest_feature].add_(1.0)
        self.context_visits[bucket].fill_(c_visits)
        self.context_last_update[bucket].copy_(self.assignment_updates)

        self.predictor_updates.add_(1)
        step = int(self.predictor_updates.item())
        if context.numel() == 0:
            return
        c64 = context.detach().to(device=self.rls_weights.device, dtype=torch.float64)
        z64 = feature.detach().to(device=self.rls_weights.device, dtype=torch.float64)
        cbar = torch.cat((c64, c64.new_ones(1)), dim=0)
        pc = self.rls_cov @ cbar
        denom = 1.0 + float((cbar @ pc).clamp_min(0.0).item())
        gain = pc / denom
        prior_err = z64 - self.rls_weights @ cbar
        self.rls_weights.add_(torch.outer(prior_err, gain))
        updated_cov = self.rls_cov - torch.outer(gain, pc)
        self.rls_cov.copy_(0.5 * (updated_cov + updated_cov.t()))

        if step > c64.numel():
            post_err = z64 - self.rls_weights @ cbar
            self.residual_cov.mul_(1.0 - self.covariance_ema).add_(
                torch.outer(post_err, post_err), alpha=self.covariance_ema,
            )
            proto64 = self.feature_centroids[nearest_feature].to(dtype=torch.float64)
            q_err = (z64 - proto64).square().sum()
            self.quantization_ema.mul_(1.0 - self.covariance_ema).add_(q_err, alpha=self.covariance_ema)

    def diagnostics(self):
        result = super().diagnostics()
        result.update(
            critic_sampling="residual_context_feature_prior",
            serving_sampling=self.conditional_sampling_law,
            predictor_updates=int(self.predictor_updates.item()),
            quantization_ema=float(self.quantization_ema.item()),
            residual_cov_trace=float(torch.diagonal(self.residual_cov).sum().item()) if self.residual_cov.numel() else 0.0,
        )
        return result
