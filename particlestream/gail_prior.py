"""Stabilized GAIL conditional prior with prioritized survival, bounded style bonus, and associative fast weights.

Adapts the converging control mechanisms of ``43.gailcart.py`` to strict single-arrival
(`batch_size=1`, `replay_policy="none"`) ParticleGAN world modeling:
1. Prioritized survival/reconstruction anchor paired with a bounded sigmoid critic
   style bonus ``1.0 + alpha * sigmoid(D)`` (replacing unbounded linear Hinge generator
   slopes so uncalibrated or saturated critic scores cannot destabilize valid trajectories).
2. Causal discounted return trace ``R_t = gamma * R_{t-1} + (1 - gamma) * r_t`` and running
   moment standardization into a bounded advantage gain ``1 + beta_A * tanh(A_{t-1})``.
3. Entropy-regularized categorical policy quantiles over observed context outcomes.
4. Detached recursive associative fast-weight operator ``M_assoc`` mapping homogeneous
   context vectors to latent feature expectations — either via radial distance-gated
   blending or precision-weighted Mahalanobis spike-and-slab prototype transport with
   empirical-Bayes kernel residual quantiles and Markov context-bypass regularization —
   without storing or replaying frames.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch
from torch import nn

from .feature_prior import FeatureConditionalReconstruction


class GAILConditionalReconstruction(FeatureConditionalReconstruction):
    """Causal GAIL style-modulated posterior, return advantage, and associative fast weights."""

    additive_transform = True
    config_fields = FeatureConditionalReconstruction.config_fields + (
        "style_bonus_weight",
        "cancel_linear_gan",
        "return_discount",
        "advantage_scale",
        "policy_entropy_bonus",
        "assoc_ridge",
        "assoc_forgetting",
        "assoc_blend_scale",
        "assoc_max_blend",
        "spike_slab_scale",
        "covariance_ema",
        "covariance_eps",
        "variance_ratio_scale",
        "visit_shrinkage",
        "kernel_bandwidth",
        "context_bypass_weight",
        "gail_style_law",
        "gail_return_law",
        "assoc_extrapolation_law",
    )

    def __init__(
        self,
        generator,
        *,
        style_bonus_weight: float = 0.1,
        cancel_linear_gan: bool = True,
        return_discount: float = 0.99,
        advantage_scale: float = 0.25,
        policy_entropy_bonus: float = 0.01,
        assoc_ridge: float = 0.05,
        assoc_forgetting: float = 1.0,
        assoc_blend_scale: float = 0.25,
        assoc_max_blend: float = 1.0,
        spike_slab_scale: float = 0.0,
        covariance_ema: float = 0.05,
        covariance_eps: float = 1e-4,
        variance_ratio_scale: float = 0.15,
        visit_shrinkage: float = 0.25,
        kernel_bandwidth: float = 0.35,
        context_bypass_weight: float = 0.0,
        **kwargs,
    ):
        if not isinstance(cancel_linear_gan, bool):
            raise ValueError("cancel_linear_gan must be a boolean")
        for name, value in (
            ("style_bonus_weight", style_bonus_weight),
            ("return_discount", return_discount),
            ("advantage_scale", advantage_scale),
            ("policy_entropy_bonus", policy_entropy_bonus),
            ("assoc_ridge", assoc_ridge),
            ("assoc_forgetting", assoc_forgetting),
            ("assoc_blend_scale", assoc_blend_scale),
            ("assoc_max_blend", assoc_max_blend),
            ("spike_slab_scale", spike_slab_scale),
            ("covariance_ema", covariance_ema),
            ("covariance_eps", covariance_eps),
            ("variance_ratio_scale", variance_ratio_scale),
            ("visit_shrinkage", visit_shrinkage),
            ("kernel_bandwidth", kernel_bandwidth),
            ("context_bypass_weight", context_bypass_weight),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if style_bonus_weight < 0:
            raise ValueError("style_bonus_weight must be nonnegative")
        if not 0.0 <= return_discount < 1.0:
            raise ValueError("return_discount must be in [0, 1)")
        if not 0.0 <= advantage_scale < 1.0:
            raise ValueError("advantage_scale must be in [0, 1)")
        if not 0.0 <= policy_entropy_bonus <= 0.5:
            raise ValueError("policy_entropy_bonus must be in [0, 0.5]")
        if assoc_ridge < 0.0:
            raise ValueError("assoc_ridge must be nonnegative")
        if not 0.0 < assoc_forgetting <= 1.0:
            raise ValueError("assoc_forgetting must be in (0, 1]")
        if assoc_blend_scale <= 0.0:
            raise ValueError("assoc_blend_scale must be positive")
        if not 0.0 <= assoc_max_blend <= 1.0:
            raise ValueError("assoc_max_blend must be in [0, 1]")
        if spike_slab_scale < 0.0:
            raise ValueError("spike_slab_scale must be nonnegative")
        if spike_slab_scale > 0.0 and assoc_ridge <= 0.0:
            raise ValueError("spike_slab_scale > 0 requires positive assoc_ridge")
        if not 0.0 < covariance_ema <= 1.0:
            raise ValueError("covariance_ema must be in (0, 1]")
        if covariance_eps <= 0.0:
            raise ValueError("covariance_eps must be positive")
        if variance_ratio_scale <= 0.0:
            raise ValueError("variance_ratio_scale must be positive")
        if visit_shrinkage < 0.0:
            raise ValueError("visit_shrinkage must be nonnegative")
        if kernel_bandwidth <= 0.0:
            raise ValueError("kernel_bandwidth must be positive")
        if context_bypass_weight < 0.0:
            raise ValueError("context_bypass_weight must be nonnegative")

        super().__init__(generator, **kwargs)
        if cancel_linear_gan and self.penalty_weight <= 0.0:
            raise ValueError("cancel_linear_gan requires positive reconstruction_weight")

        self.style_bonus_weight = float(style_bonus_weight)
        self.cancel_linear_gan = cancel_linear_gan
        self.return_discount = float(return_discount)
        self.advantage_scale = float(advantage_scale)
        self.policy_entropy_bonus = float(policy_entropy_bonus)
        self.assoc_ridge = float(assoc_ridge)
        self.assoc_forgetting = float(assoc_forgetting)
        self.assoc_blend_scale = float(assoc_blend_scale)
        self.assoc_max_blend = float(assoc_max_blend)
        self.spike_slab_scale = float(spike_slab_scale)
        self.covariance_ema = float(covariance_ema)
        self.covariance_eps = float(covariance_eps)
        self.variance_ratio_scale = float(variance_ratio_scale)
        self.visit_shrinkage = float(visit_shrinkage)
        self.kernel_bandwidth = float(kernel_bandwidth)
        self.context_bypass_weight = float(context_bypass_weight)
        self.gail_style_law = (
            "prioritized_survival_sigmoid_style_hinge_cancel_v1"
            if cancel_linear_gan
            else "prioritized_survival_additive_sigmoid_style_v1"
        )
        self.gail_return_law = "causal_ema_standardized_advantage_tanh_v1"
        if self.spike_slab_scale > 0.0:
            self.assoc_extrapolation_law = "sherman_morrison_spike_slab_kernel_residual_v1"
        elif self.assoc_ridge > 0.0 and self.assoc_max_blend > 0.0:
            self.assoc_extrapolation_law = "recursive_ridge_fast_weight_radial_bound_v1"
        else:
            self.assoc_extrapolation_law = "disabled_v1"

        buf_dtype = torch.float64 if self.spike_slab_scale > 0.0 else torch.float32
        self.register_buffer("return_trace", torch.zeros(()))
        self.register_buffer("return_mean", torch.zeros(()))
        self.register_buffer("return_var", torch.zeros(()))
        self.register_buffer("gail_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("last_style_bonus", torch.zeros(()))
        self.register_buffer("last_advantage_gain", torch.ones(()))
        self.register_buffer("assoc_weights", torch.empty(0, 0, dtype=buf_dtype))
        self.register_buffer("assoc_cov_inv", torch.empty(0, 0, dtype=buf_dtype))
        self.register_buffer("residual_cov", torch.empty(0, 0, dtype=buf_dtype))
        self.register_buffer("quantization_ema", torch.zeros((), dtype=buf_dtype))
        self.register_buffer("assoc_updates", torch.zeros((), dtype=torch.long))
        self._pending_reward = None
        self._pending_style = None
        self._pending_gain = None

    def bind(self, base_prior):
        super().bind(base_prior)
        dimension = base_prior.z_dim
        context_dimension = self._current_context().numel()
        aug_dim = context_dimension + 1 if context_dimension > 0 else 0
        z_ref = base_prior.z
        device = z_ref.device
        self.return_trace = self.return_trace.to(z_ref)
        self.return_mean = self.return_mean.to(z_ref)
        self.return_var = self.return_var.to(z_ref)
        self.gail_updates = self.gail_updates.to(device)
        self.last_style_bonus = self.last_style_bonus.to(z_ref)
        self.last_advantage_gain = self.last_advantage_gain.to(z_ref)
        self.assoc_updates = self.assoc_updates.to(device)

        if self.spike_slab_scale > 0.0:
            if aug_dim > 0:
                expected_w = (dimension, aug_dim)
                expected_p = (aug_dim, aug_dim)
                if self.assoc_weights.numel():
                    if tuple(self.assoc_weights.shape) != expected_w or tuple(self.assoc_cov_inv.shape) != expected_p:
                        raise ValueError("GAIL associative fast-weight shape differs from bound context/latent dims")
                    self.assoc_weights = self.assoc_weights.to(device=device, dtype=torch.float64)
                    self.assoc_cov_inv = self.assoc_cov_inv.to(device=device, dtype=torch.float64)
                    self.residual_cov = self.residual_cov.to(device=device, dtype=torch.float64)
                    self.quantization_ema = self.quantization_ema.to(device=device, dtype=torch.float64)
                else:
                    self.assoc_weights = torch.zeros(expected_w, device=device, dtype=torch.float64)
                    self.assoc_cov_inv = (1.0 / self.assoc_ridge) * torch.eye(
                        aug_dim, device=device, dtype=torch.float64
                    )
                    self.residual_cov = 1e-2 * torch.eye(dimension, device=device, dtype=torch.float64)
                    self.quantization_ema = torch.zeros((), device=device, dtype=torch.float64)
            else:
                self.assoc_weights = torch.empty(0, 0, device=device, dtype=torch.float64)
                self.assoc_cov_inv = torch.empty(0, 0, device=device, dtype=torch.float64)
                self.residual_cov = 1e-2 * torch.eye(dimension, device=device, dtype=torch.float64)
                self.quantization_ema = torch.zeros((), device=device, dtype=torch.float64)
            return

        self.assoc_weights = self.assoc_weights.to(z_ref)
        self.assoc_cov_inv = self.assoc_cov_inv.to(z_ref)
        self.residual_cov = self.residual_cov.to(z_ref)
        self.quantization_ema = self.quantization_ema.to(z_ref)
        if aug_dim > 0 and self.assoc_ridge > 0.0:
            expected_w = (dimension, aug_dim)
            expected_p = (aug_dim, aug_dim)
            if self.assoc_weights.numel():
                if tuple(self.assoc_weights.shape) != expected_w or tuple(self.assoc_cov_inv.shape) != expected_p:
                    raise ValueError("GAIL associative fast-weight shape differs from bound context/latent dims")
            else:
                self.assoc_weights = z_ref.new_zeros(expected_w)
                self.assoc_cov_inv = (1.0 / self.assoc_ridge) * torch.eye(
                    aug_dim, device=device, dtype=z_ref.dtype
                )

    def _causal_advantage_gain(self) -> float:
        if self.advantage_scale == 0.0 or int(self.gail_updates.item()) < 2:
            return 1.0
        std = float(self.return_var.clamp_min(0.0).sqrt().item())
        adv = float((self.return_trace - self.return_mean).item()) / (std + 1e-5)
        return 1.0 + self.advantage_scale * math.tanh(adv)

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
        c64 = context.detach().to(device=self.assoc_weights.device, dtype=torch.float64)
        cbar = torch.cat((c64, c64.new_ones(1)), dim=0)
        z_lin = self.assoc_weights @ cbar
        m_count = int(self.feature_count.item())
        updates = int(self.assoc_updates.item())
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
        slab_var = self.spike_slab_scale * self.spike_slab_scale
        log_bf = (
            0.5 * float((torch.linalg.slogdet(s_reg + slab_var * eye)[1] - torch.linalg.slogdet(s_reg)[1]).item())
            - 0.5 * float(maha[best].item())
        )
        slab_gate = 1.0 / (1.0 + math.exp(-max(min(log_bf, 60.0), -60.0)))
        beta = disc_gate * slab_gate
        return (1.0 - beta) * z_lin + beta * protos[best]

    def _active_quantile_indices(self, counts: torch.Tensor, idx: torch.Tensor, num_particles: int):
        total = counts.sum()
        if int(total.item()) == 0:
            return None
        active = counts > 0
        last_active = int(torch.nonzero(active, as_tuple=False)[-1].item())
        probs = counts.double() / total
        if self.policy_entropy_bonus > 0.0:
            active_count = int(active.sum().item())
            if active_count > 1:
                uniform_active = active.double() / active_count
                probs = (1.0 - self.policy_entropy_bonus) * probs + self.policy_entropy_bonus * uniform_active
        cdf = probs.cumsum(0)
        cdf[last_active:] = 1.0
        quantiles = (idx.double() + 0.5) / num_particles
        return torch.searchsorted(cdf, quantiles, right=True).clamp_max(last_active)

    @contextmanager
    def training_target(self, frame):
        if self._target is not None:
            raise RuntimeError("a GAIL-prior update is already active")
        self._pending_reward = self._pending_style = self._pending_gain = None
        try:
            with super().training_target(frame):
                yield
        finally:
            self._pending_reward = self._pending_style = self._pending_gain = None

    def transform(self, codes, idx=None, *, base_prior, generator=None, training=True):
        if idx is None:
            raise ValueError("GAIL prior requires the task's particle row indices")
        centers = base_prior.means() if hasattr(base_prior, "means") else base_prior.z
        perturbation = codes - centers[idx]
        if training and self._target is not None and self._phase == 2:
            return self._pending_feature.unsqueeze(0) + perturbation
        context = self._pending_context if self._target is not None else self._current_context()

        if self.spike_slab_scale > 0.0 and context.numel() > 0:
            if int(self.assoc_updates.item()) == 0 or int(self.context_count.item()) == 0:
                return codes
            with torch.no_grad():
                z_ctx = self._prior_mean(context)
                bucket = self._lookup_context(context, require_match=False)
                if bucket is None:
                    return z_ctx.to(codes).unsqueeze(0) + perturbation
                features = self._active_quantile_indices(
                    self.context_outcomes[bucket], idx, base_prior.num_particles
                )
                if features is None:
                    return z_ctx.to(codes).unsqueeze(0) + perturbation
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
                gamma = visit_gate * self._disc_gate() * kernel
                if gamma <= 1e-8:
                    return z_ctx.to(codes).unsqueeze(0) + perturbation
                z_key = self._prior_mean(key)
                combined = z_ctx.unsqueeze(0) + gamma * (empirical - z_key.unsqueeze(0))
                return combined.to(codes) + perturbation

        count = int(self.context_count.item())
        if count == 0:
            return codes
        if context.numel() > 0:
            distances = torch.linalg.vector_norm(self.context_keys[:count] - context, dim=1)
            bucket = int(distances.argmin().item())
            min_dist = float(distances[bucket].item())
        else:
            bucket = 0
            min_dist = 0.0
        features = self._active_quantile_indices(
            self.context_outcomes[bucket], idx, base_prior.num_particles
        )
        if features is None:
            return codes
        selected_centers = self.feature_centroids[features]

        if (
            context.numel() > 0
            and self.assoc_ridge > 0.0
            and self.assoc_max_blend > 0.0
            and self.assoc_weights.numel() > 0
            and min_dist > self.context_tolerance
        ):
            gate = min(1.0, (min_dist - self.context_tolerance) / self.assoc_blend_scale) * self.assoc_max_blend
            aug = torch.cat([context.double(), context.new_ones(1, dtype=torch.float64)])
            assoc_pred = (self.assoc_weights.double() @ aug).to(selected_centers)
            max_norm = 4.0 * self.feature_scale
            pred_norm = float(torch.linalg.vector_norm(assoc_pred).item())
            if pred_norm > max_norm:
                assoc_pred = assoc_pred * (max_norm / pred_norm)
            selected_centers = (1.0 - gate) * selected_centers + gate * assoc_pred.unsqueeze(0)

        return selected_centers + perturbation

    def regularize(self, raw_rows, *, sampled_codes=None, indices=None, generated=None, logits=None):
        anchor_loss = super().regularize(raw_rows, generated=generated)
        if logits is None:
            if self.style_bonus_weight == 0.0 and not self.cancel_linear_gan:
                total = anchor_loss
            else:
                raise ValueError("GAIL style regularization requires critic logits")
        else:
            if not isinstance(logits, torch.Tensor) or not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
                raise ValueError("GAIL style regularization requires finite floating critic logits")
            style_bonus = torch.sigmoid(logits).mean()
            gain = self._causal_advantage_gain()
            total = anchor_loss - (self.style_bonus_weight * gain) * style_bonus
            if self.cancel_linear_gan:
                total = total + logits.mean() / self.penalty_weight
            with torch.no_grad():
                survival = math.exp(-min(50.0, float(anchor_loss.detach().item())))
                style_val = float(style_bonus.detach().item())
                self._pending_reward = survival + self.style_bonus_weight * style_val
                self._pending_style = style_val
                self._pending_gain = gain
        if self.context_bypass_weight > 0.0 and self.penalty_weight > 0.0:
            model = self._generator()
            fc = getattr(model, "fc", None)
            z_dim = int(getattr(model, "z_dim", 0))
            c_dim = int(getattr(model, "context_dim", 0))
            if c_dim > 0 and isinstance(fc, nn.Linear) and fc.weight.shape[1] == z_dim + c_dim:
                w_ctx = fc.weight[:, z_dim:]
                total = total + (self.context_bypass_weight / self.penalty_weight) * w_ctx.square().sum()
        return total

    @torch.no_grad()
    def _commit_observation(self, context, feature):
        super()._commit_observation(context, feature)
        if self._pending_reward is not None:
            reward = float(self._pending_reward)
            n = int(self.gail_updates.item())
            alpha = 1.0 - self.return_discount
            if n == 0:
                self.return_trace.fill_(reward)
                self.return_mean.fill_(reward)
                self.return_var.zero_()
            else:
                r_new = self.return_discount * float(self.return_trace.item()) + alpha * reward
                mean_old = float(self.return_mean.item())
                diff = r_new - mean_old
                mean_new = mean_old + alpha * diff
                var_new = (1.0 - alpha) * (float(self.return_var.item()) + alpha * diff * diff)
                self.return_trace.fill_(r_new)
                self.return_mean.fill_(mean_new)
                self.return_var.fill_(var_new)
            self.last_style_bonus.fill_(float(self._pending_style))
            self.last_advantage_gain.fill_(float(self._pending_gain))
            self.gail_updates.add_(1)

        self.assoc_updates.add_(1)
        step = int(self.assoc_updates.item())
        if context.numel() == 0 or self.assoc_ridge <= 0.0 or self.assoc_weights.numel() == 0:
            return
        aug = torch.cat([context.double(), context.new_ones(1, dtype=torch.float64)])
        cov = self.assoc_cov_inv.double()
        weights = self.assoc_weights.double()
        p_aug = cov @ aug
        denom = 1.0 + float((aug @ p_aug).clamp_min(0.0).item())
        kalman = p_aug / denom
        residual = feature.double() - (weights @ aug)
        weights = weights + torch.outer(residual, kalman)
        cov_next = cov - torch.outer(kalman, p_aug)
        if self.assoc_forgetting < 1.0:
            p0 = (1.0 / self.assoc_ridge) * torch.eye(len(aug), device=aug.device, dtype=torch.float64)
            cov_next = self.assoc_forgetting * cov_next + (1.0 - self.assoc_forgetting) * p0
        cov_next = 0.5 * (cov_next + cov_next.t())
        self.assoc_weights.copy_(weights.to(self.assoc_weights))
        self.assoc_cov_inv.copy_(cov_next.to(self.assoc_cov_inv))

        if self.spike_slab_scale > 0.0 and step > context.numel():
            post_err = feature.double() - (self.assoc_weights @ aug)
            self.residual_cov.mul_(1.0 - self.covariance_ema).add_(
                torch.outer(post_err, post_err), alpha=self.covariance_ema
            )
            f_count = int(self.feature_count.item())
            nearest_proto = int(
                torch.linalg.vector_norm(self.feature_centroids[:f_count] - feature, dim=1).argmin().item()
            )
            proto64 = self.feature_centroids[nearest_proto].to(dtype=torch.float64)
            q_err = (feature.double() - proto64).square().sum()
            self.quantization_ema.mul_(1.0 - self.covariance_ema).add_(q_err, alpha=self.covariance_ema)

    def diagnostics(self):
        result = super().diagnostics()
        result.update(
            critic_sampling="gail_context_conditional_prior",
            generator_sampling="current_observation_gail_posterior",
            gail_style_law=self.gail_style_law,
            gail_return_law=self.gail_return_law,
            assoc_extrapolation_law=self.assoc_extrapolation_law,
            gail_updates=int(self.gail_updates.item()),
            assoc_updates=int(self.assoc_updates.item()),
            return_trace=float(self.return_trace.item()),
            return_mean=float(self.return_mean.item()),
            return_std=float(self.return_var.clamp_min(0.0).sqrt().item()),
            last_style_bonus=float(self.last_style_bonus.item()),
            last_advantage_gain=float(self.last_advantage_gain.item()),
            assoc_active=bool(
                self.assoc_weights.numel() > 0 and (self.assoc_max_blend > 0.0 or self.spike_slab_scale > 0.0)
            ),
            assoc_weight_norm=(
                float(torch.linalg.vector_norm(self.assoc_weights).item())
                if self.assoc_weights.numel()
                else 0.0
            ),
            quantization_ema=float(self.quantization_ema.item()),
            residual_cov_trace=(
                float(torch.diagonal(self.residual_cov).sum().item())
                if self.residual_cov.numel()
                else 0.0
            ),
            observation_state_retained=(
                self._pending_context is not None
                or self._pending_feature is not None
                or self._pending_reward is not None
            ),
        )
        return result
