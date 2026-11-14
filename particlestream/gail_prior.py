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
   context vectors to latent feature expectations — via radial distance-gated blending,
   precision-weighted Mahalanobis spike-and-slab prototype transport, optional causally
   anchored two-pass Gram-Schmidt innovation subspaces, Matérn-3/2 kernel ridge regression
   over unquantized per-bucket latent means, Markov context-bypass regularization, and
   primal-dual polynomial Schedule-Free generator iterate averaging — without storing or
   replaying frames.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch
from torch import nn

from .feature_prior import FeatureConditionalReconstruction
from .transport import CompetitiveReconstruction


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
        "subspace_dims",
        "subspace_abs_tol",
        "subspace_rel_tol",
        "subspace_nu0",
        "residual_kernel",
        "residual_bandwidth",
        "residual_ridge",
        "krr_capacity",
        "subspace_birth_debias",
        "consistent_fork_branch",
        "disc_gate_style",
        "sobolev_weight",
        "sobolev_sigma",
        "dirichlet_bypass_weight",
        "boundary_quartic_gain",
        "dirichlet_c_boost",
        "schedule_free_beta",
        "schedule_free_poly_k",
        "schedule_free_c_min",
        "gail_style_law",
        "gail_return_law",
        "assoc_extrapolation_law",
        "schedule_free_law",
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
        subspace_dims: int = 0,
        subspace_abs_tol: float = 0.008,
        subspace_rel_tol: float = 0.12,
        subspace_nu0: float = 0.10,
        residual_kernel: str = "none",
        residual_bandwidth: float = 1.15,
        residual_ridge: float = 0.06,
        krr_capacity: int = 0,
        subspace_birth_debias: bool = False,
        consistent_fork_branch: bool = False,
        disc_gate_style: bool = False,
        sobolev_weight: float = 0.0,
        sobolev_sigma: float = 1e-7,
        dirichlet_bypass_weight: float = 0.0,
        boundary_quartic_gain: float = 0.0,
        dirichlet_c_boost: float = 0.0,
        schedule_free_beta: float = 0.0,
        schedule_free_poly_k: float = 4.0,
        schedule_free_c_min: float = 0.0,
        **kwargs,
    ):
        if not isinstance(cancel_linear_gan, bool):
            raise ValueError("cancel_linear_gan must be a boolean")
        if not isinstance(subspace_birth_debias, bool):
            raise ValueError("subspace_birth_debias must be a boolean")
        if not isinstance(consistent_fork_branch, bool):
            raise ValueError("consistent_fork_branch must be a boolean")
        if not isinstance(disc_gate_style, bool):
            raise ValueError("disc_gate_style must be a boolean")
        if type(subspace_dims) is not int or not 0 <= subspace_dims <= 64:
            raise ValueError("subspace_dims must be an integer in [0, 64]")
        if type(krr_capacity) is not int or not 0 <= krr_capacity <= 2048:
            raise ValueError("krr_capacity must be an integer in [0, 2048]")
        if residual_kernel not in ("none", "nw", "matern32_krr"):
            raise ValueError("residual_kernel must be 'none', 'nw', or 'matern32_krr'")
        if krr_capacity > 0 and residual_kernel != "matern32_krr":
            raise ValueError("krr_capacity > 0 requires residual_kernel='matern32_krr'")
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
            ("subspace_abs_tol", subspace_abs_tol),
            ("subspace_rel_tol", subspace_rel_tol),
            ("subspace_nu0", subspace_nu0),
            ("residual_bandwidth", residual_bandwidth),
            ("residual_ridge", residual_ridge),
            ("sobolev_weight", sobolev_weight),
            ("sobolev_sigma", sobolev_sigma),
            ("dirichlet_bypass_weight", dirichlet_bypass_weight),
            ("boundary_quartic_gain", boundary_quartic_gain),
            ("dirichlet_c_boost", dirichlet_c_boost),
            ("schedule_free_beta", schedule_free_beta),
            ("schedule_free_poly_k", schedule_free_poly_k),
            ("schedule_free_c_min", schedule_free_c_min),
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
        if subspace_birth_debias and (spike_slab_scale <= 0.0 or assoc_ridge <= 0.0):
            raise ValueError("subspace_birth_debias requires positive spike_slab_scale and assoc_ridge")
        if residual_kernel != "none" and spike_slab_scale <= 0.0:
            raise ValueError("continuous residual_kernel requires spike_slab_scale > 0")
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
        if sobolev_weight < 0.0 or dirichlet_bypass_weight < 0.0 or boundary_quartic_gain < 0.0:
            raise ValueError("Dirichlet Sobolev and calibration weights must be nonnegative")
        if not 0.0 <= dirichlet_c_boost <= 1.0:
            raise ValueError("dirichlet_c_boost must be in [0, 1]")
        for name, value in (
            ("subspace_abs_tol", subspace_abs_tol),
            ("subspace_rel_tol", subspace_rel_tol),
            ("subspace_nu0", subspace_nu0),
            ("residual_bandwidth", residual_bandwidth),
            ("residual_ridge", residual_ridge),
            ("sobolev_sigma", sobolev_sigma),
            ("schedule_free_poly_k", schedule_free_poly_k),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= schedule_free_beta < 1.0:
            raise ValueError("schedule_free_beta must be in [0, 1)")
        if not 0.0 <= schedule_free_c_min <= 1.0:
            raise ValueError("schedule_free_c_min must be in [0, 1]")

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
        self.subspace_dims = subspace_dims
        self.subspace_abs_tol = float(subspace_abs_tol)
        self.subspace_rel_tol = float(subspace_rel_tol)
        self.subspace_nu0 = float(subspace_nu0)
        self.residual_kernel = residual_kernel
        self.residual_bandwidth = float(residual_bandwidth)
        self.residual_ridge = float(residual_ridge)
        self.krr_capacity = krr_capacity
        self.subspace_birth_debias = subspace_birth_debias
        self.consistent_fork_branch = consistent_fork_branch
        self.disc_gate_style = disc_gate_style
        self.sobolev_weight = float(sobolev_weight)
        self.sobolev_sigma = float(sobolev_sigma)
        self.dirichlet_bypass_weight = float(dirichlet_bypass_weight)
        self.boundary_quartic_gain = float(boundary_quartic_gain)
        self.dirichlet_c_boost = float(dirichlet_c_boost)
        self.schedule_free_beta = float(schedule_free_beta)
        self.schedule_free_poly_k = float(schedule_free_poly_k)
        self.schedule_free_c_min = float(schedule_free_c_min)
        if self.subspace_dims > 0:
            self.encoder_law = "precision_whitened_gram_schmidt_subspace_plus_jl_v1"
        has_dirichlet = (
            self.sobolev_weight > 0.0
            or self.dirichlet_bypass_weight > 0.0
            or self.boundary_quartic_gain > 0.0
        )
        if has_dirichlet:
            self.gail_style_law = (
                "dirichlet_sobolev_quartic_survival_sigmoid_style_hinge_cancel_v1"
                if cancel_linear_gan
                else "dirichlet_sobolev_quartic_survival_additive_sigmoid_style_v1"
            )
        else:
            self.gail_style_law = (
                "prioritized_survival_sigmoid_style_hinge_cancel_v1"
                if cancel_linear_gan
                else "prioritized_survival_additive_sigmoid_style_v1"
            )
        self.gail_return_law = "causal_ema_standardized_advantage_tanh_v1"
        if self.spike_slab_scale > 0.0:
            if self.residual_kernel == "matern32_krr":
                if self.krr_capacity > 0 or self.subspace_birth_debias:
                    self.assoc_extrapolation_law = "debiased_subspace_rls_spike_slab_unbucketed_matern32_krr_v1"
                else:
                    self.assoc_extrapolation_law = "sherman_morrison_spike_slab_matern32_krr_v1"
            elif self.residual_kernel == "nw":
                self.assoc_extrapolation_law = "sherman_morrison_spike_slab_nw_residual_v1"
            else:
                self.assoc_extrapolation_law = "sherman_morrison_spike_slab_kernel_residual_v1"
        elif self.assoc_ridge > 0.0 and self.assoc_max_blend > 0.0:
            self.assoc_extrapolation_law = "recursive_ridge_fast_weight_radial_bound_v1"
        else:
            self.assoc_extrapolation_law = "disabled_v1"
        if self.schedule_free_beta > 0.0 and self.dirichlet_c_boost > 0.0:
            self.schedule_free_law = "dirichlet_gated_primal_dual_polynomial_schedule_free_v1"
        else:
            self.schedule_free_law = (
                "primal_dual_polynomial_schedule_free_v1"
                if self.schedule_free_beta > 0.0
                else "live_weights_v1"
            )

        use_f64 = self.spike_slab_scale > 0.0 or has_dirichlet or self.dirichlet_c_boost > 0.0
        buf_dtype = torch.float64 if use_f64 else torch.float32
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
        self.register_buffer("subspace_basis", torch.empty(0, 0))
        self.register_buffer("subspace_gains", torch.empty(0))
        self.register_buffer("subspace_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("context_features", torch.empty(0, 0, dtype=buf_dtype))
        self.register_buffer("krr_keys", torch.empty(0, 0, dtype=buf_dtype))
        self.register_buffer("krr_vals", torch.empty(0, 0, dtype=buf_dtype))
        self.register_buffer("krr_sub_k", torch.empty(0, dtype=torch.long))
        self.register_buffer("krr_step", torch.empty(0, dtype=torch.long))
        self.register_buffer("krr_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("row_cov_inv", torch.empty(0, 0, 0, dtype=buf_dtype))
        self.register_buffer("dirichlet_ema", torch.zeros((), dtype=buf_dtype))
        self.register_buffer("dirichlet_steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("sf_steps", torch.zeros((), dtype=torch.long))
        self._sf_param_count = 0
        if self.schedule_free_beta > 0.0:
            g_params = [p for p in generator.parameters() if p.requires_grad]
            self._sf_param_count = len(g_params)
            for idx_p, param in enumerate(g_params):
                self.register_buffer(f"sf_z_{idx_p}", param.detach().clone())
                self.register_buffer(f"sf_x_{idx_p}", param.detach().clone())
        self._pending_reward = None
        self._pending_style = None
        self._pending_gain = None
        self._pending_basis_row = None
        self._pending_basis_gain = None
        self._pending_dirichlet_energy = None

    def bind(self, base_prior):
        super().bind(base_prior)
        dimension = base_prior.z_dim
        if self.subspace_dims > dimension:
            raise ValueError("subspace_dims cannot exceed z_dim")
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
        self.subspace_count = self.subspace_count.to(device)
        self.krr_count = self.krr_count.to(device)
        self.dirichlet_steps = self.dirichlet_steps.to(device)
        self.sf_steps = self.sf_steps.to(device)
        use_f64 = (
            self.spike_slab_scale > 0.0
            or self.sobolev_weight > 0.0
            or self.dirichlet_bypass_weight > 0.0
            or self.boundary_quartic_gain > 0.0
            or self.dirichlet_c_boost > 0.0
        )
        dir_dtype = torch.float64 if use_f64 else z_ref.dtype
        self.dirichlet_ema = self.dirichlet_ema.to(device=device, dtype=dir_dtype)

        if self.subspace_dims > 0:
            if self.subspace_basis.numel():
                if (
                    tuple(self.subspace_basis.shape) != (self.subspace_dims, 64)
                    or tuple(self.subspace_gains.shape) != (self.subspace_dims,)
                ):
                    raise ValueError("GAIL subspace capacity differs from bound configuration")
                self.subspace_basis = self.subspace_basis.to(z_ref)
                self.subspace_gains = self.subspace_gains.to(z_ref)
            else:
                self.subspace_basis = z_ref.new_zeros(self.subspace_dims, 64)
                self.subspace_gains = z_ref.new_ones(self.subspace_dims)
        else:
            self.subspace_basis = self.subspace_basis.to(z_ref)
            self.subspace_gains = self.subspace_gains.to(z_ref)

        if self.residual_kernel != "none":
            if self.context_features.numel():
                if tuple(self.context_features.shape) != (self.context_capacity, dimension):
                    raise ValueError("GAIL context_features capacity differs from task prior")
                self.context_features = self.context_features.to(device=device, dtype=torch.float64)
            else:
                self.context_features = torch.zeros(
                    self.context_capacity, dimension, device=device, dtype=torch.float64
                )
        else:
            self.context_features = self.context_features.to(device=device, dtype=dir_dtype)

        if self.krr_capacity > 0:
            k_dim = max(1, context_dimension)
            if self.krr_keys.numel():
                if (
                    tuple(self.krr_keys.shape) != (self.krr_capacity, k_dim)
                    or tuple(self.krr_vals.shape) != (self.krr_capacity, dimension)
                    or tuple(self.krr_sub_k.shape) != (self.krr_capacity,)
                    or tuple(self.krr_step.shape) != (self.krr_capacity,)
                ):
                    raise ValueError("GAIL KRR buffer capacity differs from bound configuration")
                self.krr_keys = self.krr_keys.to(device=device, dtype=torch.float64)
                self.krr_vals = self.krr_vals.to(device=device, dtype=torch.float64)
                self.krr_sub_k = self.krr_sub_k.to(device=device)
                self.krr_step = self.krr_step.to(device=device)
            else:
                self.krr_keys = torch.zeros(self.krr_capacity, k_dim, device=device, dtype=torch.float64)
                self.krr_vals = torch.zeros(self.krr_capacity, dimension, device=device, dtype=torch.float64)
                self.krr_sub_k = torch.zeros(self.krr_capacity, device=device, dtype=torch.long)
                self.krr_step = torch.zeros(self.krr_capacity, device=device, dtype=torch.long)
        else:
            self.krr_keys = self.krr_keys.to(device=device, dtype=dir_dtype)
            self.krr_vals = self.krr_vals.to(device=device, dtype=dir_dtype)
            self.krr_sub_k = self.krr_sub_k.to(device=device)
            self.krr_step = self.krr_step.to(device=device)

        if self.subspace_birth_debias and self.spike_slab_scale > 0.0 and aug_dim > 0:
            expected_rc = (dimension, aug_dim, aug_dim)
            if self.row_cov_inv.numel():
                if tuple(self.row_cov_inv.shape) != expected_rc:
                    raise ValueError("GAIL row_cov_inv shape differs from bound configuration")
                self.row_cov_inv = self.row_cov_inv.to(device=device, dtype=torch.float64)
            else:
                base_eye = (1.0 / self.assoc_ridge) * torch.eye(aug_dim, device=device, dtype=torch.float64)
                self.row_cov_inv = base_eye.unsqueeze(0).repeat(dimension, 1, 1)
        else:
            self.row_cov_inv = self.row_cov_inv.to(device=device, dtype=dir_dtype)

        if self.schedule_free_beta > 0.0:
            model = self._generator()
            g_params = [] if model is None else [p for p in model.parameters() if p.requires_grad]
            if g_params and len(g_params) != self._sf_param_count:
                raise ValueError("generator trainable parameter count differs from Schedule-Free buffers")
            for idx_p in range(self._sf_param_count):
                ref = g_params[idx_p] if g_params else z_ref
                z_buf = getattr(self, f"sf_z_{idx_p}")
                x_buf = getattr(self, f"sf_x_{idx_p}")
                if g_params and tuple(z_buf.shape) != tuple(ref.shape):
                    raise ValueError("generator parameter shape differs from Schedule-Free buffer")
                setattr(self, f"sf_z_{idx_p}", z_buf.to(ref))
                setattr(self, f"sf_x_{idx_p}", x_buf.to(ref))

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
        if self.subspace_dims == 0:
            return super().encode(frame)
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
        rows[: self.subspace_dims].zero_()
        if k > 0:
            rows[:k].copy_(self.subspace_basis[:k] * self.subspace_gains[:k].unsqueeze(1))
        if self._pending_basis_row is not None and k < self.subspace_dims:
            rows[k].copy_(self._pending_basis_row * float(self._pending_basis_gain))
        return self.feature_scale * (v @ rows.t())

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
    def _prior_mean_with_branch(
        self, context: torch.Tensor, *, forced_branch: int | None = None
    ) -> tuple[torch.Tensor, int]:
        """Compute precision-gated spike-and-slab prior mean and active prototype branch."""
        c64 = context.detach().to(device=self.assoc_weights.device, dtype=torch.float64)
        cbar = torch.cat((c64, c64.new_ones(1)), dim=0)
        z_lin = self.assoc_weights @ cbar
        m_count = int(self.feature_count.item())
        updates = int(self.assoc_updates.item())
        if m_count == 0 or updates <= c64.numel():
            return z_lin, -1
        d_z = z_lin.numel()
        eye = torch.eye(d_z, device=z_lin.device, dtype=torch.float64)
        s_reg = 0.5 * (self.residual_cov + self.residual_cov.t()) + self.covariance_eps * eye
        protos = self.feature_centroids[:m_count].to(dtype=torch.float64)
        diff = protos - z_lin.unsqueeze(0)
        solved = torch.linalg.solve(s_reg, diff.t()).t()
        maha = (diff * solved).sum(dim=-1).clamp_min(0.0)
        branch = (
            int(maha.argmin().item())
            if forced_branch is None or forced_branch < 0 or forced_branch >= m_count
            else forced_branch
        )
        disc_gate = self._disc_gate()
        slab_var = self.spike_slab_scale * self.spike_slab_scale
        log_bf = (
            0.5 * float((torch.linalg.slogdet(s_reg + slab_var * eye)[1] - torch.linalg.slogdet(s_reg)[1]).item())
            - 0.5 * float(maha[branch].item())
        )
        slab_gate = 1.0 / (1.0 + math.exp(-max(min(log_bf, 60.0), -60.0)))
        beta = disc_gate * slab_gate
        return (1.0 - beta) * z_lin + beta * protos[branch], branch

    @torch.no_grad()
    def _prior_mean(self, context: torch.Tensor) -> torch.Tensor:
        """Compute precision-gated spike-and-slab prior mean for one 1D context."""
        return self._prior_mean_with_branch(context)[0]

    @torch.no_grad()
    def causal_dirichlet_precision(self, target: torch.Tensor) -> float:
        """Empirical-Bayes smoothness precision separating spatially constant vs structured streams."""
        t_row = target[:, :, 1:, :] - target[:, :, :-1, :]
        t_col = target[:, :, :, 1:] - target[:, :, :, :-1]
        e_now = 0.5 * (float(t_row.square().mean().item()) + float(t_col.square().mean().item()))
        e_ema = float(self.dirichlet_ema.item()) if int(self.dirichlet_steps.item()) > 0 else e_now
        e_eff = max(e_now, e_ema)
        return 1.0 / (1.0 + e_eff / (self.sobolev_sigma * self.sobolev_sigma))

    @staticmethod
    def _fisher_sobolev_seminorm(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Bernoulli-Fisher preconditioned discrete Sobolev H^1 spatial gradient seminorm."""
        t_row = target[:, :, 1:, :] - target[:, :, :-1, :]
        t_col = target[:, :, :, 1:] - target[:, :, :, :-1]
        with torch.no_grad():
            slope = (generated.detach() * (1.0 - generated.detach())).clamp_min(0.0)
            w_row = 0.25 / (0.5 * (slope[:, :, 1:, :] + slope[:, :, :-1, :]) + 0.015)
            w_col = 0.25 / (0.5 * (slope[:, :, :, 1:] + slope[:, :, :, :-1]) + 0.015)
        d_row = (generated[:, :, 1:, :] - generated[:, :, :-1, :]) - t_row
        d_col = (generated[:, :, :, 1:] - generated[:, :, :, :-1]) - t_col
        return 0.5 * ((w_row * d_row.square()).mean() + (w_col * d_col.square()).mean())

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
        self._pending_basis_row = self._pending_basis_gain = None
        self._pending_dirichlet_energy = None
        try:
            with CompetitiveReconstruction.training_target(self, frame):
                self._pending_context = self._current_context()
                if (
                    self.sobolev_weight > 0.0
                    or self.dirichlet_bypass_weight > 0.0
                    or self.boundary_quartic_gain > 0.0
                    or self.dirichlet_c_boost > 0.0
                ):
                    with torch.no_grad():
                        t_row = frame[:, :, 1:, :] - frame[:, :, :-1, :]
                        t_col = frame[:, :, :, 1:] - frame[:, :, :, :-1]
                        self._pending_dirichlet_energy = 0.5 * (
                            float(t_row.square().mean().item()) + float(t_col.square().mean().item())
                        )
                if self.subspace_dims > 0:
                    cand = self._candidate_innovation(frame)
                    if cand is not None:
                        self._pending_basis_row, self._pending_basis_gain = cand
                self._pending_feature = self.encode(frame)[0]
                yield
            self._commit_observation(self._pending_context, self._pending_feature)
        finally:
            self._pending_context = self._pending_feature = None
            self._pending_reward = self._pending_style = self._pending_gain = None
            self._pending_basis_row = self._pending_basis_gain = None
            self._pending_dirichlet_energy = None

    def transform(self, codes, idx=None, *, base_prior, generator=None, training=True):
        if idx is None:
            raise ValueError("GAIL prior requires the task's particle row indices")
        centers = base_prior.means() if hasattr(base_prior, "means") else base_prior.z
        perturbation = codes - centers[idx]
        if training and self._target is not None and self._phase == 2:
            return self._pending_feature.unsqueeze(0) + perturbation
        context = self._pending_context if self._target is not None else self._current_context()

        if self.spike_slab_scale > 0.0 and context.numel() > 0:
            updates = int(self.assoc_updates.item())
            c_dim = context.numel()
            if self.krr_capacity > 0:
                k_cnt = int(self.krr_count.item())
                if updates == 0 or k_cnt == 0:
                    return codes
                with torch.no_grad():
                    z_ctx, branch_c = self._prior_mean_with_branch(context)
                    g_disc = self._disc_gate()
                    z_base = z_ctx
                    if self.residual_kernel == "matern32_krr" and updates > c_dim and (1.0 - g_disc) > 1e-6:
                        if self.subspace_birth_debias:
                            now_k = int(self.subspace_count.item())
                            valid = self.krr_sub_k[:k_cnt] >= now_k
                            if int(valid.sum().item()) > c_dim:
                                keys64 = self.krr_keys[:k_cnt, :c_dim][valid]
                                vals64 = self.krr_vals[:k_cnt][valid]
                            else:
                                keys64 = self.krr_keys[:k_cnt, :c_dim]
                                vals64 = self.krr_vals[:k_cnt]
                        else:
                            keys64 = self.krr_keys[:k_cnt, :c_dim]
                            vals64 = self.krr_vals[:k_cnt]
                        n_pts = len(keys64)
                        c64 = context.detach().to(device=keys64.device, dtype=torch.float64)
                        k_aug = torch.cat((keys64, keys64.new_ones(n_pts, 1)), dim=-1)
                        res_keys = vals64 - (k_aug @ self.assoc_weights.t())
                        s3 = math.sqrt(3.0)
                        pw = s3 * torch.cdist(keys64, keys64) / self.residual_bandwidth
                        k_mat = (1.0 + pw) * torch.exp(-pw.clamp_max(80.0))
                        alpha = torch.linalg.solve(
                            k_mat + self.residual_ridge * torch.eye(n_pts, device=keys64.device, dtype=torch.float64),
                            res_keys,
                        )
                        dc = s3 * torch.linalg.vector_norm(keys64 - c64.unsqueeze(0), dim=-1) / self.residual_bandwidth
                        k_vec = (1.0 + dc) * torch.exp(-dc.clamp_max(80.0))
                        z_base = z_ctx + (1.0 - g_disc) * (k_vec @ alpha)

                    bucket = self._lookup_context(context, require_match=False)
                    if bucket is None:
                        return z_base.to(codes).unsqueeze(0) + perturbation
                    counts = self.context_outcomes[bucket]
                    features = self._active_quantile_indices(counts, idx, base_prior.num_particles)
                    if features is None:
                        return z_base.to(codes).unsqueeze(0) + perturbation
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
                    if (
                        self.consistent_fork_branch
                        and updates > 2 * c_dim
                        and int(self.feature_count.item()) < len(self.feature_centroids)
                        and g_disc > 0.5
                        and int((counts > 0).sum().item()) >= 2
                    ):
                        z_key, _ = self._prior_mean_with_branch(key, forced_branch=branch_c)
                    else:
                        z_key = self._prior_mean(key)
                    combined = z_base.unsqueeze(0) + gamma * (empirical - z_key.unsqueeze(0))
                    return combined.to(codes) + perturbation

            b_count = int(self.context_count.item())
            if updates == 0 or b_count == 0:
                return codes
            with torch.no_grad():
                z_ctx, branch_c = self._prior_mean_with_branch(context)
                g_disc = self._disc_gate()
                z_base = z_ctx
                if self.residual_kernel != "none" and updates > c_dim and (1.0 - g_disc) > 1e-6:
                    keys64 = self.context_keys[:b_count].to(dtype=torch.float64)
                    c64 = context.detach().to(device=keys64.device, dtype=torch.float64)
                    k_aug = torch.cat((keys64, keys64.new_ones(b_count, 1)), dim=-1)
                    res_keys = self.context_features[:b_count] - (k_aug @ self.assoc_weights.t())
                    visit_cap = self.context_visits[:b_count].to(dtype=torch.float64).clamp(min=1.0, max=4.0)
                    if self.residual_kernel == "matern32_krr":
                        pw_dist = torch.cdist(keys64, keys64)
                        s3 = math.sqrt(3.0)
                        scaled_pw = s3 * pw_dist / self.residual_bandwidth
                        k_mat = (1.0 + scaled_pw) * torch.exp(-scaled_pw.clamp_max(80.0))
                        alpha = torch.linalg.solve(
                            k_mat + torch.diag(self.residual_ridge / visit_cap), res_keys
                        )
                        d_c = torch.linalg.vector_norm(keys64 - c64.unsqueeze(0), dim=-1)
                        scaled_dc = s3 * d_c / self.residual_bandwidth
                        k_vec = (1.0 + scaled_dc) * torch.exp(-scaled_dc.clamp_max(80.0))
                        z_base = z_ctx + (1.0 - g_disc) * (k_vec @ alpha)
                    else:
                        dist_sq_all = (keys64 - c64.unsqueeze(0)).square().sum(dim=-1)
                        weights = visit_cap * torch.exp(
                            -(dist_sq_all / (2.0 * self.residual_bandwidth * self.residual_bandwidth)).clamp_max(80.0)
                        )
                        w_sum = float(weights.sum().item())
                        if w_sum > 1e-12:
                            r_nw = (weights.unsqueeze(-1) * res_keys).sum(dim=0) / (self.residual_ridge + w_sum)
                            z_base = z_ctx + (1.0 - g_disc) * r_nw

                bucket = self._lookup_context(context, require_match=False)
                if bucket is None:
                    return z_base.to(codes).unsqueeze(0) + perturbation
                counts = self.context_outcomes[bucket]
                features = self._active_quantile_indices(
                    counts, idx, base_prior.num_particles
                )
                if features is None:
                    return z_base.to(codes).unsqueeze(0) + perturbation
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
                if (
                    self.consistent_fork_branch
                    and updates > 2 * c_dim
                    and int(self.feature_count.item()) < len(self.feature_centroids)
                    and g_disc > 0.5
                    and int((counts > 0).sum().item()) >= 2
                ):
                    z_key, _ = self._prior_mean_with_branch(key, forced_branch=branch_c)
                else:
                    z_key = self._prior_mean(key)
                combined = z_base.unsqueeze(0) + gamma * (empirical - z_key.unsqueeze(0))
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
        raw_anchor = super().regularize(raw_rows, generated=generated)
        has_dirichlet = (
            self.sobolev_weight > 0.0
            or self.dirichlet_bypass_weight > 0.0
            or self.boundary_quartic_gain > 0.0
        )
        alpha_dir = 0.0
        if has_dirichlet and self._target is not None and generated is not None:
            alpha_dir = self.causal_dirichlet_precision(self._target)
            if alpha_dir > 1e-6:
                with torch.no_grad():
                    t_bar = float(self._target.mean().item())
                    w_bnd = 1.0 + self.boundary_quartic_gain * ((2.0 * t_bar - 1.0) ** 4)
                bce_scale = (1.0 - alpha_dir) + alpha_dir * w_bnd
                anchor_loss = bce_scale * raw_anchor
                if self.sobolev_weight > 0.0:
                    anchor_loss = anchor_loss + (alpha_dir * self.sobolev_weight) * self._fisher_sobolev_seminorm(
                        generated, self._target
                    )
            else:
                anchor_loss = raw_anchor
        else:
            anchor_loss = raw_anchor

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
            style_gate = (
                max(self._disc_gate(), alpha_dir)
                if (self.disc_gate_style and int(self.assoc_updates.item()) > 0 and self.spike_slab_scale > 0.0)
                else 1.0
            )
            eff_style_weight = self.style_bonus_weight * style_gate
            total = anchor_loss - (eff_style_weight * gain) * style_bonus
            if self.cancel_linear_gan:
                total = total + logits.mean() / self.penalty_weight
            with torch.no_grad():
                survival = math.exp(-min(50.0, float(raw_anchor.detach().item())))
                style_val = float(style_bonus.detach().item())
                self._pending_reward = survival + eff_style_weight * style_val
                self._pending_style = style_val
                self._pending_gain = gain
        eff_cbw = self.context_bypass_weight + (alpha_dir * self.dirichlet_bypass_weight if alpha_dir > 1e-6 else 0.0)
        if eff_cbw > 0.0 and self.penalty_weight > 0.0:
            model = self._generator()
            fc = getattr(model, "fc", None)
            z_dim = int(getattr(model, "z_dim", 0))
            c_dim = int(getattr(model, "context_dim", 0))
            if c_dim > 0 and isinstance(fc, nn.Linear) and fc.weight.shape[1] == z_dim + c_dim:
                w_ctx = fc.weight[:, z_dim:]
                total = total + (eff_cbw / self.penalty_weight) * w_ctx.square().sum()
        return total

    @torch.no_grad()
    def _commit_observation(self, context, feature):
        born_k = int(self.subspace_count.item()) if self._pending_basis_row is not None else -1
        prev_w = (
            self.assoc_weights.clone()
            if self.subspace_birth_debias and self.assoc_weights.numel() > 0
            else None
        )
        if self._pending_dirichlet_energy is not None:
            e_grad = float(self._pending_dirichlet_energy)
            if int(self.dirichlet_steps.item()) == 0:
                self.dirichlet_ema.fill_(e_grad)
            else:
                self.dirichlet_ema.mul_(0.9).add_(0.1 * e_grad)
            self.dirichlet_steps.add_(1)

        k = int(self.subspace_count.item())
        if self._pending_basis_row is not None and k < self.subspace_dims:
            self.subspace_basis[k].copy_(self._pending_basis_row)
            self.subspace_gains[k].fill_(float(self._pending_basis_gain))
            self.subspace_count.add_(1)
        super()._commit_observation(context, feature)
        if self.residual_kernel != "none":
            b_count = int(self.context_count.item())
            if context.numel() > 0 and b_count > 0:
                bucket = int(self.context_last_update[:b_count].argmax().item())
                c_visits = int(self.context_visits[bucket].item())
                z64 = feature.detach().to(device=self.context_features.device, dtype=torch.float64)
                if c_visits <= 1:
                    self.context_features[bucket].copy_(z64)
                else:
                    self.context_features[bucket].add_((z64 - self.context_features[bucket]) / c_visits)

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

        now_k = int(self.subspace_count.item())
        if self.subspace_birth_debias and prev_w is not None and self.row_cov_inv.numel() > 0:
            d_z = self.assoc_weights.shape[0]
            if 0 <= born_k < self.subspace_dims:
                prev_w[born_k].zero_()
                self.row_cov_inv[born_k].copy_(
                    (1.0 / self.assoc_ridge) * torch.eye(len(aug), device=aug.device, dtype=torch.float64)
                )
            for row_i in range(d_z):
                if row_i < self.subspace_dims and row_i >= now_k:
                    self.assoc_weights[row_i].zero_()
                    continue
                cov_i = self.row_cov_inv[row_i]
                p_aug_i = cov_i @ aug
                denom_i = 1.0 + float((aug @ p_aug_i).clamp_min(0.0).item())
                kalman_i = p_aug_i / denom_i
                res_i = float(feature[row_i].double().item()) - float((prev_w[row_i] @ aug).item())
                self.assoc_weights[row_i].copy_(prev_w[row_i] + res_i * kalman_i)
                cov_i_raw = cov_i - torch.outer(kalman_i, p_aug_i)
                if self.assoc_forgetting < 1.0:
                    p0 = (1.0 / self.assoc_ridge) * torch.eye(len(aug), device=aug.device, dtype=torch.float64)
                    cov_i_raw = self.assoc_forgetting * cov_i_raw + (1.0 - self.assoc_forgetting) * p0
                self.row_cov_inv[row_i].copy_(0.5 * (cov_i_raw + cov_i_raw.t()))

        if self.krr_capacity > 0:
            cnt = int(self.krr_count.item())
            c64 = context.detach().to(device=self.krr_keys.device, dtype=torch.float64)
            z64 = feature.detach().to(device=self.krr_vals.device, dtype=torch.float64)
            slot = cnt if cnt < self.krr_capacity else int(self.krr_step.argmin().item())
            if cnt < self.krr_capacity:
                self.krr_count.add_(1)
            self.krr_keys[slot, : len(c64)].copy_(c64)
            self.krr_vals[slot].copy_(z64)
            self.krr_sub_k[slot].fill_(now_k)
            self.krr_step[slot].copy_(self.assoc_updates)

    def diagnostics(self):
        result = super().diagnostics()
        k = int(self.subspace_count.item())
        result.update(
            critic_sampling="gail_context_conditional_prior",
            generator_sampling="current_observation_gail_posterior",
            gail_style_law=self.gail_style_law,
            gail_return_law=self.gail_return_law,
            assoc_extrapolation_law=self.assoc_extrapolation_law,
            schedule_free_law=self.schedule_free_law,
            gail_updates=int(self.gail_updates.item()),
            assoc_updates=int(self.assoc_updates.item()),
            subspace_count=k,
            subspace_gains=self.subspace_gains[:k].detach().cpu().tolist(),
            krr_count=int(self.krr_count.item()),
            dirichlet_steps=int(self.dirichlet_steps.item()),
            dirichlet_ema=float(self.dirichlet_ema.item()),
            schedule_free_steps=int(self.sf_steps.item()),
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
                or self._pending_basis_row is not None
                or self._pending_basis_gain is not None
                or self._pending_dirichlet_energy is not None
            ),
        )
        return result
