"""A bounded observed-feature prior learned from one current image per update.

The generator learns a decoder through the current-observation posterior. The
critic and serving draw from context-conditioned feature prototypes learned
only from earlier successful arrivals. This changes the recipe's prior update
and loss policies, without storing images or replacing the task's networks.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch
import torch.nn.functional as F
from torch import nn

from particlegan import init

from .transport import CompetitiveReconstruction


class FeatureConditionalReconstruction(CompetitiveReconstruction):
    """Fixed feature encoding, online prototype means, and conditional counts.

    Every prior row retains its original MoG perturbation. G-phase rows use
    the feature code of the single arrived target. Other rows use deterministic
    empirical quantiles of previously learned features at the nearest observed
    context, falling back to the unchanged task prior before any observations.
    """

    additive_transform = True
    config_fields = CompetitiveReconstruction.config_fields + (
        "encoder_seed", "feature_scale", "image_center", "feature_tolerance",
        "context_capacity", "context_tolerance", "context_clip", "encoder_law",
        "prototype_law", "conditional_sampling_law",
        "serving_context_law",
    )

    def __init__(self, generator, *, encoder_seed=0, feature_scale=2.0,
                 image_center=0.5, feature_tolerance=0.15, context_capacity=64,
                 context_tolerance=0.25, context_clip=8.0, **kwargs):
        if type(encoder_seed) is not int or encoder_seed < 0:
            raise ValueError("encoder_seed must be a nonnegative named initialization seed")
        if type(context_capacity) is not int or context_capacity < 1:
            raise ValueError("context_capacity must be a positive integer")
        for name, value in (("feature_scale", feature_scale), ("feature_tolerance", feature_tolerance),
                            ("context_tolerance", context_tolerance), ("context_clip", context_clip)):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if feature_scale == 0 or context_clip == 0 or not math.isfinite(image_center):
            raise ValueError("feature scale and context clip must be positive; image center must be finite")
        super().__init__(generator, **kwargs)
        if self.temperature != 0 or self.usage_penalty != 0 or self.usage_decay != 1:
            raise ValueError("feature prior uses uniform rows; competition settings must retain defaults")
        self.encoder_seed = encoder_seed
        self.feature_scale = float(feature_scale)
        self.image_center = float(image_center)
        self.feature_tolerance = float(feature_tolerance)
        self.context_capacity = context_capacity
        self.context_tolerance = float(context_tolerance)
        self.context_clip = float(context_clip)
        self.encoder_law = "named_deterministic_orthogonal_unit_rows_v1"
        self.prototype_law = "bounded_nearest_mean_v1"
        self.conditional_sampling_law = "row_midpoint_empirical_quantiles_v1"
        self.serving_context_law = "nearest_observed_v1"
        self.probe_law = "no_generator_probes_v1"
        self.register_buffer("encoder", torch.empty(0, 0))
        self.register_buffer("feature_centroids", torch.empty(0, 0))
        self.register_buffer("feature_visits", torch.empty(0, dtype=torch.long))
        self.register_buffer("feature_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("context_keys", torch.empty(0, 0))
        self.register_buffer("context_outcomes", torch.empty(0, 0, dtype=torch.long))
        self.register_buffer("context_visits", torch.empty(0, dtype=torch.long))
        self.register_buffer("context_last_update", torch.empty(0, dtype=torch.long))
        self.register_buffer("context_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("context_evictions", torch.zeros((), dtype=torch.long))
        self._pending_context = self._pending_feature = None

    def _current_context(self):
        model = self._generator()
        if model is None:
            raise RuntimeError("the generator model no longer exists")
        active = getattr(model, "active_context", None)
        dimension = int(getattr(model, "context_dim", 0 if active is None else active.shape[-1]))
        if dimension == 0:
            return self.assignment_mass.new_empty(0)
        if active is None or tuple(active.shape) != (1, dimension) or not bool(torch.isfinite(active).all()):
            raise ValueError("feature prior requires one finite already-observed conditioning vector")
        return active[0].detach().to(self.assignment_mass).clamp(-self.context_clip, self.context_clip).clone()

    def bind(self, base_prior):
        super().bind(base_prior)
        count, dimension = base_prior.num_particles, base_prior.z_dim
        context_dimension = self._current_context().numel()
        if dimension > 64:
            raise ValueError("the fixed image encoder requires latent dimension <=64")
        if self.encoder.numel():
            if tuple(self.encoder.shape) != (dimension, 64) or len(self.feature_centroids) != count:
                raise ValueError("feature prior capacity differs from the task prior")
            return
        # Construction is isolated; only the named deterministic initializer
        # determines this fixed projection. The encoder contributes no params.
        with torch.random.fork_rng(devices=[]):
            layer = nn.Linear(64, dimension, bias=False)
            init.deterministic_orthogonal_(layer, parameter_seeds={"weight": self.encoder_seed})
            projection = F.normalize(layer.weight.detach(), dim=1)
        self.encoder = projection.to(base_prior.z).clone()
        self.feature_centroids = base_prior.z.new_zeros(count, dimension)
        self.feature_visits = torch.zeros(count, device=base_prior.z.device, dtype=torch.long)
        self.feature_count = self.feature_count.to(base_prior.z.device)
        self.context_keys = base_prior.z.new_zeros(self.context_capacity, context_dimension)
        self.context_outcomes = torch.zeros(self.context_capacity, count, device=base_prior.z.device, dtype=torch.long)
        self.context_visits = torch.zeros(self.context_capacity, device=base_prior.z.device, dtype=torch.long)
        self.context_last_update = torch.zeros_like(self.context_visits)
        self.context_count = self.context_count.to(base_prior.z.device)
        self.context_evictions = self.context_evictions.to(base_prior.z.device)

    @torch.no_grad()
    def encode(self, frame):
        if (not isinstance(frame, torch.Tensor) or tuple(frame.shape) != (1, 1, 8, 8)
                or not frame.is_floating_point() or not bool(torch.isfinite(frame).all())):
            raise ValueError("feature encoder requires one finite image")
        pixels = frame.detach().reshape(1, 64).to(self.encoder)
        return self.feature_scale * ((pixels - self.image_center) @ self.encoder.t())

    @torch.no_grad()
    def _lookup_context(self, key, *, require_match=True):
        count = int(self.context_count.item())
        if count == 0:
            return None
        distances = torch.linalg.vector_norm(self.context_keys[:count] - key, dim=1)
        nearest = int(distances.argmin().item())
        return nearest if not require_match or float(distances[nearest]) <= self.context_tolerance else None

    @contextmanager
    def training_target(self, frame):
        if self._target is not None:
            raise RuntimeError("a feature-prior update is already active")
        try:
            with super().training_target(frame):
                self._pending_context = self._current_context()
                self._pending_feature = self.encode(frame)[0]
                yield
            self._commit_observation(self._pending_context, self._pending_feature)
        finally:
            self._pending_context = self._pending_feature = None

    def sample_indices(self, base_prior, batch_size, *, generator=None):
        if self._target is None:
            return None
        if batch_size != 1:
            raise ValueError("feature-prior optimizer sampling requires batch_size=1")
        self._phase += 1
        if self._phase == 1:
            return None
        if self._phase != 2:
            raise RuntimeError("unexpected extra optimizer sampling phase")
        indices = base_prior.sample_indices(batch_size, generator=generator)
        self._selected = int(indices.item())
        return indices.detach().clone()

    def transform(self, codes, idx=None, *, base_prior, generator=None, training=True):
        if idx is None:
            raise ValueError("feature prior requires the task's particle row indices")
        centers = base_prior.means() if hasattr(base_prior, "means") else base_prior.z
        perturbation = codes - centers[idx]
        if training and self._target is not None and self._phase == 2:
            return self._pending_feature.unsqueeze(0) + perturbation
        context = self._pending_context if self._target is not None else self._current_context()
        bucket = self._lookup_context(context, require_match=False)
        if bucket is None:
            return codes
        counts = self.context_outcomes[bucket]
        total = counts.sum()
        if int(total.item()) == 0:
            return codes
        cdf = counts.double().cumsum(0) / total
        quantiles = (idx.double() + 0.5) / base_prior.num_particles
        features = torch.searchsorted(cdf, quantiles, right=True).clamp_max(base_prior.num_particles - 1)
        return self.feature_centroids[features] + perturbation

    @torch.no_grad()
    def _commit_observation(self, context, feature):
        count = int(self.feature_count.item())
        if count:
            distances = torch.linalg.vector_norm(self.feature_centroids[:count] - feature, dim=1)
            nearest = int(distances.argmin().item())
        else:
            distances, nearest = None, 0
        if count < len(self.feature_centroids) and (count == 0 or float(distances[nearest]) > self.feature_tolerance):
            nearest = count
            self.feature_count.add_(1)
        visits = int(self.feature_visits[nearest].item()) + 1
        self.feature_centroids[nearest].add_((feature - self.feature_centroids[nearest]) / visits)
        self.feature_visits[nearest].fill_(visits)
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
            self.context_visits[bucket].zero_()
        visits = int(self.context_visits[bucket].item()) + 1
        self.context_keys[bucket].add_((context - self.context_keys[bucket]) / visits)
        self.context_outcomes[bucket, nearest].add_(1)
        self.context_visits[bucket].fill_(visits)
        self.context_last_update[bucket].copy_(self.assignment_updates)

    def diagnostics(self):
        result = super().diagnostics()
        count = int(self.context_count.item())
        result.update(
            critic_sampling="observed_context_feature_prior", generator_sampling="current_observation_feature_posterior",
            serving_sampling=self.conditional_sampling_law, feature_count=int(self.feature_count.item()),
            feature_visits=self.feature_visits.detach().cpu().tolist(), context_count=count,
            context_evictions=int(self.context_evictions.item()),
            context_visits=self.context_visits[:count].detach().cpu().tolist(),
            context_outcomes=self.context_outcomes[:count].detach().cpu().tolist(),
            observation_state_retained=self._pending_context is not None or self._pending_feature is not None,
        )
        return result
