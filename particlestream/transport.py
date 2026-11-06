"""Opt-in current-observation routing through ParticleGAN's public prior hooks.

The critic always samples the uniform model mixture. The generator phase may
select a particle using the single observation that has just arrived, and add
a reconstruction loss on that same generated image. No target survives the
update; serving retains the task's original prior sampling contract.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import weakref

import torch
import torch.nn.functional as F

from particlegan import GANTrainer
from particlegan.priors import ManipulatedPrior, PriorManipulator, unwrap_prior


class CompetitiveReconstruction(PriorManipulator):
    """One uniform D draw, then one target-routed G draw per public update."""

    config_fields = ("temperature", "usage_penalty", "usage_decay", "probe_law", "reconstruction_loss")

    def __init__(self, generator, *, reconstruction_weight=10.0, temperature=0.0,
                 usage_penalty=0.0, usage_decay=1.0, routing_generator=None,
                 reconstruction_loss="mse"):
        super().__init__(penalty_weight=reconstruction_weight)
        for name, value in (("temperature", temperature), ("usage_penalty", usage_penalty)):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if isinstance(usage_decay, bool) or not math.isfinite(usage_decay) or not 0 < usage_decay <= 1:
            raise ValueError("usage_decay must be in (0,1]")
        if temperature > 0 and routing_generator is None:
            raise ValueError("stochastic routing requires an explicit routing_generator")
        if reconstruction_loss not in ("mse", "binary_cross_entropy"):
            raise ValueError("reconstruction_loss must be 'mse' or 'binary_cross_entropy'")
        self.reconstruction_loss = reconstruction_loss
        self.temperature = float(temperature)
        self.usage_penalty = float(usage_penalty)
        self.usage_decay = float(usage_decay)
        self.probe_law = "sequential_clean_centers_mse_v1"
        self._generator = weakref.ref(generator)
        self.routing_generator = routing_generator
        self.register_buffer("assignment_mass", torch.empty(0))
        self.register_buffer("assignment_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("probe_forwards", torch.zeros((), dtype=torch.long))
        self.register_buffer("last_reconstruction", torch.zeros(()))
        self._target = None
        self._phase = 0
        self._selected = None

    def bind(self, base_prior):
        if self.assignment_mass.numel():
            if len(self.assignment_mass) != base_prior.num_particles:
                raise ValueError("router capacity differs from the storage prior")
        else:
            self.assignment_mass = base_prior.z.new_zeros(base_prior.num_particles)
            self.assignment_updates = self.assignment_updates.to(base_prior.z.device)
            self.probe_forwards = self.probe_forwards.to(base_prior.z.device)
            self.last_reconstruction = self.last_reconstruction.to(base_prior.z)

    @contextmanager
    def training_target(self, frame):
        if self._target is not None:
            raise RuntimeError("a transport update is already active")
        if (not isinstance(frame, torch.Tensor) or tuple(frame.shape) != (1, 1, 8, 8)
                or frame.requires_grad or not bool(torch.isfinite(frame).all())):
            raise ValueError("transport requires one detached finite arriving frame")
        if self.reconstruction_loss == "binary_cross_entropy" and (
                not frame.is_floating_point() or bool(((frame < 0) | (frame > 1)).any())):
            raise ValueError("binary_cross_entropy targets must be floating values in [0,1]")
        self._target = frame.detach().clone()
        self._phase, self._selected = 0, None
        try:
            yield
            if self._phase != 2 or self._selected is None:
                raise RuntimeError("transport requires one D draw and one G draw")
            with torch.no_grad():
                self.assignment_mass.mul_(self.usage_decay)
                self.assignment_mass[self._selected].add_(1)
                self.assignment_updates.add_(1)
        finally:
            self._target, self._selected, self._phase = None, None, 0

    @torch.no_grad()
    def _costs(self, base_prior):
        model = self._generator()
        if model is None:
            raise RuntimeError("the routing generator model no longer exists")
        modes = [(module, module.training) for module in model.modules()]
        buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
        device = base_prior.z.device
        devices = [device.index] if device.type == "cuda" else []
        centers = base_prior.means() if hasattr(base_prior, "means") else base_prior.z
        costs = []
        try:
            model.eval()
            with torch.random.fork_rng(devices=devices):
                for row in range(base_prior.num_particles):
                    prediction = model(centers[row:row + 1])
                    if tuple(prediction.shape) != tuple(self._target.shape):
                        raise ValueError("routing probe must generate exactly one image")
                    costs.append((prediction - self._target).square().mean())
                    self.probe_forwards.add_(1)
        finally:
            for buffer, saved in buffers:
                buffer.copy_(saved)
            for module, training in modes:
                module.training = training
        scores = torch.stack(costs)
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("routing probes must be finite")
        return scores + self.usage_penalty * torch.log1p(self.assignment_mass)

    def sample_indices(self, base_prior, batch_size, *, generator=None):
        if self._target is None:
            return None
        if batch_size != 1:
            raise ValueError("transport optimizer sampling requires batch_size=1")
        self._phase += 1
        if self._phase == 1:
            return None  # Preserve a uniform mixture for the critic's fake draw.
        if self._phase != 2:
            raise RuntimeError("unexpected extra optimizer sampling phase")
        with torch.no_grad():
            scores = self._costs(base_prior)
            if self.temperature:
                probabilities = torch.softmax(-scores / self.temperature, dim=0)
                selected = torch.multinomial(probabilities, 1, generator=self.routing_generator)
            else:
                selected = scores.argmin().reshape(1)
            self._selected = int(selected.item())
            return selected.detach().clone()

    def regularize(self, raw_rows, *, generated=None, **kwargs):
        if self._target is None or self._phase != 2 or generated is None:
            raise RuntimeError("reconstruction requires the active G-phase observation")
        if tuple(generated.shape) != tuple(self._target.shape):
            raise ValueError("reconstruction requires one generated image")
        if self.reconstruction_loss == "mse":
            loss = (generated - self._target).square().mean()
        else:
            if not generated.is_floating_point() or not bool(torch.isfinite(generated).all()):
                raise ValueError("binary_cross_entropy reconstruction requires finite floating predictions")
            epsilon = torch.finfo(generated.dtype).eps
            loss = F.binary_cross_entropy(generated.clamp(epsilon, 1 - epsilon), self._target)
        with torch.no_grad():
            self.last_reconstruction.copy_(loss.detach())
        return loss

    def diagnostics(self):
        return {
            "manipulator": self.manipulator_spec(),
            "critic_sampling": "uniform", "generator_sampling": "current_target_competition",
            "probe_forwards": int(self.probe_forwards.item()),
            "assignment_updates": int(self.assignment_updates.item()),
            "assignment_mass": self.assignment_mass.detach().cpu().tolist(),
            f"last_reconstruction_{self.reconstruction_loss}": float(self.last_reconstruction.item()),
            "target_retained": self._target is not None,
        }


class TransportGANTrainer(GANTrainer):
    """Public GANTrainer with an observation-scoped prior manipulator.

    The public trainer remains responsible for its one critic update, one
    generator/prior update, losses, schedules and optimizer state. Routing
    contributes only the declared prior hooks and their inference-only probes.
    """

    def __init__(self, recipe, generator, discriminator, *, prior, transport=None,
                 routing_generator=None, router_class=CompetitiveReconstruction, **kwargs):
        if isinstance(prior, ManipulatedPrior):
            raise ValueError("transport currently requires an unwrapped storage prior")
        if recipe.batch_size != 1:
            raise ValueError("transport requires recipe.batch_size=1")
        if not isinstance(router_class, type) or not issubclass(router_class, CompetitiveReconstruction):
            raise ValueError("router_class must extend CompetitiveReconstruction")
        self.transport = router_class(
            generator, routing_generator=routing_generator, **dict(transport or {}),
        )
        wrapped = ManipulatedPrior(unwrap_prior(prior), [self.transport])
        super().__init__(recipe, generator, discriminator, prior=wrapped, **kwargs)
        self.routing_generator = routing_generator
        if routing_generator is not None:
            if any(routing_generator is getattr(self, name) for name in self._STREAMS):
                raise ValueError("routing requires a stream separate from training and evaluation")
            self._STREAMS += ("routing_generator",)

    def step(self, real, *, generator_real=None, collect_stats=False):
        if generator_real is not None:
            raise ValueError("transport uses only the current arriving target")
        with self.transport.training_target(real):
            return super().step(real, collect_stats=collect_stats)

    def transport_diagnostics(self):
        return self.transport.diagnostics()
