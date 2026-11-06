"""Serve and learn from one chronological observation at a time.

The lock serializes a serving call with an optimizer update. It does not batch,
queue, shuffle, replay, or accumulate examples. Predictive sessions retain only
the detached context prepared before the next observation arrives.
"""
from __future__ import annotations

from contextlib import contextmanager
from threading import RLock

import torch


def validate_execution_policy(candidate) -> None:
    """Historical replay declarations remain readable but cannot execute here."""
    policy = candidate.memory_policy
    if policy.get("replay_policy") != "none" or int(policy.get("rehearsal_batch", 0)) != 0:
        raise ValueError(
            f"{candidate.id}: execution requires explicit replay_policy='none' and "
            "rehearsal_batch=0; historical replay candidates are read-only"
        )


class OnlineSession:
    """One observation/update at a time, with inference between arrivals.

    ``observe`` trains the arriving frame against the preceding context in a
    predictive session; the first arrival only initializes that context. For
    marginal density tasks every arrival trains immediately. ``predict`` is
    optional and never changes the learning trajectory. Use a dedicated RNG,
    separate from the trainer's training generators, for serving.
    """

    def __init__(self, trainer, generator, critic, memory=None, *, predictive=False,
                 serving_generator=None):
        if getattr(trainer.recipe, "batch_size", 1) != 1:
            raise ValueError("online training requires recipe.batch_size=1")
        self.trainer = trainer
        self.generator = generator
        self.critic = critic
        self.memory = memory
        self.predictive = bool(predictive)
        effective_serving = serving_generator if serving_generator is not None else getattr(trainer, "eval_generator", None)
        for name, stream in vars(trainer).items():
            if name.endswith("generator") and name != "eval_generator" and stream is not None and stream is effective_serving:
                raise ValueError(f"serving RNG must be separate from training stream {name}")
        self.serving_generator = effective_serving
        self.pending_context = None
        self.arrivals = self.updates = self.predictions = 0
        self.max_real_batch = self.max_fake_batch = 0
        self._lock = RLock()

    @property
    def context(self):
        """An independent copy of the context used for the next target."""
        with self._lock:
            return None if self.pending_context is None else self.pending_context.detach().clone()

    def _set_context(self, context):
        if context is not None:
            self.generator.set_context(context)
            self.critic.set_context(context)

    @contextmanager
    def isolated_state(self):
        """Restore live memory, module buffers and modes after diagnostic probes."""
        with self._lock:
            roots = [self.generator, self.critic, self.memory]
            roots.extend(getattr(self.trainer, name, None) for name in ("prior", "ema_G", "ema_prior"))
            roots = list({id(root): root for root in roots if root is not None}.values())
            modules = [m for root in roots if root is not None for m in root.modules()]
            buffers = [(b, b.detach().clone()) for root in roots if root is not None for b in root.buffers()]
            modes = [(module, module.training) for module in modules]
            pending = None if self.pending_context is None else self.pending_context.clone()
            try:
                yield
            finally:
                with torch.no_grad():
                    for buffer, saved in buffers:
                        buffer.copy_(saved)
                for module, training in modes:
                    module.training = training
                self.pending_context = pending

    def predict(self):
        """Return one clean live prediction without learning or writing memory."""
        with self.isolated_state(), torch.no_grad():
            self._set_context(self.pending_context)
            result = self.trainer.sample(1, generator=self.serving_generator)
            if result.shape[0] != 1:
                raise ValueError("serving prediction must contain one sample")
            self.predictions += 1
            return result.detach().clone()

    def observe(self, frame):
        """Consume exactly one arriving frame; return update stats or warm-up None."""
        if not isinstance(frame, torch.Tensor) or tuple(frame.shape) != (1, 1, 8, 8):
            raise ValueError("observe requires one frame with shape (1,1,8,8)")
        if frame.requires_grad or not bool(torch.isfinite(frame).all()):
            raise ValueError("observations must be detached and finite")
        with self._lock:
            stats = None
            if not self.predictive or self.arrivals:
                self._set_context(self.pending_context)
                self.max_real_batch = max(self.max_real_batch, len(frame))

                def check_fake(module, args):
                    batch = len(args[0])
                    self.max_fake_batch = max(self.max_fake_batch, batch)
                    if batch != 1:
                        raise ValueError("optimizer generation requires batch_size=1")

                def check_critic(module, args):
                    if len(args[0]) != 1:
                        raise ValueError("optimizer critic requires batch_size=1")

                guard = self.generator.register_forward_pre_hook(check_fake)
                critic_guard = self.critic.register_forward_pre_hook(check_critic)
                try:
                    stats = self.trainer.step(frame.detach())
                finally:
                    guard.remove()
                    critic_guard.remove()
                self.updates += 1
            self.arrivals += 1
            if self.memory is not None:
                self.pending_context = self.memory.write(frame).detach().clone()
                self._set_context(self.pending_context)
            return stats

    def diagnostics(self):
        return {
            "arrivals": self.arrivals, "updates": self.updates,
            "warmup_arrivals": int(self.predictive), "live_predictions": self.predictions,
            "max_real_batch": self.max_real_batch, "max_fake_batch": self.max_fake_batch,
            "replayed_examples": 0, "gradient_accumulation_steps": 1,
        }
