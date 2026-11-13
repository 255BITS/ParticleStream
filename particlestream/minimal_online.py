"""Causal source/target observations and isolated probes for minimal tasks.

Independent pairs reset observation memory before their source arrives. A source
is available to serving, but never becomes an optimizer target. The later target
trains once against that source's detached context, and is not written to memory.
Named RNG isolation stays the caller's responsibility: use Forge's
``streams.isolated_evaluation()`` so forbidden advances remain visible to its audit.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import torch

from .online import OnlineSession


def _frame(frame):
    if (not isinstance(frame, torch.Tensor) or tuple(frame.shape) != (1, 1, 8, 8)
            or not frame.is_floating_point() or frame.requires_grad
            or not bool(torch.isfinite(frame).all())):
        raise ValueError("source observations require one detached finite floating (1,1,8,8) frame")
    return frame


def _zero_context(session):
    active = getattr(session.generator, "active_context", None)
    dimension = getattr(session.generator, "context_dim", 0)
    return torch.zeros_like(active) if active is not None and dimension else None


@torch.no_grad()
def prepare_prefix(session, frames, *, reset=True, memory=None):
    """Write already-arrived frames without learning; return the memory used.

    Supply a cloned ``memory`` for a simulation, or omit it to prepare the live
    training prefix. This function does not count arrivals: callers must record
    source-only/warm-up arrivals explicitly, separately from optimizer updates.
    """
    observed = []
    for frame in frames:
        if isinstance(frame, torch.Tensor) and tuple(frame.shape) == (1, 8, 8):
            frame = frame.unsqueeze(0)
        observed.append(_frame(frame))
    if not observed:
        raise ValueError("a source prefix requires at least one observed frame")
    memory = session.memory if memory is None else memory
    with session._lock:
        if memory is not None and reset:
            memory.reset_state()
        context = _zero_context(session)
        for frame in observed:
            if memory is not None:
                context = memory.write(frame).detach().clone()
        session.pending_context = context
        session._set_context(context)
        return memory


class PairedOnlineSession(OnlineSession):
    """One source-only arrival, prediction, and one later optimizer observation.

    Call ``source(frame)``, ``predict()``, then ``observe_target(target)``. The
    method boundary makes the target unavailable while preparing a prediction.
    Independent source pairs always reset memory, preserving the recipe's writer
    and reader within the declared one-source context.
    """

    def __init__(self, trainer, generator, critic, memory=None, *, serving_generator=None):
        super().__init__(trainer, generator, critic, memory, predictive=False,
                         serving_generator=serving_generator)
        self.source_only_arrivals = 0
        self._pair_open = False
        self._prediction_ready = False

    def source(self, frame):
        _frame(frame)
        with self._lock:
            if self._pair_open:
                raise RuntimeError("the previous source still requires its target")
            prepare_prefix(self, [frame], reset=True)
            self.source_only_arrivals += 1
            self._pair_open = True
            self._prediction_ready = False
            return self.context

    def predict(self):
        with self._lock:
            if not self._pair_open:
                raise RuntimeError("prediction requires an already-arrived source")
            prediction = super().predict()
            self._prediction_ready = True
            return prediction

    def observe_target(self, target):
        _frame(target)
        with self._lock:
            if not self._pair_open or not self._prediction_ready:
                raise RuntimeError("target learning requires source then prediction")
            # Reuse the original public step and batch guards while suppressing
            # its ordinary target-as-next-context write for independent pairs.
            memory = self.memory
            self.memory = None
            try:
                result = super().observe(target)
            finally:
                self.memory = memory
            self._pair_open = False
            self._prediction_ready = False
            return result

    def observe(self, frame, *, context_frame=None):
        raise RuntimeError("paired sessions require source(), predict(), observe_target()")

    def diagnostics(self):
        result = super().diagnostics()
        result.update(
            arrivals=self.arrivals + self.source_only_arrivals,
            target_arrivals=self.arrivals,
            source_only_arrivals=self.source_only_arrivals,
            warmup_arrivals=self.source_only_arrivals,
            observation_contract="independent_source_then_prediction_then_target_v1",
            target_memory_writes=0,
        )
        return result


def _module_roots(session):
    trainer = session.trainer
    roots = [session.generator, session.critic, session.memory]
    roots.extend(getattr(trainer, name, None)
                 for name in ("prior", "ema_G", "ema_prior", "transport"))
    return [root for root in roots if isinstance(root, torch.nn.Module)]


def _copy_state(value):
    return value.detach().clone() if isinstance(value, torch.Tensor) else deepcopy(value)


@contextmanager
def isolated_session(session):
    """Restore all learned/observation state and prohibit learning during a probe.

    Includes registered feature/prior statistics, parameters and gradients,
    per-module modes, contexts, diagnostic counts, and transport's transient
    Python state. Named generators are deliberately neither advanced nor restored
    here: an outer Forge evaluation audit detects any forbidden RNG consumption.
    """
    with session._lock:
        modules = {}
        for root in _module_roots(session):
            modules.update({id(module): module for module in root.modules()})
        tensors, modes, attributes = [], [], []
        for module in modules.values():
            modes.append((module, module.training))
            for name, parameter in module._parameters.items():
                if parameter is not None:
                    tensors.append((module, "_parameters", name, parameter,
                                    parameter.detach().clone(), parameter.requires_grad,
                                    None if parameter.grad is None else parameter.grad.detach().clone()))
            for name, buffer in module._buffers.items():
                if buffer is not None:
                    tensors.append((module, "_buffers", name, buffer,
                                    buffer.detach().clone(), None, None))
            for name, value in vars(module).items():
                if name in {"_target", "_phase", "_selected", "_pending_context", "_pending_feature",
                            "_pending_reward", "_pending_style", "_pending_gain",
                            "_pending_basis_row", "_pending_basis_gain"}:
                    attributes.append((module, name, _copy_state(value)))
        session_fields = ("pending_context", "arrivals", "updates", "predictions", "max_real_batch",
                          "max_fake_batch", "source_only_arrivals", "_pair_open", "_prediction_ready")
        saved_session = {name: _copy_state(getattr(session, name))
                         for name in session_fields if hasattr(session, name)}
        trainer = session.trainer
        had_instance_step = "step" in vars(trainer)
        instance_step = vars(trainer).get("step")

        def prohibit_learning(*args, **kwargs):
            raise RuntimeError("learning is disabled during isolated evaluation")

        trainer.step = prohibit_learning
        try:
            with torch.no_grad():
                yield session
        finally:
            if had_instance_step:
                trainer.step = instance_step
            else:
                del trainer.step
            with torch.no_grad():
                for module, kind, name, original, saved, requires_grad, grad in tensors:
                    getattr(module, kind)[name] = original
                    original.copy_(saved)
                    if kind == "_parameters":
                        original.requires_grad_(requires_grad)
                        original.grad = grad
                for module, name, value in attributes:
                    setattr(module, name, value)
                for module, training in modes:
                    module.training = training
                for name, value in saved_session.items():
                    setattr(session, name, value)


@contextmanager
def prefix_context(session, frames):
    """Condition on a prefix in cloned memory, then restore the live session.

    The yielded memory may consume generated frames to continue a rollout. Set
    the session's context from each cloned-memory write before the next sample.
    """
    with isolated_session(session):
        simulated = deepcopy(session.memory) if session.memory is not None else None
        prepare_prefix(session, frames, reset=True, memory=simulated)
        yield simulated
