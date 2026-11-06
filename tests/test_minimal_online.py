"""Causal pair boundaries and evaluation isolation without training experiments."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from particlegan_forge import NamedStreams

from particlestream.memory import CausalStateMemory
from particlestream.minimal_online import (
    PairedOnlineSession, isolated_session, prefix_context, prepare_prefix,
)
from particlestream.online import OnlineSession


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))
        self.context_dim = 16
        self.register_buffer("active_context", torch.zeros(1, 16))

    def set_context(self, context):
        self.active_context.copy_(context)

    def forward(self, value):
        return self.weight.expand(len(value), 1, 8, 8)


class Prior(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("feature_visits", torch.zeros(4, dtype=torch.long))
        self._pending_feature = None
        self._pending_context = None
        self._target = None
        self._selected = None
        self._phase = 0


class Trainer:
    def __init__(self, generator, critic):
        self.recipe = SimpleNamespace(batch_size=1)
        self.generator = generator
        self.critic = critic
        self.prior = Prior()
        self.transport = self.prior
        self.noise_generator = torch.Generator().manual_seed(0)
        self.eval_generator = torch.Generator().manual_seed(1)
        self.trace = []
        self.fake_batch = 1

    def sample(self, count, *, generator=None):
        return self.generator(torch.zeros(count, 1))

    def step(self, frame):
        self.generator(torch.zeros(self.fake_batch, 1))
        self.critic(frame)
        self.trace.append((frame.clone(), self.generator.active_context.clone()))
        with torch.no_grad():
            self.generator.weight.add_(frame.mean())
            self.prior.feature_visits[0].add_(1)
        return {"finite": True}


def session(*, paired=True):
    generator, critic = Network(), Network()
    memory = CausalStateMemory()
    trainer = Trainer(generator, critic)
    kind = PairedOnlineSession if paired else OnlineSession
    return kind(trainer, generator, critic, memory, serving_generator=trainer.eval_generator)


def frame(index):
    value = torch.zeros(1, 1, 8, 8)
    value[0, 0, index // 8, index % 8] = 1
    return value


def test_source_prediction_then_single_target_update_without_target_memory_write():
    pair = session()
    source, target = frame(1), frame(30)
    context = pair.source(source)
    assert pair.updates == pair.arrivals == 0
    assert pair.source_only_arrivals == 1
    assert not pair.trainer.trace
    prediction = pair.predict()
    assert prediction.shape == target.shape
    assert not pair.trainer.trace
    pair.observe_target(target)
    assert len(pair.trainer.trace) == 1
    assert torch.equal(pair.trainer.trace[0][0], target)
    assert torch.equal(pair.trainer.trace[0][1], context)
    assert torch.equal(pair.memory.last_frame, source)
    assert torch.equal(pair.context, context)
    receipt = pair.diagnostics()
    assert receipt["arrivals"] == 2
    assert receipt["target_arrivals"] == receipt["updates"] == 1
    assert receipt["source_only_arrivals"] == 1
    assert receipt["target_memory_writes"] == receipt["replayed_examples"] == 0
    assert receipt["max_real_batch"] == receipt["max_fake_batch"] == 1


def test_independent_sources_reset_history_without_erasing_learned_prior():
    pair = session()
    pair.source(frame(1))
    pair.predict()
    pair.observe_target(frame(30))
    pair.source(frame(2))
    fresh = CausalStateMemory()
    fresh.load_state_dict(pair.memory.state_dict())
    fresh.reset_state()
    expected = fresh.write(frame(2))
    assert torch.equal(pair.context, expected)
    assert int(pair.memory.reservoir_count) == 1
    assert int(pair.trainer.prior.feature_visits[0]) == 1


def test_pair_rejects_out_of_order_or_duplicate_target():
    pair = session()
    with pytest.raises(RuntimeError, match="source"):
        pair.predict()
    with pytest.raises(RuntimeError, match="source then prediction"):
        pair.observe_target(frame(1))
    pair.source(frame(2))
    with pytest.raises(RuntimeError, match="requires its target"):
        pair.source(frame(3))
    with pytest.raises(RuntimeError, match="source then prediction"):
        pair.observe_target(frame(1))
    with pytest.raises(RuntimeError, match="paired sessions"):
        pair.observe(frame(1))
    pair.predict()
    pair.observe_target(frame(1))
    with pytest.raises(RuntimeError, match="source then prediction"):
        pair.observe_target(frame(1))


@pytest.mark.parametrize("bad", [torch.zeros(2, 1, 8, 8),
                                  torch.full((1, 1, 8, 8), float("nan")),
                                  torch.zeros(1, 1, 8, 8, requires_grad=True)])
def test_source_rejects_batched_nonfinite_or_attached_inputs(bad):
    pair = session()
    with pytest.raises(ValueError, match="detached finite"):
        pair.source(bad)
    assert pair.source_only_arrivals == pair.updates == 0


def test_public_step_rejects_fake_optimizer_batch_above_one():
    pair = session()
    pair.source(frame(2))
    pair.predict()
    pair.trainer.fake_batch = 2
    with pytest.raises(ValueError, match="batch_size=1"):
        pair.observe_target(frame(3))
    assert pair.updates == pair.arrivals == 0
    assert pair.memory is not None


def test_isolation_restores_parameters_gradients_modes_features_and_transients():
    pair = session()
    pair.source(frame(3))
    pair.generator.weight.grad = torch.tensor(0.3)
    before = {name: value.clone() for name, value in pair.memory.state_dict().items()}
    source_context = pair.context
    prior = pair.trainer.transport
    with isolated_session(pair):
        pair.generator.weight.add_(1)
        pair.generator.weight.grad.zero_()
        pair.generator.eval()
        pair.memory.write(frame(20))
        prior.feature_visits.fill_(99)
        prior._pending_feature = torch.ones(3)
        prior._pending_context = torch.ones(16)
        prior._target = frame(20)
        prior._selected, prior._phase = 2, 7
        pair.predict()
        with pytest.raises(RuntimeError, match="disabled"):
            pair.trainer.step(frame(2))
    assert pair.generator.training
    assert float(pair.generator.weight.detach()) == pytest.approx(0.1)
    assert float(pair.generator.weight.grad) == pytest.approx(0.3)
    assert all(torch.equal(value, before[name]) for name, value in pair.memory.state_dict().items())
    assert torch.equal(pair.context, source_context)
    assert int(prior.feature_visits.sum()) == 0
    assert prior._pending_feature is prior._pending_context is prior._target is None
    assert prior._selected is None and prior._phase == 0
    assert pair.predictions == 0 and not pair._prediction_ready
    pair.predict()
    pair.observe_target(frame(2))
    assert pair.updates == 1


def test_prefix_context_writes_only_clone_and_returns_live_state_after_rollout():
    live = session(paired=False)
    live.memory.write(frame(10))
    before = {name: value.clone() for name, value in live.memory.state_dict().items()}
    with prefix_context(live, torch.cat([frame(1), frame(2), frame(3)])) as simulated:
        assert simulated is not live.memory
        assert int(simulated.reservoir_count) == 3
        assert torch.equal(simulated.last_frame, frame(3))
        simulated.write(frame(4))
        assert int(simulated.reservoir_count) == 4
        assert all(torch.equal(value, before[name]) for name, value in live.memory.state_dict().items())
    assert live.context is None
    assert not live.trainer.trace
    assert all(torch.equal(value, before[name]) for name, value in live.memory.state_dict().items())


def test_training_prefix_is_source_only_and_caller_counts_warmups():
    live = session(paired=False)
    prepare_prefix(live, [frame(1), frame(2), frame(3)])
    assert int(live.memory.reservoir_count) == 3
    assert live.context is not None
    assert live.arrivals == live.updates == 0
    assert not live.trainer.trace


def test_invalid_prefix_does_not_partially_overwrite_live_memory():
    live = session(paired=False)
    live.memory.write(frame(20))
    before = {name: value.clone() for name, value in live.memory.state_dict().items()}
    with pytest.raises(ValueError, match="at least one"):
        prepare_prefix(live, [])
    with pytest.raises(ValueError, match="detached finite"):
        prepare_prefix(live, [frame(1), torch.zeros(2, 1, 8, 8)])
    assert all(torch.equal(value, before[name]) for name, value in live.memory.state_dict().items())


def test_rng_misuse_remains_visible_to_forge_evaluation_audit():
    pair = session()
    streams = NamedStreams(0)
    training_rng = streams.generator("noise")
    with streams.isolated_evaluation(), isolated_session(pair):
        torch.rand(1, generator=training_rng)
    assert streams.last_eval_audit["unintended_rng_deviations"] > 0


def test_extra_serving_does_not_change_subsequent_pair_learning():
    direct, served = session(), session()
    served.memory.load_state_dict(direct.memory.state_dict())
    for source, target in [(frame(1), frame(3)), (frame(2), frame(4))]:
        for pair in (direct, served):
            pair.source(source)
            pair.predict()
        with prefix_context(served, [frame(40), frame(50)]):
            served.predict()
        served.predict()
        direct.observe_target(target)
        served.observe_target(target)
    assert torch.equal(direct.generator.weight, served.generator.weight)
    assert torch.equal(direct.trainer.prior.feature_visits, served.trainer.prior.feature_visits)
    for left, right in zip(direct.trainer.trace, served.trainer.trace):
        assert all(torch.equal(a, b) for a, b in zip(left, right))
