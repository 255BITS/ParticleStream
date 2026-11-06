"""Causal memory preserves useful history without replay or cross-step gradients."""
from copy import deepcopy

import pytest
import torch

from particlegan_forge import NamedStreams
from particlestream.challenges import DelayedCueStream, PersistentForkStream
from particlestream.memory import CausalStateMemory, make_stream_memory
from particlestream.models import FastWeightMemory
from particlestream.streams import translating_bar_cycle
from particlestream.tier1 import _init_module


def initialized_memory(**kwargs):
    memory = CausalStateMemory(**kwargs)
    _init_module(memory, NamedStreams(seed=0), "memory")
    return memory


def contexts_for_history(memory, history):
    return [memory.write(frame.unsqueeze(0)) for frame in history]


def test_factory_is_explicit_and_preserves_legacy_parameter_shapes():
    legacy = make_stream_memory({"kind": "fast_weight", "dim": 16})
    causal = make_stream_memory({"kind": "fast_weight", "dim": 16, "reader": "causal_state"})
    assert type(legacy) is FastWeightMemory
    assert isinstance(causal, CausalStateMemory)
    assert make_stream_memory({"kind": "none"}) is None
    assert {name: tuple(value.shape) for name, value in legacy.named_parameters()} == {
        name: tuple(value.shape) for name, value in causal.named_parameters()}
    assert causal.active_context.shape == (1, 16)
    with pytest.raises(ValueError, match="unknown memory reader"):
        make_stream_memory({"kind": "fast_weight", "reader": "typo"})
    with pytest.raises(ValueError, match="unknown memory kind"):
        make_stream_memory({"kind": "typo"})


def test_delayed_cue_survives_four_identical_distractors_and_erasure_is_destructive():
    stream = DelayedCueStream()
    prototype = initialized_memory()
    retained = []
    erased = []
    for cue in (0, 1):
        memory = deepcopy(prototype)
        context = contexts_for_history(memory, stream.clean_history(cue))[-1]
        assert context.shape == (1, 16)
        assert memory.repetition_age.item() == 3
        assert torch.allclose(memory.prev_key, torch.zeros_like(memory.prev_key))
        assert memory.retained_key.norm() > 0.99
        retained.append(context)
        memory.reset_state()
        erased.append(contexts_for_history(memory, stream.distractor.repeat(4, 1, 1, 1))[-1])
    assert (retained[0] - retained[1]).norm() > 1
    assert torch.equal(erased[0], erased[1]), "erasing the cue must remove the only distinguishing observation"


def test_noise_robust_cue_identity_and_delay_age():
    stream = DelayedCueStream()
    prototype = initialized_memory()
    clean = [contexts_for_history(deepcopy(prototype), stream.clean_history(cue))[-1]
             for cue in (0, 1)]
    rng = torch.Generator().manual_seed(0)
    for cue in (0, 1):
        for _ in range(16):
            memory = deepcopy(prototype)
            history = stream.clean_history(cue)
            history = (history + 0.01 * torch.randn(history.shape, generator=rng)).clamp(0, 1)
            context = contexts_for_history(memory, history)[-1]
            distances = [(context - template).norm() for template in clean]
            assert distances[cue] < distances[1 - cue]
            assert memory.repetition_age.item() == 3
    memory = deepcopy(prototype)
    contexts = contexts_for_history(memory, stream.clean_history(0))
    assert [round(float(context[0, -1]), 6) for context in contexts] == [0, 0, .375, .75, 1.125]


def test_current_position_and_direction_remain_distinguishable():
    positions, _ = translating_bar_cycle()
    prototype = initialized_memory()
    forward = contexts_for_history(deepcopy(prototype), positions[[0, 1]])[-1]
    backward = contexts_for_history(deepcopy(prototype), positions[[2, 1]])[-1]
    other_position = contexts_for_history(deepcopy(prototype), positions[[1, 2]])[-1]
    assert (forward - backward).norm() > .5
    assert (forward - other_position).norm() > .5
    static = contexts_for_history(deepcopy(prototype), positions[[1, 1]])[-1]
    assert static[:, :-1].norm() > 1, "static appearance must survive when velocity is zero"


def test_associative_matrix_is_actually_read_and_query_never_mutates_state():
    memory = initialized_memory()
    positions, _ = translating_bar_cycle()
    contexts_for_history(memory, positions[[0, 1, 2, 1]])
    before = {name: value.clone() for name, value in memory.state_dict().items()}
    preview = memory.read_context(positions[2:3])
    assert all(torch.equal(value, memory.state_dict()[name]) for name, value in before.items())
    ablated = deepcopy(memory)
    ablated.matrix.zero_()
    assert not torch.equal(preview, ablated.read_context(positions[2:3])), "matrix ablation must affect the read"
    clone = deepcopy(memory)
    assert torch.equal(preview, clone.write(positions[2:3]))
    copied_context = memory.read_context()
    copied_context.zero_()
    assert memory.active_context.norm() > 0


def test_write_detaches_all_history_and_returned_context_is_independent():
    memory = initialized_memory()
    positions, _ = translating_bar_cycle()
    incoming = positions[0:1].clone().requires_grad_()
    context = memory.write(incoming)
    assert not context.requires_grad and context.grad_fn is None
    memory.write(positions[1:2])
    assert all(not buffer.requires_grad and buffer.grad_fn is None for buffer in memory.buffers())
    assert all(parameter.grad is None for parameter in memory.parameters())
    context.zero_()
    assert memory.active_context.norm() > 0
    assert incoming.grad is None


def test_clone_can_follow_generated_frames_without_changing_live_history():
    stream = PersistentForkStream()
    memory = initialized_memory()
    memory.write(stream.branch_paths[0, -1:])
    memory.write(stream.stem)
    before = {name: value.clone() for name, value in memory.state_dict().items()}
    simulations = [deepcopy(memory), deepcopy(memory)]
    for branch, simulated in enumerate(simulations):
        for image in stream.branch_paths[branch]:
            generated = (.9 * image + .1 * .1).unsqueeze(0).detach()
            context = simulated.write(generated)
            assert torch.isfinite(context).all()
            assert context.abs().max() <= simulated.context_scale
    assert not torch.equal(simulations[0].active_context, simulations[1].active_context)
    assert all(torch.equal(value, memory.state_dict()[name]) for name, value in before.items())


def test_equal_observed_prefixes_have_equal_state_regardless_of_unseen_future():
    stream = DelayedCueStream()
    first, second = initialized_memory(), initialized_memory()
    for frame in stream.clean_history(0):
        assert torch.equal(first.write(frame.unsqueeze(0)), second.write(frame.unsqueeze(0)))
    assert all(torch.equal(value, second.state_dict()[name]) for name, value in first.state_dict().items())
    # Only after a different observation actually arrives can the state differ.
    assert not torch.equal(first.write(stream.outcomes[0:1]), second.write(stream.outcomes[1:2]))


def test_reset_clears_all_mutable_history_and_long_repeats_stay_bounded():
    memory = initialized_memory(reservoir_size=3)
    stream = DelayedCueStream()
    memory.write(stream.cues[0:1])
    for _ in range(128):
        context = memory.write(stream.distractor)
        assert torch.isfinite(context).all()
        assert context.abs().max() <= memory.context_scale
    assert memory.repetition_age.item() == memory.age_horizon
    assert memory.reservoir.shape == (3, 1, 8, 8)
    weights = {name: parameter.clone() for name, parameter in memory.named_parameters()}
    memory.reset_state()
    assert all(torch.count_nonzero(buffer) == 0 for buffer in memory.buffers())
    assert all(torch.equal(parameter, weights[name]) for name, parameter in memory.named_parameters())


@pytest.mark.parametrize("frame", [torch.zeros(2, 1, 8, 8), torch.zeros(1, 8, 8),
                                   torch.full((1, 1, 8, 8), float("nan")),
                                   torch.zeros(1, 1, 8, 8, dtype=torch.long)])
def test_invalid_observations_fail_without_mutation(frame):
    memory = initialized_memory()
    before = {name: value.clone() for name, value in memory.state_dict().items()}
    with pytest.raises(ValueError, match="one finite floating frame"):
        memory.write(frame)
    assert all(torch.equal(value, memory.state_dict()[name]) for name, value in before.items())


@pytest.mark.parametrize("settings", [{"dim": 1}, {"age_horizon": 0}, {"decay": 1},
                                     {"context_scale": float("nan")}, {"contrast_threshold": -.01}])
def test_invalid_policy_settings_fail(settings):
    with pytest.raises(ValueError):
        CausalStateMemory(**settings)
