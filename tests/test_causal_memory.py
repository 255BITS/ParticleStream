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
                                     {"context_scale": float("nan")}, {"contrast_threshold": -.01},
                                     {"normalize_key": "yes"}, {"key_gain": 0.0}, {"key_gain": True},
                                     {"prev_val_weight": -0.1}, {"prev_val_weight": True},
                                     {"prev_val_weight": float("nan")},
                                     {"barycentric_kinematics": "yes"}, {"kinematic_alpha": 0.0},
                                     {"kinematic_alpha": 1.2}, {"kinematic_alpha": True},
                                     {"kinematic_beta": -0.01}, {"kinematic_beta": 1.2},
                                     {"kinematic_beta": True}])
def test_invalid_policy_settings_fail(settings):
    with pytest.raises(ValueError):
        CausalStateMemory(**settings)


def test_prev_val_weight_retains_preceding_frame_orthogonally_without_new_parameters():
    legacy = make_stream_memory({"kind": "fast_weight", "dim": 16})
    mem_pv = make_stream_memory({
        "kind": "fast_weight",
        "dim": 16,
        "reader": "causal_state",
        "association_weight": 0.0,
        "normalize_key": False,
        "key_gain": 0.5,
        "prev_val_weight": 1.0,
    })
    _init_module(mem_pv, NamedStreams(seed=0), "memory")
    assert {name: tuple(v.shape) for name, v in legacy.named_parameters()} == {
        name: tuple(v.shape) for name, v in mem_pv.named_parameters()
    }
    state_a = torch.full((1, 1, 8, 8), 0.5)
    state_a[:, :, :2, :] = 0.9
    state_b = torch.full((1, 1, 8, 8), 0.5)
    state_b[:, :, 4:6, :] = 0.1
    action = torch.full((1, 1, 8, 8), 0.5)
    action[:, :, 6:8, :2] = 0.8

    mem_pv.reset_state()
    _ = mem_pv.write(state_a)
    saved = {k: v.clone() for k, v in mem_pv.state_dict().items()}
    preview_a = mem_pv.read_context(action)
    assert all(torch.equal(v, mem_pv.state_dict()[k]) for k, v in saved.items())
    ctx_after_a = mem_pv.write(action)
    assert torch.equal(preview_a, ctx_after_a)

    mem_pv.reset_state()
    assert torch.count_nonzero(mem_pv.prev_val) == 0
    _ = mem_pv.write(state_b)
    ctx_after_b = mem_pv.write(action)
    assert (ctx_after_a - ctx_after_b).norm() > 0.5


def test_linear_key_mode_preserves_additive_superposition_and_gates_distractors():
    memory = initialized_memory(association_weight=0.0, normalize_key=False, key_gain=0.5)
    via_factory = make_stream_memory({
        "kind": "fast_weight", "dim": 16, "reader": "causal_state",
        "association_weight": 0.0, "normalize_key": False, "key_gain": 0.5,
    })
    assert via_factory.normalize_key is False and via_factory.key_gain == 0.5

    # Disjoint zero-mean patches should add linearly before clipping when normalize_key=False.
    patch_a = torch.full((1, 1, 8, 8), 0.5)
    patch_a[:, :, :2, :2] = 0.8
    patch_a[:, :, :2, 2:4] = 0.2
    patch_b = torch.full((1, 1, 8, 8), 0.5)
    patch_b[:, :, 4:6, :2] = 0.8
    patch_b[:, :, 4:6, 2:4] = 0.2
    combined = 0.5 + (patch_a - 0.5) + (patch_b - 0.5)

    memory.reset_state()
    ctx_a = memory.write(patch_a)
    memory.reset_state()
    ctx_b = memory.write(patch_b)
    memory.reset_state()
    ctx_ab = memory.write(combined)
    assert (ctx_ab[:, :-1].abs().max().item()) < memory.context_scale
    assert torch.allclose(ctx_ab[:, :-1], ctx_a[:, :-1] + ctx_b[:, :-1], atol=1e-6)

    # Low-contrast distractor still preserves retained salient key instead of overwriting with noise.
    distractor = torch.full((1, 1, 8, 8), 0.5)
    after_distractor = memory.write(distractor)
    assert torch.allclose(memory.prev_key, torch.zeros_like(memory.prev_key))
    assert memory.retained_key.norm() > 0.05
    assert after_distractor[:, :-1].norm() > 0.1


def test_barycentric_kinematics_tracks_wavepackets_and_leaves_non_wavepackets_unchanged():
    from particlestream.minimal_velocity import render_point

    legacy = make_stream_memory({"kind": "fast_weight", "dim": 16})
    mem_base = make_stream_memory({
        "kind": "fast_weight",
        "dim": 16,
        "reader": "causal_state",
        "association_weight": 0.0,
        "normalize_key": False,
        "key_gain": 0.5,
        "prev_val_weight": 1.0,
    })
    mem_kin = make_stream_memory({
        "kind": "fast_weight",
        "dim": 16,
        "reader": "causal_state",
        "association_weight": 0.0,
        "normalize_key": False,
        "key_gain": 0.5,
        "prev_val_weight": 1.0,
        "barycentric_kinematics": True,
        "kinematic_alpha": 0.20,
        "kinematic_beta": 0.04,
    })
    _init_module(mem_base, NamedStreams(seed=0), "memory")
    _init_module(mem_kin, NamedStreams(seed=0), "memory")
    assert {name: tuple(v.shape) for name, v in legacy.named_parameters()} == {
        name: tuple(v.shape) for name, v in mem_kin.named_parameters()
    }

    # Non-wavepacket frames (e.g. translating bars) produce byte-identical contexts, and single-frame
    # wavepacket writes (wp_age == 0 before write) also leave the first emitted context unchanged.
    bars, _ = translating_bar_cycle()
    for frame in bars[:4]:
        assert torch.equal(mem_base.write(frame.unsqueeze(0)), mem_kin.write(frame.unsqueeze(0)))
    assert int(mem_kin.wp_age.item()) == 0
    mem_base.reset_state()
    mem_kin.reset_state()

    wp0 = render_point(0.30)
    wp1 = render_point(0.35)
    wp2 = render_point(0.40)
    wp3 = render_point(0.45)
    assert torch.equal(mem_base.write(wp0), mem_kin.write(wp0))
    assert int(mem_kin.wp_age.item()) == 1

    # Preview read_context matches clone.write without mutating live buffers
    saved = {k: v.clone() for k, v in mem_kin.state_dict().items()}
    preview1 = mem_kin.read_context(wp1)
    assert all(torch.equal(v, mem_kin.state_dict()[k]) for k, v in saved.items())
    ctx1 = mem_kin.write(wp1)
    assert torch.equal(preview1, ctx1)
    ctx2 = mem_kin.write(wp2)
    ctx3 = mem_kin.write(wp3)

    # Constant-velocity trajectory has zero innovation at k=3 (exact 0.05 normalized/grid velocity = +0.05)
    assert float(ctx2[0, 0].item()) == pytest.approx(3.0 * (0.40 - 0.50), abs=1e-5)
    assert float(ctx2[0, 2].item()) == pytest.approx(3.0 * 4.0 * 0.05, abs=1e-5)
    assert float(ctx3[0, 0].item()) == pytest.approx(3.0 * (0.45 - 0.50), abs=1e-5)
    assert float(ctx3[0, 2].item()) == pytest.approx(3.0 * 4.0 * 0.05, abs=1e-5)

    # Closed-loop perturbed frame at k=4 damps position jitter by kinematic_alpha=0.20 (5x reduction)
    wp4_perturbed = (render_point(0.50 + 0.02) + 0.0005).clamp(0.0, 1.0)
    ctx4 = mem_kin.write(wp4_perturbed)
    tracked_x = float(ctx4[0, 0].item()) / 3.0 + 0.50
    assert abs(tracked_x - 0.50) < 0.006

    mem_kin.reset_state()
    assert all(torch.count_nonzero(buf) == 0 for buf in mem_kin.buffers())
