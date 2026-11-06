"""Context-local capacity tests with one current target and no training replay."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan.particle_prior import ParticlePrior
from particlegan.priors import ManipulatedPrior
from particlestream.balanced_transport import ContextBalancedReconstruction


class ColorTableGenerator(nn.Module):
    """A tiny test model whose selected scalar coefficient can fit the current target."""

    def __init__(self, conditional=False):
        super().__init__()
        self.context_dim = 2 if conditional else 0
        self.register_buffer("active_context", torch.zeros(1, max(1, self.context_dim)))
        self.register_buffer("values", torch.full((2 if conditional else 1, 32), .5))
        self.calls = []

    def context_row(self):
        return int(self.context_dim > 0 and self.active_context[0, 0] > .5)

    def forward(self, codes):
        self.calls.append((len(codes), self.training, torch.is_grad_enabled()))
        values = self.values[self.context_row(), codes[:, 0].round().long()]
        return values.reshape(-1, 1, 1, 1).expand(-1, 1, 8, 8)


def setup_router(conditional=False, **settings):
    model = ColorTableGenerator(conditional)
    storage = ParticlePrior(num_particles=32, z_dim=1, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        storage.z.copy_(torch.arange(32).reshape(32, 1))
    router = ContextBalancedReconstruction(model, **settings)
    prior = ManipulatedPrior(storage, [router])
    return model, router, prior


def current_update(model, router, prior, value, *, context=0, fit=True):
    if model.context_dim:
        model.active_context[0, 0] = context
    target = torch.full((1, 1, 8, 8), float(value))
    with router.training_target(target):
        prior.sample(1, generator=torch.Generator().manual_seed(0))
        code, selected = prior.sample(1, generator=torch.Generator().manual_seed(0))
        prior.regularize(prior.z, generated=model(code), indices=selected)
        if fit:
            # A model coefficient fits only the observation arriving in this call.
            model.values[model.context_row(), int(selected.item())] = value
    model.calls.clear()
    return int(selected.item())


def test_deterministic_context_trains_every_equal_mass_row():
    model, router, prior = setup_router()
    selected = [current_update(model, router, prior, .5) for _ in range(96)]
    assert set(selected) == set(range(32))
    assert router.context_count == 1
    assert router.context_visits[0] == 96
    assert torch.equal(router.context_assignment_mass[0], torch.full((32,), 3.))
    assert router.context_keys.shape == (64, 0), "unconditional density uses one global context"


@pytest.mark.parametrize("cycle,expected", [
    ((0, 1, 2, 3), (8, 8, 8, 8)),
    ((0, 1, 2) * 5 + (3,), (10, 10, 10, 2)),
])
def test_current_observation_frequency_allocates_particle_capacity(cycle, expected):
    model, router, prior = setup_router()
    colors = (.1, .35, .65, .9)
    for update in range(480):
        current_update(model, router, prior, colors[cycle[update % len(cycle)]])
    values = model.values[0]
    counts = torch.tensor([int(torch.isclose(values, torch.tensor(color)).sum()) for color in colors])
    target = torch.tensor(expected) / 32
    assert .5 * (counts / 32 - target).abs().sum() <= .1, counts.tolist()
    if expected[-1] == 2:
        assert 1 <= counts[-1] <= 3, "rare outcome must retain its declared small probability mass"
    else:
        assert torch.equal(counts, torch.tensor(expected))
    assert torch.count_nonzero(router.context_assignment_mass[0]) == 32
    assert int(counts.sum()) == 32
    assert router._target is None and router._pending_key is None


def test_capacity_balance_is_local_to_each_conditioning_state():
    model, router, prior = setup_router(conditional=True)
    for update in range(128):
        current_update(model, router, prior, .5, context=update % 2)
    assert router.context_count == 2
    assert torch.equal(router.context_assignment_mass[:2], torch.full((2, 32), 2.))
    assert torch.equal(router.context_duals[0], router.context_duals[1])
    before_second = router.context_duals[1].clone()
    current_update(model, router, prior, .5, context=0)
    assert not torch.equal(router.context_duals[0], before_second)
    assert torch.equal(router.context_duals[1], before_second)


@pytest.mark.parametrize("dual_rate,updates", [(.01, 1024), (.05, 480)])
def test_each_conditional_fork_allocates_both_outputs_with_equal_particle_mass(dual_rate, updates):
    model, router, prior = setup_router(conditional=True, dual_rate=dual_rate)
    for update in range(updates):
        context = update % 2
        # Opposite phase in two contexts must not be mistaken for global balance.
        color = .1 if ((update // 2 + context) % 2 == 0) else .9
        current_update(model, router, prior, color, context=context)
    for values in model.values:
        assert abs(int(torch.isclose(values, torch.tensor(.1)).sum()) - 16) <= 1
        assert abs(int(torch.isclose(values, torch.tensor(.9)).sum()) - 16) <= 1
    assert torch.count_nonzero(router.context_assignment_mass[:2], dim=1).tolist() == [32, 32]


def test_nearby_contexts_update_centroid_but_distinct_repetition_ages_stay_separate():
    model, router, prior = setup_router(conditional=True)
    current_update(model, router, prior, .5, context=.1)
    current_update(model, router, prior, .5, context=.2)
    assert router.context_count == 1
    assert router.context_keys[0, 0] == pytest.approx(.15)
    model.active_context[0, 1] = .375
    current_update(model, router, prior, .5, context=.15)
    assert router.context_count == 2, "a separate observed age must retain its own capacity budget"


def test_context_was_captured_before_d_and_target_is_never_saved_after_update():
    model, router, prior = setup_router(conditional=True)
    target = torch.full((1, 1, 8, 8), .5)
    with router.training_target(target):
        model.active_context[0, 0] = 1  # Simulate an accidental later buffer change.
        prior.sample(1, generator=torch.Generator().manual_seed(0))
        prior.sample(1, generator=torch.Generator().manual_seed(0))
    assert router.context_keys[0].tolist() == [0, 0]
    assert router._target is None and router._pending_key is None
    assert not any("target" in name for name in router.state_dict())
    assert all(not buffer.requires_grad and buffer.grad_fn is None for buffer in router.buffers())


def test_failed_update_does_not_allocate_evict_or_change_centroids_and_duals():
    model, router, prior = setup_router(conditional=True, context_capacity=1)
    current_update(model, router, prior, .5, context=0)
    before = {name: value.clone() for name, value in router.named_buffers() if not name.startswith("probe_")}
    model.active_context[0, 0] = 1
    with pytest.raises(RuntimeError, match="interrupted"):
        with router.training_target(torch.full((1, 1, 8, 8), .5)):
            prior.sample(1, generator=torch.Generator().manual_seed(0))
            prior.sample(1, generator=torch.Generator().manual_seed(0))
            raise RuntimeError("interrupted")
    assert all(torch.equal(value, dict(router.named_buffers())[name]) for name, value in before.items())
    assert router._target is None and router._pending_key is None
    current_update(model, router, prior, .5, context=1)
    assert router.context_evictions == 1 and router.context_count == 1
    assert router.context_visits[0] == 1
    assert router.context_keys[0, 0] == 1


def test_serving_preserves_all_context_state_and_prior_sampling_law():
    model, router, prior = setup_router(conditional=True)
    current_update(model, router, prior, .5)
    before = deepcopy(router.state_dict())
    serving = torch.Generator().manual_seed(0)
    reference = torch.Generator().manual_seed(0)
    _, row = prior.sample(1, generator=serving)
    assert torch.equal(row, torch.randint(0, 32, (1,), generator=reference))
    _, rows = prior.sample(32, fixed_first_n=True)
    assert rows.tolist() == list(range(32))
    assert all(torch.equal(value, router.state_dict()[name]) if isinstance(value, torch.Tensor)
               else value == router.state_dict()[name] for name, value in before.items())
    assert torch.equal(serving.get_state(), reference.get_state())


def test_checkpoint_roundtrip_and_reset_preserve_configuration():
    model, router, prior = setup_router(conditional=True)
    current_update(model, router, prior, .5, context=0)
    current_update(model, router, prior, .5, context=1)
    checkpoint = deepcopy(prior.state_dict())
    other_model, restored, restored_prior = setup_router(conditional=True)
    restored_prior.load_state_dict(checkpoint)
    assert restored.manipulator_spec() == router.manipulator_spec()
    assert all(torch.equal(value, dict(restored.named_buffers())[name]) for name, value in router.named_buffers())
    assert current_update(model, router, prior, .5, context=0, fit=False) == current_update(
        other_model, restored, restored_prior, .5, context=0, fit=False)
    restored.reset_state()
    assert all(torch.count_nonzero(buffer) == 0 for buffer in restored.buffers())
    assert restored.context_keys.shape == (64, 2) and restored.context_duals.shape == (64, 32)
    assert restored.manipulator_spec() == router.manipulator_spec()
    _, _, differently_configured = setup_router(conditional=True, dual_rate=.02)
    with pytest.raises(ValueError, match="configuration mismatch"):
        differently_configured.load_state_dict(checkpoint)


def test_duals_and_context_storage_remain_bounded_under_extreme_repeated_costs():
    model, router, prior = setup_router(conditional=True, context_capacity=2, dual_rate=10., dual_clip=.02)
    model.values.fill_(.9)
    model.values[:, 0] = .1
    for update in range(48):
        current_update(model, router, prior, .1, context=(update % 3) * 100, fit=False)
    assert router.context_count <= 2
    assert router.context_duals.abs().max() <= .0200001
    assert router.context_duals.sum(dim=1).abs().max() < 1e-6
    assert router.context_keys.abs().max() <= 8
    assert router.context_duals.shape == (2, 32)
    assert all(torch.isfinite(buffer).all() for buffer in router.buffers())


@pytest.mark.parametrize("settings", [{"context_capacity": 0}, {"context_capacity": 1.5}, {"context_capacity": float("inf")},
                                     {"context_tolerance": -.1}, {"dual_rate": float("nan")},
                                     {"dual_clip": 0}, {"context_clip": 0}])
def test_invalid_configurations_fail(settings):
    with pytest.raises(ValueError):
        setup_router(**settings)
