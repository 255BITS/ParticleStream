"""Current-target routing remains causal, batch-one, and absent during serving."""
from contextlib import contextmanager

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe
from particlegan.particle_prior import ParticlePrior
from particlegan.priors import ManipulatedPrior
from particlegan_forge import NamedStreams
from particlestream.challenge_adapter import _history_context
from particlestream.models import FastWeightMemory, StreamImageCritic, StreamImageGenerator
from particlestream.online import OnlineSession
from particlestream.streams import bars4_templates
from particlestream.tier1 import _init_module
from particlestream.transport import CompetitiveReconstruction, TransportGANTrainer


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class ProbeGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.register_buffer("forward_count", torch.zeros((), dtype=torch.long))
        self.calls = []

    def forward(self, codes):
        self.calls.append((len(codes), self.training, torch.is_grad_enabled()))
        self.forward_count.add_(1)
        return (codes[:, :1] * self.scale).reshape(-1, 1, 1, 1).expand(-1, 1, 8, 8)


def make_router(**settings):
    model = ProbeGenerator()
    prior = ParticlePrior(num_particles=3, z_dim=1, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        prior.z.copy_(torch.tensor([[.1], [.4], [.8]]))
    router = CompetitiveReconstruction(model, **settings)
    return model, router, ManipulatedPrior(prior, [router])


def test_only_g_routes_and_reconstruction_uses_single_current_target():
    model, router, prior = make_router(reconstruction_weight=10.)
    latent = torch.Generator().manual_seed(0)
    reference = torch.Generator().manual_seed(0)
    expected_d = torch.randint(0, 3, (1,), generator=reference)
    target = torch.full((1, 1, 8, 8), .75)
    with router.training_target(target):
        _, d_row = prior.sample(1, generator=latent)
        code, g_row = prior.sample(1, generator=latent)
        assert torch.equal(d_row, expected_d)
        assert g_row.tolist() == [2]
        assert torch.equal(latent.get_state(), reference.get_state()), "G routing must not redraw D's RNG"
        assert model.forward_count == 0, "inference probes leaked model buffer writes"
        assert model.training is True
        assert model.calls == [(1, False, False)] * 3
        generated = model(code)
        loss = prior.regularize(prior.z, generated=generated, indices=g_row)
        assert torch.allclose(loss, torch.tensor(.025))
        loss.backward()
        assert model.scale.grad is not None and torch.isfinite(model.scale.grad)
        assert prior.z.grad is not None and torch.isfinite(prior.z.grad).all()
    diagnostics = router.diagnostics()
    assert diagnostics["assignment_mass"] == [0., 0., 1.]
    assert diagnostics["probe_forwards"] == 3
    assert diagnostics["assignment_updates"] == 1
    assert diagnostics["target_retained"] is False
    assert router._target is None


def test_serving_samples_original_prior_without_target_or_probes():
    _, router, prior = make_router()
    with router.training_target(torch.full((1, 1, 8, 8), .75)):
        prior.sample(1, generator=torch.Generator().manual_seed(0))
        prior.sample(1, generator=torch.Generator().manual_seed(0))
    before = router.probe_forwards.clone()
    serving = torch.Generator().manual_seed(0)
    reference = torch.Generator().manual_seed(0)
    _, sampled = prior.sample(1, generator=serving)
    assert torch.equal(sampled, torch.randint(0, 3, (1,), generator=reference))
    _, enumerated = prior.sample(3, fixed_first_n=True)
    assert enumerated.tolist() == [0, 1, 2]
    assert torch.equal(router.probe_forwards, before)


def test_target_does_not_survive_a_failed_public_update(monkeypatch):
    model = StreamImageGenerator()
    critic = StreamImageCritic()
    recipe = get_recipe("bcap_adam", batch_size=1, num_particles=32, z_dim=8)
    prior = recipe.make_prior(generator=torch.Generator().manual_seed(0))
    trainer = TransportGANTrainer(recipe, model, critic, prior=prior)

    def fail(*args, **kwargs):
        raise RuntimeError("interrupted optimizer update")

    monkeypatch.setattr(GANTrainer, "step", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        trainer.step(torch.full((1, 1, 8, 8), .8))
    assert trainer.transport._target is None
    assert trainer.transport._phase == 0
    assert trainer.transport.assignment_updates == 0
    assert not any("target" in key for key in trainer.prior.state_dict())


def test_usage_penalty_can_allocate_initially_identical_particles():
    model, router, prior = make_router(usage_penalty=.01)
    with torch.no_grad():
        prior.z.fill_(.5)
    selected = []
    for _ in range(3):
        with router.training_target(torch.full((1, 1, 8, 8), .5)):
            prior.sample(1, generator=torch.Generator().manual_seed(0))
            _, row = prior.sample(1, generator=torch.Generator().manual_seed(0))
            selected.append(int(row.item()))
    assert selected == [0, 1, 2]
    assert router.assignment_mass.tolist() == [1., 1., 1.]


def test_stochastic_routing_has_explicit_rng_and_checkpointed_configuration():
    with pytest.raises(ValueError, match="explicit routing_generator"):
        make_router(temperature=.01)
    _, router, prior = make_router(temperature=.01, usage_penalty=.02,
                                  routing_generator=torch.Generator().manual_seed(0))
    _, _, differently_configured = make_router(temperature=.01, usage_penalty=.03,
                                               routing_generator=torch.Generator().manual_seed(0))
    with pytest.raises(ValueError, match="configuration mismatch"):
        differently_configured.load_state_dict(prior.state_dict())
    assert router.manipulator_spec()["config"]["usage_penalty"] == .02


@contextmanager
def real_trainer(*, frozen=False):
    streams = NamedStreams(seed=0)
    with streams.fork("init", component="models", purpose="construction"):
        generator = StreamImageGenerator(context_dim=16)
        critic = StreamImageCritic(context_dim=16)
        memory = FastWeightMemory()
        for module, component in ((generator, "generator"), (critic, "discriminator"), (memory, "memory")):
            _init_module(module, streams, component)
    recipe = get_recipe("bcap_adam", batch_size=1, num_particles=32, z_dim=8,
                        total_steps=480, prior_kind="mog", standardize=False,
                        prior_update="frozen" if frozen else "learned", loss="non_saturating")
    prior = recipe.replace(prior_update="learned").make_prior(
        sigma=.025, generator=streams.generator("init", component="prior", purpose="locations"))
    _init_module(prior, streams, "prior")
    recipe.apply_prior_policy(prior)
    trainer = TransportGANTrainer(
        recipe, generator, critic, prior=prior, transport={"usage_penalty": .01}, max_steps=2,
        routing_generator=streams.generator("prior", component="routing", purpose="assignment"),
        latent_generator=streams.generator("prior", component="latent", purpose="indices"),
        penalty_generator=streams.generator("noise", component="penalty", purpose="training"),
        noise_generator=streams.generator("noise", component="generator", purpose="output"),
        input_noise_generator=streams.generator("noise", component="critic", purpose="input"),
        model_generator=streams.generator("noise", component="models", purpose="stochastic_layers"),
        prior_noise_generator=streams.generator("noise", component="prior", purpose="gaussian"),
        eval_generator=streams.generator("eval", component="sampler", purpose="samples"),
    )
    yield trainer, memory, streams


@pytest.mark.parametrize("frozen", [False, True])
def test_public_optimizer_updates_remain_batch_one_and_clear_targets(frozen):
    with real_trainer(frozen=frozen) as (trainer, memory, streams):
        session = OnlineSession(
            trainer, trainer.G, trainer.D, memory, predictive=True,
            serving_generator=streams.generator("eval", component="serving", purpose="sample"),
        )
        frames = bars4_templates()
        session.observe(frames[:1])
        expected_context = session.context
        seen_contexts = []
        hook = trainer.G.register_forward_pre_hook(
            lambda module, args: seen_contexts.append(module.active_context.clone()))
        try:
            stats = session.observe(frames[1:2])
        finally:
            hook.remove()
        assert all(torch.equal(context, expected_context) for context in seen_contexts)
        assert trainer.completed_steps == session.updates == 1
        assert session.max_real_batch == session.max_fake_batch == 1
        assert trainer.transport.assignment_updates == 1
        assert trainer.transport.probe_forwards == 32
        assert torch.isfinite(stats["prior_manipulation"])
        assert trainer.transport._target is None
        before = trainer.transport.probe_forwards.clone()
        with streams.isolated_evaluation():
            session.predict()
            trainer.sample(32, fixed_first_n=True)
        assert streams.last_eval_audit["unintended_rng_deviations"] == 0
        assert torch.equal(trainer.transport.probe_forwards, before)


def test_eval_history_clone_preserves_initialized_projection_coordinates():
    streams = NamedStreams(seed=0)
    memory = FastWeightMemory()
    _init_module(memory, streams, "memory")
    generator = StreamImageGenerator(context_dim=16)
    critic = StreamImageCritic(context_dim=16)
    history = bars4_templates()[:3]
    for frame in history:
        expected = memory.write(frame.unsqueeze(0))
    original = {key: value.clone() for key, value in memory.state_dict().items()}
    rng = torch.random.get_rng_state()
    with torch.no_grad():
        simulated = _history_context(memory, history, generator, critic)
    assert torch.equal(generator.active_context, expected)
    assert torch.equal(critic.active_context, expected)
    assert torch.equal(simulated.key_proj.weight, memory.key_proj.weight)
    assert simulated.key_proj.weight.data_ptr() != memory.key_proj.weight.data_ptr()
    assert all(torch.equal(memory.state_dict()[key], value) for key, value in original.items())
    assert torch.equal(torch.random.get_rng_state(), rng)
