"""Feature-prior sufficient statistics are causal and serving remains target-free."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import get_recipe
from particlegan.particle_prior import MoGParticlePrior
from particlegan.priors import ManipulatedPrior
from particlegan_forge import NamedStreams
from particlestream.feature_prior import FeatureConditionalReconstruction
from particlestream.models import StreamImageCritic, StreamImageGenerator
from particlestream.streams import bars4_templates
from particlestream.tier1 import _init_module
from particlestream.transport import TransportGANTrainer


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class ContextModel(nn.Module):
    def __init__(self, context_dim=0):
        super().__init__()
        self.context_dim, self.z_dim = context_dim, 8
        self.register_buffer("active_context", torch.zeros(1, max(1, context_dim)))


def make_prior(*, context_dim=0, **config):
    model = ContextModel(context_dim)
    base = MoGParticlePrior(num_particles=32, z_dim=8, sigma=.025, standardize=False,
                            generator=torch.Generator().manual_seed(0))
    base.requires_grad_(False)
    router = FeatureConditionalReconstruction(model, **config)
    wrapped = ManipulatedPrior(base, [router])
    return model, router, wrapped


def observe(router, prior, frame):
    with router.training_target(frame):
        prior.sample(1, generator=torch.Generator().manual_seed(0))
        prior.sample(1, generator=torch.Generator().manual_seed(0))


def clean_codes(prior):
    prior.eval()
    return prior.sample(32, fixed_first_n=True, eps=torch.zeros(32, 8))[0]


def assignments(codes, router, frames):
    features = torch.cat([router.encode(frame.unsqueeze(0)) for frame in frames])
    return (codes[:, None] - features[None]).square().sum(-1).argmin(-1)


def test_fixed_encoder_is_orthogonal_buffer_and_does_not_consume_global_rng():
    model = ContextModel()
    base = MoGParticlePrior(num_particles=32, z_dim=8, sigma=.025,
                            generator=torch.Generator().manual_seed(0))
    before = torch.random.get_rng_state()
    router = FeatureConditionalReconstruction(model, encoder_seed=123)
    wrapped = ManipulatedPrior(base, [router])
    assert torch.equal(torch.random.get_rng_state(), before)
    assert torch.allclose(router.encoder @ router.encoder.t(), torch.eye(8), atol=1e-6)
    assert router.encoder.shape == (8, 64)
    assert list(router.parameters()) == []
    assert list(wrapped.parameters()) == list(base.parameters())
    second_model, second, _ = make_prior(encoder_seed=123)
    assert torch.equal(router.encoder, second.encoder)


def test_g_uses_current_feature_plus_original_mog_perturbation_d_does_not():
    model, router, prior = make_prior()
    target = bars4_templates()[1:2]
    latent = torch.Generator().manual_seed(0)
    noise = torch.Generator().manual_seed(0)
    with router.training_target(target):
        d_codes, d_rows = prior.sample(1, generator=latent, noise_generator=noise)
        assert torch.linalg.vector_norm(d_codes - router.encode(target)) > .1
        reference_noise = torch.Generator()
        reference_noise.set_state(noise.get_state())
        expected_eps = torch.randn(1, 8, generator=reference_noise)
        g_codes, g_rows = prior.sample(1, generator=latent, noise_generator=noise)
        assert torch.allclose(g_codes, router.encode(target) + .025 * expected_eps, atol=1e-6)
        assert router.feature_count == 0 and router.context_count == 0
    assert router.feature_count == router.context_count == 1
    assert router._target is router._pending_feature is router._pending_context is None
    assert router.probe_forwards == 0


@pytest.mark.parametrize("frequencies,expected", [((1, 1, 1, 1), (8, 8, 8, 8)),
                                                ((5, 5, 5, 1), (10, 10, 10, 2))])
def test_unconditional_rows_represent_observed_frequencies(frequencies, expected):
    model, router, prior = make_prior()
    frames = bars4_templates()
    for index, count in enumerate(frequencies):
        for _ in range(count):
            observe(router, prior, frames[index:index + 1])
    codes = clean_codes(prior)
    labels = assignments(codes, router, frames)
    assert torch.bincount(labels, minlength=4).tolist() == list(expected)
    assert router.feature_count == 4
    assert router.assignment_updates == sum(frequencies)


def test_conditional_rows_use_only_the_already_observed_context():
    model, router, prior = make_prior(context_dim=16)
    frames = bars4_templates()
    for _ in range(4):
        for context, outcome in ((-1., 0), (1., 1)):
            model.active_context.fill_(context)
            observe(router, prior, frames[outcome:outcome + 1])
    assert router.context_count == 2
    for context, outcome in ((-1., 0), (1., 1)):
        model.active_context.fill_(context)
        labels = assignments(clean_codes(prior), router, frames)
        assert labels.tolist() == [outcome] * 32
    # Serving classifies a displaced context using the nearest observed state;
    # the matching radius governs learning new states, not serving rejection.
    model.active_context.fill_(3.)
    assert assignments(clean_codes(prior), router, frames).tolist() == [1] * 32


def test_initial_prior_is_unchanged_before_any_successful_observations():
    model, router, prior = make_prior(context_dim=16)
    model.active_context.fill_(3.)
    assert torch.equal(clean_codes(prior), prior.base.means())


def test_multimodal_context_maps_equal_rows_to_both_observed_outcomes():
    model, router, prior = make_prior(context_dim=16)
    model.active_context.fill_(.5)
    frames = bars4_templates()
    for _ in range(3):
        for outcome in (0, 1):
            observe(router, prior, frames[outcome:outcome + 1])
    labels = assignments(clean_codes(prior), router, frames)
    assert torch.bincount(labels, minlength=4).tolist() == [16, 16, 0, 0]


def test_failed_observation_does_not_commit_statistics_or_retain_target():
    model, router, prior = make_prior(context_dim=16)
    before = deepcopy(router.state_dict())
    with pytest.raises(RuntimeError, match="failed"):
        with router.training_target(bars4_templates()[:1]):
            prior.sample(1, generator=torch.Generator().manual_seed(0))
            prior.sample(1, generator=torch.Generator().manual_seed(0))
            raise RuntimeError("failed optimizer step")
    for key, value in router.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, before[key])
        else:
            assert value == before[key]
    assert router._target is router._pending_feature is router._pending_context is None


def test_serving_changes_no_prior_statistics_and_never_uses_current_target():
    model, router, prior = make_prior(context_dim=16)
    frames = bars4_templates()
    observe(router, prior, frames[:1])
    prior.eval()
    before = deepcopy(router.state_dict())
    # Even an explicit inference request inside a target scope must use the old
    # predictive distribution, rather than revealing the newly arrived target.
    with pytest.raises(RuntimeError, match="abort"):
        with router.training_target(frames[1:2]):
            codes = clean_codes(prior)
            assert assignments(codes, router, frames).tolist() == [0] * 32
            raise RuntimeError("abort diagnostic scope")
    assert all(torch.equal(value, before[key]) for key, value in router.state_dict().items()
               if isinstance(value, torch.Tensor))


def test_context_capacity_is_bounded_and_eviction_uses_no_images():
    model, router, prior = make_prior(context_dim=16, context_capacity=2)
    for context in (-1., 0., 1.):
        model.active_context.fill_(context)
        observe(router, prior, bars4_templates()[:1])
    assert router.context_count == 2 and router.context_evictions == 1
    assert router.context_outcomes.shape == (2, 32)
    assert router.feature_centroids.shape == (32, 8)
    assert not any(tuple(buffer.shape) == (1, 1, 8, 8) for buffer in router.buffers())


def test_encoder_configuration_is_checkpoint_bound():
    model, router, prior = make_prior(encoder_seed=123)
    other_model, other, other_prior = make_prior(encoder_seed=124)
    with pytest.raises(ValueError, match="configuration mismatch"):
        other_prior.load_state_dict(prior.state_dict())
    assert router.manipulator_spec()["config"]["encoder_seed"] == 123


def test_public_batch_one_decoder_updates_with_frozen_storage_and_no_output_bypass(monkeypatch):
    streams = NamedStreams(seed=0)
    with streams.fork("init", component="models", purpose="construction"):
        generator = StreamImageGenerator(context_dim=16)
        critic = StreamImageCritic(context_dim=16)
        _init_module(generator, streams, "generator")
        _init_module(critic, streams, "discriminator")
    recipe = get_recipe("bcap_adam", batch_size=1, num_particles=32, z_dim=8,
                        prior_kind="mog", standardize=False, prior_update="frozen", total_steps=480,
                        loss="non_saturating")
    base = recipe.replace(prior_update="learned").make_prior(
        sigma=.025, generator=streams.generator("init", component="prior", purpose="locations"))
    _init_module(base, streams, "prior")
    recipe.apply_prior_policy(base)
    before_storage = base.z.clone()
    trainer = TransportGANTrainer(
        recipe, generator, critic, prior=base, max_steps=2,
        router_class=FeatureConditionalReconstruction,
        transport={"encoder_seed": streams.seed_for("init", component="feature_encoder", purpose="projection")},
        latent_generator=streams.generator("prior", component="latent", purpose="indices"),
        penalty_generator=streams.generator("noise", component="penalty", purpose="training"),
        noise_generator=streams.generator("noise", component="generator", purpose="output"),
        prior_noise_generator=streams.generator("noise", component="prior", purpose="gaussian"),
        model_generator=streams.generator("noise", component="models", purpose="stochastic_layers"),
        eval_generator=streams.generator("eval", component="sampler", purpose="samples"),
    )
    before_decoder = [parameter.detach().clone() for parameter in generator.parameters()]
    batches = []
    handle = generator.register_forward_pre_hook(lambda module, args: batches.append(len(args[0])))
    try:
        for frame in bars4_templates()[:2]:
            stats = trainer.step(frame.unsqueeze(0))
            assert torch.isfinite(stats["prior_manipulation"])
    finally:
        handle.remove()
    assert batches == [1, 1, 1, 1]
    assert any(not torch.equal(before, after) for before, after in zip(before_decoder, generator.parameters()))
    assert torch.equal(base.z, before_storage)
    assert trainer.transport.assignment_updates == 2 and trainer.transport.probe_forwards == 0
    assert trainer.transport._target is trainer.transport._pending_context is trainer.transport._pending_feature is None
    with streams.isolated_evaluation():
        trainer.sample(32, fixed_first_n=True)
    assert streams.last_eval_audit["unintended_rng_deviations"] == 0
    monkeypatch.setattr(generator, "forward", lambda codes: codes.new_zeros(len(codes), 1, 8, 8))
    assert torch.count_nonzero(trainer.sample(32, fixed_first_n=True)) == 0, "serving bypassed the trained decoder"
