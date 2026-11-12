"""Unit and lifecycle audits for ResidualFeatureConditionalReconstruction."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe
from particlegan.particle_prior import MoGParticlePrior
from particlegan.priors import ManipulatedPrior
from particlegan_forge import NamedStreams
from particlestream import minimal_tasks
from particlestream.challenges import DelayedCueStream
from particlestream.feature_prior import FeatureConditionalReconstruction
from particlestream.memory import CausalStateMemory
from particlestream.models import StreamImageCritic, StreamImageGenerator
from particlestream.online import OnlineSession
from particlestream.residual_prior import ResidualFeatureConditionalReconstruction
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


def make_prior(*, context_dim=0, router_cls=ResidualFeatureConditionalReconstruction, **config):
    model = ContextModel(context_dim)
    base = MoGParticlePrior(num_particles=32, z_dim=8, sigma=.025, standardize=False,
                            generator=torch.Generator().manual_seed(0))
    base.requires_grad_(False)
    router = router_cls(model, **config)
    wrapped = ManipulatedPrior(base, [router])
    return model, router, wrapped


def observe(router, prior, frame):
    with router.training_target(frame):
        prior.sample(1, generator=torch.Generator().manual_seed(0))
        prior.sample(1, generator=torch.Generator().manual_seed(0))


def clean_codes(prior):
    prior.eval()
    return prior.sample(32, fixed_first_n=True, eps=torch.zeros(32, 8))[0]


def test_residual_prior_has_zero_parameters_and_preserves_global_rng():
    before = torch.random.get_rng_state()
    _, router, wrapped = make_prior(context_dim=16, encoder_seed=123)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert list(router.parameters()) == []
    assert list(wrapped.parameters()) == list(wrapped.base.parameters())
    assert router.rls_weights.shape == (8, 17)
    assert router.rls_cov.shape == (17, 17)
    assert router.residual_cov.shape == (8, 8)
    assert router.rls_weights.dtype == torch.float64


@pytest.mark.parametrize("frequencies", [(1, 1, 1, 1), (5, 5, 5, 1)])
def test_zero_context_reduces_exactly_to_feature_conditional_prior(frequencies):
    res_model, res_router, res_prior = make_prior(context_dim=0, encoder_seed=77)
    base_model, base_router, base_prior = make_prior(context_dim=0, router_cls=FeatureConditionalReconstruction, encoder_seed=77)
    assert res_model is not base_model
    frames = bars4_templates()
    for index, count in enumerate(frequencies):
        for _ in range(count):
            observe(res_router, res_prior, frames[index:index + 1])
            observe(base_router, base_prior, frames[index:index + 1])
    assert torch.equal(clean_codes(res_prior), clean_codes(base_prior))


@pytest.mark.parametrize("mode", ["kernel2x2", "addition2bit"])
def test_heldout_compositional_and_carry_latent_extrapolation_with_linear_keys(mode):
    streams = NamedStreams(seed=0)
    memory = CausalStateMemory(association_weight=0.0, normalize_key=False, key_gain=0.5)
    _init_module(memory, streams, "memory")
    model, router, prior = make_prior(
        context_dim=16,
        encoder_seed=streams.seed_for("init", component="feature_encoder", purpose="projection"),
    )
    train_cases = minimal_tasks.training_cases(mode)
    for _ in range(12):
        for case in train_cases:
            memory.reset_state()
            ctx = memory.write(case.source)
            model.active_context.copy_(ctx)
            observe(router, prior, case.target)
    heldout_cases = minimal_tasks.paired_cases(mode, split="heldout")
    for case in heldout_cases:
        memory.reset_state()
        ctx = memory.write(case.source)
        model.active_context.copy_(ctx)
        codes = clean_codes(prior)
        expected = router.encode(case.target)[0]
        max_err = (codes - expected.unsqueeze(0)).abs().max().item()
        assert max_err < 1e-4, f"{mode} {case.label}: max latent error {max_err}"


def test_recurrent_multimodal_context_recovers_exact_branch_split():
    model, router, prior = make_prior(context_dim=16, encoder_seed=19)
    frames = bars4_templates()
    # Warm up across distinct contexts so step > context_dim, then revisit a bimodal context.
    for step in range(20):
        model.active_context.zero_()
        model.active_context[0, step % 8] = 0.2 * ((step % 4) + 1)
        observe(router, prior, frames[(step % 2) + 2:(step % 2) + 3])
    model.active_context.fill_(0.5)
    for _ in range(12):
        for outcome in (0, 1):
            observe(router, prior, frames[outcome:outcome + 1])
    codes = clean_codes(prior)
    encoded = torch.cat([router.encode(f.unsqueeze(0)) for f in frames])
    labels = (codes[:, None] - encoded[None]).square().sum(-1).argmin(-1)
    assert torch.bincount(labels, minlength=4).tolist() == [16, 16, 0, 0]


def test_continuous_single_visit_manifold_extrapolates_smoothly_without_prototype_snapping():
    model, router, prior = make_prior(context_dim=16, encoder_seed=23)
    rng = torch.Generator().manual_seed(7)
    direction = torch.randn(1, 1, 8, 8, generator=rng)
    direction = direction / direction.abs().max()
    for step in range(60):
        u = -0.8 + 1.6 * (step / 59.0)
        ctx = torch.zeros(1, 16)
        ctx[0, 0] = u
        ctx[0, 1] = 0.5 * u
        model.active_context.copy_(ctx)
        frame = (0.5 + 0.35 * u * direction).clamp(0.05, 0.95)
        observe(router, prior, frame)
    # Test at an unseen intermediate continuous point u_test = 0.313
    u_test = 0.313
    model.active_context.zero_()
    model.active_context[0, 0] = u_test
    model.active_context[0, 1] = 0.5 * u_test
    expected_frame = (0.5 + 0.35 * u_test * direction).clamp(0.05, 0.95)
    expected_code = router.encode(expected_frame)[0]
    predicted_codes = clean_codes(prior)
    assert torch.allclose(predicted_codes[0], predicted_codes[-1], atol=1e-6)
    assert torch.linalg.vector_norm(predicted_codes[0] - expected_code).item() < 1e-3


def _online_runtime():
    streams = NamedStreams(seed=0)
    with streams.fork("init", component="models", purpose="construction"):
        generator = StreamImageGenerator(context_dim=16)
        critic = StreamImageCritic(context_dim=16)
        memory = CausalStateMemory(association_weight=0.0, normalize_key=False, key_gain=0.5)
        for module, component in ((generator, "generator"), (critic, "discriminator"), (memory, "memory")):
            _init_module(module, streams, component)
    recipe = get_recipe("bcap_adam", batch_size=1, num_particles=32, z_dim=8,
                        prior_kind="mog", standardize=False, prior_update="frozen",
                        total_steps=64, loss="non_saturating", lr=.005)
    base = recipe.replace(prior_update="learned").make_prior(
        sigma=.025, generator=streams.generator("init", component="prior", purpose="locations"))
    _init_module(base, streams, "prior")
    recipe.apply_prior_policy(base)
    trainer = TransportGANTrainer(
        recipe, generator, critic, prior=base, max_steps=16,
        router_class=ResidualFeatureConditionalReconstruction,
        transport={
            "encoder_seed": streams.seed_for("init", component="feature_encoder", purpose="projection"),
            "reconstruction_loss": "binary_cross_entropy",
            "context_bypass_weight": 5.0,
        },
        latent_generator=streams.generator("prior", component="latent", purpose="indices"),
        penalty_generator=streams.generator("noise", component="penalty", purpose="training"),
        noise_generator=streams.generator("noise", component="generator", purpose="output"),
        input_noise_generator=streams.generator("noise", component="critic", purpose="input"),
        prior_noise_generator=streams.generator("noise", component="prior", purpose="gaussian"),
        model_generator=streams.generator("noise", component="models", purpose="stochastic_layers"),
        eval_generator=streams.generator("eval", component="sampler", purpose="samples"),
    )
    session = OnlineSession(
        trainer, generator, critic, memory, predictive=True,
        serving_generator=streams.generator("eval", component="serving", purpose="prediction"),
    )
    return trainer, session, streams


def _assert_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_equal(first[key], second[key])
    else:
        assert first == second


def test_serving_isolation_failed_step_rollback_and_context_bypass_shrinkage():
    ref_trainer, ref_session, _ = _online_runtime()
    srv_trainer, srv_session, srv_streams = _online_runtime()
    init_ctx_norm = float(ref_trainer.G.fc.weight[:, 8:].square().sum().item())
    stream = DelayedCueStream(noise_std=0)
    frames = torch.cat((stream.clean_history(0), stream.outcomes[:1]))
    for frame in frames:
        with srv_streams.isolated_evaluation():
            srv_session.predict()
            srv_session.predict()
        assert srv_streams.last_eval_audit["unintended_rng_deviations"] == 0
        ref_session.observe(frame.unsqueeze(0))
        srv_session.observe(frame.unsqueeze(0))
    for left, right in ((ref_trainer.G, srv_trainer.G), (ref_trainer.D, srv_trainer.D),
                        (ref_trainer.prior, srv_trainer.prior), (ref_session.memory, srv_session.memory)):
        _assert_equal(left.state_dict(), right.state_dict())
    assert float(ref_trainer.G.fc.weight[:, 8:].square().sum().item()) < init_ctx_norm
    diag = ref_trainer.transport_diagnostics()
    assert diag["target_retained"] is False and diag["observation_state_retained"] is False

    # Interrupted optimizer update must roll back without mutating RLS or prototype buffers.
    saved_prior = deepcopy(ref_trainer.prior.state_dict())
    def boom(self, real, **kwargs):
        self.prior.sample(1, generator=self.latent_generator, noise_generator=self.prior_noise_generator)
        raise RuntimeError("boom")

    orig_step = GANTrainer.step
    GANTrainer.step = boom
    try:
        with pytest.raises(RuntimeError, match="boom"):
            ref_session.observe(stream.outcomes[1:2])
    finally:
        GANTrainer.step = orig_step
    _assert_equal(saved_prior, ref_trainer.prior.state_dict())


@pytest.mark.parametrize("bad_kwargs", [
    {"rls_ridge": 0.0},
    {"rls_ridge": True},
    {"covariance_ema": 0.0},
    {"covariance_ema": 1.5},
    {"slab_scale": -1.0},
    {"variance_ratio_scale": 0.0},
    {"visit_shrinkage": -0.1},
    {"kernel_bandwidth": 0.0},
    {"outcome_decay": 0.0},
    {"context_bypass_weight": -1.0},
])
def test_invalid_residual_prior_settings_rejected(bad_kwargs):
    with pytest.raises(ValueError):
        make_prior(context_dim=16, **bad_kwargs)
