"""Unit and public-trainer contracts for the GAIL-BCAP conditional prior."""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import GANLoss, get_recipe
from particlegan.particle_prior import MoGParticlePrior
from particlegan.priors import ManipulatedPrior
from particlegan_forge import NamedStreams
from particlestream.gail_prior import GAILConditionalReconstruction
from particlestream.minimal_online import PairedOnlineSession, isolated_session
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
    def __init__(self, context_dim: int = 0):
        super().__init__()
        self.context_dim = context_dim
        self.z_dim = 8
        self.register_buffer("active_context", torch.zeros(1, max(1, context_dim)))
        if context_dim > 0:
            self.fc = nn.Linear(8 + context_dim, 16)


def make_gail_prior(*, context_dim: int = 0, **config):
    model = ContextModel(context_dim)
    base = MoGParticlePrior(
        num_particles=32,
        z_dim=8,
        sigma=0.025,
        standardize=False,
        generator=torch.Generator().manual_seed(0),
    )
    base.requires_grad_(False)
    router = GAILConditionalReconstruction(model, **config)
    wrapped = ManipulatedPrior(base, [router])
    return model, router, wrapped


def observe_gail(router, prior, frame, *, logit_value: float = 0.25):
    with router.training_target(frame):
        prior.sample(1, generator=torch.Generator().manual_seed(0))
        prior.sample(1, generator=torch.Generator().manual_seed(0))
        generated = frame.detach().clone().requires_grad_(True)
        logits = torch.tensor([[logit_value]], dtype=frame.dtype, requires_grad=True)
        loss = prior.regularize(prior.z, generated=generated, logits=logits)
        assert torch.isfinite(loss)


def clean_codes(prior):
    prior.eval()
    return prior.sample(32, fixed_first_n=True, eps=torch.zeros(32, 8))[0]


def assignments(codes, router, frames):
    features = torch.cat([router.encode(frame.unsqueeze(0)) for frame in frames])
    return (codes[:, None] - features[None]).square().sum(-1).argmin(-1)


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"cancel_linear_gan": 1},
        {"style_bonus_weight": True},
        {"style_bonus_weight": -0.1},
        {"return_discount": 1.0},
        {"return_discount": -0.01},
        {"advantage_scale": 1.0},
        {"policy_entropy_bonus": 0.6},
        {"assoc_ridge": -1.0},
        {"assoc_forgetting": 0.0},
        {"assoc_forgetting": 1.1},
        {"assoc_blend_scale": 0.0},
        {"assoc_max_blend": 1.2},
        {"spike_slab_scale": -0.1},
        {"spike_slab_scale": 1.0, "assoc_ridge": 0.0},
        {"covariance_ema": 0.0},
        {"covariance_ema": 1.1},
        {"covariance_eps": 0.0},
        {"variance_ratio_scale": 0.0},
        {"visit_shrinkage": -0.1},
        {"kernel_bandwidth": 0.0},
        {"context_bypass_weight": -1.0},
        {"reconstruction_weight": 0.0, "cancel_linear_gan": True},
    ],
)
def test_invalid_gail_hyperparameters_fail_closed(bad_kwargs):
    with pytest.raises(ValueError):
        make_gail_prior(context_dim=16, **bad_kwargs)


def test_hinge_cancellation_replaces_unbounded_slope_with_bounded_sigmoid_style():
    model, router, prior = make_gail_prior(
        reconstruction_weight=10.0,
        style_bonus_weight=0.1,
        cancel_linear_gan=True,
        reconstruction_loss="mse",
    )
    target = bars4_templates()[:1]
    hinge = GANLoss("hinge")
    with router.training_target(target):
        prior.sample(1, generator=torch.Generator().manual_seed(0))
        prior.sample(1, generator=torch.Generator().manual_seed(0))
        generated = torch.full_like(target, 0.25, requires_grad=True)
        logits = torch.tensor([[2.0]], requires_grad=True)
        loss_g = hinge.g_loss(logits) + prior.regularize(prior.z, generated=generated, logits=logits)
        (grad_logit,) = torch.autograd.grad(loss_g, logits)
        sig = torch.sigmoid(logits.detach())
        expected_grad = -10.0 * 0.1 * sig * (1.0 - sig)
        assert torch.allclose(grad_logit, expected_grad, atol=1e-6)


def test_causal_return_advantage_commits_only_after_successful_step():
    model, router, prior = make_gail_prior(
        context_dim=16,
        return_discount=0.90,
        advantage_scale=0.25,
        style_bonus_weight=0.1,
    )
    frames = bars4_templates()
    observe_gail(router, prior, frames[0:1], logit_value=-1.0)
    assert router.gail_updates == 1
    assert router.last_advantage_gain.item() == pytest.approx(1.0)
    observe_gail(router, prior, frames[1:2], logit_value=1.5)
    assert router.gail_updates == 2
    observe_gail(router, prior, frames[2:3], logit_value=2.0)
    assert router.gail_updates == 3
    gain = router.last_advantage_gain.item()
    assert 0.75 <= gain <= 1.25 and gain > 1.0

    saved = deepcopy(router.state_dict())
    with pytest.raises(RuntimeError, match="aborted"):
        with router.training_target(frames[3:4]):
            prior.sample(1, generator=torch.Generator().manual_seed(0))
            prior.sample(1, generator=torch.Generator().manual_seed(0))
            _ = prior.regularize(
                prior.z,
                generated=frames[3:4].clone().requires_grad_(True),
                logits=torch.tensor([[0.5]], requires_grad=True),
            )
            raise RuntimeError("aborted optimizer step")
    for key, value in router.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, saved[key])
    assert router.diagnostics()["observation_state_retained"] is False
    assert router.diagnostics()["target_retained"] is False


@pytest.mark.parametrize("frequencies,expected", [((1, 1, 1, 1), (8, 8, 8, 8)), ((5, 5, 5, 1), (10, 10, 10, 2))])
def test_entropy_regularized_quantiles_preserve_multimodal_and_rare_frequencies(frequencies, expected):
    model, router, prior = make_gail_prior(policy_entropy_bonus=0.01)
    frames = bars4_templates()
    for index, count in enumerate(frequencies):
        for _ in range(count):
            observe_gail(router, prior, frames[index : index + 1])
    codes = clean_codes(prior)
    labels = assignments(codes, router, frames)
    assert torch.bincount(labels, minlength=4).tolist() == list(expected)


def test_associative_fast_weight_extrapolates_unseen_contexts_without_perturbing_seen_buckets():
    model, router, prior = make_gail_prior(
        context_dim=4,
        context_tolerance=0.20,
        assoc_ridge=0.01,
        assoc_forgetting=0.95,
        assoc_blend_scale=0.25,
        assoc_max_blend=1.0,
    )
    assert list(router.parameters()) == []
    assert all(b.dtype == torch.float32 for b in router.buffers() if b.is_floating_point())
    frames = bars4_templates()
    # Train linear interpolation between context = -1 -> frame 0 and context = +1 -> frame 2
    for _ in range(12):
        model.active_context.copy_(torch.tensor([[-1.0, 0.0, 0.0, 0.0]]))
        observe_gail(router, prior, frames[0:1])
        model.active_context.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        observe_gail(router, prior, frames[2:3])

    # Seen bucket (-1, 0, 0, 0) uses exact discrete prototype 0
    model.active_context.copy_(torch.tensor([[-1.0, 0.0, 0.0, 0.0]]))
    seen_codes = clean_codes(prior)
    assert torch.allclose(seen_codes, router.encode(frames[0:1]).expand_as(seen_codes), atol=1e-5)

    # Unseen midpoint (0, 0, 0, 0) is at distance 1.0 > 0.20 + 0.25 -> smooth associative midpoint!
    model.active_context.copy_(torch.tensor([[0.0, 0.0, 0.0, 0.0]]))
    mid_codes = clean_codes(prior)
    expected_mid = 0.5 * (router.encode(frames[0:1]) + router.encode(frames[2:3]))
    assert torch.allclose(mid_codes[0:1], expected_mid, atol=2e-2)

    # Directional forgetting keeps covariance bounded by 1 / assoc_ridge even on rank-deficient contexts
    eigvals = torch.linalg.eigvalsh(router.assoc_cov_inv)
    assert bool((eigvals > 0).all()) and float(eigvals.max().item()) <= (1.0 / 0.01) + 1e-4


def test_public_hinge_gail_trainer_preserves_batch_one_rng_and_session_isolation():
    streams = NamedStreams(seed=0)
    with streams.fork("init", component="models", purpose="construction"):
        generator = StreamImageGenerator(context_dim=16)
        critic = StreamImageCritic(context_dim=16)
        _init_module(generator, streams, "generator")
        _init_module(critic, streams, "discriminator")
    recipe = get_recipe(
        "bcap_adam",
        batch_size=1,
        num_particles=32,
        z_dim=8,
        prior_kind="mog",
        standardize=False,
        prior_update="frozen",
        total_steps=16,
        loss="hinge",
    )
    base = recipe.replace(prior_update="learned").make_prior(
        sigma=0.025,
        generator=streams.generator("init", component="prior", purpose="locations"),
    )
    _init_module(base, streams, "prior")
    recipe.apply_prior_policy(base)
    trainer = TransportGANTrainer(
        recipe,
        generator,
        critic,
        prior=base,
        max_steps=4,
        router_class=GAILConditionalReconstruction,
        transport={"encoder_seed": streams.seed_for("init", component="feature_encoder", purpose="projection")},
        latent_generator=streams.generator("prior", component="latent", purpose="indices"),
        penalty_generator=streams.generator("noise", component="penalty", purpose="training"),
        noise_generator=streams.generator("noise", component="generator", purpose="output"),
        prior_noise_generator=streams.generator("noise", component="prior", purpose="gaussian"),
        model_generator=streams.generator("noise", component="models", purpose="stochastic_layers"),
        eval_generator=streams.generator("eval", component="sampler", purpose="samples"),
    )
    session = PairedOnlineSession(
        trainer,
        generator,
        critic,
        serving_generator=streams.generator("eval", component="serving", purpose="prediction"),
    )
    frames = bars4_templates()
    for idx in range(2):
        session.source(frames[idx : idx + 1])
        with streams.isolated_evaluation():
            _ = session.predict()
        assert streams.last_eval_audit["unintended_rng_deviations"] == 0
        stats = session.observe_target(frames[(idx + 1) % 4 : (idx + 1) % 4 + 1])
        assert torch.isfinite(stats["loss_g"]) and torch.isfinite(stats["loss_d"])

    saved_transport = deepcopy(trainer.transport.state_dict())
    with streams.isolated_evaluation(), isolated_session(session):
        _ = trainer.sample(32, fixed_first_n=True)
    assert streams.last_eval_audit["unintended_rng_deviations"] == 0
    for key, value in trainer.transport.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, saved_transport[key])


def test_spike_slab_mahalanobis_and_context_bypass_regularizer():
    model, router, prior = make_gail_prior(
        context_dim=2,
        assoc_ridge=0.1,
        spike_slab_scale=1.0,
        context_bypass_weight=5.0,
        reconstruction_weight=10.0,
        style_bonus_weight=0.05,
        advantage_scale=0.0,
        reconstruction_loss="mse",
    )
    assert router.assoc_extrapolation_law == "sherman_morrison_spike_slab_kernel_residual_v1"
    assert router.assoc_weights.dtype == torch.float64
    frames = bars4_templates()

    # Check exact context-bypass gradient without committing an observation (or on a fresh router)
    with torch.no_grad():
        model.fc.weight.fill_(0.25)
    model_grad, router_grad, prior_grad = make_gail_prior(
        context_dim=2,
        assoc_ridge=0.1,
        spike_slab_scale=1.0,
        context_bypass_weight=5.0,
        reconstruction_weight=10.0,
        style_bonus_weight=0.05,
        advantage_scale=0.0,
        reconstruction_loss="mse",
    )
    with torch.no_grad():
        model_grad.fc.weight.fill_(0.25)
    with router_grad.training_target(frames[0:1]):
        prior_grad.sample(1, generator=torch.Generator().manual_seed(0))
        prior_grad.sample(1, generator=torch.Generator().manual_seed(0))
        generated = frames[0:1].clone().requires_grad_(True)
        logits = torch.tensor([[0.0]], requires_grad=True)
        reg = prior_grad.regularize(prior_grad.z, generated=generated, logits=logits)
        (grad_fc,) = torch.autograd.grad(reg, model_grad.fc.weight)
        assert torch.allclose(grad_fc[:, :8], torch.zeros_like(grad_fc[:, :8]))
        assert torch.allclose(grad_fc[:, 8:], 10.0 * model_grad.fc.weight[:, 8:].detach(), atol=1e-6)

    # Train 2D linear manifold on seen contexts (-1,0), (+1,0), (0,1); evaluate on unseen midpoint (0,0)
    f_mid = 0.5 * (frames[0:1] + frames[2:3])
    for _ in range(10):
        model.active_context.copy_(torch.tensor([[-1.0, 0.0]]))
        observe_gail(router, prior, frames[0:1])
        model.active_context.copy_(torch.tensor([[1.0, 0.0]]))
        observe_gail(router, prior, frames[2:3])
        model.active_context.copy_(torch.tensor([[0.0, 1.0]]))
        observe_gail(router, prior, f_mid)

    # Seen context (-1, 0) snaps cleanly to discrete prototype frames[0:1]
    model.active_context.copy_(torch.tensor([[-1.0, 0.0]]))
    assert torch.allclose(clean_codes(prior)[0:1], router.encode(frames[0:1]), atol=1e-4)

    # Unseen midpoint (0, 0) extrapolates via RLS + spike-and-slab to 0.5 * (f0 + f2)
    model.active_context.copy_(torch.tensor([[0.0, 0.0]]))
    mid_codes = clean_codes(prior)
    assert torch.allclose(mid_codes[0:1], router.encode(f_mid), atol=2e-2)
    diag = router.diagnostics()
    assert diag["assoc_active"] is True
    assert diag["assoc_updates"] == 30
    assert diag["observation_state_retained"] is False


def test_high_snr_feature_scale_quadruples_latent_energy_without_splitting_prototypes():
    m_base, r_base, p_base = make_gail_prior(
        feature_scale=2.0, feature_tolerance=0.15, spike_slab_scale=1.0, assoc_ridge=0.1
    )
    m_snr, r_snr, p_snr = make_gail_prior(
        feature_scale=4.0, feature_tolerance=0.30, spike_slab_scale=2.0, assoc_ridge=0.1
    )
    assert m_base is not None and m_snr is not None
    frames = bars4_templates()
    for _ in range(3):
        for i in range(4):
            observe_gail(r_base, p_base, frames[i : i + 1])
            observe_gail(r_snr, p_snr, frames[i : i + 1])
    assert int(r_base.feature_count.item()) == int(r_snr.feature_count.item()) == 4
    e_base = r_base.encode(frames[0:1]).square().sum().item()
    e_snr = r_snr.encode(frames[0:1]).square().sum().item()
    assert e_snr == pytest.approx(4.0 * e_base, rel=1e-6)
