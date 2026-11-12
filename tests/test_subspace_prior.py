"""Unit and lifecycle audits for SubspaceResidualConditionalReconstruction."""
from copy import deepcopy
import math

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe
from particlegan.particle_prior import MoGParticlePrior
from particlegan.priors import ManipulatedPrior
from particlegan_forge import NamedStreams
from particlestream.challenges import DelayedCueStream
from particlestream.memory import CausalStateMemory
from particlestream.models import StreamImageCritic, StreamImageGenerator
from particlestream.online import OnlineSession
from particlestream.streams import bars4_templates
from particlestream.subspace_prior import SubspaceResidualConditionalReconstruction
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
    base = MoGParticlePrior(
        num_particles=32,
        z_dim=8,
        sigma=0.025,
        standardize=False,
        generator=torch.Generator().manual_seed(0),
    )
    base.requires_grad_(False)
    router = SubspaceResidualConditionalReconstruction(model, **config)
    wrapped = ManipulatedPrior(base, [router])
    return model, router, wrapped


def observe(router, prior, frame):
    with router.training_target(frame):
        prior.sample(1, generator=torch.Generator().manual_seed(0))
        prior.sample(1, generator=torch.Generator().manual_seed(0))


def clean_codes(prior):
    prior.eval()
    return prior.sample(32, fixed_first_n=True, eps=torch.zeros(32, 8))[0]


def test_subspace_prior_has_zero_parameters_and_preserves_global_rng():
    before = torch.random.get_rng_state()
    model, router, wrapped = make_prior(context_dim=16, encoder_seed=123)
    assert model is not None
    assert torch.equal(torch.random.get_rng_state(), before)
    assert list(router.parameters()) == []
    assert list(wrapped.parameters()) == list(wrapped.base.parameters())
    assert router.subspace_basis.shape == (6, 64)
    assert router.subspace_gains.shape == (6,)
    assert router.context_features.shape == (64, 8)
    assert router.context_features.dtype == torch.float64


def test_gram_schmidt_orthonormality_precision_whitening_and_zero_retroactive_drift():
    model, router, prior = make_prior(
        context_dim=0,
        encoder_seed=17,
        feature_scale=2.1,
        subspace_dims=6,
        subspace_nu0=0.10,
    )
    assert model is not None
    # Construct 6 mutually orthogonal micro- and macro-scale patterns across a 50x dynamic range.
    amplitudes = (0.45, 0.30, 0.12, 0.04, 0.015, 0.009)
    frames = []
    for idx, amp in enumerate(amplitudes):
        vec = torch.full((64,), 0.5)
        vec[8 * idx : 8 * (idx + 1)] += amp
        vec[8 * idx + 4 : 8 * (idx + 1)] -= 2.0 * amp
        frames.append(vec.reshape(1, 1, 8, 8))

    first_codes = []
    for k, frame in enumerate(frames):
        observe(router, prior, frame)
        assert int(router.subspace_count.item()) == k + 1
        first_codes.append(router.encode(frame).clone())

    basis = router.subspace_basis.to(dtype=torch.float64)
    gram = basis @ basis.t()
    assert torch.allclose(gram, torch.eye(6, dtype=torch.float64), atol=1e-6)

    # Earlier encoded frames suffer zero retroactive coordinate drift on their discovered subspace coordinates.
    for k, frame in enumerate(frames):
        later_code = router.encode(frame)
        assert torch.allclose(later_code[:, : k + 1], first_codes[k][:, : k + 1], atol=1e-6)
        # Even the smallest micro-displacement achieves SNR >> 1 over prior_sigma = 0.025.
        assert float(later_code[0, k].abs().item()) > 0.40


def test_zero_context_preserves_exact_marginal_empirical_quantiles():
    model, router, prior = make_prior(context_dim=0, encoder_seed=41)
    assert model is not None
    frames = bars4_templates()
    for index, count in enumerate((5, 5, 5, 1)):
        for _ in range(count):
            observe(router, prior, frames[index : index + 1])
    codes = clean_codes(prior)
    encoded = torch.cat([router.encode(f.unsqueeze(0)) for f in frames])
    labels = (codes[:, None] - encoded[None]).square().sum(-1).argmin(-1)
    assert torch.bincount(labels, minlength=4).tolist() == [10, 10, 10, 2]


def test_semiparametric_nw_residual_beats_pure_linear_rls_on_curved_continuous_manifold():
    nw_model, nw_router, nw_prior = make_prior(
        context_dim=16, encoder_seed=29, nw_weight=1.0, nw_bandwidth=0.55, nw_ridge=0.04,
    )
    lin_model, lin_router, lin_prior = make_prior(
        context_dim=16, encoder_seed=29, nw_weight=0.0, nw_bandwidth=0.55, nw_ridge=0.04,
    )
    dirs = []
    for idx in range(5):
        vec = torch.zeros(64)
        vec[8 * idx : 8 * idx + 4] = 1.0
        vec[8 * idx + 4 : 8 * idx + 8] = -1.0
        dirs.append(vec.reshape(1, 1, 8, 8))

    def manifold_frame(u: torch.Tensor) -> torch.Tensor:
        linear_part = sum(0.22 * float(u[i].item()) * dirs[i] for i in range(4))
        nonlinear_part = 0.14 * math.sin(2.2 * float(u[0].item())) * dirs[4]
        return (0.5 + linear_part + nonlinear_part).clamp(0.05, 0.95)

    rng = torch.Generator().manual_seed(11)
    samples_u = 2.0 * torch.rand(96, 4, generator=rng) - 1.0
    samples_u[:, 2:] *= 0.25
    for step in range(96):
        u = samples_u[step]
        ctx = torch.zeros(1, 16)
        ctx[0, :4] = 3.0 * u
        nw_model.active_context.copy_(ctx)
        lin_model.active_context.copy_(ctx)
        frame = manifold_frame(u)
        observe(nw_router, nw_prior, frame)
        observe(lin_router, lin_prior, frame)

    assert nw_router._disc_gate() < 0.05
    u_test = samples_u[-3] + torch.tensor([0.03, -0.02, 0.01, -0.01])
    for model in (nw_model, lin_model):
        model.active_context.zero_()
        model.active_context[0, :4] = 3.0 * u_test
    expected = nw_router.encode(manifold_frame(u_test))[0]
    nw_pred = clean_codes(nw_prior)[0]
    lin_pred = clean_codes(lin_prior)[0]
    nw_err = float(torch.linalg.vector_norm(nw_pred - expected).item())
    lin_err = float(torch.linalg.vector_norm(lin_pred - expected).item())
    assert nw_err < 0.55 * lin_err


def _online_runtime():
    streams = NamedStreams(seed=0)
    with streams.fork("init", component="models", purpose="construction"):
        generator = StreamImageGenerator(context_dim=16)
        critic = StreamImageCritic(context_dim=16)
        memory = CausalStateMemory(
            association_weight=0.0, normalize_key=False, key_gain=0.5, contrast_threshold=0.002,
        )
        for module, component in ((generator, "generator"), (critic, "discriminator"), (memory, "memory")):
            _init_module(module, streams, component)
    recipe = get_recipe(
        "bcap_adam",
        batch_size=1,
        num_particles=32,
        z_dim=8,
        prior_kind="mog",
        standardize=False,
        prior_update="frozen",
        total_steps=64,
        loss="non_saturating",
        lr=0.007,
        d_lr_mult=0.02,
        betas=[0.4, 0.999],
    )
    base = recipe.replace(prior_update="learned").make_prior(
        sigma=0.025, generator=streams.generator("init", component="prior", purpose="locations"),
    )
    _init_module(base, streams, "prior")
    recipe.apply_prior_policy(base)
    trainer = TransportGANTrainer(
        recipe,
        generator,
        critic,
        prior=base,
        max_steps=16,
        router_class=SubspaceResidualConditionalReconstruction,
        transport={
            "encoder_seed": streams.seed_for("init", component="feature_encoder", purpose="projection"),
            "reconstruction_loss": "binary_cross_entropy",
            "reconstruction_weight": 100.0,
            "context_bypass_weight": 5.0,
            "feature_scale": 2.1,
            "subspace_dims": 6,
            "subspace_nu0": 0.10,
            "nw_weight": 1.0,
            "nw_bandwidth": 0.65,
            "nw_ridge": 0.06,
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
        trainer,
        generator,
        critic,
        memory,
        predictive=True,
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


def test_serving_isolation_and_failed_step_transactional_rollback():
    ref_trainer, ref_session, _ = _online_runtime()
    srv_trainer, srv_session, srv_streams = _online_runtime()
    stream = DelayedCueStream(noise_std=0)
    frames = torch.cat((stream.clean_history(0), stream.outcomes[:1]))
    for frame in frames:
        with srv_streams.isolated_evaluation():
            srv_session.predict()
            srv_session.predict()
        assert srv_streams.last_eval_audit["unintended_rng_deviations"] == 0
        ref_session.observe(frame.unsqueeze(0))
        srv_session.observe(frame.unsqueeze(0))
    for left, right in (
        (ref_trainer.G, srv_trainer.G),
        (ref_trainer.D, srv_trainer.D),
        (ref_trainer.prior, srv_trainer.prior),
        (ref_session.memory, srv_session.memory),
    ):
        _assert_equal(left.state_dict(), right.state_dict())
    diag = ref_trainer.transport_diagnostics()
    assert diag["target_retained"] is False and diag["observation_state_retained"] is False
    assert diag["subspace_count"] >= 1

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
    assert ref_trainer.transport_diagnostics()["observation_state_retained"] is False


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"subspace_dims": -1},
        {"subspace_dims": 9},
        {"subspace_dims": True},
        {"subspace_abs_tol": 0.0},
        {"subspace_rel_tol": -0.1},
        {"subspace_nu0": 0.0},
        {"nw_weight": -0.5},
        {"nw_weight": False},
        {"nw_bandwidth": 0.0},
        {"nw_ridge": 0.0},
    ],
)
def test_invalid_subspace_prior_hyperparameters_raise(bad_kwargs):
    with pytest.raises(ValueError):
        make_prior(context_dim=16, **bad_kwargs)
