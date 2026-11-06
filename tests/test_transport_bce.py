"""Opt-in probability reconstruction keeps saturated foreground gradients useful."""
from copy import deepcopy
from contextlib import contextmanager

import pytest
import torch
from torch import nn

from particlegan import get_recipe
from particlegan.particle_prior import ParticlePrior
from particlegan.priors import ManipulatedPrior
from particlegan_forge import NamedStreams
from particlestream.memory import CausalStateMemory
from particlestream.models import StreamImageCritic, StreamImageGenerator
from particlestream.online import OnlineSession
from particlestream.streams import bars4_templates
from particlestream.tier1 import _init_module
from particlestream.transport import CompetitiveReconstruction, TransportGANTrainer


class ConstantProbability(nn.Module):
    def forward(self, codes):
        return torch.sigmoid(codes[:, :1]).reshape(-1, 1, 1, 1).expand(-1, 1, 8, 8)


def make_router(loss="mse"):
    model = ConstantProbability()
    storage = ParticlePrior(num_particles=2, z_dim=1, generator=torch.Generator().manual_seed(0))
    router = CompetitiveReconstruction(model, reconstruction_weight=1, reconstruction_loss=loss)
    return model, router, ManipulatedPrior(storage, [router])


@contextmanager
def active_reconstruction(router, prior, target):
    with router.training_target(target):
        prior.sample(1, generator=torch.Generator().manual_seed(0))
        prior.sample(1, generator=torch.Generator().manual_seed(0))
        yield


def test_default_mse_matches_original_value_and_gradient_bitwise():
    _, router, prior = make_router()
    target = torch.linspace(.1, .9, 64).reshape(1, 1, 8, 8)
    generated = torch.linspace(.8, .2, 64).reshape(1, 1, 8, 8).requires_grad_()
    reference = generated.detach().clone().requires_grad_()
    expected = (reference - target).square().mean()
    with active_reconstruction(router, prior, target):
        actual = router.regularize(prior.z, generated=generated)
        assert torch.equal(actual, expected)
        actual.backward()
        expected.backward()
        assert torch.equal(generated.grad, reference.grad)
    assert router.manipulator_spec()["config"]["reconstruction_loss"] == "mse"
    assert router.diagnostics()["last_reconstruction_mse"] == actual.item()


def test_bce_foreground_logit_gradient_avoids_sigmoid_mse_suppression():
    gradients = {}
    target = torch.full((1, 1, 8, 8), .1)
    target[0, 0, 0, 0] = .9
    for kind in ("mse", "binary_cross_entropy"):
        _, router, prior = make_router(kind)
        logits = torch.full_like(target, torch.logit(torch.tensor(.01)), requires_grad=True)
        generated = torch.sigmoid(logits)
        with active_reconstruction(router, prior, target):
            loss = router.regularize(prior.z, generated=generated)
            loss.backward()
            gradients[kind] = float(logits.grad[0, 0, 0, 0])
    assert gradients["binary_cross_entropy"] < 0 and gradients["mse"] < 0
    assert abs(gradients["binary_cross_entropy"]) > 40 * abs(gradients["mse"])
    assert gradients["binary_cross_entropy"] == pytest.approx((.01 - .9) / 64, rel=1e-5)


@pytest.mark.parametrize("kind", ["mse", "binary_cross_entropy"])
def test_both_reconstruction_losses_have_optimum_at_target_probability(kind):
    _, router, prior = make_router(kind)
    target = torch.full((1, 1, 8, 8), .9)
    losses = []
    for value in (.8, .9, .99):
        generated = torch.full_like(target, value, requires_grad=True)
        with active_reconstruction(router, prior, target):
            loss = router.regularize(prior.z, generated=generated)
            losses.append(float(loss.detach()))
            if value == .9:
                loss.backward()
                assert generated.grad.abs().max() < 1e-6
    assert losses[1] < losses[0] and losses[1] < losses[2]


@pytest.mark.parametrize("prediction", [0., 1., -1., 2.])
def test_bce_probability_clamp_keeps_boundary_loss_and_gradients_finite(prediction):
    _, router, prior = make_router("binary_cross_entropy")
    target = torch.full((1, 1, 8, 8), .9)
    generated = torch.full_like(target, prediction, requires_grad=True)
    with active_reconstruction(router, prior, target):
        loss = router.regularize(prior.z, generated=generated)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(generated.grad).all()
    diagnostics = router.diagnostics()
    assert "last_reconstruction_binary_cross_entropy" in diagnostics
    assert "last_reconstruction_mse" not in diagnostics


@pytest.mark.parametrize("value", [-.01, 1.01])
def test_invalid_bce_target_rejected_before_any_routing_state_changes(value):
    _, router, prior = make_router("binary_cross_entropy")
    before = deepcopy(router.state_dict())
    with pytest.raises(ValueError, match=r"targets must be floating values in \[0,1\]"):
        with router.training_target(torch.full((1, 1, 8, 8), value)):
            pytest.fail("out-of-range target entered the optimizer scope")
    assert router._target is None and router._phase == 0
    assert all(torch.equal(saved, router.state_dict()[name]) if isinstance(saved, torch.Tensor)
               else saved == router.state_dict()[name] for name, saved in before.items())


def test_bce_metric_is_checkpointed_and_mismatched_metrics_cannot_resume():
    _, router, prior = make_router("binary_cross_entropy")
    target = torch.full((1, 1, 8, 8), .9)
    with active_reconstruction(router, prior, target):
        router.regularize(prior.z, generated=torch.full_like(target, .7))
    checkpoint = deepcopy(prior.state_dict())
    _, restored, restored_prior = make_router("binary_cross_entropy")
    restored_prior.load_state_dict(checkpoint)
    assert restored.manipulator_spec() == router.manipulator_spec()
    assert restored.last_reconstruction == router.last_reconstruction
    _, _, mse_prior = make_router("mse")
    with pytest.raises(ValueError, match="configuration mismatch"):
        mse_prior.load_state_dict(checkpoint)


@pytest.mark.parametrize("value", ["bce", "unknown", None, False])
def test_unknown_reconstruction_metric_rejected(value):
    with pytest.raises(ValueError, match="reconstruction_loss"):
        make_router(value)


def test_public_frozen_prior_bce_update_is_batch_one_and_invalid_targets_do_not_advance():
    streams = NamedStreams(seed=0)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with streams.fork("init", component="models", purpose="construction"):
            generator = StreamImageGenerator(context_dim=16)
            critic = StreamImageCritic(context_dim=16)
            memory = CausalStateMemory()
            for module, component in ((generator, "generator"), (critic, "discriminator"), (memory, "memory")):
                _init_module(module, streams, component)
        recipe = get_recipe("bcap_adam", batch_size=1, num_particles=32, z_dim=8,
                            total_steps=480, prior_kind="mog", standardize=False,
                            prior_update="frozen", loss="non_saturating")
        prior = recipe.replace(prior_update="learned").make_prior(
            sigma=.025, generator=streams.generator("init", component="prior", purpose="locations"))
        _init_module(prior, streams, "prior")
        recipe.apply_prior_policy(prior)
        trainer = TransportGANTrainer(
            recipe, generator, critic, prior=prior, max_steps=2,
            transport={"reconstruction_loss": "binary_cross_entropy"},
            latent_generator=streams.generator("prior", component="latent", purpose="indices"),
            penalty_generator=streams.generator("noise", component="penalty", purpose="training"),
            noise_generator=streams.generator("noise", component="generator", purpose="output"),
            input_noise_generator=streams.generator("noise", component="critic", purpose="input"),
            model_generator=streams.generator("noise", component="models", purpose="stochastic_layers"),
            prior_noise_generator=streams.generator("noise", component="prior", purpose="gaussian"),
            eval_generator=streams.generator("eval", component="sampler", purpose="samples"),
        )
        frozen_rows = trainer.prior.z.detach().clone()
        session = OnlineSession(trainer, generator, critic, memory, predictive=True,
                                serving_generator=streams.generator("eval", component="serving", purpose="sample"))
        images = bars4_templates()
        assert session.observe(images[:1]) is None
        with streams.isolated_evaluation():
            session.predict()
        assert streams.last_eval_audit["unintended_rng_deviations"] == 0
        stats = session.observe(images[1:2])
        assert torch.isfinite(stats["prior_manipulation"])
        assert trainer.completed_steps == session.updates == trainer.transport.assignment_updates == 1
        assert session.max_real_batch == session.max_fake_batch == 1
        assert torch.equal(trainer.prior.z, frozen_rows)
        assert trainer.transport._target is None
        weights = [parameter.detach().clone() for module in (generator, critic) for parameter in module.parameters()]
        training_rng = {name: stream.get_state() for name, stream in vars(trainer).items()
                        if name.endswith("generator") and isinstance(stream, torch.Generator)}
        with pytest.raises(ValueError, match="targets must be floating values"):
            trainer.step(torch.full((1, 1, 8, 8), 1.1))
        assert trainer.completed_steps == trainer.transport.assignment_updates == 1
        assert trainer.transport._target is None
        current = [parameter for module in (generator, critic) for parameter in module.parameters()]
        assert all(torch.equal(before, after) for before, after in zip(weights, current))
        assert all(torch.equal(state, getattr(trainer, name).get_state()) for name, state in training_rng.items())
    finally:
        torch.set_num_threads(previous_threads)
