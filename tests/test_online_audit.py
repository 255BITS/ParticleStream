"""Independent adversarial checks of serving and chronological target matching."""
from types import SimpleNamespace
from threading import Event, Thread

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe
from particlegan_forge import NamedStreams
from particlestream.models import FastWeightMemory, StreamImageCritic, StreamImageGenerator
from particlestream.online import OnlineSession
from particlestream.tier1 import _init_module


class MarkerMemory(nn.Module):
    """Expose the arriving frame's value so temporal mistakes are observable."""

    def __init__(self):
        super().__init__()
        self.register_buffer("marker", torch.zeros(1, 16))
        self.register_buffer("writes", torch.zeros((), dtype=torch.long))

    def write(self, frame):
        self.marker.fill_(float(frame[0, 0, 0, 0]))
        self.writes.add_(1)
        return self.marker


class RecordingTrainer:
    def __init__(self, generator, critic):
        self.recipe = SimpleNamespace(batch_size=1)
        self.G, self.D = generator, critic
        self.examples = []
        self.input_noise_generator = torch.Generator().manual_seed(0)

    def step(self, frame):
        self.examples.append((frame.clone(), self.G.active_context.clone(), self.D.active_context.clone()))
        self.G(torch.zeros(1, 8))
        return {"step": len(self.examples)}

    def sample(self, n, *, generator=None):
        return torch.zeros(n, 1, 8, 8)


def recording_session():
    generator = StreamImageGenerator(context_dim=16)
    critic = StreamImageCritic(context_dim=16)
    trainer = RecordingTrainer(generator, critic)
    memory = MarkerMemory()
    session = OnlineSession(trainer, generator, critic, memory, predictive=True,
                            serving_generator=torch.Generator().manual_seed(0))
    return session, trainer


def test_arrival_trains_previous_context_and_detaches_pending_memory():
    session, trainer = recording_session()
    assert session.observe(torch.full((1, 1, 8, 8), 0.2)) is None
    session.predict()
    session.observe(torch.full((1, 1, 8, 8), 0.8))
    session.observe(torch.full((1, 1, 8, 8), 0.4))
    assert len(trainer.examples) == 2
    for (frame, generator_context, critic_context), target, preceding in zip(
        trainer.examples, (0.8, 0.4), (0.2, 0.8), strict=True
    ):
        assert torch.allclose(frame, torch.full_like(frame, target))
        assert torch.allclose(generator_context, torch.full_like(generator_context, preceding))
        assert torch.equal(generator_context, critic_context)
    assert session.pending_context.data_ptr() != session.memory.marker.data_ptr()
    assert session.pending_context.grad_fn is None
    assert session.diagnostics()["arrivals"] == 3
    assert session.diagnostics()["updates"] == 2
    assert session.diagnostics()["max_real_batch"] == 1
    assert session.diagnostics()["max_fake_batch"] == 1


@pytest.mark.parametrize("raise_during_sampling", [False, True])
def test_serving_restores_every_live_buffer_and_module_mode(raise_during_sampling):
    session, trainer = recording_session()
    session.observe(torch.full((1, 1, 8, 8), 0.2))
    roots = (session.generator, session.critic, session.memory)
    before = [(module, module.training) for root in roots for module in root.modules()]
    buffers = [(buffer, buffer.clone()) for root in roots for buffer in root.buffers()]
    pending = session.pending_context.clone()

    def corrupting_sample(n, *, generator=None):
        for root in roots:
            root.eval()
            for buffer in root.buffers():
                buffer.zero_()
        if raise_during_sampling:
            raise RuntimeError("sampling interrupted")
        return torch.zeros(n, 1, 8, 8)

    trainer.sample = corrupting_sample
    if raise_during_sampling:
        with pytest.raises(RuntimeError, match="sampling interrupted"):
            session.predict()
    else:
        assert session.predict().shape == (1, 1, 8, 8)
    assert all(torch.equal(buffer, saved) for buffer, saved in buffers)
    assert all(module.training == flag for module, flag in before)
    assert torch.equal(session.pending_context, pending)


def test_serving_cannot_use_an_auxiliary_training_rng():
    generator = StreamImageGenerator(context_dim=16)
    critic = StreamImageCritic(context_dim=16)
    trainer = RecordingTrainer(generator, critic)
    with pytest.raises(ValueError, match="(serving|sampling|training|separate)"):
        session = OnlineSession(trainer, generator, critic,
                                serving_generator=trainer.input_noise_generator)
        session.predict()


def test_serving_request_waits_for_an_in_progress_update():
    """A concurrent serving request must not observe half-applied weights."""
    session, trainer = recording_session()
    session.observe(torch.full((1, 1, 8, 8), 0.2))
    updating, release_update, sampled = Event(), Event(), Event()
    original_step, original_sample = trainer.step, trainer.sample
    failures = []

    def blocking_step(frame):
        updating.set()
        if not release_update.wait(2):
            raise RuntimeError("test did not release training update")
        return original_step(frame)

    def observed_sample(n, *, generator=None):
        sampled.set()
        return original_sample(n, generator=generator)

    def run(call):
        try:
            call()
        except BaseException as error:
            failures.append(error)

    trainer.step, trainer.sample = blocking_step, observed_sample
    training = Thread(target=run, args=(lambda: session.observe(torch.full((1, 1, 8, 8), 0.8)),))
    serving = Thread(target=run, args=(session.predict,))
    training.start()
    try:
        assert updating.wait(2)
        serving.start()
        assert not sampled.wait(0.05)
    finally:
        release_update.set()
        training.join(2)
        if serving.ident is not None:
            serving.join(2)
    assert not training.is_alive() and not serving.is_alive()
    assert sampled.is_set() and not failures
    assert session.updates == 1 and session.predictions == 1


def real_session():
    streams = NamedStreams(seed=0)
    with streams.fork("init", component="models", purpose="construction"):
        generator = StreamImageGenerator(context_dim=16)
        critic = StreamImageCritic(context_dim=16)
        memory = FastWeightMemory(dim=16)
        for module, component in ((generator, "generator"), (critic, "discriminator"), (memory, "memory")):
            _init_module(module, streams, component)
    recipe = get_recipe("bcap_adam", batch_size=1, num_particles=32, z_dim=8,
                        total_steps=480, lr=0.005, loss="non_saturating",
                        prior_kind="mog", standardize=False, sigma_rel=0.0)
    prior = recipe.make_prior(sigma=0.025, generator=streams.generator("init", component="prior", purpose="locations"))
    _init_module(prior, streams, "prior")
    trainer = GANTrainer(
        recipe, generator, critic, prior=prior, seed=0, max_steps=4,
        latent_generator=streams.generator("prior", component="latent", purpose="indices"),
        penalty_generator=streams.generator("noise", component="penalty", purpose="training"),
        noise_generator=streams.generator("noise", component="generator", purpose="output"),
        input_noise_generator=streams.generator("noise", component="critic", purpose="input"),
        model_generator=streams.generator("noise", component="models", purpose="stochastic_layers"),
        prior_noise_generator=streams.generator("noise", component="prior", purpose="gaussian"),
        eval_generator=streams.generator("eval", component="sampler", purpose="samples"),
    )
    session = OnlineSession(trainer, generator, critic, memory, predictive=True,
                            serving_generator=streams.generator("eval", component="serving", purpose="samples"))
    return session, streams


def assert_identical(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_identical(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_identical(a, b)
    else:
        assert left == right


def test_extra_serving_calls_do_not_change_later_actual_training():
    """A serving request count change must leave the learned state identical."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        first, first_streams = real_session()
        second, second_streams = real_session()
        for index, value in enumerate((0.1, 0.8, 0.3, 0.6, 0.2)):
            frame = torch.full((1, 1, 8, 8), value)
            first.predict()
            for _ in range(index + 3):
                second.predict()
            first.observe(frame)
            second.observe(frame)
        for left, right in ((first.generator, second.generator), (first.critic, second.critic),
                            (first.memory, second.memory), (first.trainer.prior, second.trainer.prior)):
            assert_identical(left.state_dict(), right.state_dict())
        for name in ("opt_g", "opt_d"):
            assert_identical(getattr(first.trainer, name).state_dict(), getattr(second.trainer, name).state_dict())
        first_rng, second_rng = first_streams.audit(), second_streams.audit()
        assert {key: value for key, value in first_rng.items() if key not in first_streams.eval_stream_keys()} == {
            key: value for key, value in second_rng.items() if key not in second_streams.eval_stream_keys()
        }
        assert first.updates == second.updates == 4
        assert first.max_real_batch == second.max_real_batch == 1
        assert first.max_fake_batch == second.max_fake_batch == 1
    finally:
        torch.set_num_threads(previous_threads)
