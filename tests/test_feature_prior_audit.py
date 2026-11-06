"""Independent lifecycle audits for the observed-feature conditional world model."""
from copy import deepcopy

import pytest
import torch

from particlegan import GANTrainer, get_recipe
from particlegan_forge import NamedStreams
from particlestream.challenge_adapter import evaluate_challenge
from particlestream.challenges import CircleOrbitStream, DelayedCueStream, PersistentForkStream, SquareCircleCycleStream
from particlestream.feature_prior import FeatureConditionalReconstruction
from particlestream.memory import CausalStateMemory
from particlestream.models import StreamImageCritic, StreamImageGenerator
from particlestream.online import OnlineSession
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


def runtime():
    streams = NamedStreams(seed=0)
    with streams.fork("init", component="models", purpose="construction"):
        generator = StreamImageGenerator(context_dim=16)
        critic = StreamImageCritic(context_dim=16)
        memory = CausalStateMemory(association_weight=0)
        for module, component in ((generator, "generator"), (critic, "discriminator"), (memory, "memory")):
            _init_module(module, streams, component)
    recipe = get_recipe("bcap_adam", batch_size=1, num_particles=32, z_dim=8,
                        prior_kind="mog", standardize=False, prior_update="frozen",
                        total_steps=480, loss="non_saturating", lr=.005)
    base = recipe.replace(prior_update="learned").make_prior(
        sigma=.025, generator=streams.generator("init", component="prior", purpose="locations"))
    _init_module(base, streams, "prior")
    recipe.apply_prior_policy(base)
    trainer = TransportGANTrainer(
        recipe, generator, critic, prior=base, max_steps=16,
        router_class=FeatureConditionalReconstruction,
        transport={"encoder_seed": streams.seed_for("init", component="feature_encoder", purpose="projection"),
                   "reconstruction_loss": "binary_cross_entropy"},
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


def assert_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for name in first:
            assert_equal(first[name], second[name])
    else:
        assert first == second


def test_extra_serving_calls_leave_actual_training_trajectory_bitwise_unchanged():
    reference, quiet, reference_streams = runtime()
    observed, serving, serving_streams = runtime()
    stream = DelayedCueStream(noise_std=0)
    frames = torch.cat((stream.clean_history(0), stream.outcomes[:1],
                        stream.clean_history(1), stream.outcomes[1:2]))
    for frame in frames:
        with serving_streams.isolated_evaluation():
            for _ in range(3):
                serving.predict()
        assert serving_streams.last_eval_audit["unintended_rng_deviations"] == 0
        expected_stats = quiet.observe(frame.unsqueeze(0))
        actual_stats = serving.observe(frame.unsqueeze(0))
        if expected_stats is not None:
            assert_equal(expected_stats, actual_stats)
    for first, second in ((reference.G, observed.G), (reference.D, observed.D),
                          (reference.prior, observed.prior), (quiet.memory, serving.memory)):
        assert_equal(first.state_dict(), second.state_dict())
    for name in reference._STREAMS:
        if name != "eval_generator":
            assert torch.equal(getattr(reference, name).get_state(), getattr(observed, name).get_state())
    assert reference.completed_steps == observed.completed_steps == 11
    assert quiet.arrivals == serving.arrivals == 12
    assert serving.predictions == 36 and quiet.predictions == 0
    assert quiet.max_real_batch == quiet.max_fake_batch == serving.max_real_batch == serving.max_fake_batch == 1
    assert observed.transport._target is observed.transport._pending_context is observed.transport._pending_feature is None


def test_prior_statistics_commit_preceding_context_before_memory_advances():
    trainer, session, _ = runtime()
    stream = DelayedCueStream(noise_std=0)
    session.observe(stream.cues[:1])
    preceding = session.context
    assert trainer.transport.context_count == 0
    session.observe(stream.distractor)
    assert trainer.transport.context_count == trainer.transport.feature_count == 1
    assert torch.equal(trainer.transport.context_keys[0], preceding[0])
    assert not torch.equal(session.context, preceding)
    assert torch.equal(trainer.transport.feature_centroids[0], trainer.transport.encode(stream.distractor)[0])
    assert session.arrivals == 2 and session.updates == 1


def test_interrupted_public_update_preserves_feature_history_and_online_memory(monkeypatch):
    trainer, session, _ = runtime()
    stream = DelayedCueStream(noise_std=0)
    session.observe(stream.cues[:1])
    session.observe(stream.distractor)
    prior_state = deepcopy(trainer.prior.state_dict())
    memory_state = deepcopy(session.memory.state_dict())
    pending = session.context
    arrivals, updates = session.arrivals, session.updates

    def interrupt(self, real, **kwargs):
        for _ in range(2):
            self.prior.sample(1, generator=self.latent_generator, noise_generator=self.prior_noise_generator)
        raise RuntimeError("interrupted after both optimizer draws")

    monkeypatch.setattr(GANTrainer, "step", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        session.observe(stream.outcomes[1:2])
    assert_equal(trainer.prior.state_dict(), prior_state)
    assert_equal(session.memory.state_dict(), memory_state)
    assert torch.equal(session.context, pending)
    assert session.arrivals == arrivals and session.updates == updates
    assert trainer.transport._target is trainer.transport._pending_context is trainer.transport._pending_feature is None


def test_invalid_bce_arrival_does_not_change_world_model_or_retain_future_feature():
    trainer, session, _ = runtime()
    stream = DelayedCueStream(noise_std=0)
    session.observe(stream.cues[:1])
    prior_state = deepcopy(trainer.prior.state_dict())
    memory_state = deepcopy(session.memory.state_dict())
    training_rng = {name: getattr(trainer, name).get_state() for name in trainer._STREAMS}
    with pytest.raises(ValueError, match="targets must be floating values"):
        session.observe(torch.full((1, 1, 8, 8), 1.01))
    assert_equal(trainer.prior.state_dict(), prior_state)
    assert_equal(session.memory.state_dict(), memory_state)
    assert all(torch.equal(value, getattr(trainer, name).get_state()) for name, value in training_rng.items())
    assert session.arrivals == 1 and session.updates == 0
    assert trainer.transport.feature_count == trainer.transport.context_count == 0
    assert trainer.transport._target is trainer.transport._pending_context is trainer.transport._pending_feature is None


def test_closed_loop_fork_writes_generated_frames_only_to_cloned_history(monkeypatch):
    trainer, session, streams = runtime()
    stream = PersistentForkStream(noise_std=0)
    session.observe(stream.stem)
    prior_state = deepcopy(trainer.prior.state_dict())
    memory_state = deepcopy(session.memory.state_dict())
    writes = []
    original_write = CausalStateMemory.write

    def track_write(memory, frame):
        writes.append((id(memory), frame.detach().clone()))
        return original_write(memory, frame)

    monkeypatch.setattr(CausalStateMemory, "write", track_write)
    monkeypatch.setattr(trainer.G, "forward", lambda codes: codes.new_full((len(codes), 1, 8, 8), .37))
    with streams.isolated_evaluation(), session.isolated_state():
        metrics, samples = evaluate_challenge(
            "persistent_fork", trainer, trainer.G, trainer.D, session.memory, stream, 32,
            streams.generator("eval", component="rollout", purpose="audit"),
        )
    assert streams.last_eval_audit["unintended_rng_deviations"] == 0
    assert metrics["hq"] == 0 and metrics["trajectory_consistency"] == 0
    assert samples.shape == (32, 1, 8, 8) and torch.all(samples == .37)
    assert len(writes) == 2 * 32 * 4
    for path in range(64):
        history = writes[4 * path:4 * path + 4]
        previous = path // 32
        assert all(identifier != id(session.memory) for identifier, _ in history)
        assert torch.equal(history[0][1], stream.branch_paths[previous, -1:])
        assert torch.equal(history[1][1], stream.stem)
        assert torch.all(history[2][1] == .37) and torch.all(history[3][1] == .37)
    assert_equal(trainer.prior.state_dict(), prior_state)
    assert_equal(session.memory.state_dict(), memory_state)


def test_closed_loop_circle_orbit_writes_generated_frames_only_to_cloned_history(monkeypatch):
    trainer, session, streams = runtime()
    stream = CircleOrbitStream(noise_std=0)
    session.observe(stream.templates[:1])
    prior_state = deepcopy(trainer.prior.state_dict())
    memory_state = deepcopy(session.memory.state_dict())
    writes = []
    original_write = CausalStateMemory.write

    def track_write(memory, frame):
        writes.append((id(memory), frame.detach().clone()))
        return original_write(memory, frame)

    monkeypatch.setattr(CausalStateMemory, "write", track_write)
    monkeypatch.setattr(trainer.G, "forward", lambda codes: codes.new_full((len(codes), 1, 8, 8), .37))
    with streams.isolated_evaluation(), session.isolated_state():
        metrics, samples = evaluate_challenge(
            "circle_orbit", trainer, trainer.G, trainer.D, session.memory, stream, 32,
            streams.generator("eval", component="rollout", purpose="audit"),
        )
    assert streams.last_eval_audit["unintended_rng_deviations"] == 0
    assert metrics["orbit_completion"] == 0.0 and metrics["step_accuracy"] == 0.0
    assert samples.shape == (32, 1, 8, 8) and torch.all(samples == .37)
    # 32 particles * (2 warmup frames + 7 self-fed rollout steps) = 288 writes to cloned memory
    assert len(writes) == 32 * 9
    warmup = stream.clean_warmup(0)
    for particle in range(32):
        history = writes[9 * particle:9 * particle + 9]
        assert all(identifier != id(session.memory) for identifier, _ in history)
        assert torch.equal(history[0][1], warmup[0:1])
        assert torch.equal(history[1][1], warmup[1:2])
        assert all(torch.all(step_frame == .37) for _, step_frame in history[2:])
    assert_equal(trainer.prior.state_dict(), prior_state)
    assert_equal(session.memory.state_dict(), memory_state)


def test_closed_loop_square_circle_cycle_writes_generated_frames_only_to_cloned_history(monkeypatch):
    trainer, session, streams = runtime()
    rev_trainer, rev_session, _ = runtime()
    trainer.cycle_partner = {
        "trainer": rev_trainer,
        "generator": rev_trainer.G,
        "critic": rev_trainer.D,
        "memory": rev_session.memory,
        "session": rev_session,
    }
    stream = SquareCircleCycleStream(noise_std=0)
    s0, c0, _ = stream.next_pair(streams.generator("data", component="stream", purpose="audit"))
    session.observe(c0, context_frame=s0)
    rev_session.observe(s0, context_frame=c0)
    prior_state = deepcopy(trainer.prior.state_dict())
    rev_prior_state = deepcopy(rev_trainer.prior.state_dict())
    memory_state = deepcopy(session.memory.state_dict())
    rev_memory_state = deepcopy(rev_session.memory.state_dict())
    writes = []
    original_write = CausalStateMemory.write

    def track_write(memory, frame):
        writes.append((id(memory), frame.detach().clone()))
        return original_write(memory, frame)

    monkeypatch.setattr(CausalStateMemory, "write", track_write)
    monkeypatch.setattr(trainer.G, "forward", lambda codes: codes.new_full((len(codes), 1, 8, 8), .37))
    monkeypatch.setattr(rev_trainer.G, "forward", lambda codes: codes.new_full((len(codes), 1, 8, 8), .43))
    with streams.isolated_evaluation(), session.isolated_state(), rev_session.isolated_state():
        metrics, samples = evaluate_challenge(
            "square_circle_cycle", trainer, trainer.G, trainer.D, session.memory, stream, 32,
            streams.generator("eval", component="rollout", purpose="audit_cycle"),
        )
    assert streams.last_eval_audit["unintended_rng_deviations"] == 0
    assert metrics["cycle_completion"] == 0.0 and metrics["step_accuracy"] == 0.0
    assert samples.shape == (32, 1, 8, 8)
    assert torch.all(samples[:16] == .37) and torch.all(samples[16:] == .43)
    # 32 particles * 2 domains * (2 warmup frames + 7 cross-fed rollout steps) = 576 writes to cloned memories
    assert len(writes) == 32 * 18
    live_ids = {id(session.memory), id(rev_session.memory)}
    assert all(identifier not in live_ids for identifier, _ in writes)
    assert_equal(trainer.prior.state_dict(), prior_state)
    assert_equal(rev_trainer.prior.state_dict(), rev_prior_state)
    assert_equal(session.memory.state_dict(), memory_state)
    assert_equal(rev_session.memory.state_dict(), rev_memory_state)
