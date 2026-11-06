"""The declared gates and actual evaluation path agree with challenge controls."""
import torch
import pytest

from particlegan_forge import grade_sustained_observations
from particlestream.challenge_adapter import (
    CHALLENGE_TASKS, evaluate_challenge, make_challenge_stream,
)
from particlestream.challenges import (
    score_circle_orbit, score_delayed_cue, score_persistent_fork, score_rare_mode_bars,
)
from particlestream.models import FastWeightMemory, StreamImageCritic, StreamImageGenerator
from particlestream.streams import score_multimodal_bars


def _controls(task):
    mode = task.host_definition["mode"]
    stream = make_challenge_stream(mode, "cpu")
    if mode == "variable_dwell":
        return (score_multimodal_bars(stream.templates.repeat(8, 1, 1, 1), stream.templates),
                score_multimodal_bars(stream.templates[:1].repeat(32, 1, 1, 1), stream.templates))
    if mode == "rare_mode":
        oracle = torch.cat([stream.templates[i:i + 1].repeat(n, 1, 1, 1) for i, n in enumerate((10, 10, 10, 2))])
        missing = oracle.clone()
        missing[-2:] = stream.templates[:1]
        return score_rare_mode_bars(oracle, stream.templates), score_rare_mode_bars(missing, stream.templates)
    if mode == "delayed_cue":
        left, right = [stream.outcomes[i:i + 1].repeat(32, 1, 1, 1) for i in (0, 1)]
        return score_delayed_cue(left, right, stream.outcomes), score_delayed_cue(left, left, stream.outcomes)
    if mode == "circle_orbit":
        oracle = stream.templates.unsqueeze(0).repeat(32, 1, 1, 1, 1)
        incomplete = oracle.clone()
        incomplete[:, -1] = stream.templates[0]
        return score_circle_orbit(oracle, stream.templates), score_circle_orbit(incomplete, stream.templates)
    oracle = stream.branch_paths.repeat(16, 1, 1, 1, 1)
    switching = oracle.clone()
    switching[:, 1] = oracle.flip(0)[:, 1]
    return score_persistent_fork(oracle, stream.branch_paths), score_persistent_fork(switching, stream.branch_paths)


@pytest.mark.parametrize("task", CHALLENGE_TASKS, ids=lambda task: task.id)
def test_declared_challenge_gates_accept_oracle_reject_destructive(task):
    passing, failing = _controls(task)
    for metrics, expected in ((passing, "PASS"), (failing, "FAIL")):
        result = grade_sustained_observations(
            task, [{"step": step, "metrics": metrics} for step in task.observation_steps()],
            completed_steps=480, all_finite=True, unintended_rng_deviations=0,
        )
        assert result["status"] == expected


def test_closed_loop_fork_keeps_particle_identity_and_live_memory_unchanged():
    stream = make_challenge_stream("persistent_fork", "cpu")
    generator = StreamImageGenerator(context_dim=16)
    critic = StreamImageCritic(context_dim=16)
    memory = FastWeightMemory()
    memory.write(torch.full((1, 1, 8, 8), .2))
    before = {name: value.clone() for name, value in memory.state_dict().items()}
    contexts = (generator.active_context.clone(), critic.active_context.clone())

    class OracleSampler:
        def __init__(self):
            self.calls = []
            self.draws = []

        def sample(self, count, *, fixed_first_n, offset, generator):
            assert count == 1 and fixed_first_n
            depth = len(self.calls) % 3
            self.calls.append(offset)
            self.draws.append(torch.rand(1, generator=generator).item())
            return stream.branch_paths[offset % 2, depth:depth + 1].clone()

    sampler = OracleSampler()
    metrics, samples = evaluate_challenge(
        "persistent_fork", sampler, generator, critic, memory, stream, 32,
        torch.Generator().manual_seed(0),
    )
    assert metrics["trajectory_consistency"] == 1
    assert metrics["branch_tv"] == 0
    assert sampler.calls == [particle for particle in range(32) for _ in range(3)] * 2
    assert all(len(set(sampler.draws[i:i + 3])) == 1 for i in range(0, 192, 3))
    assert samples.shape == (32, 1, 8, 8)
    assert all(torch.equal(value, before[name]) for name, value in memory.state_dict().items())
    assert torch.equal(generator.active_context, contexts[0])
    assert torch.equal(critic.active_context, contexts[1])


def test_closed_loop_circle_orbit_keeps_particle_identity_and_live_memory_unchanged():
    stream = make_challenge_stream("circle_orbit", "cpu")
    generator = StreamImageGenerator(context_dim=16)
    critic = StreamImageCritic(context_dim=16)
    memory = FastWeightMemory()
    memory.write(torch.full((1, 1, 8, 8), .2))
    before = {name: value.clone() for name, value in memory.state_dict().items()}
    contexts = (generator.active_context.clone(), critic.active_context.clone())

    class OracleOrbitSampler:
        def __init__(self):
            self.calls = []
            self.draws = []

        def sample(self, count, *, fixed_first_n, offset, generator):
            assert count == 1 and fixed_first_n
            phase = len(self.calls) % 8
            self.calls.append(offset)
            self.draws.append(torch.rand(1, generator=generator).item())
            return stream.templates[phase:phase + 1].clone()

    sampler = OracleOrbitSampler()
    metrics, samples = evaluate_challenge(
        "circle_orbit", sampler, generator, critic, memory, stream, 32,
        torch.Generator().manual_seed(0),
    )
    assert metrics["orbit_completion"] == 1.0
    assert metrics["step_accuracy"] == 1.0
    assert metrics["waypoint_modes"] == 8.0
    assert metrics["orbit_rmse"] == 0.0
    assert metrics["xy_rmse"] == 0.0
    assert sampler.calls == [particle for particle in range(32) for _ in range(8)]
    assert all(len(set(sampler.draws[i:i + 8])) == 1 for i in range(0, 256, 8))
    assert samples.shape == (32, 1, 8, 8)
    assert all(torch.equal(value, before[name]) for name, value in memory.state_dict().items())
    assert torch.equal(generator.active_context, contexts[0])
    assert torch.equal(critic.active_context, contexts[1])
