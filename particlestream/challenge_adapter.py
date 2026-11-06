"""Task declarations and isolated evaluation for the online Tier 1 challenges."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import torch
from particlegan_forge import MetricBound, TaskSpec

from .challenges import (
    DelayedCueStream,
    PersistentForkStream,
    RareModeBarsStream,
    VariableDwellBarsStream,
    score_delayed_cue,
    score_persistent_fork,
    score_rare_mode_bars,
)
from .streams import score_fork_rollout, score_multimodal_bars


def _task(task_id, description, mode, bounds, *, predictive=False, **host):
    return TaskSpec(
        id=task_id, description=description, adapter="seq_image", steps=480,
        batch_size=1, num_particles=32, z_dim=8, prior_kind="mog", prior_sigma=0.025,
        thresholds=[MetricBound(*bound) for bound in bounds],
        host_definition={
            "mode": mode, "width": 16, "context_dim": 16 if predictive else 0,
            "online_protocol": "causal_single_arrival_v1", **host,
        },
    )


CHALLENGE_TASKS = [
    _task(
        "seq_img_variable_dwell_bars4", "Retain four modes across balanced variable dwell blocks of 4–32 arriving frames.",
        "variable_dwell", [("hq", ">=", .85), ("modes", "==", 4.), ("finite_template_tv", "<=", .20)],
    ),
    _task(
        "seq_img_rare_mode_bars4", "Retain a rare fourth mode in a sequential stream with exact 5:5:5:1 frequencies.",
        "rare_mode", [("hq", ">=", .85), ("finite_template_tv", "<=", .15),
                      ("support_modes", "==", 4.), ("rare_mass", ">=", 1 / 32), ("rare_mass", "<=", 3 / 32)],
    ),
    _task(
        "seq_img_delayed_cue", "Predict a cue-dependent outcome after four pixel-identical intervening observations.",
        "delayed_cue", [("outcome_accuracy", ">=", .90), ("prediction_rmse", "<=", .12)],
        predictive=True, delay=4,
    ),
    _task(
        "seq_img_persistent_fork", "Generate both three-step fork futures while preserving each particle's branch without teacher forcing.",
        "persistent_fork", [("hq", ">=", .85), ("within_state_modes", "==", 2.),
                            ("branch_tv", "<=", .20), ("trajectory_consistency", ">=", .95)],
        predictive=True, rollout_horizon=3, rollout_latent_policy="fixed_particle_and_noise_v1",
        branch_law="independent_fair_on_arrival", evaluation_histories="previous_terminal_then_stem",
    ),
]

CHALLENGE_MODES = {task.host_definition["mode"] for task in CHALLENGE_TASKS}
PREDICTIVE_CHALLENGE_MODES = {"delayed_cue", "persistent_fork"}


def make_challenge_stream(mode, device):
    return {
        "variable_dwell": VariableDwellBarsStream,
        "rare_mode": RareModeBarsStream,
        "delayed_cue": DelayedCueStream,
        "persistent_fork": PersistentForkStream,
    }[mode](device=device)


def challenge_target_bank(mode, stream):
    if mode == "delayed_cue":
        return stream.outcomes
    if mode == "persistent_fork":
        return stream.branch_paths.reshape(-1, 1, 8, 8)
    return stream.templates


@contextmanager
def _saved_context(generator, critic):
    """Evaluation must leave the live session's serving/training context intact."""
    states = [(module, module.active_context.detach().clone()) for module in (generator, critic)]
    try:
        yield
    finally:
        for module, state in states:
            module.active_context.copy_(state)


def _history_context(memory, history, generator, critic):
    simulated = deepcopy(memory) if memory is not None else None
    if simulated is None:
        return None
    simulated.reset_state()
    for frame in history:
        context = simulated.write(frame.unsqueeze(0))
        generator.set_context(context)
        critic.set_context(context)
    return simulated


def _worst_states(states):
    return {
        key: (max if key.endswith("_tv") or key in {"branch_tv", "mean_rmse"} else min)(state[key] for state in states)
        for key in states[0]
    }


@torch.no_grad()
def evaluate_original_fork(trainer, generator, critic, memory, stream, num_particles, eval_generator):
    """Both futures must be represented from each attainable previous-branch state."""
    states, predictions = [], []
    with _saved_context(generator, critic):
        for previous in (1, 2):
            _history_context(memory, stream.templates[[previous, 0]], generator, critic)
            samples = trainer.sample(num_particles, fixed_first_n=True, generator=eval_generator)
            states.append(score_fork_rollout(samples, stream.templates))
            predictions.append(samples)
    return _worst_states(states), torch.cat([samples[:16] for samples in predictions])


@torch.no_grad()
def evaluate_challenge(mode, trainer, generator, critic, memory, stream, num_particles, eval_generator):
    """Return metrics and actual learned images; simulation only mutates cloned state."""
    with _saved_context(generator, critic):
        if mode in {"variable_dwell", "rare_mode"}:
            samples = trainer.sample(num_particles, fixed_first_n=True, generator=eval_generator)
            scorer = score_multimodal_bars if mode == "variable_dwell" else score_rare_mode_bars
            return scorer(samples, stream.templates), samples
        if mode == "delayed_cue":
            predictions = []
            for cue in (0, 1):
                _history_context(memory, stream.clean_history(cue), generator, critic)
                predictions.append(trainer.sample(num_particles, fixed_first_n=True, generator=eval_generator))
            metrics = score_delayed_cue(*predictions, stream.outcomes)
            return metrics, torch.cat([predictions[0][:16], predictions[1][:16]])

        states, predictions = [], []
        for previous in (0, 1):
            paths = []
            history = torch.cat([stream.branch_paths[previous, -1:], stream.stem])
            for particle in range(num_particles):
                simulated = _history_context(memory, history, generator, critic)
                frames = []
                # Reuse both the MoG row and its Gaussian perturbation.
                latent_rng = eval_generator.get_state()
                for depth in range(stream.branch_paths.shape[1]):
                    eval_generator.set_state(latent_rng)
                    image = trainer.sample(1, fixed_first_n=True, offset=particle, generator=eval_generator)
                    frames.append(image[0])
                    if simulated is not None and depth + 1 < stream.branch_paths.shape[1]:
                        context = simulated.write(image)
                        generator.set_context(context)
                        critic.set_context(context)
                paths.append(torch.stack(frames))
            trajectories = torch.stack(paths)
            states.append(score_persistent_fork(trajectories, stream.branch_paths))
            predictions.append(trajectories[:, -1])
        return _worst_states(states), torch.cat([samples[:16] for samples in predictions])
