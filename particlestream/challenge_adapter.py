"""Task declarations and isolated evaluation for the online Tier 1 challenges."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import torch
from particlegan_forge import MetricBound, TaskSpec

from .challenges import (
    CircleOrbitStream,
    DelayedCueStream,
    PersistentForkStream,
    RareModeBarsStream,
    SquareCircleCycleStream,
    VariableDwellBarsStream,
    score_circle_orbit,
    score_delayed_cue,
    score_persistent_fork,
    score_rare_mode_bars,
    score_square_circle_cycle,
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
    _task(
        "seq_img_circle_orbit", "Trace an 8-waypoint circular orbit in real time by predicting subsequent (x, y) point frames across a closed-loop 8-step rollout without teacher forcing.",
        "circle_orbit", [("orbit_rmse", "<=", .12), ("xy_rmse", "<=", .15),
                         ("step_accuracy", ">=", .90), ("orbit_completion", ">=", .90),
                         ("waypoint_modes", "==", 8.)],
        predictive=True, rollout_horizon=8, rollout_latent_policy="fixed_particle_and_noise_v1",
        orbit_waypoints=8,
    ),
    _task(
        "seq_img_square_circle_cycle", "Translate bidirectionally between an 8-waypoint square perimeter and an 8-waypoint circular orbit in closed loop without teacher forcing (CycleGAN S<->C).",
        "square_circle_cycle", [("cycle_rmse", "<=", .12), ("s2c_rmse", "<=", .12),
                                ("c2s_rmse", "<=", .12), ("xy_rmse", "<=", .15),
                                ("step_accuracy", ">=", .90), ("cycle_completion", ">=", .90),
                                ("waypoint_modes", "==", 8.)],
        predictive=True, rollout_horizon=8, rollout_latent_policy="fixed_particle_and_noise_v1",
        orbit_waypoints=8, cycle_domains=["square", "circle"],
    ),
]

CHALLENGE_MODES = {task.host_definition["mode"] for task in CHALLENGE_TASKS}
PREDICTIVE_CHALLENGE_MODES = {"delayed_cue", "persistent_fork", "circle_orbit", "square_circle_cycle"}


def make_challenge_stream(mode, device):
    return {
        "variable_dwell": VariableDwellBarsStream,
        "rare_mode": RareModeBarsStream,
        "delayed_cue": DelayedCueStream,
        "persistent_fork": PersistentForkStream,
        "circle_orbit": CircleOrbitStream,
        "square_circle_cycle": SquareCircleCycleStream,
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
        if mode == "circle_orbit":
            paths = []
            warmup = stream.clean_warmup(0)
            for particle in range(num_particles):
                simulated = _history_context(memory, warmup, generator, critic)
                frames = []
                latent_rng = eval_generator.get_state()
                for step_idx in range(stream.period):
                    eval_generator.set_state(latent_rng)
                    image = trainer.sample(1, fixed_first_n=True, offset=particle, generator=eval_generator)
                    frames.append(image[0])
                    if simulated is not None and step_idx + 1 < stream.period:
                        context = simulated.write(image)
                        generator.set_context(context)
                        critic.set_context(context)
                paths.append(torch.stack(frames))
            trajectories = torch.stack(paths)
            return score_circle_orbit(trajectories, stream.templates), trajectories.reshape(-1, 1, 8, 8)[:num_particles]
        if mode == "square_circle_cycle":
            partner = getattr(trainer, "cycle_partner", None) or {}
            c2s_trainer = partner.get("trainer", trainer)
            c2s_generator = partner.get("generator", generator)
            c2s_critic = partner.get("critic", critic)
            c2s_memory = partner.get("memory", memory)
            s2c_paths, c2s_paths = [], []
            sq_warmup = stream.clean_square_warmup(0)
            ci_warmup = stream.clean_circle_warmup(0)
            with _saved_context(c2s_generator, c2s_critic):
                for particle in range(num_particles):
                    sim_s = _history_context(memory, sq_warmup, generator, critic)
                    sim_c = _history_context(c2s_memory, ci_warmup, c2s_generator, c2s_critic)
                    s2c_frames, c2s_frames = [], []
                    latent_rng = eval_generator.get_state()
                    for step_idx in range(stream.period):
                        eval_generator.set_state(latent_rng)
                        c_img = trainer.sample(1, fixed_first_n=True, offset=particle, generator=eval_generator)
                        eval_generator.set_state(latent_rng)
                        s_img = c2s_trainer.sample(1, fixed_first_n=True, offset=particle, generator=eval_generator)
                        s2c_frames.append(c_img[0])
                        c2s_frames.append(s_img[0])
                        if step_idx + 1 < stream.period:
                            # Cross-domain CycleGAN closed-loop feedback:
                            # C->S translated Square frame drives S->C; S->C translated Circle frame drives C->S.
                            if sim_s is not None:
                                ctx_s = sim_s.write(s_img)
                                generator.set_context(ctx_s)
                                critic.set_context(ctx_s)
                            if sim_c is not None:
                                ctx_c = sim_c.write(c_img)
                                c2s_generator.set_context(ctx_c)
                                c2s_critic.set_context(ctx_c)
                    s2c_paths.append(torch.stack(s2c_frames))
                    c2s_paths.append(torch.stack(c2s_frames))
            s2c_trajectories = torch.stack(s2c_paths)
            c2s_trajectories = torch.stack(c2s_paths)
            metrics = score_square_circle_cycle(
                s2c_trajectories, c2s_trajectories, stream.square_templates, stream.circle_templates
            )
            half = max(1, num_particles // 2)
            samples = torch.cat(
                [
                    s2c_trajectories.reshape(-1, 1, 8, 8)[:half],
                    c2s_trajectories.reshape(-1, 1, 8, 8)[: num_particles - half],
                ],
                dim=0,
            )
            return metrics, samples

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
