"""Real Gymnasium-Robotics transitions and frozen physical-unit scorers.

The simulator supplies chronological observations, never training replays.
Held-out episodes use only the named eval stream and a separate simulator.
The learned output is displacement of the gripper/object/fingertips, in metres.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version

import numpy as np
import torch

ENVIRONMENTS = {
    "fetch_reach": "FetchReach-v4",
    "fetch_push": "FetchPush-v4",
    "fetch_slide": "FetchSlide-v4",
    "fetch_pick_and_place": "FetchPickAndPlace-v4",
    "hand_reach": "HandReach-v3",
}
VERSIONS = {"gymnasium": "1.4.0", "gymnasium-robotics": "1.4.2", "mujoco": "3.3.7"}
STEPS = 480
EPISODE_LENGTH = 40
DELTA_SCALE_METRES = .10
BOUNDS = (("position_delta_rmse_m", "<=", .005),
          ("relative_mse", "<=", .80),
          ("moving_direction_accuracy", ">=", .75),
          ("finite_fraction", "==", 1.))


def make_env(mode, *, render=False):
    import gymnasium as gym
    import gymnasium_robotics
    for package, expected in VERSIONS.items():
        if version(package) != expected:
            raise ValueError(f"robotics requires {package}=={expected}; install .[robotics]")
    gym.register_envs(gymnasium_robotics)
    return gym.make(ENVIRONMENTS[mode], **({"render_mode": "rgb_array", "width": 320, "height": 240} if render else {}))


def tracked_positions(mode, observation):
    if mode == "hand_reach":
        return np.asarray(observation["achieved_goal"], dtype=np.float64).copy()
    state = observation["observation"]
    return np.asarray(state[:3] if mode == "fetch_reach" else state[:6], dtype=np.float64).copy()


def _frame(values, *, repeat=False, device="cpu"):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not 0 < len(values) <= 64 or not np.isfinite(values).all():
        raise ValueError("robotics encoding requires 1..64 finite coordinates")
    flat = np.resize(values, 64) if repeat else np.full(64, .5, dtype=np.float32)
    if not repeat:
        flat[:len(values)] = values
    return torch.tensor(flat.copy(), device=device).reshape(1, 1, 8, 8).detach()


def encode_source(observation, action, device="cpu"):
    # Two source arrivals retain the complete 63-dimensional HandReach state
    # and its 20-dimensional action without truncating either to fit one frame.
    state = .5 + .5 * np.tanh(np.asarray(observation["observation"]))
    return (_frame(state, device=device), _frame(.5 + .5 * np.asarray(action), device=device))


def encode_delta(delta, device="cpu"):
    delta = np.asarray(delta)
    if not np.isfinite(delta).all():
        raise ValueError("physical displacement must be finite")
    # A shared invertible map covers fast sliding contacts without clipping or
    # task-specific limits. DELTA_SCALE_METRES sets curvature, not a hard cap.
    return _frame(.5 + np.arctan(delta / DELTA_SCALE_METRES) / np.pi, repeat=True, device=device)


def decode_delta(frames, dimensions):
    if frames.ndim != 4 or tuple(frames.shape[1:]) != (1, 8, 8):
        raise ValueError("expected (N,1,8,8) predictions")
    flat = frames.detach().flatten(1).double()
    code = torch.stack([flat[:, index::dimensions].mean(1) for index in range(dimensions)], 1)
    return torch.tan((code - .5) * torch.pi) * DELTA_SCALE_METRES


def policy_action(mode, observation, step, rng, action_dim):
    """Fixed excitation, with contact/lift phases owned by the task, not a learner.

    No reward, success flag or future state is used to choose an action.
    """
    noise = torch.rand(action_dim, generator=rng).numpy() * 2 - 1
    if mode in {"fetch_reach", "hand_reach"}:
        return noise.astype(np.float32)
    state = observation["observation"]
    grip, obj = state[:3], state[3:6]
    phase = step % EPISODE_LENGTH
    if mode == "fetch_pick_and_place":
        offset = np.array([0., 0., .025 if phase < 8 else 0. if phase < 20 else .15])
        action = np.r_[np.clip((obj + offset - grip) * 12, -1, 1), 1. if phase < 14 else -1.]
    else:
        direction = np.asarray(observation["desired_goal"]) - obj
        direction[2] = 0.
        direction /= max(np.linalg.norm(direction), 1e-6)
        target = obj - .06 * direction
        target[2] += .06 if phase < 8 else 0.
        if phase >= 20:
            target = obj + .12 * direction
        action = np.r_[np.clip((target - grip) * 12, -1, 1), 0.]
    return np.clip(action + .1 * noise, -1, 1).astype(np.float32)


def reset_env(env, rng):
    # These are named per-episode reset draws under protocol seed 0, not
    # alternate experiment seeds or repeated seed-only comparisons.
    seed = int(torch.randint(0, 2**31 - 1, (), generator=rng))
    observation, _ = env.reset(seed=seed)
    return observation, seed


@dataclass(frozen=True)
class Transition:
    source: tuple[torch.Tensor, torch.Tensor]
    delta: np.ndarray
    opposite_delta: np.ndarray
    episode: int
    step: int


def heldout_transitions(mode, rng):
    """Sixteen fixed probes from four separate eval-only episodes.

    Counterfactuals restore MuJoCo's complete integration state before applying
    the opposite action; they never use or alter the live training simulator.
    """
    import mujoco
    env = make_env(mode)
    cases, reset_seeds = [], []
    try:
        for episode in range(4):
            observation, seed = reset_env(env, rng)
            reset_seeds.append(seed)
            for step in range(EPISODE_LENGTH):
                action = policy_action(mode, observation, step, rng, env.action_space.shape[0])
                before = tracked_positions(mode, observation)
                if step in (7, 17, 27, 37):
                    sim = env.unwrapped
                    mask = mujoco.mjtState.mjSTATE_INTEGRATION
                    snapshot = np.empty(mujoco.mj_stateSize(sim.model, mask))
                    mujoco.mj_getState(sim.model, sim.data, snapshot, mask)
                    alternative, *_ = sim.step(-action)
                    opposite_delta = tracked_positions(mode, alternative) - before
                    mujoco.mj_setState(sim.model, sim.data, snapshot, mask)
                    mujoco.mj_forward(sim.model, sim.data)
                following, _, terminated, truncated, _ = env.step(action)
                if step in (7, 17, 27, 37):
                    delta = tracked_positions(mode, following) - before
                    encode_delta(delta)  # Reject nonfinite simulator data before training.
                    cases.append(Transition(encode_source(observation, action), delta, opposite_delta, episode, step))
                observation = following
                if terminated or truncated:
                    raise ValueError("unexpected termination inside the declared 40-step robotics episode")
        return tuple(cases), reset_seeds
    finally:
        env.close()


def score_predictions(predictions, cases):
    """Every particle is scored; no nearest-case, target snapping or best-of-N."""
    dimensions = len(cases[0].delta)
    if predictions.ndim != 5 or predictions.shape[0] != len(cases):
        raise ValueError("expected (cases,particles,1,8,8) predictions")
    finite = torch.isfinite(predictions).flatten(2).all(2)
    in_range = ((predictions > 0) & (predictions < 1)).flatten(2).all(2)
    predicted = decode_delta(torch.nan_to_num(predictions.reshape(-1, 1, 8, 8), nan=1e6, posinf=1e6, neginf=-1e6), dimensions).reshape(len(cases), -1, dimensions).double()
    target = torch.tensor(np.stack([case.delta for case in cases]), device=predicted.device)[:, None]
    errors = (predicted - target).square()
    mse = errors.mean()
    baseline_mse = target.square().mean()
    moving = (target.abs() >= .001).expand_as(predicted)
    correct = (predicted * target > 0) & finite[..., None] & in_range[..., None]
    metrics = {
        "position_delta_rmse_m": float(mse.sqrt()),
        "relative_mse": float(mse / baseline_mse.clamp_min(1e-16)),
        "persistence_rmse_m": float(baseline_mse.sqrt()),
        "moving_direction_accuracy": float(correct[moving].double().mean()) if bool(moving.any()) else 0.,
        "moving_coordinate_fraction": float(moving.double().mean()),
        "finite_fraction": float((finite & in_range).double().mean()),
    }
    if dimensions == 6:
        metrics.update(gripper_delta_rmse_m=float(errors[..., :3].mean().sqrt()),
                       object_delta_rmse_m=float(errors[..., 3:].mean().sqrt()),
                       object_motion_rms_m=float(target[..., 3:].square().mean().sqrt()))
    return metrics


def control_scores(cases):
    targets = torch.cat([encode_delta(case.delta) for case in cases])[:, None]
    opposite = torch.cat([encode_delta(case.opposite_delta) for case in cases])[:, None]
    outputs = {"oracle": targets,
               "persistence": torch.full_like(targets, .5),
               "reversed_motion": 1 - targets,
               "opposite_action_oracle": opposite,
               "shuffled_targets": targets.roll(1, 0)}
    return {name: score_predictions(output, cases) for name, output in outputs.items()}


def validate_controls(task, cases):
    results = {}
    for name, metrics in control_scores(cases).items():
        passed = all(bound.check(metrics[bound.metric]) for bound in task.thresholds)
        if passed != (name == "oracle"):
            raise ValueError(f"{task.id}: {name} control unexpectedly {'passed' if passed else 'failed'}")
        results[name] = {"verdict": "PASS" if passed else "FAIL", "metrics": metrics}
    return results
