"""Five sequential simulator experiments using the unchanged global recipe."""
from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import resource
import shutil
import time

import torch
from particlegan_forge import MetricBound, TaskSpec, file_hash, has_nonfinite, render_training_gif

from . import robotics_tasks as data
from .minimal_adapter import _build, _probe
from .minimal_online import PairedOnlineSession, isolated_session, prepare_prefix, prefix_context
from .online import validate_execution_policy


def implementation(base):
    from .minimal_adapter import implementation as minimal_implementation
    return {**minimal_implementation(base), **{name: file_hash(Path(__file__).with_name(name))
            for name in ("robotics_adapter.py", "robotics_tasks.py")}}


def make_tasks(base):
    return [TaskSpec(
        id=f"seq_robotics_{mode}",
        description=f"Predict action-conditioned physical motion in {env_id} on held-out simulator episodes.",
        adapter="gym_robotics", steps=data.STEPS, batch_size=1, num_particles=32,
        z_dim=8, prior_kind="mog", prior_sigma=.025,
        thresholds=[MetricBound(*bound) for bound in data.BOUNDS],
        host_definition={
            "mode": mode, "environment_id": env_id, "width": 16, "context_dim": 16,
            "qualification_tier": 2, "implementation": implementation(base),
            "protocol": "gym_robotics_action_dynamics_v2", "dependencies": data.VERSIONS,
            "episode_length": data.EPISODE_LENGTH, "training_episodes": 12,
            "target": "gripper_xyz" if mode == "fetch_reach" else "fingertips_xyz" if mode == "hand_reach" else "gripper_xyz_and_object_xyz",
            "encoding": {"source": "complete_tanh_state_then_affine_action_two_frames",
                         "target": "repeated_displacement_coordinates_invertible_atan_no_clipping",
                         "delta_scale_metres": data.DELTA_SCALE_METRES},
            "policy": "fixed_task_excitation_v1; no reward or success feedback",
            "evaluation": {"episodes": 4, "steps_per_episode": [7, 17, 27, 37],
                           "split": "separate_named_eval_resets; never trained",
                           "metric_units": "metres; all particles scored",
                           "moving_coordinate_threshold_m": .001,
                           "frozen_episodes": True, "horizon": 1},
            "sampling": {"law": "four_clean_centers_v1", "rows": [0, 8, 16, 24],
                         "eval_output_noise": "none", "weights": "live", "best_of_n": False},
            "online_contract": {"optimizer_batch": 1, "arrival_batch": 1,
                "replay": "none", "gradient_accumulation_steps": 1,
                "source_only_arrivals": 2 * data.STEPS,
                "target_timing": "predict_before_simulator_step", "target_memory_writes": 0},
        }) for mode, env_id in data.ENVIRONMENTS.items()]


class RoboticsSession(PairedOnlineSession):
    def source_pair(self, state, action):
        super().source(state)
        prepare_prefix(self, [action], reset=False)
        self.source_only_arrivals += 1


def validate_controls(tasks):
    from particlegan_forge import NamedStreams
    results = {}
    for task in tasks:
        streams = NamedStreams(seed=0)
        rng = streams.generator("eval", component="robotics", purpose="heldout_episodes", device="cpu")
        with streams.isolated_evaluation():
            cases, _ = data.heldout_transitions(task.host_definition["mode"], rng)
            results[task.id] = data.validate_controls(task, cases)
    return results


def _evaluate(session, cases, eval_rng, device):
    predictions = []
    zero = torch.zeros(1, session.trainer.prior.z_dim, device=device)
    for case in cases:
        with prefix_context(session, [frame.to(device) for frame in case.source]):
            predictions.append(torch.cat([_probe(session, row, zero, eval_rng)
                                          for row in (0, 8, 16, 24)]))
    predictions = torch.stack(predictions)
    return data.score_predictions(predictions, cases), predictions


def run_robotics_task(*, task, candidate, streams, attempt_dir, workspace_root=None,
                      device="cpu", steps_override=None, log_event=None):
    validate_execution_policy(candidate)
    if task.batch_size != 1:
        raise ValueError("robotics requires sequential batch_size=1")
    mode = task.host_definition["mode"]
    steps = task.steps if steps_override is None else int(steps_override)
    started = time.perf_counter()
    eval_rng = streams.generator("eval", component="robotics", purpose="heldout_episodes", device="cpu")
    with streams.isolated_evaluation():
        cases, eval_seeds = data.heldout_transitions(mode, eval_rng)
        controls = data.validate_controls(task, cases)
    deviations = streams.last_eval_audit["unintended_rng_deviations"]
    trainer, g, d, memory, recipe = _build(task, candidate, streams, steps, device)
    session = RoboticsSession(trainer, g, d, memory,
        serving_generator=streams.generator("eval", component="serving", purpose="prediction", device=device))
    probe_rng = streams.generator("eval", component="robotics", purpose="probes", device=device)
    data_rng = streams.generator("data", component="robotics", purpose="training_episodes", device="cpu")
    phase = ["training"]
    counters = {f"{phase}_{model}_forwards": 0 for phase in ("training", "serving", "evaluation") for model in ("g", "d")}
    hooks = []
    for model, name in ((g, "g"), (d, "d")):
        def count(module, args, name=name):
            if len(args[0]) != 1:
                raise ValueError("robotics model forwards require batch_size=1")
            counters[f"{phase[0]}_{name}_forwards"] += 1
        hooks.append(model.register_forward_pre_hook(count))
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    checkpoints = set(task.observation_steps()) if steps_override is None else {
        max(1, round(i * steps / min(steps, task.observations))) for i in range(1, min(steps, task.observations) + 1)}
    observations, frames, train_seeds, errors, trace = [], [], [], [], []
    all_finite = True
    evaluation_seconds = 0.
    env = data.make_env(mode)
    try:
        for step in range(1, steps + 1):
            episode_step = (step - 1) % data.EPISODE_LENGTH
            if episode_step == 0:
                current, seed = data.reset_env(env, data_rng)
                train_seeds.append(seed)
            action = data.policy_action(mode, current, episode_step, data_rng, env.action_space.shape[0])
            session.source_pair(*data.encode_source(current, action, device))
            phase[0] = "serving"
            # The simulator does not advance until serving has completed.
            with streams.isolated_evaluation():
                prediction = session.predict()
            deviations += streams.last_eval_audit["unintended_rng_deviations"]
            following, reward, terminated, truncated, info = env.step(action)
            delta = data.tracked_positions(mode, following) - data.tracked_positions(mode, current)
            target = data.encode_delta(delta, device)
            errors.append(float((data.decode_delta(prediction, len(delta)).double() - torch.tensor(delta, device=device)).square().mean().sqrt()))
            phase[0] = "training"
            stats = session.observe_target(target)
            all_finite &= not has_nonfinite(stats)
            all_finite &= all(bool(torch.isfinite(p).all()) and (p.grad is None or bool(torch.isfinite(p.grad).all()))
                              for model in (g, d, trainer.prior) for p in model.parameters())
            trace.append({"step": step, "episode": (step - 1) // data.EPISODE_LENGTH,
                          "action": action.tolist(), "delta_m": delta.tolist(),
                          "reward": float(reward), "is_success": bool(info.get("is_success", False))})
            current = following
            if (terminated or truncated) and episode_step != data.EPISODE_LENGTH - 1:
                raise ValueError("unexpected simulator termination")
            if step not in checkpoints:
                continue
            phase[0] = "evaluation"
            t0 = time.perf_counter()
            with streams.isolated_evaluation(), isolated_session(session):
                metrics, samples = _evaluate(session, cases, probe_rng, device)
            evaluation_seconds += time.perf_counter() - t0
            deviations += streams.last_eval_audit["unintended_rng_deviations"]
            metrics["live_prequential_rmse_m"] = sum(errors) / len(errors)
            errors.clear()
            passed = all(bound.check(metrics[bound.metric]) for bound in task.thresholds)
            observations.append({"step": step, "metrics": metrics})
            frames.append({"step": step, "passed": passed, "metrics": metrics,
                           "current_stream_image": target.detach().cpu().numpy(),
                           "target_images": torch.cat([data.encode_delta(case.delta) for case in cases]).numpy(),
                           "sample_images": samples[:, 0].detach().cpu().numpy()})
            if log_event:
                log_event("observation", step=step, passed=passed, metrics=metrics)
    finally:
        env.close()
        for hook in hooks:
            hook.remove()
    elapsed = time.perf_counter() - started
    stable = len(frames) >= task.minimum_stable_checks and all(frame["passed"] for frame in frames[-task.minimum_stable_checks:])
    verdict = "INVALID" if deviations else "INCOMPLETE" if steps != task.steps else "PASS" if stable and all_finite else "FAIL"
    gif = render_training_gif(frames, attempt_dir / "goal.gif", task_id=task.id,
        candidate_id=candidate.id, verdict=verdict, primary_metric="position_delta_rmse_m")
    relative = f"gifs/{candidate.id}_{task.id}.gif"
    if workspace_root is not None and steps == task.steps:
        destination = Path(workspace_root) / "reports/forge" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(attempt_dir / "goal.gif", destination)
    heldout = [{"episode": case.episode, "step": case.step,
                "source": [frame.flatten().tolist() for frame in case.source],
                "delta_m": case.delta.tolist(), "opposite_delta_m": case.opposite_delta.tolist()} for case in cases]
    manifest = {"training_reset_seeds": train_seeds, "eval_reset_seeds": eval_seeds,
                "heldout": heldout, "training_trace": trace}
    payload = json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n"
    (attempt_dir / "simulator_transitions.json").write_text(payload)
    # Preserve a compact, source-bound simulator trace alongside the receipt.
    trace_relative = f"traces/{candidate.id}_{task.id}.json"
    if workspace_root is not None and steps == task.steps:
        destination = Path(workspace_root) / "reports/forge" / trace_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload)
    return {"observations": observations, "completed_steps": steps, "all_finite": all_finite,
        "unintended_rng_deviations": deviations, "artifacts": {
            "goal_gif": relative, "gif_sha256": gif["sha256"],
            "implementation": task.host_definition["implementation"],
            "particlegan_version": version("particlegan"), "torch_version": torch.__version__,
            "simulator_versions": {name: version(name) for name in data.VERSIONS},
            "resolved_recipe": asdict(recipe), "recipe_delta": {}, "memory_policy": candidate.memory_policy,
            "online_execution": {**session.diagnostics(), "episode_resets": len(train_seeds),
                                 "observation_contract": "state_then_action_then_predict_then_simulator_step_then_target_v1"},
            "sampling": task.host_definition["sampling"], "controls": controls,
            "simulator_trace": trace_relative, "simulator_trace_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "evaluation_cases": len(cases), "training_reset_seeds": train_seeds, "eval_reset_seeds": eval_seeds,
            "compute": {**counters, "optimizer_updates": session.updates,
                "training_and_evaluation_seconds": elapsed, "evaluation_seconds": evaluation_seconds,
                "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else 0,
                "generator_parameters": sum(p.numel() for p in g.parameters()),
                "critic_parameters": sum(p.numel() for p in d.parameters())},
            **({"transport": trainer.transport_diagnostics()} if hasattr(trainer, "transport_diagnostics") else {})}}
