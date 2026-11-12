"""Five bounded capability probes executed by the public Forge workspace."""
from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import resource
import shutil
import time

import torch
from particlegan import GANTrainer
from particlegan_forge import MetricBound, TaskSpec, file_hash, has_nonfinite, render_training_gif

from .balanced_transport import ContextBalancedReconstruction
from .feature_prior import FeatureConditionalReconstruction
from .residual_prior import ResidualFeatureConditionalReconstruction
from .memory import make_stream_memory
from .models import StreamImageCritic, StreamImageGenerator
from .online import OnlineSession, validate_execution_policy
from .transport import TransportGANTrainer
from . import minimal_tasks as data
from . import minimal_velocity as velocity
from .minimal_online import PairedOnlineSession, isolated_session, prepare_prefix, prefix_context


def implementation(base):
    return {**base, **{name: file_hash(Path(__file__).with_name(name)) for name in (
        "minimal_adapter.py", "minimal_tasks.py", "minimal_velocity.py", "minimal_online.py")}}


def make_tasks(base):
    tasks = []
    specs = list(data.PROBE_SPECS.items()) + [("velocity", velocity.VELOCITY_SPEC)]
    for mode, spec in specs:
        bounds = velocity.VELOCITY_BOUNDS if mode == "velocity" else spec["bounds"]
        tasks.append(TaskSpec(
            id=spec["id"], description=spec["description"], adapter="minimal_probe",
            steps=spec["steps"], batch_size=1, num_particles=32, z_dim=8,
            prior_kind="mog", prior_sigma=.025,
            thresholds=[MetricBound(*bound) for bound in bounds],
            host_definition={
                "mode": mode, "width": 16,
                "context_dim": 0 if mode in {"dense4", "tokens4"} else 16,
                "qualification_tier": spec["tier"],
                "implementation": implementation(base),
                "protocol": "minimal_capability_proofs_v1",
                "data_manifest": (velocity.velocity_manifest() if mode == "velocity" else data.probe_manifest(mode)),
                "sampling": {
                    "law": "stratified_rows_frozen_mog_eps_v1" if mode in {"dense4", "tokens4"}
                           else "four_clean_centers_v1",
                    "rows": list(range(32)) if mode in {"dense4", "tokens4"} else [0, 8, 16, 24],
                    "eval_output_noise": "none", "weights": "live", "best_of_n": False,
                },
                "online_contract": {
                    "optimizer_batch": 1, "arrival_batch": 1, "replay": "none",
                    "gradient_accumulation_steps": 1,
                    "observation_unit": "source_target_pair" if mode in {"kernel2x2", "addition2bit"}
                                        else "target_frame",
                    "source_only_arrivals": spec["steps"] if mode in {"kernel2x2", "addition2bit", "velocity"} else 0,
                    "target_timing": "predict_before_target_arrival",
                },
            },
        ))
    return tasks


def validate_controls(tasks):
    results = {}
    for task in tasks:
        mode = task.host_definition["mode"]
        scores = velocity.velocity_controls() if mode == "velocity" else data.control_scores(mode)
        verdicts = {}
        for name, metrics in scores.items():
            passed = all(bound.check(metrics.get(bound.metric, float("nan"))) for bound in task.thresholds)
            expected = name == "oracle"
            if passed != expected:
                raise ValueError(f"{task.id}: {name} control {'passed' if passed else 'failed'} unexpectedly")
            verdicts[name] = {"verdict": "PASS" if passed else "FAIL", "metrics": metrics}
        if "oracle" not in verdicts or len(verdicts) < 2:
            raise ValueError(f"{task.id}: missing oracle/destructive controls")
        results[task.id] = verdicts
    return results


def _build(task, candidate, streams, steps, device):
    # Reuse the exact existing recipe builder and named initializer. New task
    # adapters do not rewrite old modules or their evidence fingerprints.
    from .tier1 import _build_recipe, _init_module
    with streams.fork("init", component="models", purpose="construction", device=device):
        g = StreamImageGenerator(z_dim=task.z_dim, width=16, context_dim=task.host_definition["context_dim"]).to(device)
        d = StreamImageCritic(width=16, context_dim=task.host_definition["context_dim"]).to(device)
        memory = make_stream_memory(candidate.memory_policy)
        if memory is not None:
            memory = memory.to(device)
        for module, component in ((g, "generator"), (d, "discriminator"), (memory, "memory")):
            if module is not None:
                _init_module(module, streams, component)
    recipe = _build_recipe(task, candidate, steps)
    prior = recipe.replace(prior_update="learned").make_prior(
        device=device, dtype=torch.float32, init_std=1., sigma=task.prior_sigma,
        generator=streams.generator("init", component="prior", purpose="locations", device=device))
    _init_module(prior, streams, "prior")
    recipe.apply_prior_policy(prior)
    transport = candidate.memory_policy.get("transport")
    options = {}
    trainer_type = GANTrainer
    if transport is not None:
        trainer_type = TransportGANTrainer
        options["transport"] = dict(transport)
        router = options["transport"].pop("router", "competitive")
        if router in {"feature_conditional", "residual_feature_conditional"}:
            options["router_class"] = (
                ResidualFeatureConditionalReconstruction
                if router == "residual_feature_conditional"
                else FeatureConditionalReconstruction
            )
            if "encoder_seed" in options["transport"]:
                raise ValueError("feature encoder initialization uses the protocol's named init stream")
            options["transport"]["encoder_seed"] = streams.seed_for("init", component="feature_encoder", purpose="projection")
        elif router == "context_balanced":
            options["router_class"] = ContextBalancedReconstruction
        elif router != "competitive":
            raise ValueError(f"unknown router: {router}")
        options["routing_generator"] = streams.generator("prior", component="routing", purpose="assignment", device=device)
    trainer = trainer_type(
        recipe, g, d, prior=prior, seed=streams.seed, max_steps=steps,
        latent_generator=streams.generator("prior", component="latent", purpose="indices", device=device),
        penalty_generator=streams.generator("noise", component="penalty", purpose="training", device=device),
        noise_generator=streams.generator("noise", component="generator", purpose="output", device=device),
        input_noise_generator=streams.generator("noise", component="critic", purpose="input", device=device),
        eval_generator=streams.generator("eval", component="sampler", purpose="samples", device=device),
        model_generator=streams.generator("noise", component="models", purpose="stochastic_layers", device=device),
        prior_noise_generator=streams.generator("noise", component="prior", purpose="gaussian", device=device),
        require_latent_damping=recipe.latent_damping_max_rate > 0, **options)
    return trainer, g, d, memory, recipe


@torch.no_grad()
def _probe(session, row, eps, eval_rng):
    """Public prior API supplies explicit MoG noise; every forward has batch 1."""
    session._set_context(session.pending_context)
    session.generator.eval()
    session.trainer.prior.eval()
    latent, _ = session.trainer.prior.sample(1, fixed_first_n=True, offset=row, eps=eps, generator=eval_rng)
    image = session.generator(latent)
    postprocess = getattr(session.trainer.prior, "postprocess_generated", None)
    return postprocess(image) if postprocess is not None else image


def _paired_predictions(mode, split, session, eval_rng, zero_eps, device):
    cases = data.paired_cases(mode, split=split, device=device)
    predictions = []
    for case in cases:
        with prefix_context(session, [case.source]):
            predictions.append(torch.cat([_probe(session, row, zero_eps, eval_rng)
                                          for row in (0, 8, 16, 24)], dim=0))
    return torch.stack(predictions)


def _evaluate(mode, session, eval_rng, frozen_eps, device, *, include_seen):
    zero = frozen_eps[:1].new_zeros(1, frozen_eps.shape[-1])
    if mode in {"dense4", "tokens4"}:
        samples = torch.cat([_probe(session, row, frozen_eps[row:row + 1], eval_rng)
                             for row in range(32)], dim=0)
        return data.score_distribution(mode, samples), samples
    if mode in {"kernel2x2", "addition2bit"}:
        held = _paired_predictions(mode, "heldout", session, eval_rng, zero, device)
        held_cases = data.paired_cases(mode, split="heldout", device=device)
        metrics = {f"heldout_{key}": value for key, value in data.score_paired(mode, held, held_cases).items()}
        if include_seen:
            seen = _paired_predictions(mode, "seen", session, eval_rng, zero, device)
            seen_cases = data.paired_cases(mode, split="seen", device=device)
            metrics.update({f"seen_{key}": value for key, value in data.score_paired(mode, seen, seen_cases).items()})
        return metrics, held.reshape(-1, 1, 8, 8)
    cases = velocity.velocity_cases(split="all", device=device)
    paths = []
    for case in cases:
        particles = []
        for row in (0, 8, 16, 24):
            with prefix_context(session, case["prefix"]) as simulated:
                frames = []
                for depth in range(3):
                    frame = _probe(session, row, zero, eval_rng)
                    frames.append(frame[0])
                    if depth < 2 and simulated is not None:
                        session.pending_context = simulated.write(frame).detach().clone()
                particles.append(torch.stack(frames))
        paths.append(torch.stack(particles))
    predictions = torch.stack(paths)
    return velocity.score_velocity(predictions, cases), predictions.reshape(-1, 1, 8, 8)[:32]


def run_minimal_task(*, task, candidate, streams, attempt_dir, workspace_root=None,
                     device="cpu", steps_override=None, log_event=None):
    validate_execution_policy(candidate)
    if task.batch_size != 1:
        raise ValueError("minimal probes require batch_size=1")
    controls = validate_controls([task])[task.id]
    mode = task.host_definition["mode"]
    steps = task.steps if steps_override is None else int(steps_override)
    trainer, g, d, memory, recipe = _build(task, candidate, streams, steps, device)
    eval_rng = streams.generator("eval", component="minimal", purpose="probes", device=device)
    frozen_eps = torch.randn(32, task.z_dim, generator=eval_rng, device=device)
    session_type = PairedOnlineSession if mode in {"kernel2x2", "addition2bit"} else OnlineSession
    session = session_type(trainer, g, d, memory,
        serving_generator=streams.generator("eval", component="serving", purpose="prediction", device=device))
    counters = {name: 0 for name in ("training_g_forwards", "training_d_forwards", "evaluation_g_forwards",
                                    "evaluation_d_forwards", "serving_g_forwards", "serving_d_forwards")}
    phase = ["training"]
    hooks = []
    for model, name in ((g, "g"), (d, "d")):
        def count(module, args, name=name):
            if len(args[0]) != 1:
                raise ValueError("all minimal-probe model forwards require batch_size=1")
            counters[f"{phase[0]}_{name}_forwards"] += 1
        hooks.append(model.register_forward_pre_hook(count))
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    eval_seconds = 0.
    observations, gif_frames = [], []
    checkpoints = set(task.observation_steps()) if steps_override is None else {
        max(1, round(i * steps / min(steps, task.observations))) for i in range(1, min(steps, task.observations) + 1)}
    terminal_start = sorted(checkpoints)[-min(task.minimum_stable_checks, len(checkpoints))]
    all_finite, deviations, warmups = True, 0, 0
    examples = velocity.velocity_cases(split="train", device=device) if mode == "velocity" else data.training_cases(mode, device=device)
    errors = []
    try:
        for step in range(1, steps + 1):
            if mode == "velocity":
                clip = examples[((step - 1) // 3) % len(examples)]
                depth = (step - 1) % 3
                if depth == 0:
                    prepare_prefix(session, clip["prefix"], reset=True)
                    warmups += 3
            else:
                case = examples[(step - 1) % len(examples)]
                if isinstance(session, PairedOnlineSession):
                    session.source(case.source)
            phase[0] = "serving"
            with streams.isolated_evaluation():
                prediction = session.predict()
            deviations += streams.last_eval_audit["unintended_rng_deviations"]
            target = clip["targets"][depth:depth + 1] if mode == "velocity" else case.target
            errors.append(float((prediction - target).square().mean().sqrt()))
            phase[0] = "training"
            stats = session.observe_target(target) if isinstance(session, PairedOnlineSession) else session.observe(target)
            all_finite &= not has_nonfinite(stats)
            all_finite &= all(bool(torch.isfinite(p).all()) for module in (g, d, trainer.prior) for p in module.parameters())
            if step not in checkpoints:
                continue
            phase[0] = "evaluation"
            t0 = time.perf_counter()
            with streams.isolated_evaluation(), isolated_session(session):
                metrics, samples = _evaluate(mode, session, eval_rng, frozen_eps, device, include_seen=step >= terminal_start)
            eval_seconds += time.perf_counter() - t0
            deviations += streams.last_eval_audit["unintended_rng_deviations"]
            metrics["live_prequential_rmse"] = sum(errors) / len(errors)
            errors.clear()
            passed = all(bound.check(metrics.get(bound.metric, float("nan"))) for bound in task.thresholds)
            observations.append({"step": step, "metrics": metrics})
            targets = (torch.cat([c["targets"] for c in velocity.velocity_cases(device=device)], dim=0)
                       if mode == "velocity" else data.target_bank(mode, device=device))
            gif_frames.append({"step": step, "passed": passed, "metrics": metrics,
                "current_stream_image": target.detach().cpu().numpy(),
                "target_images": targets.detach().cpu().numpy(), "sample_images": samples[:32].detach().cpu().numpy()})
            if log_event is not None:
                log_event("observation", step=step, passed=passed, metrics=metrics)
    finally:
        for hook in hooks:
            hook.remove()
    training_and_eval_seconds = time.perf_counter() - start
    stable = len(gif_frames) >= task.minimum_stable_checks and all(f["passed"] for f in gif_frames[-task.minimum_stable_checks:])
    verdict = "INVALID" if deviations else "INCOMPLETE" if steps != task.steps else "PASS" if stable and all_finite else "FAIL"
    gif = render_training_gif(gif_frames, attempt_dir / "goal.gif", task_id=task.id,
                             candidate_id=candidate.id, verdict=verdict, primary_metric=task.thresholds[0].metric)
    relative = f"gifs/{candidate.id}_{task.id}.gif"
    if workspace_root is not None and steps == task.steps:
        destination = Path(workspace_root) / "reports/forge" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(attempt_dir / "goal.gif", destination)
    execution = session.diagnostics()
    if mode == "velocity":
        execution.update(source_only_arrivals=warmups, target_arrivals=steps,
                         arrivals=steps + warmups, warmup_arrivals=warmups,
                         total_arrivals=steps + warmups, episode_resets=(steps + 2) // 3)
    cost = {**counters, "training_and_evaluation_seconds": training_and_eval_seconds,
            "evaluation_seconds": eval_seconds, "optimizer_updates": session.updates,
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else 0,
            "generator_parameters": sum(p.numel() for p in g.parameters()),
            "critic_parameters": sum(p.numel() for p in d.parameters())}
    return {"observations": observations, "completed_steps": steps, "all_finite": all_finite,
        "unintended_rng_deviations": deviations, "artifacts": {
            "goal_gif": relative, "gif_sha256": gif["sha256"],
            "implementation": task.host_definition["implementation"],
            "particlegan_version": version("particlegan"), "torch_version": torch.__version__,
            "resolved_recipe": asdict(recipe), "recipe_delta": {}, "memory_policy": candidate.memory_policy,
            "online_execution": execution, "compute": cost, "controls": controls,
            "sampling": task.host_definition["sampling"],
            "frozen_probe_eps": frozen_eps.detach().cpu().tolist() if mode in {"dense4", "tokens4"} else "zero",
            **({"transport": trainer.transport_diagnostics()} if hasattr(trainer, "transport_diagnostics") else {})}}
