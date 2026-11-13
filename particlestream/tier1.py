"""Tier 1 declarations, adapter, and bootstrap CLI for ParticleStream."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path

import torch
from particlegan import GANTrainer, Recipe, get_recipe, init
from particlegan_forge import (
    CandidateSpec,
    ForgeRegistry,
    MetricBound,
    NamedStreams,
    TaskSpec,
    TierSpec,
    TierView,
    atomic_text,
    file_hash,
    has_nonfinite,
    render_training_gif,
)

from .models import FastWeightMemory, StreamImageCritic, StreamImageGenerator
from .memory import make_stream_memory
from .transport import TransportGANTrainer
from .balanced_transport import ContextBalancedReconstruction
from .feature_prior import FeatureConditionalReconstruction
from .gail_prior import GAILConditionalReconstruction
from .residual_prior import ResidualFeatureConditionalReconstruction
from .subspace_prior import SubspaceResidualConditionalReconstruction
from .forge import StreamWorkspace
from .challenge_adapter import (
    CHALLENGE_TASKS, CHALLENGE_MODES, PREDICTIVE_CHALLENGE_MODES,
    make_challenge_stream, challenge_target_bank, evaluate_challenge, evaluate_original_fork,
)
from .challenges import render_circle_orbit_gif, render_square_circle_cycle_gif
from .online import OnlineSession, validate_execution_policy
from .streams import (
    ForkStream,
    SequentialBarsStream,
    TranslatingBarStream,
    score_fork_rollout,
    score_multimodal_bars,
    score_translating_next,
)

REGISTRY = ForgeRegistry()

TIER1_TASKS: list[TaskSpec] = [
    TaskSpec(
        id="seq_img_round_robin_bars4",
        description="Acquire and hold all 4 8x8 bar modes from a deterministic round-robin stream (batch_size=1, dwell=1).",
        adapter="seq_image",
        steps=480,
        batch_size=1,
        num_particles=32,
        z_dim=8,
        prior_kind="mog",
        prior_sigma=0.025,
        thresholds=[
            MetricBound("hq", ">=", 0.90),
            MetricBound("modes", "==", 4.0),
            MetricBound("finite_template_tv", "<=", 0.15),
        ],
        host_definition={"mode": "bars4", "dwell": 1, "width": 16},
    ),
    TaskSpec(
        id="seq_img_dwell_bars4",
        description="Retain all 4 8x8 bar modes across correlated blocks of dwell=16 identical frames (batch_size=1).",
        adapter="seq_image",
        steps=480,
        batch_size=1,
        num_particles=32,
        z_dim=8,
        prior_kind="mog",
        prior_sigma=0.025,
        thresholds=[
            MetricBound("hq", ">=", 0.85),
            MetricBound("modes", "==", 4.0),
            MetricBound("finite_template_tv", "<=", 0.20),
        ],
        host_definition={"mode": "bars4", "dwell": 16, "width": 16},
    ),
    TaskSpec(
        id="seq_img_translating_bar_next",
        description="Predict the next 8x8 frame of a bouncing bar from ambiguous interior frames using memory (batch_size=1).",
        adapter="seq_image",
        steps=480,
        batch_size=1,
        num_particles=32,
        z_dim=8,
        prior_kind="mog",
        prior_sigma=0.025,
        thresholds=[
            MetricBound("next_frame_rmse", "<=", 0.12),
            MetricBound("direction_acc", ">=", 0.90),
        ],
        host_definition={"mode": "translating_next", "width": 16, "context_dim": 16},
    ),
    TaskSpec(
        id="seq_img_fork_rollout",
        description="Cover both Left and Right future branches from the same stem memory state across particles (batch_size=1).",
        adapter="seq_image",
        steps=480,
        batch_size=1,
        num_particles=32,
        z_dim=8,
        prior_kind="mog",
        prior_sigma=0.025,
        thresholds=[
            MetricBound("hq", ">=", 0.85),
            MetricBound("within_state_modes", "==", 2.0),
            MetricBound("branch_tv", "<=", 0.20),
        ],
        host_definition={"mode": "fork_rollout", "width": 16, "context_dim": 16},
    ),
]

BASE_TASK_IDS = [task.id for task in TIER1_TASKS]
TIER1_TASKS.extend(CHALLENGE_TASKS)
PREDICTIVE_MODES = {"translating_next", "fork_rollout"} | PREDICTIVE_CHALLENGE_MODES

# Bind dependency code as well as the adapter body into each task fingerprint.
IMPLEMENTATION = {
    name: file_hash(Path(__file__).with_name(name))
    for name in (
        "models.py", "streams.py", "online.py", "challenges.py", "challenge_adapter.py",
        "memory.py", "transport.py", "balanced_transport.py", "feature_prior.py",
        "residual_prior.py", "gail_prior.py", "subspace_prior.py",
    )
}

PRIMARY_VIEW = TierView.from_tiers(
    id="single_stream_stability",
    goal="single_stream_stability",
    revision=7,
    policy_change_reason="Add seq_img_square_circle_cycle as the tenth required Tier 1 task in single_stream_stability so one global strict online feature-prior recipe qualifies across all ten sequential tasks, including bidirectional Square <-> Circle CycleGAN orbit translation.",
    tiers=[
        TierSpec(
            tier=1,
            required=[task.id for task in TIER1_TASKS],
        )
    ],
)

CANDIDATES: list[CandidateSpec] = [
    CandidateSpec(
        id="bcap",
        technique="BCAP",
        changed_factors=["Stateless BCAP baseline on batch_size=1 sequential stream"],
        recipe_preset="bcap_adam",
        recipe_overrides={},
        memory_policy={"kind": "none"},
    ),
    CandidateSpec(
        id="ka2",
        technique="KA2",
        changed_factors=["KA2 critic EMA anchor on batch_size=1 sequential stream"],
        recipe_preset="ka2",
        recipe_overrides={},
        memory_policy={"kind": "none"},
    ),
    CandidateSpec(
        id="fast-weight-bcap",
        technique="FastWeight-BCAP",
        changed_factors=[
            "Detached FastWeightMemory writer/reader context + bounded 64-frame reservoir replay on batch_size=1 stream"
        ],
        parent="bcap",
        recipe_preset="bcap_adam",
        recipe_overrides={},
        memory_policy={"kind": "fast_weight", "dim": 16, "decay": 0.90, "write_lr": 0.35, "reservoir_size": 64},
    ),
    CandidateSpec(
        id="fast-weight-bcap-ns",
        technique="Global-Stream-BCAP",
        changed_factors=[
            "Non-saturating GAN loss for batch_size=1 stability",
            "Cosine learning rate annealing (lr=0.005, betas=(0.0, 0.999), lr_schedule=cosine, lr_anneal_start=0.5, lr_floor=0.05)",
            "64-frame rehearsal reservoir spanning full dwell=16 4-mode cycle",
            "Temporal velocity direction memory with static noise gating (||diff|| > 0.35)",
            "Joint non-linear conditional critic architecture",
            "Dwell-aware rehearsal replay and task-isolated context conditioning",
        ],
        parent="fast-weight-bcap",
        recipe_preset="bcap_adam",
        recipe_overrides={
            "loss": "non_saturating",
            "lr": 0.005,
            "betas": [0.0, 0.999],
            "lr_schedule": "cosine",
            "lr_anneal_start": 0.5,
            "lr_floor": 0.05,
            "network_lr_floor": 0.05,
            "d_lr_mult": 1.0,
            "prior_lr_mult": 2.0,
            "prior_regularizer": "none",
            "reg_arm": "b_cap",
            "reg_coeff": 1.0,
            "reg_kappa": 1.0,
        },
        memory_policy={
            "kind": "fast_weight",
            "dim": 16,
            "decay": 0.90,
            "write_lr": 0.35,
            "reservoir_size": 64,
            "rehearsal_batch": 7,
        },
    ),
]

# Each candidate uses this bounded memory policy across the entire task suite.
# A family is identified by the resolved critic formulation, not its label.
STREAM_MEMORY = {
    "kind": "fast_weight", "dim": 16, "decay": 0.90, "write_lr": 0.35,
    "reservoir_size": 64, "rehearsal_batch": 7,
}
STREAM_SCHEDULE = {
    "loss": "non_saturating", "lr_schedule": "cosine",
    "lr_anneal_start": 0.5, "lr_floor": 0.05, "network_lr_floor": 0.05,
}
FAMILY_PRESETS = {
    "stream-bcap-adam": ("BCAP", "bcap_adam"),
    "stream-bcap": ("BCAP", "bcap"),
    "stream-ka2": ("KA2", "ka2"),
    "stream-bispf": ("BISPF", "bispf"),
    "stream-bispf-optimized": ("BISPF", "bispf_optimized"),
}
for candidate_id, (family, preset) in FAMILY_PRESETS.items():
    overrides = dict(STREAM_SCHEDULE)
    if preset in {"bcap_adam", "ka2", "bispf"}:
        overrides.update(lr=0.005, betas=[0.0, 0.999])
    if family == "BISPF":
        # Earlier discrete-mode experiments found repulsive prior penalties
        # obstruct clustering. Declare this ablation instead of hiding it.
        overrides.update(prior_regularizer="none", prior_reg=0.0,
                         training_precision="float32")
    CANDIDATES.append(CandidateSpec(
        id=candidate_id, technique=family, recipe_preset=preset,
        recipe_overrides=overrides, memory_policy=dict(STREAM_MEMORY),
        changed_factors=[
            f"ParticleGAN 0.9.6 public {preset} trainer preset",
            "Shared 64-frame FIFO memory, seven replay samples and velocity context",
            "Global non-saturating loss and cosine schedule (start=0.5, floor=0.05)",
            *(["SBNP disabled for discrete target modes; FP32 training"] if family == "BISPF" else []),
        ],
    ))

# The historical representatives retain their original identities and evidence.
# New identities declare the replay delta; all optimization overrides are copied.
ONLINE_PARENTS = ("stream-bcap-adam", "stream-bispf-early", "stream-ka2-ready")
for parent_id in ONLINE_PARENTS:
    declaration = Path(__file__).resolve().parents[1] / "configs/forge/ideas" / f"{parent_id}.json"
    parent = CandidateSpec.from_dict(json.loads(declaration.read_text()))
    policy = dict(parent.memory_policy)
    policy.update(replay_policy="none", rehearsal_batch=0)
    CANDIDATES.append(CandidateSpec(
        id=parent_id.replace("stream-", "online-", 1), technique=parent.technique,
        parent=parent.id, recipe_preset=parent.recipe_preset,
        recipe_overrides=dict(parent.recipe_overrides), memory_policy=policy,
        changed_factors=[
            *parent.changed_factors,
            "Explicit replay_policy=none: train every arriving observation once, batch_size=1",
            "Predict before arrival; learn next-frame targets from the preceding detached context",
        ],
    ))

TIER1_CANDIDATES = {c.id: c for c in CANDIDATES}


def _build_recipe(task: TaskSpec, candidate: CandidateSpec, steps: int) -> Recipe:
    task_owned = {"total_steps", "batch_size", "num_particles", "z_dim", "prior_kind",
                  "sigma_rel", "standardize"}
    if task_owned.intersection(candidate.recipe_overrides):
        raise ValueError("candidate overrides task-owned resource or initial-prior fields")
    prior_kind = "mog" if task.prior_kind == "mog" else "particles"
    options = {
        "total_steps": steps,
        "batch_size": task.batch_size,
        "num_particles": task.num_particles,
        "z_dim": task.z_dim,
        "prior_kind": prior_kind,
        "sigma_rel": 0.0,
        "standardize": task.prior_standardize,
        **candidate.recipe_overrides,
    }
    preset = candidate.recipe_preset
    return get_recipe(preset, **options)


def _init_module(module: torch.nn.Module, streams: NamedStreams, component: str) -> None:
    seeds = {
        name: streams.seed_for("init", component=component, purpose=name)
        for name, p in module.named_parameters()
        if p.requires_grad and p.numel()
    }
    if seeds:
        init.deterministic_orthogonal_(module, parameter_seeds=seeds)


for helper in (_build_recipe, _init_module):
    IMPLEMENTATION[helper.__name__] = hashlib.sha256(inspect.getsource(helper).encode()).hexdigest()
for task in TIER1_TASKS:
    task.host_definition["implementation"] = dict(IMPLEMENTATION)
    task.host_definition["online_contract"] = {
        "version": 1, "arrival_batch": 1, "optimizer_batch": 1,
        "replay": "none", "gradient_accumulation_steps": 1,
        "warmup_arrivals": int(task.host_definition["mode"] in PREDICTIVE_MODES),
        "target_timing": "train_on_arrival_with_preceding_context",
    }
    if task.host_definition["mode"] == "fork_rollout":
        task.host_definition.update(branch_law="independent_fair_on_arrival",
                                    evaluation_histories="previous_branch_then_stem")


@REGISTRY.adapter("seq_image")
def run_seq_image_task(
    *,
    task: TaskSpec,
    candidate: CandidateSpec,
    streams: NamedStreams,
    attempt_dir: Path,
    workspace_root: Path | None = None,
    device: str = "cpu",
    steps_override: int | None = None,
    log_event=None,
) -> dict:
    validate_execution_policy(candidate)
    if task.batch_size != 1:
        raise ValueError("sequential task execution requires batch_size=1")
    steps = int(steps_override or task.steps)
    host = task.host_definition
    mode = host["mode"]
    width = int(host.get("width", 16))
    task_context_dim = int(host.get("context_dim", 0))
    model_context_dim = task_context_dim

    with streams.fork("init", component="models", purpose="construction", device=device):
        g = StreamImageGenerator(z_dim=task.z_dim, width=width, context_dim=model_context_dim).to(device)
        d = StreamImageCritic(width=width, context_dim=model_context_dim).to(device)
        memory = make_stream_memory(candidate.memory_policy)
        if memory is not None:
            memory = memory.to(device)
        _init_module(g, streams, "generator")
        _init_module(d, streams, "discriminator")
        if memory is not None:
            _init_module(memory, streams, "memory")

    recipe = _build_recipe(task, candidate, steps)
    prior_init_stream = streams.generator("init", component="prior", purpose="locations", device=device)
    prior_opts = {"device": device, "dtype": torch.float32, "generator": prior_init_stream, "init_std": 1.0}
    if task.prior_kind == "mog":
        prior_opts["sigma"] = task.prior_sigma
    prior = recipe.replace(prior_update="learned").make_prior(**prior_opts)
    _init_module(prior, streams, "prior")
    recipe.apply_prior_policy(prior)

    transport_options = candidate.memory_policy.get("transport")
    trainer_class = TransportGANTrainer if transport_options is not None else GANTrainer
    trainer_options = {} if transport_options is None else {
        "transport": dict(transport_options),
        "routing_generator": streams.generator("prior", component="routing", purpose="assignment", device=device),
    }
    if transport_options is not None:
        router = trainer_options["transport"].pop("router", "competitive")
        if router == "context_balanced":
            trainer_options["router_class"] = ContextBalancedReconstruction
        elif router in {
            "feature_conditional",
            "residual_feature_conditional",
            "gail_conditional",
            "subspace_residual_conditional",
        }:
            trainer_options["router_class"] = {
                "feature_conditional": FeatureConditionalReconstruction,
                "residual_feature_conditional": ResidualFeatureConditionalReconstruction,
                "gail_conditional": GAILConditionalReconstruction,
                "subspace_residual_conditional": SubspaceResidualConditionalReconstruction,
            }[router]
            if "encoder_seed" in trainer_options["transport"]:
                raise ValueError("feature encoder initialization uses the protocol's named init stream")
            trainer_options["transport"]["encoder_seed"] = streams.seed_for(
                "init", component="feature_encoder", purpose="projection")
        elif router != "competitive":
            raise ValueError(f"unknown transport router: {router}")
    trainer = trainer_class(
        recipe,
        g,
        d,
        prior=prior,
        seed=streams.seed,
        max_steps=steps,
        latent_generator=streams.generator("prior", component="latent", purpose="indices", device=device),
        penalty_generator=streams.generator("noise", component="penalty", purpose="training", device=device),
        noise_generator=streams.generator("noise", component="generator", purpose="output", device=device),
        input_noise_generator=streams.generator("noise", component="critic", purpose="input", device=device),
        eval_generator=streams.generator("eval", component="sampler", purpose="samples", device=device),
        model_generator=streams.generator("noise", component="models", purpose="stochastic_layers", device=device),
        prior_noise_generator=streams.generator("noise", component="prior", purpose="gaussian", device=device),
        require_latent_damping=recipe.latent_damping_max_rate > 0,
        **trainer_options,
    )

    data_gen = streams.generator("data", component="stream", purpose="training", device=device)
    eval_gen = streams.generator("eval", component="live", purpose="enumerated_samples", device=device)

    if mode == "bars4":
        stream = SequentialBarsStream(dwell=int(host.get("dwell", 1)), device=device)
        target_bank = stream.templates
    elif mode == "translating_next":
        stream = TranslatingBarStream(device=device)
        target_bank = stream.positions
    elif mode == "fork_rollout":
        stream = ForkStream(device=device)
        target_bank = stream.templates[1:3]
    elif mode in CHALLENGE_MODES:
        stream = make_challenge_stream(mode, device)
        target_bank = challenge_target_bank(mode, stream)
    else:
        raise ValueError(f"unknown seq_image host mode: {mode!r}")

    checkpoints = set(task.observation_steps())
    if steps_override is not None:
        checkpoints = {
            max(1, round(i * steps / min(steps, task.observations)))
            for i in range(1, min(steps, task.observations) + 1)
        }

    observations = []
    gif_frames = []
    unintended_rng = 0
    all_finite = True
    session = OnlineSession(
        trainer, g, d, memory, predictive=(mode in PREDICTIVE_MODES),
        serving_generator=streams.generator("eval", component="serving", purpose="prediction", device=device),
    )
    reverse_trainer = None
    reverse_session = None
    g_c2s = d_s = prior_c2s = memory_c = None
    if mode == "square_circle_cycle":
        with streams.fork("init", component="models_c2s", purpose="construction", device=device):
            g_c2s = StreamImageGenerator(z_dim=task.z_dim, width=width, context_dim=model_context_dim).to(device)
            d_s = StreamImageCritic(width=width, context_dim=model_context_dim).to(device)
            memory_c = make_stream_memory(candidate.memory_policy)
            if memory_c is not None:
                memory_c = memory_c.to(device)
            _init_module(g_c2s, streams, "generator_c2s")
            _init_module(d_s, streams, "discriminator_c2s")
            if memory_c is not None:
                _init_module(memory_c, streams, "memory_c2s")
        prior_c2s_init = streams.generator("init", component="prior_c2s", purpose="locations", device=device)
        prior_c2s_opts = {"device": device, "dtype": torch.float32, "generator": prior_c2s_init, "init_std": 1.0}
        if task.prior_kind == "mog":
            prior_c2s_opts["sigma"] = task.prior_sigma
        prior_c2s = recipe.replace(prior_update="learned").make_prior(**prior_c2s_opts)
        _init_module(prior_c2s, streams, "prior_c2s")
        recipe.apply_prior_policy(prior_c2s)
        rev_trainer_options = {} if transport_options is None else {
            "transport": dict(trainer_options["transport"]),
            "routing_generator": streams.generator("prior", component="routing_c2s", purpose="assignment", device=device),
            **({"router_class": trainer_options["router_class"]} if "router_class" in trainer_options else {}),
        }
        reverse_trainer = trainer_class(
            recipe,
            g_c2s,
            d_s,
            prior=prior_c2s,
            seed=streams.seed,
            max_steps=steps,
            latent_generator=streams.generator("prior", component="latent_c2s", purpose="indices", device=device),
            penalty_generator=streams.generator("noise", component="penalty_c2s", purpose="training", device=device),
            noise_generator=streams.generator("noise", component="generator_c2s", purpose="output", device=device),
            input_noise_generator=streams.generator("noise", component="critic_c2s", purpose="input", device=device),
            eval_generator=streams.generator("eval", component="sampler_c2s", purpose="samples", device=device),
            model_generator=streams.generator("noise", component="models_c2s", purpose="stochastic_layers", device=device),
            prior_noise_generator=streams.generator("noise", component="prior_c2s", purpose="gaussian", device=device),
            require_latent_damping=recipe.latent_damping_max_rate > 0,
            **rev_trainer_options,
        )
        reverse_session = OnlineSession(
            reverse_trainer, g_c2s, d_s, memory_c, predictive=True,
            serving_generator=streams.generator("eval", component="serving_c2s", purpose="prediction", device=device),
        )
        trainer.cycle_partner = {
            "trainer": reverse_trainer,
            "generator": g_c2s,
            "critic": d_s,
            "memory": memory_c,
            "session": reverse_session,
        }

    if session.predictive:
        if mode == "square_circle_cycle":
            s_init, c_init, _ = stream.next_pair(data_gen)
            session.observe(c_init, context_frame=s_init)
            reverse_session.observe(s_init, context_frame=c_init)
        else:
            initial_frame, _ = stream.next_frame(data_gen)
            session.observe(initial_frame)
    prequential_errors = []

    for step in range(1, steps + 1):
        # Serving happens before the stream reveals its next real observation.
        with streams.isolated_evaluation():
            live_prediction = session.predict()
            live_rev_prediction = reverse_session.predict() if reverse_session is not None else None
        unintended_rng += streams.last_eval_audit["unintended_rng_deviations"]
        if mode == "square_circle_cycle":
            s_cur, c_cur, _ = stream.next_pair(data_gen)
            x_cur = c_cur
            s2c_err = float((live_prediction - c_cur).square().mean().sqrt().item())
            c2s_err = float((live_rev_prediction - s_cur).square().mean().sqrt().item())
            prequential_errors.append(0.5 * (s2c_err + c2s_err))
            stats = session.observe(c_cur, context_frame=s_cur)
            rev_stats = reverse_session.observe(s_cur, context_frame=c_cur)
            all_finite &= not has_nonfinite(stats) and not has_nonfinite(rev_stats)
        else:
            x_cur, _ = stream.next_frame(data_gen)
            prequential_errors.append(float((live_prediction - x_cur).square().mean().sqrt().item()))
            stats = session.observe(x_cur)
            all_finite &= not has_nonfinite(stats)
        for module in (g, d, prior, memory, g_c2s, d_s, prior_c2s, memory_c):
            if module is not None:
                all_finite &= all(bool(torch.isfinite(p).all()) for p in module.parameters())
                all_finite &= all(bool(torch.isfinite(p.grad).all())
                                  for p in module.parameters() if p.grad is not None)

        if step in checkpoints:
            from contextlib import nullcontext
            rev_ctx = reverse_session.isolated_state() if reverse_session is not None else nullcontext()
            with streams.isolated_evaluation(), session.isolated_state(), rev_ctx:
                if mode == "bars4":
                    samples = trainer.sample(task.num_particles, fixed_first_n=True, generator=eval_gen)
                    metrics = score_multimodal_bars(samples, stream.templates)
                elif mode == "translating_next":
                    if memory is not None:
                        saved_mat = memory.matrix.clone()
                        saved_prev = memory.prev_key.clone()
                        saved_cnt = memory.reservoir_count.clone()
                        saved_res = memory.reservoir.clone()
                        saved_g_ctx = g.active_context.clone()
                        saved_d_ctx = d.active_context.clone()
                        # Rightward warm-start: 0 -> 1 (expect 2)
                        memory.reset_state()
                        memory.write(stream.positions[0:1])
                        ctx_r = memory.write(stream.positions[1:2])
                        g.set_context(ctx_r)
                        pred_r = trainer.sample(task.num_particles, fixed_first_n=True, generator=eval_gen)
                        # Leftward warm-start: 2 -> 1 (expect 0)
                        memory.reset_state()
                        memory.write(stream.positions[2:3])
                        ctx_l = memory.write(stream.positions[1:2])
                        g.set_context(ctx_l)
                        pred_l = trainer.sample(task.num_particles, fixed_first_n=True, generator=eval_gen)
                        # Restore training memory state and active model contexts
                        memory.matrix.copy_(saved_mat)
                        memory.prev_key.copy_(saved_prev)
                        memory.reservoir_count.copy_(saved_cnt)
                        memory.reservoir.copy_(saved_res)
                        g.active_context.copy_(saved_g_ctx)
                        d.active_context.copy_(saved_d_ctx)
                    else:
                        pred_r = trainer.sample(task.num_particles, fixed_first_n=True, generator=eval_gen)
                        pred_l = trainer.sample(task.num_particles, fixed_first_n=True, generator=eval_gen)
                    metrics = score_translating_next(pred_r, pred_l, stream.positions)
                    samples = torch.cat([pred_r[:16], pred_l[:16]], dim=0)
                elif mode == "fork_rollout":
                    metrics, samples = evaluate_original_fork(trainer, g, d, memory, stream, task.num_particles, eval_gen)
                else:
                    metrics, samples = evaluate_challenge(mode, trainer, g, d, memory, stream, task.num_particles, eval_gen)

            unintended_rng += streams.last_eval_audit["unintended_rng_deviations"]
            metrics["live_prequential_rmse"] = round(sum(prequential_errors) / len(prequential_errors), 6)
            metrics["live_predictions"] = float(session.predictions)
            prequential_errors.clear()
            step_passed = all(b.check(metrics.get(b.metric, float("nan"))) for b in task.thresholds)
            observations.append({"step": step, "metrics": metrics})
            gif_frames.append(
                {
                    "step": step,
                    "passed": step_passed,
                    "metrics": metrics,
                    "current_stream_image": x_cur.detach().cpu().numpy(),
                    "target_images": target_bank.detach().cpu().numpy(),
                    "sample_images": samples.detach().cpu().numpy(),
                }
            )
            if log_event is not None:
                log_event("observation", step=step, passed=step_passed, metrics=metrics)

    # Render actual-training goal.gif
    window_passed = (
        len(gif_frames) >= task.minimum_stable_checks
        and all(f["passed"] for f in gif_frames[-task.minimum_stable_checks :])
    )
    if unintended_rng:
        verdict_str = "INVALID"
    elif steps < task.steps:
        verdict_str = "INCOMPLETE"
    else:
        verdict_str = "PASS" if (window_passed and all_finite) else "FAIL"
    gif_out = attempt_dir / "goal.gif"
    gif_rel = f"gifs/{candidate.id}_{task.id}.gif"
    gif_meta = render_training_gif(
        gif_frames,
        gif_out,
        task_id=task.id,
        candidate_id=candidate.id,
        verdict=verdict_str,
        primary_metric=task.thresholds[0].metric if task.thresholds else None,
    )
    gif_sha256 = gif_meta["sha256"]
    orbit_gif_meta = None
    if mode == "circle_orbit":
        orbit_gif_out = attempt_dir / "circle_orbit.gif"
        orbit_gif_meta = render_circle_orbit_gif(
            gif_frames,
            orbit_gif_out,
            task_id=task.id,
            candidate_id=candidate.id,
            verdict=verdict_str,
        )
    cycle_gif_meta = None
    if mode == "square_circle_cycle":
        cycle_gif_out = attempt_dir / "square_circle_cycle.gif"
        cycle_gif_meta = render_square_circle_cycle_gif(
            gif_frames,
            cycle_gif_out,
            task_id=task.id,
            candidate_id=candidate.id,
            verdict=verdict_str,
        )

    root_dir = workspace_root if workspace_root is not None else None
    if root_dir is not None:
        is_production_root = (root_dir.resolve() == Path(__file__).resolve().parents[1])
        should_promote = (not is_production_root) or (steps_override is None or steps_override == task.steps)
        if should_promote:
            canonical_gif = root_dir / "reports/forge" / gif_rel
            canonical_gif.parent.mkdir(parents=True, exist_ok=True)
            tmp_gif = canonical_gif.with_suffix(".tmp")
            import shutil
            shutil.copyfile(gif_out, tmp_gif)
            tmp_gif.replace(canonical_gif)
            if orbit_gif_meta is not None and is_production_root and candidate.id == "online-feature-prior-bce":
                readme_gif = root_dir / "docs/media/readme/circle-orbit.gif"
                readme_gif.parent.mkdir(parents=True, exist_ok=True)
                tmp_readme_gif = readme_gif.with_suffix(".tmp")
                shutil.copyfile(orbit_gif_out, tmp_readme_gif)
                tmp_readme_gif.replace(readme_gif)
            if cycle_gif_meta is not None and is_production_root and candidate.id == "online-feature-prior-bce":
                readme_cycle_gif = root_dir / "docs/media/readme/square-circle-cycle.gif"
                readme_cycle_gif.parent.mkdir(parents=True, exist_ok=True)
                tmp_readme_cycle = readme_cycle_gif.with_suffix(".tmp")
                shutil.copyfile(cycle_gif_out, tmp_readme_cycle)
                tmp_readme_cycle.replace(readme_cycle_gif)

    transport_artifacts = {} if transport_options is None else {
        "transport": trainer.transport_diagnostics(),
    }
    orbit_artifacts = {} if orbit_gif_meta is None else {
        "orbit_gif_sha256": orbit_gif_meta["sha256"],
    }
    cycle_artifacts = {} if cycle_gif_meta is None else {
        "cycle_gif_sha256": cycle_gif_meta["sha256"],
        "cyclegan": {
            "forward_direction": "square_to_circle",
            "backward_direction": "circle_to_square",
            "reverse_online_execution": reverse_session.diagnostics(),
            **({"reverse_transport": reverse_trainer.transport_diagnostics()} if transport_options is not None else {}),
        },
    }
    return {
        "observations": observations,
        "all_finite": all_finite,
        "unintended_rng_deviations": unintended_rng,
        "completed_steps": steps,
        "artifacts": {"goal_gif": gif_rel, "gif_sha256": gif_sha256,
                      "particlegan_version": version("particlegan"),
                      "torch_version": torch.__version__,
                      "resolved_recipe": asdict(recipe),
                      "implementation": IMPLEMENTATION,
                      "memory_policy": candidate.memory_policy,
                      "online_execution": session.diagnostics(),
                      **orbit_artifacts,
                      **cycle_artifacts,
                      **transport_artifacts},
    }


def bootstrap_workspace(root: Path | str) -> StreamWorkspace:
    ws = StreamWorkspace(root, registry=REGISTRY)
    view_path = ws.views_dir / f"{PRIMARY_VIEW.id}.json"
    if view_path.is_file():
        previous = view_path.read_text()
        previous_revision = json.loads(previous)["revision"]
        if previous_revision > PRIMARY_VIEW.revision:
            raise ValueError("bootstrap would replace a newer view; reconcile its policy first")
        if previous_revision < PRIMARY_VIEW.revision:
            archive = ws.view_history_dir / f"{PRIMARY_VIEW.id}-v{previous_revision}.json"
            if archive.exists() and archive.read_text() != previous:
                raise ValueError("archived view revision differs; refusing to overwrite history")
            atomic_text(archive, previous)
    for t in TIER1_TASKS:
        ws.register_task(t)
    ws.register_view(PRIMARY_VIEW)
    for c in CANDIDATES:
        ws.register_candidate(c)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ws.compile_reports(view_id=PRIMARY_VIEW.id, device=device)
    return ws


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ParticleStream Tier 1 bootstrapper")
    p.add_argument("action", choices=("bootstrap",), help="Write Tier 1 tasks, view, and candidates to configs/forge/")
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args(argv)
    ws = bootstrap_workspace(args.root)
    print(f"Bootstrapped ParticleStream at {ws.root} ({len(TIER1_TASKS)} tasks, {len(CANDIDATES)} candidates).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
