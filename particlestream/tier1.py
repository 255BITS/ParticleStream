"""Tier 1 declarations, adapter, and bootstrap CLI for ParticleStream."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from particlegan import GANTrainer, Recipe, get_recipe, init
from particlegan_forge import (
    CandidateSpec,
    ForgeRegistry,
    ForgeWorkspace,
    MetricBound,
    NamedStreams,
    TaskSpec,
    TierSpec,
    TierView,
    render_training_gif,
)

from .models import FastWeightMemory, StreamImageCritic, StreamImageGenerator
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

PRIMARY_VIEW = TierView.from_tiers(
    id="single_stream_stability",
    goal="single_stream_stability",
    revision=1,
    policy_change_reason="Initial Tier 1 single-stream batch_size=1 sequential image suite.",
    tiers=[
        TierSpec(
            tier=1,
            required=[t.id for t in TIER1_TASKS],
        )
    ],
)

if not hasattr(TaskSpec, "host"):
    TaskSpec.host = property(lambda self: self.host_definition)

CANDIDATES: list[CandidateSpec] = [
    CandidateSpec(
        id="bcap",
        technique="BCAP",
        changed_factors=["Stateless BCAP baseline on batch_size=1 sequential stream"],
        recipe_preset="bcap",
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
        recipe_preset="bcap",
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
        recipe_preset="bcap",
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

TIER1_CANDIDATES = {c.id: c for c in CANDIDATES}


def _build_recipe(task: TaskSpec, candidate: CandidateSpec, steps: int) -> Recipe:
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
    if preset == "bcap":
        return get_recipe("bcap_adam", **options).replace(name="bcap")
    return get_recipe(preset, **options)


def _init_module(module: torch.nn.Module, streams: NamedStreams, component: str) -> None:
    seeds = {
        name: streams.seed_for("init", component=component, purpose=name)
        for name, p in module.named_parameters()
        if p.requires_grad and p.numel()
    }
    if seeds:
        init.deterministic_orthogonal_(module, parameter_seeds=seeds)


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
    steps = int(steps_override or task.steps)
    host = getattr(task, "host", task.host_definition)
    mode = host["mode"]
    width = int(host.get("width", 16))
    task_context_dim = int(host.get("context_dim", 0))
    use_memory = candidate.memory_policy.get("kind") == "fast_weight"
    mem_dim = int(candidate.memory_policy.get("dim", 16)) if use_memory else 0
    model_context_dim = task_context_dim if use_memory else 0

    with streams.fork("init", component="models", purpose="construction", device=device):
        g = StreamImageGenerator(z_dim=task.z_dim, width=width, context_dim=model_context_dim).to(device)
        d = StreamImageCritic(width=width, context_dim=model_context_dim).to(device)
        memory = (
            FastWeightMemory(
                dim=mem_dim,
                decay=float(candidate.memory_policy.get("decay", 0.90)),
                write_lr=float(candidate.memory_policy.get("write_lr", 0.35)),
                reservoir_size=int(candidate.memory_policy.get("reservoir_size", 64)),
            ).to(device)
            if use_memory
            else None
        )
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

    trainer = GANTrainer(
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
    )

    data_gen = streams.generator("data", component="stream", purpose="training", device=device)
    mem_gen = streams.generator("data", component="memory", purpose="reservoir", device=device)
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
    prev_frame = None

    for step in range(1, steps + 1):
        if mode == "bars4":
            x_cur, _ = stream.next_frame(data_gen)
            if memory is not None:
                rehearsal_batch = int(candidate.memory_policy.get("rehearsal_batch", 0))
                if rehearsal_batch > 0:
                    ctx = memory.write(x_cur)
                    recalled = memory.sample_reservoir(mem_gen, n=rehearsal_batch)
                    train_real = torch.cat([x_cur, recalled], dim=0) if recalled is not None else x_cur
                else:
                    is_dwelling = (prev_frame is not None) and ((x_cur - prev_frame).abs().max().item() < 0.10)
                    recalled = memory.sample_reservoir(mem_gen)
                    ctx = memory.write(x_cur)
                    prev_frame = x_cur.clone()
                    if is_dwelling and (recalled is not None) and (int(memory.reservoir_count.item()) >= memory.reservoir_size):
                        train_real = recalled
                    else:
                        train_real = x_cur
                if task_context_dim > 0:
                    g.set_context(ctx)
                    d.set_context(ctx)
            else:
                train_real = x_cur
        elif mode == "translating_next":
            x_t, x_next, _, _ = stream.next_pair(data_gen)
            if memory is not None:
                ctx = memory.write(x_t)
                g.set_context(ctx)
                d.set_context(ctx)
            x_cur = x_t
            train_real = x_next
        else:  # fork_rollout
            stem, branch, _ = stream.next_transition(data_gen)
            if memory is not None:
                ctx = memory.write(stem)
                g.set_context(ctx)
                d.set_context(ctx)
            x_cur = stem
            train_real = branch

        trainer.step(train_real)
        all_finite &= all(bool(torch.isfinite(p).all()) for p in g.parameters())
        all_finite &= all(bool(torch.isfinite(p).all()) for p in d.parameters())

        if step in checkpoints:
            with streams.isolated_evaluation():
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
                else:  # fork_rollout
                    if memory is not None:
                        saved_g_ctx = g.active_context.clone()
                        ctx_stem = memory.read_context(stream.templates[0:1])
                        g.set_context(ctx_stem)
                    samples = trainer.sample(task.num_particles, fixed_first_n=True, generator=eval_gen)
                    if memory is not None:
                        g.active_context.copy_(saved_g_ctx)
                    metrics = score_fork_rollout(samples, stream.templates)

            unintended_rng += streams.last_eval_audit["unintended_rng_deviations"]
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
    verdict_str = "PASS" if (window_passed and all_finite and unintended_rng == 0) else "FAIL"
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

    return {
        "observations": observations,
        "all_finite": all_finite,
        "unintended_rng_deviations": unintended_rng,
        "completed_steps": steps,
        "artifacts": {"goal_gif": gif_rel, "gif_sha256": gif_sha256},
    }


def bootstrap_workspace(root: Path | str) -> ForgeWorkspace:
    ws = ForgeWorkspace(root, registry=REGISTRY)
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
