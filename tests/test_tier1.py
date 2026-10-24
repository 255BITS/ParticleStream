"""Tests for particlegan_forge Tier API and ParticleStream Tier 1 contracts."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from particlegan_forge import (
    ForgeWorkspace,
    NamedStreams,
    grade_sustained_observations,
)
from particlegan_forge.rng import STREAM_NAMES
from particlestream.models import (
    FastWeightMemory,
    StreamImageCritic,
    StreamImageGenerator,
)
from particlestream.streams import (
    ForkStream,
    SequentialBarsStream,
    TranslatingBarStream,
    bars4_templates,
    fork_templates,
    score_fork_rollout,
    score_multimodal_bars,
    score_translating_next,
    translating_bar_cycle,
)
from particlestream.tier1 import (
    CANDIDATES,
    REGISTRY,
    TIER1_CANDIDATES,
    TIER1_TASKS,
    _init_module,
    bootstrap_workspace,
)


# ---------------------------------------------------------------------------
# Stream Law, Memory Detachment & Deterministic Initialization (Milestone M2)
# ---------------------------------------------------------------------------

def test_stream_nograd_and_memory_detach_contract():
    """Smoke test for basic stream nograd and memory read invariance."""
    gen = torch.Generator().manual_seed(0)
    s1 = SequentialBarsStream(dwell=4)
    frame, _ = s1.next_frame(gen)
    assert frame.shape == (1, 1, 8, 8)
    assert frame.requires_grad is False

    mem = FastWeightMemory(dim=8)
    ctx1 = mem.write(frame)
    assert ctx1.requires_grad is False
    before = mem.matrix.clone()
    ctx_read = mem.read_context(frame)
    assert ctx_read.requires_grad is False
    assert torch.equal(mem.matrix, before), "read_context must never mutate memory"


def test_all_streams_nograd_and_batch_size_contract():
    """Verify all data streams and static templates enforce batch_size=1 and requires_grad=False."""
    gen = torch.Generator().manual_seed(0)

    # Static templates
    b_temp = bars4_templates()
    assert b_temp.requires_grad is False
    assert b_temp.grad_fn is None
    assert b_temp.shape == (4, 1, 8, 8)

    p_temp, cycle = translating_bar_cycle()
    assert p_temp.requires_grad is False
    assert p_temp.grad_fn is None
    assert p_temp.shape == (4, 1, 8, 8)
    assert cycle == [0, 1, 2, 3, 2, 1]

    f_temp = fork_templates()
    assert f_temp.requires_grad is False
    assert f_temp.grad_fn is None
    assert f_temp.shape == (3, 1, 8, 8)

    # SequentialBarsStream across multiple modes
    s_bars = SequentialBarsStream(dwell=2)
    for _ in range(8):
        frame, mode = s_bars.next_frame(gen)
        assert frame.shape == (1, 1, 8, 8)
        assert frame.requires_grad is False
        assert frame.grad_fn is None
        assert 0 <= mode < 4

    # TranslatingBarStream across full cycle
    s_trans = TranslatingBarStream()
    for _ in range(12):
        xt, xnext, pt, pnext = s_trans.next_pair(gen)
        assert xt.shape == (1, 1, 8, 8)
        assert xnext.shape == (1, 1, 8, 8)
        assert xt.requires_grad is False
        assert xnext.requires_grad is False
        assert xt.grad_fn is None
        assert xnext.grad_fn is None

    # ForkStream across branching choices
    s_fork = ForkStream()
    for _ in range(6):
        stem, branch, bid = s_fork.next_transition(gen)
        assert stem.shape == (1, 1, 8, 8)
        assert branch.shape == (1, 1, 8, 8)
        assert stem.requires_grad is False
        assert branch.requires_grad is False
        assert stem.grad_fn is None
        assert branch.grad_fn is None
        assert bid in (1, 2)


def test_model_memory_inter_step_detachment_and_backward_isolation():
    """Verify FastWeightMemory buffers and context updates never leak autograd graphs across steps."""
    gen = torch.Generator().manual_seed(0)
    s = SequentialBarsStream(dwell=1)
    mem = FastWeightMemory(dim=16, reservoir_size=4)

    # 1. State buffers must be non-differentiable
    for buf_name in ("matrix", "prev_key", "reservoir", "reservoir_count"):
        buf = getattr(mem, buf_name)
        assert buf.requires_grad is False
        assert buf.grad_fn is None

    # 2. Multi-step sequential writes must not link computation graphs
    f1, _ = s.next_frame(gen)
    ctx1 = mem.write(f1)
    assert ctx1.requires_grad is False
    assert ctx1.grad_fn is None

    f2, _ = s.next_frame(gen)
    ctx2 = mem.write(f2)
    assert ctx2.requires_grad is False
    assert ctx2.grad_fn is None

    # 3. Reservoir sample must be detached
    sampled = mem.sample_reservoir(gen)
    assert sampled is not None
    assert sampled.shape == (1, 1, 8, 8)
    assert sampled.requires_grad is False
    assert sampled.grad_fn is None

    # 4. Generator & Critic backward pass must not backprop into memory
    g = StreamImageGenerator(z_dim=8, width=16, context_dim=16)
    d = StreamImageCritic(width=16, context_dim=16)
    g.set_context(ctx2)
    d.set_context(ctx2)

    z = torch.randn(2, 8, requires_grad=True)
    fake = g(z)
    loss_g = fake.sum()
    loss_g.backward()
    assert mem.matrix.grad is None
    assert mem.prev_key.grad is None
    assert ctx2.grad is None

    loss_d = d(fake.detach()).sum()
    loss_d.backward()
    assert mem.matrix.grad is None


def test_deterministic_orthogonal_initialization_seed0():
    """Verify deterministic orthogonal initialization (particlegan.init) under protocol seed 0."""
    # Instance A under seed 0
    streams0_a = NamedStreams(seed=0)
    g0_a = StreamImageGenerator(z_dim=8, width=16, context_dim=16)
    d0_a = StreamImageCritic(width=16, context_dim=16)
    m0_a = FastWeightMemory(dim=16)
    _init_module(g0_a, streams0_a, "generator")
    _init_module(d0_a, streams0_a, "discriminator")
    _init_module(m0_a, streams0_a, "memory")

    # Instance B under seed 0
    streams0_b = NamedStreams(seed=0)
    g0_b = StreamImageGenerator(z_dim=8, width=16, context_dim=16)
    d0_b = StreamImageCritic(width=16, context_dim=16)
    m0_b = FastWeightMemory(dim=16)
    _init_module(g0_b, streams0_b, "generator")
    _init_module(d0_b, streams0_b, "discriminator")
    _init_module(m0_b, streams0_b, "memory")

    # 1. Bitwise parameter reproducibility across independent instances
    for p_a, p_b in zip(g0_a.parameters(), g0_b.parameters()):
        assert torch.equal(p_a, p_b)
    for p_a, p_b in zip(d0_a.parameters(), d0_b.parameters()):
        assert torch.equal(p_a, p_b)
    for p_a, p_b in zip(m0_a.parameters(), m0_b.parameters()):
        assert torch.equal(p_a, p_b)

    # 2. Seed sensitivity: seed 1 must yield different parameters
    streams1 = NamedStreams(seed=1)
    g1 = StreamImageGenerator(z_dim=8, width=16, context_dim=16)
    _init_module(g1, streams1, "generator")
    assert any(not torch.equal(p0, p1) for p0, p1 in zip(g0_a.parameters(), g1.parameters()))

    # 3. Semi-orthogonality check on all 2D and 4D weights (W W^T or W^T W proportional to identity)
    for module in (g0_a, d0_a, m0_a):
        for name, p in module.named_parameters():
            if "weight" in name and p.ndim >= 2:
                mat = p.reshape(p.shape[0], -1) if p.ndim > 2 else p
                rows, cols = mat.shape
                prod = mat @ mat.t() if rows <= cols else mat.t() @ mat
                diag = torch.diag(prod)
                off_diag = prod - torch.diag(diag)
                max_off = off_diag.abs().max().item()
                assert max_off < 1e-4, f"Weight {name} failed semi-orthogonality: {max_off}"


# ---------------------------------------------------------------------------
# Task Scorers, Oracle PASS & Destructive FAIL Controls (Milestone M2)
# ---------------------------------------------------------------------------

def test_oracle_and_destructive_controls():
    """Baseline smoke test for scorer functions."""
    bars = bars4_templates()
    oracle_samples = bars.repeat(8, 1, 1, 1)
    oracle_score = score_multimodal_bars(oracle_samples, bars)
    assert oracle_score["hq"] == 1.0
    assert oracle_score["modes"] == 4.0
    assert oracle_score["finite_template_tv"] == 0.0

    collapsed = bars[0:1].repeat(32, 1, 1, 1)
    collapsed_score = score_multimodal_bars(collapsed, bars)
    assert collapsed_score["modes"] == 1.0
    assert collapsed_score["finite_template_tv"] > 0.50

    positions, _ = translating_bar_cycle()
    trans_oracle = score_translating_next(
        positions[2:3].repeat(16, 1, 1, 1),
        positions[0:1].repeat(16, 1, 1, 1),
        positions,
    )
    assert trans_oracle["next_frame_rmse"] == 0.0
    assert trans_oracle["direction_acc"] == 1.0

    trans_wrong = score_translating_next(
        positions[0:1].repeat(16, 1, 1, 1),
        positions[2:3].repeat(16, 1, 1, 1),
        positions,
    )
    assert trans_wrong["direction_acc"] == 0.0

    ftemplates = fork_templates()
    fork_oracle = score_fork_rollout(ftemplates[1:3].repeat(16, 1, 1, 1), ftemplates)
    assert fork_oracle["within_state_modes"] == 2.0
    assert fork_oracle["branch_tv"] == 0.0

    fork_collapsed = score_fork_rollout(ftemplates[1:2].repeat(32, 1, 1, 1), ftemplates)
    assert fork_collapsed["within_state_modes"] == 1.0


def test_task1_round_robin_bars4_oracle_and_destructive_controls():
    """Verify Task 1 threshold bounds against oracle PASS and destructive FAIL controls."""
    task = next(t for t in TIER1_TASKS if t.id == "seq_img_round_robin_bars4")
    bars = bars4_templates()

    # 1. Perfect Oracle (PASS)
    oracle_samples = bars.repeat(8, 1, 1, 1)
    oracle_score = score_multimodal_bars(oracle_samples, bars)
    assert all(b.check(oracle_score[b.metric]) for b in task.thresholds)
    assert oracle_score["hq"] == 1.0
    assert oracle_score["modes"] == 4.0
    assert oracle_score["finite_template_tv"] == 0.0

    # 2. Noisy Oracle (sigma=0.02, PASS)
    noisy_oracle = (oracle_samples + 0.02 * torch.randn_like(oracle_samples)).clamp(0.0, 1.0)
    noisy_score = score_multimodal_bars(noisy_oracle, bars)
    assert all(b.check(noisy_score[b.metric]) for b in task.thresholds)

    # 3. Destructive: Single-mode collapse onto template 0 (FAIL)
    collapsed = bars[0:1].repeat(32, 1, 1, 1)
    collapsed_score = score_multimodal_bars(collapsed, bars)
    assert not all(b.check(collapsed_score[b.metric]) for b in task.thresholds)
    assert collapsed_score["modes"] == 1.0
    assert collapsed_score["finite_template_tv"] > 0.50

    # 4. Destructive: Incomplete mode coverage (3 of 4 modes, FAIL)
    three_modes = torch.cat(
        [
            bars[0:1].repeat(11, 1, 1, 1),
            bars[1:2].repeat(11, 1, 1, 1),
            bars[2:3].repeat(10, 1, 1, 1),
        ],
        dim=0,
    )
    three_score = score_multimodal_bars(three_modes, bars)
    assert not all(b.check(three_score[b.metric]) for b in task.thresholds)
    assert three_score["modes"] == 3.0
    assert three_score["finite_template_tv"] > 0.15

    # 5. Destructive: Untrained random noise (FAIL)
    noise_score = score_multimodal_bars(torch.rand(32, 1, 8, 8), bars)
    assert not all(b.check(noise_score[b.metric]) for b in task.thresholds)
    assert noise_score["hq"] == 0.0
    assert noise_score["modes"] == 0.0


def test_task2_dwell_bars4_oracle_and_destructive_controls():
    """Verify Task 2 threshold bounds against oracle PASS and dwell-specific destructive FAIL controls."""
    task = next(t for t in TIER1_TASKS if t.id == "seq_img_dwell_bars4")
    bars = bars4_templates()

    # 1. Oracle (PASS)
    oracle_samples = bars.repeat(8, 1, 1, 1)
    oracle_score = score_multimodal_bars(oracle_samples, bars)
    assert all(b.check(oracle_score[b.metric]) for b in task.thresholds)

    # 2. Noisy Oracle (sigma=0.02, PASS)
    noisy_oracle = (oracle_samples + 0.02 * torch.randn_like(oracle_samples)).clamp(0.0, 1.0)
    noisy_score = score_multimodal_bars(noisy_oracle, bars)
    assert all(b.check(noisy_score[b.metric]) for b in task.thresholds)

    # 3. Destructive: Dwell collapse onto current dwelling mode (FAIL)
    collapsed = bars[0:1].repeat(32, 1, 1, 1)
    collapsed_score = score_multimodal_bars(collapsed, bars)
    assert not all(b.check(collapsed_score[b.metric]) for b in task.thresholds)

    # 4. Destructive: Forgetting earlier modes (3 of 4 modes retained, FAIL)
    three_modes = torch.cat(
        [
            bars[0:1].repeat(11, 1, 1, 1),
            bars[1:2].repeat(11, 1, 1, 1),
            bars[2:3].repeat(10, 1, 1, 1),
        ],
        dim=0,
    )
    three_score = score_multimodal_bars(three_modes, bars)
    assert not all(b.check(three_score[b.metric]) for b in task.thresholds)
    assert three_score["finite_template_tv"] > 0.20

    # 5. Destructive: Untrained random noise (FAIL)
    noise_score = score_multimodal_bars(torch.rand(32, 1, 8, 8), bars)
    assert not all(b.check(noise_score[b.metric]) for b in task.thresholds)


def test_task3_translating_next_oracle_and_destructive_controls():
    """Verify Task 3 threshold bounds against oracle PASS and temporal destructive FAIL controls."""
    task = next(t for t in TIER1_TASKS if t.id == "seq_img_translating_bar_next")
    positions, _ = translating_bar_cycle()

    # 1. Oracle: pos 2 moving right, pos 0 moving left (PASS)
    pred_r = positions[2:3].repeat(16, 1, 1, 1)
    pred_l = positions[0:1].repeat(16, 1, 1, 1)
    trans_oracle = score_translating_next(pred_r, pred_l, positions)
    assert all(b.check(trans_oracle[b.metric]) for b in task.thresholds)
    assert trans_oracle["next_frame_rmse"] == 0.0
    assert trans_oracle["direction_acc"] == 1.0

    # 2. Noisy Oracle (sigma=0.02, PASS)
    noisy_r = (pred_r + 0.02 * torch.randn_like(pred_r)).clamp(0.0, 1.0)
    noisy_l = (pred_l + 0.02 * torch.randn_like(pred_l)).clamp(0.0, 1.0)
    noisy_trans = score_translating_next(noisy_r, noisy_l, positions)
    assert all(b.check(noisy_trans[b.metric]) for b in task.thresholds)

    # 3. Destructive: Reversed direction prediction (FAIL)
    trans_reversed = score_translating_next(pred_l, pred_r, positions)
    assert not all(b.check(trans_reversed[b.metric]) for b in task.thresholds)
    assert trans_reversed["direction_acc"] == 0.0

    # 4. Destructive: Static memoryless prediction of current ambiguous frame 1 (FAIL)
    static_frame = positions[1:2].repeat(16, 1, 1, 1)
    trans_static = score_translating_next(static_frame, static_frame, positions)
    assert not all(b.check(trans_static[b.metric]) for b in task.thresholds)
    assert trans_static["direction_acc"] == 0.0
    assert trans_static["next_frame_rmse"] > 0.12

    # 5. Destructive: Untrained random noise (FAIL)
    trans_noise = score_translating_next(
        torch.rand(16, 1, 8, 8),
        torch.rand(16, 1, 8, 8),
        positions,
    )
    assert not all(b.check(trans_noise[b.metric]) for b in task.thresholds)


def test_task4_fork_rollout_oracle_and_destructive_controls():
    """Verify Task 4 threshold bounds against oracle PASS and branching destructive FAIL controls."""
    task = next(t for t in TIER1_TASKS if t.id == "seq_img_fork_rollout")
    ftemplates = fork_templates()

    # 1. Oracle: 16 copies of branch 1 and 16 copies of branch 2 (PASS)
    pred_fork = ftemplates[1:3].repeat(16, 1, 1, 1)
    fork_oracle = score_fork_rollout(pred_fork, ftemplates)
    assert all(b.check(fork_oracle[b.metric]) for b in task.thresholds)
    assert fork_oracle["hq"] == 1.0
    assert fork_oracle["within_state_modes"] == 2.0
    assert fork_oracle["branch_tv"] == 0.0

    # 2. Noisy Oracle (sigma=0.02, PASS)
    noisy_fork = (pred_fork + 0.02 * torch.randn_like(pred_fork)).clamp(0.0, 1.0)
    noisy_fork_score = score_fork_rollout(noisy_fork, ftemplates)
    assert all(b.check(noisy_fork_score[b.metric]) for b in task.thresholds)

    # 3. Destructive: Single branch collapse (branch 1 only, FAIL)
    fork_branch_coll = score_fork_rollout(ftemplates[1:2].repeat(32, 1, 1, 1), ftemplates)
    assert not all(b.check(fork_branch_coll[b.metric]) for b in task.thresholds)
    assert fork_branch_coll["within_state_modes"] == 1.0
    assert fork_branch_coll["branch_tv"] > 0.20

    # 4. Destructive: Stem identity collapse (repeats conditioning stem 0, FAIL)
    fork_stem_coll = score_fork_rollout(ftemplates[0:1].repeat(32, 1, 1, 1), ftemplates)
    assert not all(b.check(fork_stem_coll[b.metric]) for b in task.thresholds)
    assert fork_stem_coll["hq"] == 0.0
    assert fork_stem_coll["within_state_modes"] == 0.0
    assert fork_stem_coll["branch_tv"] == 1.0

    # 5. Destructive: Untrained random noise (FAIL)
    fork_noise = score_fork_rollout(torch.rand(32, 1, 8, 8), ftemplates)
    assert not all(b.check(fork_noise[b.metric]) for b in task.thresholds)
    assert fork_noise["hq"] == 0.0


def test_tier1_grade_sustained_observations_contract():
    """Verify sustained 24-observation gate evaluates PASS for oracles and FAIL for destructive controls."""
    tasks = {t.id: t for t in TIER1_TASKS}
    bars = bars4_templates()
    positions, _ = translating_bar_cycle()
    ft = fork_templates()

    oracle_metrics = {
        "seq_img_round_robin_bars4": score_multimodal_bars(bars.repeat(8, 1, 1, 1), bars),
        "seq_img_dwell_bars4": score_multimodal_bars(bars.repeat(8, 1, 1, 1), bars),
        "seq_img_translating_bar_next": score_translating_next(
            positions[2:3].repeat(16, 1, 1, 1),
            positions[0:1].repeat(16, 1, 1, 1),
            positions,
        ),
        "seq_img_fork_rollout": score_fork_rollout(ft[1:3].repeat(16, 1, 1, 1), ft),
    }

    fail_metrics = {
        "seq_img_round_robin_bars4": score_multimodal_bars(bars[0:1].repeat(32, 1, 1, 1), bars),
        "seq_img_dwell_bars4": score_multimodal_bars(bars[0:1].repeat(32, 1, 1, 1), bars),
        "seq_img_translating_bar_next": score_translating_next(
            positions[0:1].repeat(16, 1, 1, 1),
            positions[2:3].repeat(16, 1, 1, 1),
            positions,
        ),
        "seq_img_fork_rollout": score_fork_rollout(ft[1:2].repeat(32, 1, 1, 1), ft),
    }

    for task_id, task in tasks.items():
        obs_pass = [{"step": i * 20, "metrics": oracle_metrics[task_id]} for i in range(1, 25)]
        grade_pass = grade_sustained_observations(task, obs_pass, completed_steps=task.steps)
        assert grade_pass["status"] == "PASS"
        assert grade_pass["terminal_passing"] == 5
        assert len(grade_pass["failed_bounds"]) == 0

        obs_fail = [{"step": i * 20, "metrics": fail_metrics[task_id]} for i in range(1, 25)]
        grade_fail = grade_sustained_observations(task, obs_fail, completed_steps=task.steps)
        assert grade_fail["status"] == "FAIL"
        assert grade_fail["terminal_passing"] == 0
        assert len(grade_fail["failed_bounds"]) > 0


# ---------------------------------------------------------------------------
# NamedStreams RNG Isolation & Sustained 5-Check Gate (Milestone M2)
# ---------------------------------------------------------------------------

def test_named_streams_five_families_and_isolation():
    """Verify all 5 stream families (init, data, prior, noise, eval) are isolated and deterministic."""
    assert set(STREAM_NAMES) == {"init", "data", "prior", "noise", "eval"}

    ns1 = NamedStreams(seed=42)
    ns2 = NamedStreams(seed=42)

    # 1. Derivation determinism: same arguments derive identical generators
    g_init1 = ns1.generator("init", component="models", purpose="construction")
    g_init2 = ns2.generator("init", component="models", purpose="construction")
    assert torch.equal(g_init1.get_state(), g_init2.get_state())

    # 2. Independence across families: drawing from one stream does not advance others
    g_data = ns1.generator("data", component="stream", purpose="training")
    g_prior = ns1.generator("prior", component="latent", purpose="indices")
    g_noise = ns1.generator("noise", component="generator", purpose="output")
    g_eval = ns1.generator("eval", component="live", purpose="samples")

    prior_state_before = g_prior.get_state().clone()
    noise_state_before = g_noise.get_state().clone()
    eval_state_before = g_eval.get_state().clone()

    torch.randn(100, generator=g_data)

    assert torch.equal(g_prior.get_state(), prior_state_before)
    assert torch.equal(g_noise.get_state(), noise_state_before)
    assert torch.equal(g_eval.get_state(), eval_state_before)

    # 3. Serialization and restoration round-trip
    sd = ns1.state_dict()
    assert sd["schema"] == 1
    ns_restored = NamedStreams(seed=42)
    ns_restored.load_state_dict(sd)
    g_data_restored = ns_restored.generator("data", component="stream", purpose="training")
    assert torch.equal(g_data.get_state(), g_data_restored.get_state())


def test_isolated_evaluation_audit_and_adversarial_leakage():
    """Verify isolated_evaluation permits eval draws and catches non-eval stream leakage."""
    ns = NamedStreams(seed=123)
    g_eval = ns.generator("eval", component="live", purpose="samples")
    g_data = ns.generator("data", component="stream", purpose="training")
    g_noise = ns.generator("noise", component="critic", purpose="input")

    # Case A: Legitimate evaluation draw from an eval stream
    with ns.isolated_evaluation():
        torch.randn(20, generator=g_eval)
    assert ns.last_eval_audit["unintended_rng_deviations"] == 0
    assert ns.last_eval_audit["unintended_streams"] == []

    # Case B: Adversarial leakage modifying an existing non-eval stream ('data')
    with ns.isolated_evaluation():
        torch.randn(5, generator=g_data)
    assert ns.last_eval_audit["unintended_rng_deviations"] == 1
    assert any("data" in key for key in ns.last_eval_audit["unintended_streams"])

    # Case C: Adversarial leakage dynamically instantiating a new non-eval stream ('noise')
    with ns.isolated_evaluation():
        g_dyn_noise = ns.generator("noise", component="temp", purpose="untracked")
        torch.randn(5, generator=g_dyn_noise)
    assert ns.last_eval_audit["unintended_rng_deviations"] >= 1

    # Case D: Dynamically creating an eval stream is permitted
    with ns.isolated_evaluation():
        g_dyn_eval = ns.generator("eval", component="temp", purpose="extra_eval")
        torch.randn(5, generator=g_dyn_eval)
    assert ns.last_eval_audit["unintended_rng_deviations"] == 0

    # Case E: Ambient PyTorch RNG protection via fork_rng
    ambient_before = torch.get_rng_state()
    with ns.isolated_evaluation():
        torch.randn(50)  # ambient global draw
    ambient_after = torch.get_rng_state()
    assert torch.equal(ambient_before, ambient_after), "isolated_evaluation must preserve ambient torch RNG"


def test_adversarial_rng_leakage_invalidates_task_run(tmp_path: Path):
    """Verify that illegal non-eval RNG mutation during task execution marks receipt INVALID."""
    ws = bootstrap_workspace(tmp_path)
    original_isolated_eval = NamedStreams.isolated_evaluation

    @contextmanager
    def leaky_isolated_eval(self):
        with original_isolated_eval(self):
            # Adversarially advance a non-eval stream ('data') during evaluation
            data_gen = self.generator("data", component="stream", purpose="training", device=self.device)
            torch.randn(5, generator=data_gen)
            yield self

    with patch.object(NamedStreams, "isolated_evaluation", leaky_isolated_eval):
        rec = ws.run_task("fast-weight-bcap", "seq_img_round_robin_bars4", device="cpu", steps_override=4)

    assert rec["unintended_rng_deviations"] > 0
    assert rec["verdict"] == "INVALID"
    assert rec["grade"]["status"] == "INVALID"
    assert rec["terminal_passing"] == 0
    assert "unintended_rng_deviations == 0" in rec["grade"]["failed_bounds"]


def test_tier1_24_observation_cadence_and_terminal_window():
    """Verify all 4 Tier 1 tasks have 24 observation checkpoints with a sustained 5-check terminal window."""
    assert len(TIER1_TASKS) == 4
    expected_cadence = [i * 20 for i in range(1, 25)]  # [20, 40, ..., 480]
    expected_window = [400, 420, 440, 460, 480]

    for task in TIER1_TASKS:
        assert task.steps == 480
        assert task.observations == 24
        assert task.minimum_stable_checks == 5

        obs_steps = task.observation_steps()
        assert len(obs_steps) == 24
        assert obs_steps == expected_cadence
        assert obs_steps[-5:] == expected_window


def test_grade_sustained_observations_terminal_window_edges():
    """Test all edge cases of grade_sustained_observations contract across 24 checkpoints."""
    task = TIER1_TASKS[0]  # seq_img_round_robin_bars4: hq>=0.90, modes==4.0, finite_template_tv<=0.15
    good = {"hq": 0.95, "modes": 4.0, "finite_template_tv": 0.05}
    bad = {"hq": 0.50, "modes": 1.0, "finite_template_tv": 0.60}
    partial = {"hq": 0.95, "modes": 4.0, "finite_template_tv": 0.25}  # tv fails

    # 1. Clean PASS (all 24 checks pass)
    res_clean = grade_sustained_observations(
        task, [{"step": i * 20, "metrics": good} for i in range(1, 25)], completed_steps=480
    )
    assert res_clean["status"] == "PASS"
    assert res_clean["terminal_passing"] == 5
    assert res_clean["failed_bounds"] == []

    # 2. Late Recovery PASS (checks 1..19 fail, terminal window 20..24 passes)
    obs_late = [{"step": i * 20, "metrics": bad} for i in range(1, 20)] + [
        {"step": i * 20, "metrics": good} for i in range(20, 25)
    ]
    res_late = grade_sustained_observations(task, obs_late, completed_steps=480)
    assert res_late["status"] == "PASS"
    assert res_late["terminal_passing"] == 5

    # 3. Terminal Dip FAIL (checks 1..23 pass, check 24 fails)
    obs_term_dip = [{"step": i * 20, "metrics": good} for i in range(1, 24)] + [{"step": 480, "metrics": bad}]
    res_term_dip = grade_sustained_observations(task, obs_term_dip, completed_steps=480)
    assert res_term_dip["status"] == "FAIL"
    assert res_term_dip["terminal_passing"] == 4
    assert len(res_term_dip["failed_bounds"]) > 0

    # 4. Mid-Window Dip FAIL (step 440 fails, other terminal checks pass)
    obs_mid_dip = [{"step": i * 20, "metrics": good if i != 22 else bad} for i in range(1, 25)]
    res_mid_dip = grade_sustained_observations(task, obs_mid_dip, completed_steps=480)
    assert res_mid_dip["status"] == "FAIL"
    assert res_mid_dip["terminal_passing"] == 4

    # 5. Partial Threshold Failure (one metric bound fails in terminal window)
    obs_partial = [{"step": i * 20, "metrics": good} for i in range(1, 20)] + [
        {"step": i * 20, "metrics": partial} for i in range(20, 25)
    ]
    res_partial = grade_sustained_observations(task, obs_partial, completed_steps=480)
    assert res_partial["status"] == "FAIL"
    assert res_partial["terminal_passing"] == 0
    assert "finite_template_tv <= 0.15" in res_partial["failed_bounds"]

    # 6. Catastrophic Forgetting FAIL (checks 1..19 pass, terminal window fails)
    obs_forget = [{"step": i * 20, "metrics": good} for i in range(1, 20)] + [
        {"step": i * 20, "metrics": bad} for i in range(20, 25)
    ]
    res_forget = grade_sustained_observations(task, obs_forget, completed_steps=480)
    assert res_forget["status"] == "FAIL"
    assert res_forget["terminal_passing"] == 0

    # 7. Non-Finite Failure Override
    res_nonfinite = grade_sustained_observations(
        task, [{"step": i * 20, "metrics": good} for i in range(1, 25)], completed_steps=480, all_finite=False
    )
    assert res_nonfinite["status"] == "FAIL"
    assert res_nonfinite["terminal_passing"] == 0
    assert "all_finite == 1" in res_nonfinite["failed_bounds"]

    # 8. Unintended RNG Deviation Override
    res_rng_invalid = grade_sustained_observations(
        task, [{"step": i * 20, "metrics": good} for i in range(1, 25)], completed_steps=480, unintended_rng_deviations=1
    )
    assert res_rng_invalid["status"] == "INVALID"
    assert res_rng_invalid["terminal_passing"] == 0
    assert "unintended_rng_deviations == 0" in res_rng_invalid["failed_bounds"]

    # 9. Incomplete Step Budget
    res_steps_inc = grade_sustained_observations(
        task, [{"step": i * 20, "metrics": good} for i in range(1, 25)], completed_steps=400
    )
    assert res_steps_inc["status"] == "INCOMPLETE"
    assert res_steps_inc["terminal_passing"] == 0
    assert "completed_steps == steps" in res_steps_inc["failed_bounds"]

    # 10. Too Few Observations (< minimum_stable_checks)
    res_few_obs = grade_sustained_observations(
        task, [{"step": i * 20, "metrics": good} for i in range(1, 4)], completed_steps=480
    )
    assert res_few_obs["status"] == "INCOMPLETE"
    assert res_few_obs["terminal_passing"] == 0
    assert "observations >= minimum_stable_checks" in res_few_obs["failed_bounds"]


# ---------------------------------------------------------------------------
# Workspace Bootstrap, Retiering & Execution Integration
# ---------------------------------------------------------------------------

def test_tier_api_bootstrap_retier_and_short_run(tmp_path: Path):
    """Verify workspace bootstrap, short task execution with RNG audit, and atomic retiering."""
    ws = bootstrap_workspace(tmp_path)
    val = ws.validate()
    assert val["status"] == "valid"
    assert val["tasks"] == 4
    assert val["candidates"] == 4

    # Test short 12-step run with RNG isolation check and GIF generation (grades INCOMPLETE vs full budget)
    rec = ws.run_task("fast-weight-bcap", "seq_img_round_robin_bars4", device="cpu", steps_override=12)
    assert rec["grade"]["status"] == "INCOMPLETE"
    assert rec["unintended_rng_deviations"] == 0
    assert rec["terminal_passing"] == 0
    assert rec["observations"]
    assert (tmp_path / "reports/forge" / rec["artifacts"]["goal_gif"]).is_file()

    # Test atomic view retiering & archive creation
    updated = ws.retier(
        "single_stream_stability",
        task_id="seq_img_fork_rollout",
        new_tier=2,
        reason="Move fork rollout to Tier 2 in test",
    )
    assert updated.revision == 2
    assert (tmp_path / "configs/forge/view-history/single_stream_stability-v1.json").is_file()


def test_stream_image_critic_non_linear_context_coupling():
    """Verify StreamImageCritic branches between linear and non-linear context architectures."""
    # Context-free critic uses single fc
    d_uncond = StreamImageCritic(width=16, context_dim=0)
    assert hasattr(d_uncond, "fc")
    assert not hasattr(d_uncond, "fc1")
    assert not hasattr(d_uncond, "fc2")
    x = torch.randn(2, 1, 8, 8)
    out_uncond = d_uncond(x)
    assert out_uncond.shape == (2, 1)

    # Conditional critic uses joint non-linear mixing (fc1, fc2)
    d_cond = StreamImageCritic(width=16, context_dim=16)
    assert hasattr(d_cond, "fc1")
    assert hasattr(d_cond, "fc2")
    assert not hasattr(d_cond, "fc")

    ctx_a = torch.ones(2, 16)
    ctx_b = -torch.ones(2, 16)

    # Input gradient w.r.t x must be coupled to context vector c
    x_var_a = torch.randn(1, 1, 8, 8, requires_grad=True)
    out_a = d_cond(x_var_a, context=ctx_a[:1])
    out_a.backward()
    grad_a = x_var_a.grad.clone()

    x_var_b = x_var_a.detach().clone().requires_grad_(True)
    out_b = d_cond(x_var_b, context=ctx_b[:1])
    out_b.backward()
    grad_b = x_var_b.grad.clone()

    # With non-linear mixing, grad w.r.t x depends on context c
    assert not torch.allclose(grad_a, grad_b, atol=1e-5), "Gradient must be conditioned on context"


def test_fast_weight_memory_temporal_velocity_and_gating():
    """Verify FastWeightMemory computes directional velocity and clamps static noise."""
    mem = FastWeightMemory(dim=16, reservoir_size=64)
    assert mem.reservoir_size == 64
    assert mem.reservoir.shape == (64, 1, 8, 8)

    positions, _ = translating_bar_cycle()

    # Rightward transition: pos 0 -> pos 1
    mem.reset_state()
    mem.write(positions[0:1])
    ctx_r = mem.write(positions[1:2])
    assert ctx_r.norm() > 0

    # Leftward transition: pos 2 -> pos 1
    mem.reset_state()
    mem.write(positions[2:3])
    ctx_l = mem.write(positions[1:2])
    assert ctx_l.norm() > 0

    # The two directions must produce well-separated context vectors
    diff_norm = (ctx_r - ctx_l).norm().item()
    assert diff_norm > 1.0, f"Expected well-separated context vectors, got diff norm {diff_norm}"
    cos_sim = torch.nn.functional.cosine_similarity(ctx_r, ctx_l, dim=-1).item()
    assert cos_sim < 0.90, f"Context vectors must not be collinear: {cos_sim}"

    # Static transition: same frame twice with tiny sensor noise
    mem.reset_state()
    f0 = positions[0:1]
    mem.write(f0)
    f0_noisy = f0 + 0.005 * torch.randn_like(f0)
    ctx_static = mem.write(f0_noisy)
    assert torch.equal(ctx_static, torch.zeros_like(ctx_static)), "Static frame transition must clamp context to 0"


def test_fast_weight_bcap_ns_candidate_spec():
    """Verify fast-weight-bcap-ns specification in TIER1_CANDIDATES."""
    assert "fast-weight-bcap-ns" in TIER1_CANDIDATES
    c = TIER1_CANDIDATES["fast-weight-bcap-ns"]
    assert c.recipe_preset == "bcap"
    assert c.recipe_overrides.get("loss") == "non_saturating"
    assert c.recipe_overrides.get("lr") == 0.005
    assert c.memory_policy.get("reservoir_size") == 64
    assert c.parent == "fast-weight-bcap"
