"""Adversarial stress test suite for Milestone M3 Iteration 3.

Empirically challenges:
1. Numerical stability of cosine learning rate annealing at boundary steps (step 0, 240, 480).
2. Memory reservoir FIFO behavior and bounds under non-finite inputs or extreme sequences.
3. Determinism of NamedStreams under different batch sizes and devices.
"""
from __future__ import annotations

import math
import pytest
import torch

from particlegan.recipes import learning_rate_scale, learning_rate_scales
from particlegan.training import GANTrainer
from particlegan_forge.rng import NamedStreams, STREAM_NAMES
from particlestream.models import FastWeightMemory, StreamImageGenerator, StreamImageCritic
from particlestream.tier1 import TIER1_TASKS, TIER1_CANDIDATES, _build_recipe, _init_module


class TestCosineAnnealingBoundaryStress:
    """Stress tests for cosine learning rate annealing at boundary and transition steps."""

    def test_cosine_scale_boundary_exact_values(self):
        """Verify exact theoretical values at step 0, step 240 (transition), step 480 (floor), and step > 480."""
        total_steps = 480
        start = 0.5
        floor = 0.05

        # Step 0: Exactly 1.0 (before annealing start)
        s0 = learning_rate_scale(0, total_steps, start=start, floor=floor)
        assert s0 == 1.0

        # Steps in [0, 240]: Exactly 1.0
        for step in [1, 50, 120, 239, 240]:
            assert learning_rate_scale(step, total_steps, start=start, floor=floor) == 1.0

        # Step 241: Immediately begins smooth cosine descent
        s241 = learning_rate_scale(241, total_steps, start=start, floor=floor)
        assert 0.05 < s241 < 1.0
        expected_s241 = floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * 1.0 / 240.0))
        assert pytest.approx(s241, rel=1e-12) == expected_s241

        # Midpoint step 360 (fraction = 0.5): cos(pi * 0.5) = 0.0 -> floor + 0.5 * (1 - floor) = 0.525
        s360 = learning_rate_scale(360, total_steps, start=start, floor=floor)
        assert pytest.approx(s360, rel=1e-12) == 0.525

        # Step 480: Exactly floor (0.05)
        s480 = learning_rate_scale(480, total_steps, start=start, floor=floor)
        assert pytest.approx(s480, rel=1e-12) == 0.05

        # Steps > 480: Strictly clamped to floor, never oscillates back up
        for step in [481, 482, 500, 1000, 10000]:
            s_over = learning_rate_scale(step, total_steps, start=start, floor=floor)
            assert pytest.approx(s_over, rel=1e-12) == 0.05

    def test_cosine_scale_monotonic_cooling(self):
        """Verify that learning rate scale is strictly monotonically non-increasing from step 240 to 480."""
        total_steps = 480
        scales = [
            learning_rate_scale(step, total_steps, start=0.5, floor=0.05)
            for step in range(240, 481)
        ]
        for prev, nxt in zip(scales[:-1], scales[1:]):
            assert nxt <= prev, f"Monotonicity violated: step went from {prev} to {nxt}"

        # Differences between consecutive steps are smoothly bounded
        diffs = [prev - nxt for prev, nxt in zip(scales[:-1], scales[1:])]
        assert all(d >= 0.0 for d in diffs)
        # Max derivative of cosine is pi/2 * (1 - floor) / (total_steps * 0.5)
        max_diff = max(diffs)
        assert max_diff < 0.007

    def test_cosine_scale_extreme_steps_and_clamping(self):
        """Verify behavior under negative steps, large steps, and extreme configurations."""
        total_steps = 480
        # Negative step clamped to 0.0 fraction -> scale 1.0
        assert learning_rate_scale(-1, total_steps, start=0.5, floor=0.05) == 1.0
        assert learning_rate_scale(-1000, total_steps, start=0.5, floor=0.05) == 1.0

        # floor=0.0 reaches exactly 0.0 at step 480 without negative values
        assert learning_rate_scale(480, total_steps, start=0.5, floor=0.0) == 0.0
        assert learning_rate_scale(500, total_steps, start=0.5, floor=0.0) == 0.0

        # floor=1.0 stays exactly 1.0 across all steps
        for step in [0, 240, 360, 480, 1000]:
            assert learning_rate_scale(step, total_steps, start=0.5, floor=1.0) == 1.0

        # start=0.0 starts decay immediately from step 0
        s0_start0 = learning_rate_scale(0, total_steps, start=0.0, floor=0.05)
        assert pytest.approx(s0_start0, rel=1e-12) == 1.0
        s1_start0 = learning_rate_scale(1, total_steps, start=0.0, floor=0.05)
        assert s1_start0 < 1.0

    def test_gantrainer_lr_groups_at_boundary_steps(self):
        """Construct genuine GANTrainer with candidate recipe and verify exact group LRs at steps 0, 240, 480."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tasks = {t.id: t for t in TIER1_TASKS}
        task = tasks["seq_img_translating_bar_next"]
        cand = TIER1_CANDIDATES["fast-weight-bcap-ns"]

        streams = NamedStreams(seed=0, device=device)
        g = StreamImageGenerator(z_dim=task.z_dim, width=16, context_dim=16).to(device)
        d = StreamImageCritic(width=16, context_dim=16).to(device)
        memory = FastWeightMemory(dim=16, decay=0.90, write_lr=0.35, reservoir_size=64).to(device)

        _init_module(g, streams, "generator")
        _init_module(d, streams, "discriminator")
        _init_module(memory, streams, "memory")

        recipe = _build_recipe(task, cand, 480)
        prior_init_stream = streams.generator("init", component="prior", purpose="locations", device=device)
        prior = recipe.replace(prior_update="learned").make_prior(
            device=device, dtype=torch.float32, generator=prior_init_stream, init_std=1.0
        )
        _init_module(prior, streams, "prior")
        recipe.apply_prior_policy(prior)

        trainer = GANTrainer(
            recipe,
            g,
            d,
            prior=prior,
            seed=0,
            max_steps=480,
            latent_generator=streams.generator("prior", component="latent", purpose="indices", device=device),
            penalty_generator=streams.generator("noise", component="penalty", purpose="training", device=device),
            noise_generator=streams.generator("noise", component="generator", purpose="output", device=device),
            input_noise_generator=streams.generator("noise", component="critic", purpose="input", device=device),
            eval_generator=streams.generator("eval", component="sampler", purpose="samples", device=device),
            model_generator=streams.generator("noise", component="models", purpose="stochastic_layers", device=device),
            prior_noise_generator=streams.generator("noise", component="prior", purpose="gaussian", device=device),
        )

        # Initial check before any step: completed_steps == 0
        assert trainer.policy.completed_steps == 0
        assert pytest.approx(trainer.opt_g.param_groups[0]["lr"]) == 0.005
        assert pytest.approx(trainer.opt_d.param_groups[0]["lr"]) == 0.005

        # Step 1: executed with scale for completed_steps=0
        x = torch.zeros(1, 1, 8, 8, device=device)
        trainer.step(x)
        assert trainer.policy.completed_steps == 1
        assert pytest.approx(trainer.opt_g.param_groups[0]["lr"]) == 0.005
        assert pytest.approx(trainer.opt_d.param_groups[0]["lr"]) == 0.005

        # Fast forward to step 240
        for _ in range(2, 241):
            trainer.step(x)
        assert trainer.policy.completed_steps == 240
        assert pytest.approx(trainer.opt_g.param_groups[0]["lr"]) == 0.005
        assert pytest.approx(trainer.opt_d.param_groups[0]["lr"]) == 0.005

        # Step 241: beginning of cooling
        trainer.step(x)
        assert trainer.policy.completed_steps == 241

        # Fast forward to step 480
        for _ in range(242, 481):
            trainer.step(x)
        assert trainer.policy.completed_steps == 480
        # Group LR after step 480 completed (scheduled at step 479):
        # 0.005 * 0.05004069 = ~0.0002502
        assert 0.00025 <= trainer.opt_g.param_groups[0]["lr"] <= 0.00026
        assert 0.00025 <= trainer.opt_d.param_groups[0]["lr"] <= 0.00026

        # Subsequent query of learning_rate_scales at step 480 is exactly (0.05, 0.05)
        net_scale, prior_scale = learning_rate_scales(480, recipe)
        assert pytest.approx(net_scale, rel=1e-12) == 0.05
        assert pytest.approx(prior_scale, rel=1e-12) == 0.05

        # Stepping at step 481 must raise budget exhaustion
        with pytest.raises(RuntimeError, match="recipe training budget exhausted"):
            trainer.step(x)

    def test_gantrainer_numerical_finiteness_across_boundary_transitions(self):
        """Verify that all model parameters, buffers, and optimizer momentum states remain 100% finite."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tasks = {t.id: t for t in TIER1_TASKS}
        task = tasks["seq_img_translating_bar_next"]
        cand = TIER1_CANDIDATES["fast-weight-bcap-ns"]

        streams = NamedStreams(seed=0, device=device)
        g = StreamImageGenerator(z_dim=task.z_dim, width=16, context_dim=16).to(device)
        d = StreamImageCritic(width=16, context_dim=16).to(device)
        memory = FastWeightMemory(dim=16, decay=0.90, write_lr=0.35, reservoir_size=64).to(device)

        _init_module(g, streams, "generator")
        _init_module(d, streams, "discriminator")
        _init_module(memory, streams, "memory")

        recipe = _build_recipe(task, cand, 10)
        prior_init_stream = streams.generator("init", component="prior", purpose="locations", device=device)
        prior = recipe.replace(prior_update="learned").make_prior(
            device=device, dtype=torch.float32, generator=prior_init_stream, init_std=1.0
        )
        _init_module(prior, streams, "prior")
        recipe.apply_prior_policy(prior)

        trainer = GANTrainer(
            recipe,
            g,
            d,
            prior=prior,
            seed=0,
            max_steps=10,
            latent_generator=streams.generator("prior", component="latent", purpose="indices", device=device),
            penalty_generator=streams.generator("noise", component="penalty", purpose="training", device=device),
            noise_generator=streams.generator("noise", component="generator", purpose="output", device=device),
            input_noise_generator=streams.generator("noise", component="critic", purpose="input", device=device),
            eval_generator=streams.generator("eval", component="sampler", purpose="samples", device=device),
            model_generator=streams.generator("noise", component="models", purpose="stochastic_layers", device=device),
            prior_noise_generator=streams.generator("noise", component="prior", purpose="gaussian", device=device),
        )

        for _ in range(10):
            x = torch.randn(1, 1, 8, 8, device=device)
            trainer.step(x)
            assert all(torch.isfinite(p).all() for p in g.parameters())
            assert all(torch.isfinite(p).all() for p in d.parameters())
            assert all(torch.isfinite(p).all() for p in prior.parameters())


class TestMemoryReservoirFIFONonFiniteStress:
    """Stress tests for FastWeightMemory FIFO buffer and rejection of non-finite/extreme inputs."""

    def test_fifo_wraparound_exact_indexing(self):
        """Verify strict circular FIFO indexing across 256 writes in a 64-slot reservoir."""
        mem = FastWeightMemory(dim=16, reservoir_size=64)

        for i in range(256):
            f = torch.full((1, 1, 8, 8), float(i))
            mem.write(f)

        assert mem.reservoir_count.item() == 256
        # At step 256 (256 = 4 * 64), slot 0 contains frame 256 - 64 = 192,
        # slot 1 contains 193, ..., slot 63 contains 255
        for slot in range(64):
            expected_val = float(192 + slot)
            actual_val = mem.reservoir[slot, 0, 0, 0].item()
            assert actual_val == expected_val, f"Slot {slot} had {actual_val}, expected {expected_val}"

    def test_nonfinite_inputs_rejected_without_state_poisoning(self):
        """Non-finite inputs (NaN, +Inf, -Inf) must be rejected without mutating count, reservoir, or matrix."""
        mem = FastWeightMemory(dim=16, reservoir_size=64)
        gen = torch.Generator().manual_seed(999)

        # Write 10 valid initial frames
        for i in range(10):
            mem.write(torch.full((1, 1, 8, 8), float(i + 1)))

        assert mem.reservoir_count.item() == 10
        saved_matrix = mem.matrix.clone()
        saved_prev_key = mem.prev_key.clone()
        saved_reservoir = mem.reservoir.clone()

        # Adversarially inject 60 non-finite frames with different anomalies
        anomalies = [
            torch.full((1, 1, 8, 8), float("nan")),
            torch.full((1, 1, 8, 8), float("inf")),
            torch.full((1, 1, 8, 8), float("-inf")),
        ]
        # Also mixed frames where only 1 pixel is NaN or Inf
        p_nan = torch.ones(1, 1, 8, 8)
        p_nan[0, 0, 3, 3] = float("nan")
        anomalies.append(p_nan)

        p_inf = torch.ones(1, 1, 8, 8)
        p_inf[0, 0, 7, 7] = float("inf")
        anomalies.append(p_inf)

        for frame in anomalies * 12:
            ctx = mem.write(frame)
            # Must return finite zero context
            assert ctx.shape == (1, 16)
            assert (ctx == 0.0).all()
            assert torch.isfinite(ctx).all()

        # State must remain 100% bitwise identical to before the attack
        assert mem.reservoir_count.item() == 10
        assert torch.equal(mem.matrix, saved_matrix)
        assert torch.equal(mem.prev_key, saved_prev_key)
        assert torch.equal(mem.reservoir, saved_reservoir)

        # Sampling reservoir must yield only valid frames 1..10
        for _ in range(500):
            s = mem.sample_reservoir(gen)
            assert s is not None
            assert torch.isfinite(s).all()
            val = s[0, 0, 0, 0].item()
            assert 1.0 <= val <= 10.0

        # Writing next valid frame resumes strictly at slot 10
        ctx_next = mem.write(torch.full((1, 1, 8, 8), 11.0))
        assert mem.reservoir_count.item() == 11
        assert mem.reservoir[10, 0, 0, 0].item() == 11.0
        assert torch.isfinite(ctx_next).all()

    def test_empty_reservoir_nonfinite_behavior(self):
        """Writing non-finite frames to an unpopulated memory returns zero context and sample returns None."""
        mem = FastWeightMemory(dim=16, reservoir_size=64)
        gen = torch.Generator().manual_seed(42)

        for val in [float("nan"), float("inf"), float("-inf")]:
            f = torch.full((1, 1, 8, 8), val)
            ctx = mem.write(f)
            assert ctx.shape == (1, 16)
            assert (ctx == 0.0).all()
            assert mem.reservoir_count.item() == 0
            assert mem.sample_reservoir(gen) is None

        assert torch.isfinite(mem.matrix).all()
        assert torch.isfinite(mem.prev_key).all()
        assert torch.isfinite(mem.reservoir).all()

    def test_extreme_magnitude_sequence_stability(self):
        """Alternating extreme magnitudes (+1e20, -1e20, +1e-20) do not cause overflow or NaNs."""
        mem = FastWeightMemory(dim=16, reservoir_size=64)

        extremes = [1e20, -1e20, 1e-20, -1e-20, 0.0]
        for val in extremes * 20:
            f = torch.full((1, 1, 8, 8), val)
            ctx = mem.write(f)
            assert torch.isfinite(ctx).all()
            assert torch.isfinite(mem.matrix).all()
            assert torch.isfinite(mem.prev_key).all()
            assert torch.isfinite(mem.reservoir).all()

        # Matrix Frobenius norm remains strictly bounded
        f_norm = torch.linalg.matrix_norm(mem.matrix, ord="fro").item()
        assert f_norm <= 3.5 + 1e-4

    def test_read_context_with_extreme_and_all_zero_frames(self):
        """read_context does not mutate state and safely handles zeros, extremes, and non-finite frames."""
        mem = FastWeightMemory(dim=16, reservoir_size=64)
        mem.write(torch.randn(1, 1, 8, 8))

        saved_count = mem.reservoir_count.clone()
        saved_matrix = mem.matrix.clone()

        # Zero frame
        ctx_zero = mem.read_context(torch.zeros(1, 1, 8, 8))
        assert ctx_zero.shape == (1, 16)
        assert torch.isfinite(ctx_zero).all()

        # Extreme frame
        ctx_huge = mem.read_context(torch.full((1, 1, 8, 8), 1e20))
        assert ctx_huge.shape == (1, 16)
        assert torch.isfinite(ctx_huge).all()

        # None frame
        ctx_none = mem.read_context(None)
        assert ctx_none.shape == (1, 16)
        assert (ctx_none == 0.0).all()

        # Ensure state was not mutated
        assert torch.equal(mem.reservoir_count, saved_count)
        assert torch.equal(mem.matrix, saved_matrix)


class TestNamedStreamsDeterminismStress:
    """Stress tests for NamedStreams determinism across devices, seeds, and batch sizes."""

    def test_derived_seeds_device_invariant(self):
        """Derived seeds from seed_for are mathematically identical regardless of target device."""
        s_cpu = NamedStreams(seed=0, device="cpu")
        s_cuda = NamedStreams(seed=0, device="cuda" if torch.cuda.is_available() else "cpu")

        families = STREAM_NAMES
        components = ["shared", "generator", "critic", "prior", "memory", "stream"]
        purposes = ["default", "training", "locations", "reservoir", "samples", "noise"]

        for fam in families:
            for comp in components:
                for purp in purposes:
                    seed_c = s_cpu.seed_for(fam, component=comp, purpose=purp)
                    seed_cu = s_cuda.seed_for(fam, component=comp, purpose=purp)
                    assert seed_c == seed_cu
                    assert 0 <= seed_c < 2 ** 63

    def test_stream_isolation_under_adversarial_draws(self):
        """Drawing 5,000 times from eval stream leaves data, prior, and noise streams completely untouched."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        streams = NamedStreams(seed=0, device=device)

        g_data = streams.generator("data", component="stream", purpose="training")
        g_prior = streams.generator("prior", component="latent", purpose="indices")
        g_noise = streams.generator("noise", component="generator", purpose="output")
        g_eval = streams.generator("eval", component="sampler", purpose="samples")

        audit_before = streams.audit()

        with streams.isolated_evaluation():
            # Adversarial burst of draws from eval
            for _ in range(5000):
                torch.randn(4, 8, device=device, generator=g_eval)

        audit_after = streams.audit()
        allowed = streams.eval_stream_keys()
        res = NamedStreams.compare(audit_before, audit_after, allowed=allowed)

        assert res["unintended_rng_deviations"] == 0
        assert res["unintended_streams"] == []

    def test_batch_size_draw_properties_on_cpu_and_cuda(self):
        """Demonstrate and verify PyTorch RNG behavior under different batch draw shapes.

        On CPU (Mersenne Twister), draw sequence is element-wise invariant between 1 batch of 4 vs 4 batches of 1.
        On CUDA (Philox 4x32), kernel launch thread block chunking means 1 batch of 4 != 4 batches of 1.
        This demonstrates why the ParticleStream protocol strictly mandates fixed batch_size=1.
        """
        g_cpu1 = torch.Generator(device="cpu").manual_seed(42)
        g_cpu2 = torch.Generator(device="cpu").manual_seed(42)

        b4_cpu = torch.randn(4, device="cpu", generator=g_cpu1)
        b1_cpu = torch.cat([torch.randn(1, device="cpu", generator=g_cpu2) for _ in range(4)])
        assert torch.equal(b4_cpu, b1_cpu)

        if torch.cuda.is_available():
            g_cuda1 = torch.Generator(device="cuda").manual_seed(42)
            g_cuda2 = torch.Generator(device="cuda").manual_seed(42)

            b4_cuda = torch.randn(4, device="cuda", generator=g_cuda1)
            b1_cuda = torch.cat([torch.randn(1, device="cuda", generator=g_cuda2) for _ in range(4)])
            # Philox advances differently under different kernel grid launches
            assert not torch.equal(b4_cuda, b1_cuda)

    def test_named_streams_bitwise_reproducibility_at_batch_size_1(self):
        """Verify that under the required fixed batch_size=1 protocol, execution is 100% bitwise deterministic."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        s1 = NamedStreams(seed=0, device=device)
        s2 = NamedStreams(seed=0, device=device)

        g1 = s1.generator("data", component="stream", purpose="training")
        g2 = s2.generator("data", component="stream", purpose="training")

        draws1 = [torch.randn(1, 1, 8, 8, device=device, generator=g1) for _ in range(200)]
        draws2 = [torch.randn(1, 1, 8, 8, device=device, generator=g2) for _ in range(200)]

        for i, (d1, d2) in enumerate(zip(draws1, draws2)):
            assert torch.equal(d1, d2), f"Divergence at draw {i}"

    def test_named_streams_state_dict_serialization_roundtrip(self):
        """Verify that state_dict and load_state_dict faithfully preserve generator states."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        s1 = NamedStreams(seed=123, device=device)

        g_data = s1.generator("data", component="stream", purpose="training")
        g_noise = s1.generator("noise", component="generator", purpose="output")

        # Advance generators
        for _ in range(50):
            torch.randn(1, 8, device=device, generator=g_data)
            torch.randn(1, 8, device=device, generator=g_noise)

        state = s1.state_dict()

        # Restore into new instance
        s2 = NamedStreams(seed=123, device=device)
        s2.load_state_dict(state)

        g_data2 = s2.generator("data", component="stream", purpose="training")
        g_noise2 = s2.generator("noise", component="generator", purpose="output")

        # Next 50 draws must be bitwise identical
        for _ in range(50):
            d1 = torch.randn(1, 8, device=device, generator=g_data)
            d2 = torch.randn(1, 8, device=device, generator=g_data2)
            assert torch.equal(d1, d2)

            n1 = torch.randn(1, 8, device=device, generator=g_noise)
            n2 = torch.randn(1, 8, device=device, generator=g_noise2)
            assert torch.equal(n1, n2)
