"""Adversarial stress test harness for Milestone M3.

Verifies edge cases, degenerate tensors, zero context, extreme velocities,
and batch size constraints for FastWeightMemory and StreamImageCritic.
"""
from __future__ import annotations

import pytest
import torch

from particlestream.models import FastWeightMemory, StreamImageCritic


class TestFastWeightMemoryAdversarial:
    """Adversarial tests for FastWeightMemory."""

    def test_batch_size_single_vs_multi_and_empty(self):
        """FastWeightMemory write handles B=1 and B=0 safely; B>1 raises dimensional constraint."""
        mem = FastWeightMemory(dim=16)

        # Batch size 1 succeeds
        f1 = torch.randn(1, 1, 8, 8)
        ctx1 = mem.write(f1)
        assert ctx1.shape == (1, 16)

        # Batch size 0 empty frame returns (0, 16) without error
        f0 = torch.randn(0, 1, 8, 8)
        ctx0 = mem.write(f0)
        assert ctx0.shape == (0, 16)

        # Batch size 4 raises dimensional constraint in assoc = v.t() @ self.prev_key
        f4 = torch.randn(4, 1, 8, 8)
        with pytest.raises(RuntimeError, match="mat1 and mat2 shapes cannot be multiplied"):
            mem.write(f4)

    def test_read_context_dynamic_batch_shape_consistency(self):
        """read_context returns strictly (B, dim) for any B >= 0, never collapsing to (1, dim)."""
        mem = FastWeightMemory(dim=16)
        mem.write(torch.randn(1, 1, 8, 8))

        # Distinct frames -> diff_norm > 0.35 -> returns (4, 16)
        f_diff = torch.randn(4, 1, 8, 8)
        ctx_diff = mem.read_context(f_diff)
        assert ctx_diff.shape == (4, 16)

        # Re-using reservoir frame -> diff_norm <= 0.35 -> strictly returns (4, 16) zeros
        f_same = mem.reservoir[0:1].expand(4, -1, -1, -1)
        ctx_same = mem.read_context(f_same)
        assert ctx_same.shape == (4, 16)
        assert (ctx_same == 0.0).all()

        # Empty frame B=0 -> returns (0, 16)
        f_empty = torch.randn(0, 1, 8, 8)
        ctx_empty = mem.read_context(f_empty)
        assert ctx_empty.shape == (0, 16)

        # None frame -> returns default (1, 16)
        ctx_none = mem.read_context(None)
        assert ctx_none.shape == (1, 16)

    def test_nan_and_nonfinite_poisoning_guarded(self):
        """Non-finite frames (NaN, Inf) are guarded and do not corrupt memory state."""
        mem = FastWeightMemory(dim=16)
        mem.write(torch.randn(1, 1, 8, 8))
        assert torch.isfinite(mem.matrix).all()

        # Write NaN frame
        nan_frame = torch.full((1, 1, 8, 8), float("nan"))
        ctx_nan = mem.write(nan_frame)
        assert ctx_nan.shape == (1, 16)
        assert (ctx_nan == 0.0).all()
        assert torch.isfinite(mem.matrix).all()

        # Write Inf frame
        inf_frame = torch.full((1, 1, 8, 8), float("inf"))
        ctx_inf = mem.write(inf_frame)
        assert ctx_inf.shape == (1, 16)
        assert (ctx_inf == 0.0).all()
        assert torch.isfinite(mem.matrix).all()

        # Write clean frame afterwards operates normally
        ctx_clean = mem.write(torch.randn(1, 1, 8, 8))
        assert torch.isfinite(mem.matrix).all()
        assert torch.isfinite(ctx_clean).all()

    def test_extreme_velocities_and_threshold_discontinuity(self):
        """Extreme frame inversion norm is bounded at 3.0; threshold has step discontinuity."""
        mem = FastWeightMemory(dim=16)
        f0 = torch.randn(1, 1, 8, 8)
        mem.write(f0)

        # Opposite direction frame: k - prev_key has norm ~ 2.0 (maximum)
        f_inv = -f0
        ctx_inv = mem.write(f_inv)
        assert pytest.approx(ctx_inv.norm().item(), abs=1e-4) == 3.0

        # Identical frame: diff == 0 -> context is exactly 0.0
        ctx_same = mem.write(f_inv)
        assert ctx_same.norm().item() == 0.0

    def test_sample_reservoir_detachment_and_bounds(self):
        """Reservoir returns None when empty, and detached clone when populated."""
        mem = FastWeightMemory(dim=16, reservoir_size=64)
        gen = torch.Generator().manual_seed(0)
        assert mem.sample_reservoir(gen) is None

        for _ in range(10):
            mem.write(torch.randn(1, 1, 8, 8))

        sample = mem.sample_reservoir(gen)
        assert sample is not None
        assert sample.shape == (1, 1, 8, 8)
        assert not sample.requires_grad

    def test_reservoir_wraparound_fifo_and_uniform_sampling(self):
        """Reservoir wraps around cleanly across 200 writes, preserving latest 64 frames uniformly sampled."""
        mem = FastWeightMemory(dim=16, reservoir_size=64)
        gen = torch.Generator().manual_seed(42)

        for i in range(200):
            frame = torch.full((1, 1, 8, 8), float(i))
            mem.write(frame)

        assert mem.reservoir_count.item() == 200
        # The 64 slots must contain exactly frames 136 through 199
        items = set(mem.reservoir[:, 0, 0, 0].int().tolist())
        assert items == set(range(136, 200))

        # Test uniform sampling distribution across 64 slots over 6400 draws
        counts = {val: 0 for val in range(136, 200)}
        for _ in range(6400):
            s = mem.sample_reservoir(gen)
            assert s is not None
            val = int(s[0, 0, 0, 0].item())
            counts[val] += 1
        assert min(counts.values()) > 50

    def test_memory_decay_stability_1000_steps(self):
        """Frobenius norm of memory matrix remains strictly bounded by 3.5 over 1000 steps."""
        mem = FastWeightMemory(dim=16, decay=0.90, write_lr=0.35)

        # 1000 random steps
        for _ in range(1000):
            mem.write(torch.randn(1, 1, 8, 8))
            f_norm = torch.linalg.matrix_norm(mem.matrix, ord="fro").item()
            assert f_norm <= 3.5 + 1e-4

        # Constant frame convergence
        mem.reset_state()
        c_frame = torch.randn(1, 1, 8, 8)
        for _ in range(500):
            mem.write(c_frame)
        final_norm = torch.linalg.matrix_norm(mem.matrix, ord="fro").item()
        assert pytest.approx(final_norm, abs=1e-3) == 3.5

    def test_read_context_mixed_nan_and_finite_batch(self):
        """read_context on mixed batch containing NaN and Inf safely zeros corrupted rows and stays finite."""
        mem = FastWeightMemory(dim=16)
        mem.write(torch.randn(1, 1, 8, 8))

        f = torch.randn(4, 1, 8, 8)
        f[1].fill_(float("nan"))
        f[2].fill_(float("inf"))

        ctx = mem.read_context(f)
        assert ctx.shape == (4, 16)
        assert torch.isfinite(ctx).all()
        assert (ctx[1] == 0.0).all()
        assert (ctx[2] == 0.0).all()


class TestStreamImageCriticAdversarial:
    """Adversarial tests for StreamImageCritic."""

    def test_batch_sizes_and_empty_batch_handling(self):
        """Critic safely handles B=1, B=4, and empty batch B=0 without crashing."""
        critic = StreamImageCritic(width=16, context_dim=16)

        out1 = critic(torch.randn(1, 1, 8, 8))
        assert out1.shape == (1, 1)

        out4 = critic(torch.randn(4, 1, 8, 8))
        assert out4.shape == (4, 1)

        # B=0 empty batch returns (0, 1) safely
        out0 = critic(torch.randn(0, 1, 8, 8))
        assert out0.shape == (0, 1)

    def test_context_broadcasting_when_batch_size_mismatched(self):
        """Critic broadcasts context of shape (1, dim) when x has B > 1."""
        critic = StreamImageCritic(width=16, context_dim=16)
        x4 = torch.randn(4, 1, 8, 8)
        ctx1 = torch.randn(1, 16)

        out = critic(x4, context=ctx1)
        assert out.shape == (4, 1)
        assert torch.isfinite(out).all()

    def test_zero_context_and_context_dim_switching(self):
        """context_dim=0 uses single fc layer; context_dim>0 uses non-linear fc1/fc2 coupling."""
        critic_zero = StreamImageCritic(width=16, context_dim=0)
        assert hasattr(critic_zero, "fc")
        assert not hasattr(critic_zero, "fc1")

        critic_coupled = StreamImageCritic(width=16, context_dim=16)
        assert hasattr(critic_coupled, "fc1")
        assert hasattr(critic_coupled, "fc2")
        assert not hasattr(critic_coupled, "fc")

        # Zero context input operates safely
        out_zero = critic_coupled(torch.randn(2, 1, 8, 8), context=torch.zeros(2, 16))
        assert out_zero.shape == (2, 1)
        assert torch.isfinite(out_zero).all()

    def test_gradient_isolation_and_detachment(self):
        """Critic detaches context, preventing backward gradients into memory states."""
        critic = StreamImageCritic(width=16, context_dim=16)
        x = torch.randn(2, 1, 8, 8, requires_grad=True)
        ctx = torch.randn(2, 16, requires_grad=True)

        out = critic(x, context=ctx)
        out.sum().backward()

        assert x.grad is not None
        assert ctx.grad is None  # ctx is explicitly detached in forward

    def test_extreme_context_and_spatial_mismatches(self):
        """Extreme context produces finite logits; wrong spatial resolutions crash."""
        critic = StreamImageCritic(width=16, context_dim=16)

        # Extreme magnitude context (1e6) produces finite logits
        out_huge = critic(torch.randn(1, 1, 8, 8), context=torch.full((1, 16), 1e6))
        assert torch.isfinite(out_huge).all()

        # Spatial resolution mismatch (16x16 instead of 8x8) crashes fc1
        with pytest.raises(RuntimeError, match="mat1 and mat2 shapes cannot be multiplied"):
            critic(torch.randn(1, 1, 16, 16))

        # Channel count mismatch (3 channels instead of 1) crashes conv1
        with pytest.raises(RuntimeError, match="expected input.*to have 1 channels"):
            critic(torch.randn(1, 3, 8, 8))

    def test_chained_memory_read_context_and_critic_success(self):
        """Chaining read_context on static batch with critic operates seamlessly without error."""
        mem = FastWeightMemory(dim=16)
        critic = StreamImageCritic(width=16, context_dim=16)

        # Setup memory state
        mem.write(torch.randn(1, 1, 8, 8))

        # Batch of 4 identical frames
        x_batch = mem.reservoir[0:1].expand(4, -1, -1, -1)
        # read_context returns (4, 16)
        ctx = mem.read_context(x_batch)
        assert ctx.shape == (4, 16)

        # Feeding ctx into critic(x_batch, context=ctx) succeeds cleanly
        out = critic(x_batch, context=ctx)
        assert out.shape == (4, 1)
        assert torch.isfinite(out).all()

    def test_dynamic_context_pairing_mismatched_batch_raises(self):
        """Critic raises RuntimeError when x and context have mismatched non-unit batch dimensions."""
        critic = StreamImageCritic(width=16, context_dim=16)
        x = torch.randn(4, 1, 8, 8)
        ctx_mismatched = torch.randn(2, 16)
        with pytest.raises(RuntimeError, match="Sizes of tensors must match except in dimension 1"):
            critic(x, context=ctx_mismatched)

    def test_critic_out_of_distribution_frame_values(self):
        """Critic handles large positive and negative out-of-distribution inputs without crashing."""
        critic = StreamImageCritic(width=16, context_dim=16)

        x_pos = torch.full((2, 1, 8, 8), 1e4)
        out_pos = critic(x_pos)
        assert out_pos.shape == (2, 1)
        assert torch.isfinite(out_pos).all()

        x_neg = torch.full((2, 1, 8, 8), -1e4)
        out_neg = critic(x_neg)
        assert out_neg.shape == (2, 1)
        assert torch.isfinite(out_neg).all()
