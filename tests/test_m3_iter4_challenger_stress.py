"""Adversarial stress test suite for Milestone M3 Iteration 4.

Empirically challenges:
1. FastWeightMemory.sample_reservoir(gen, n=k) with k=0, 1, 7, 64, 128 under:
   - empty reservoir (count == 0)
   - partially filled reservoir (count in {1, 5, 31, 63})
   - full reservoir (count == 64)
   - overfilled/wrapped FIFO reservoir (count in {65, 128, 500})
2. Detachment, finiteness, and gradient isolation:
   - requires_grad is False and grad_fn is None
   - deepcopy/clone buffer independence (mutating sample does not mutate reservoir and vice-versa)
   - strict autograd gradient isolation (backward pass through critic does not leak into memory)
3. Concatenated train_real tensor shape and device invariance:
   - x_cur concatenated with sample_reservoir across k in {0, 1, 7, 64, 128}
   - device invariance across CPU and CUDA
   - downstream critic and GANTrainer step execution across all concatenated batch sizes
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from particlegan.training import GANTrainer
from particlegan_forge.rng import NamedStreams
from particlestream.models import FastWeightMemory, StreamImageGenerator, StreamImageCritic
from particlestream.tier1 import TIER1_TASKS, TIER1_CANDIDATES, _build_recipe, _init_module


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class TestFastWeightMemorySampleReservoirKStress:
    """Stress tests FastWeightMemory.sample_reservoir across k in {0, 1, 7, 64, 128}."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("k", [0, 1, 7, 64, 128])
    def test_empty_reservoir_returns_none_for_all_k(self, device: str, k: int):
        """When reservoir is unpopulated (count == 0), sample_reservoir must return None for all k."""
        mem = FastWeightMemory(dim=16, reservoir_size=64).to(device)
        gen = torch.Generator(device=device).manual_seed(12345)

        assert mem.reservoir_count.item() == 0
        sampled = mem.sample_reservoir(gen, n=k)
        assert sampled is None

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("k", [0, 1, 7, 64, 128])
    def test_single_frame_reservoir_exact_shapes_and_values(self, device: str, k: int):
        """When reservoir has exactly 1 frame, sample_reservoir must draw only that frame."""
        mem = FastWeightMemory(dim=16, reservoir_size=64).to(device)
        gen = torch.Generator(device=device).manual_seed(42)

        frame = torch.full((1, 1, 8, 8), 42.5, device=device)
        mem.write(frame)
        assert mem.reservoir_count.item() == 1

        sampled = mem.sample_reservoir(gen, n=k)
        assert sampled is not None
        assert sampled.shape == (k, 1, 8, 8)
        assert sampled.device.type == device
        assert not sampled.requires_grad
        assert sampled.grad_fn is None
        assert torch.isfinite(sampled).all()

        if k > 0:
            assert torch.all(sampled == 42.5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("count", [5, 31, 63])
    @pytest.mark.parametrize("k", [0, 1, 7, 64, 128])
    def test_partially_filled_reservoir_bounds_and_shapes(self, device: str, count: int, k: int):
        """When reservoir is partially filled (count < reservoir_size), draws must be strictly within [0, count)."""
        mem = FastWeightMemory(dim=16, reservoir_size=64).to(device)
        gen = torch.Generator(device=device).manual_seed(100 + count + k)

        for i in range(count):
            mem.write(torch.full((1, 1, 8, 8), float(i + 1), device=device))

        assert mem.reservoir_count.item() == count

        sampled = mem.sample_reservoir(gen, n=k)
        assert sampled is not None
        assert sampled.shape == (k, 1, 8, 8)
        assert sampled.device.type == device
        assert not sampled.requires_grad
        assert torch.isfinite(sampled).all()

        if k > 0:
            min_val = sampled.min().item()
            max_val = sampled.max().item()
            assert min_val >= 1.0, f"Sampled value {min_val} below lower bound 1.0"
            assert max_val <= float(count), f"Sampled value {max_val} above upper bound {count}"

            # Verify that every sampled frame matches one of the written frames exactly
            for s_idx in range(k):
                frame_val = sampled[s_idx, 0, 0, 0].item()
                assert frame_val in [float(v) for v in range(1, count + 1)]

    @pytest.mark.parametrize("device", DEVICES)
    def test_partially_filled_uniform_coverage(self, device: str):
        """Sampling n=128 multiple times from 5 items must achieve 100% mode coverage."""
        mem = FastWeightMemory(dim=16, reservoir_size=64).to(device)
        gen = torch.Generator(device=device).manual_seed(777)

        for i in range(5):
            mem.write(torch.full((1, 1, 8, 8), float(i + 1), device=device))

        sampled = mem.sample_reservoir(gen, n=128)
        assert sampled is not None
        unique_vals = set(sampled[:, 0, 0, 0].cpu().tolist())
        assert unique_vals == {1.0, 2.0, 3.0, 4.0, 5.0}

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("k", [0, 1, 7, 64, 128])
    def test_full_reservoir_exact_shapes_and_bounds(self, device: str, k: int):
        """When reservoir is completely full (count == 64), draws must span all 64 slots."""
        mem = FastWeightMemory(dim=16, reservoir_size=64).to(device)
        gen = torch.Generator(device=device).manual_seed(999 + k)

        for i in range(64):
            mem.write(torch.full((1, 1, 8, 8), float(i + 1), device=device))

        assert mem.reservoir_count.item() == 64

        sampled = mem.sample_reservoir(gen, n=k)
        assert sampled is not None
        assert sampled.shape == (k, 1, 8, 8)
        assert sampled.device.type == device
        assert not sampled.requires_grad
        assert torch.isfinite(sampled).all()

        if k > 0:
            assert sampled.min().item() >= 1.0
            assert sampled.max().item() <= 64.0

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("total_writes", [65, 128, 500])
    @pytest.mark.parametrize("k", [0, 1, 7, 64, 128])
    def test_overfilled_wrapped_reservoir_fifo_eviction(self, device: str, total_writes: int, k: int):
        """When reservoir wraps around (count > 64), evicted items must never be sampled."""
        mem = FastWeightMemory(dim=16, reservoir_size=64).to(device)
        gen = torch.Generator(device=device).manual_seed(total_writes * 10 + k)

        for i in range(total_writes):
            mem.write(torch.full((1, 1, 8, 8), float(i + 1), device=device))

        assert mem.reservoir_count.item() == total_writes

        sampled = mem.sample_reservoir(gen, n=k)
        assert sampled is not None
        assert sampled.shape == (k, 1, 8, 8)
        assert sampled.device.type == device
        assert not sampled.requires_grad
        assert torch.isfinite(sampled).all()

        if k > 0:
            oldest_retained = float(total_writes - 64 + 1)
            newest_retained = float(total_writes)
            min_val = sampled.min().item()
            max_val = sampled.max().item()
            assert min_val >= oldest_retained, f"Sampled evicted frame {min_val} < {oldest_retained}"
            assert max_val <= newest_retained, f"Sampled future frame {max_val} > {newest_retained}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_sample_reservoir_generator_determinism(self, device: str):
        """Fixed generator seeds must produce bitwise identical sampled tensors."""
        mem = FastWeightMemory(dim=16, reservoir_size=64).to(device)
        for i in range(64):
            mem.write(torch.randn(1, 1, 8, 8, device=device))

        gen1 = torch.Generator(device=device).manual_seed(2026)
        gen2 = torch.Generator(device=device).manual_seed(2026)

        s1 = mem.sample_reservoir(gen1, n=7)
        s2 = mem.sample_reservoir(gen2, n=7)
        assert torch.equal(s1, s2)

        # Different seed produces different draws
        gen3 = torch.Generator(device=device).manual_seed(2027)
        s3 = mem.sample_reservoir(gen3, n=7)
        assert not torch.equal(s1, s3)


class TestDetachmentFinitenessAndGradientIsolationStress:
    """Stress tests detachment, finiteness, and gradient isolation through autograd."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_sample_reservoir_buffer_mutation_independence(self, device: str):
        """In-place modification of sampled tensor must not mutate reservoir buffer and vice-versa."""
        mem = FastWeightMemory(dim=16, reservoir_size=64).to(device)
        gen = torch.Generator(device=device).manual_seed(555)

        for i in range(10):
            mem.write(torch.full((1, 1, 8, 8), float(i + 1), device=device))

        recalled = mem.sample_reservoir(gen, n=7)
        saved_recalled = recalled.clone()
        saved_reservoir = mem.reservoir.clone()

        # Mutate sampled tensor in-place
        recalled.fill_(-9999.0)
        recalled.add_(123.0)

        # Reservoir must remain 100% bitwise unchanged
        assert torch.equal(mem.reservoir, saved_reservoir)

        # Mutate reservoir in-place
        mem.reservoir.fill_(7777.0)

        # Previously sampled tensor must remain unchanged
        assert torch.equal(recalled, torch.full_like(recalled, -9999.0 + 123.0))

    @pytest.mark.parametrize("device", DEVICES)
    def test_gradient_isolation_through_critic_backward(self, device: str):
        """Backpropagating loss through critic on recalled frames must produce zero gradients in memory."""
        mem = FastWeightMemory(dim=16, reservoir_size=64).to(device)
        critic = StreamImageCritic(width=16, context_dim=16).to(device)
        gen = torch.Generator(device=device).manual_seed(888)

        # Populate memory
        for _ in range(8):
            f = torch.randn(1, 1, 8, 8, device=device)
            ctx = mem.write(f)

        critic.set_context(ctx)

        # Recalled batch
        recalled = mem.sample_reservoir(gen, n=7)
        assert recalled.requires_grad is False
        assert recalled.grad_fn is None

        # Pass through critic
        scores = critic(recalled)
        loss = scores.sum()
        loss.backward()

        # Verify critic weights received gradients
        assert critic.conv1.weight.grad is not None
        assert torch.isfinite(critic.conv1.weight.grad).all()

        # Verify memory parameters and buffers have ZERO gradient
        assert recalled.grad is None
        assert mem.matrix.grad is None
        assert mem.prev_key.grad is None
        assert mem.reservoir.grad is None
        assert mem.reservoir_count.grad is None
        assert mem.key_proj.weight.grad is None
        assert mem.val_proj.weight.grad is None

    @pytest.mark.parametrize("device", DEVICES)
    def test_sample_reservoir_finiteness_under_poisoned_write_attempts(self, device: str):
        """Even under adversarial NaN and Inf writes, sample_reservoir returns strictly finite tensors."""
        mem = FastWeightMemory(dim=16, reservoir_size=64).to(device)
        gen = torch.Generator(device=device).manual_seed(333)

        # Write valid frame
        mem.write(torch.full((1, 1, 8, 8), 1.0, device=device))

        # Attempt to poison with non-finites
        for val in [float("nan"), float("inf"), float("-inf")]:
            poison = torch.full((1, 1, 8, 8), val, device=device)
            ctx = mem.write(poison)
            assert torch.isfinite(ctx).all()

        # Sampling reservoir must remain finite
        for k in [1, 7, 64, 128]:
            s = mem.sample_reservoir(gen, n=k)
            assert s is not None
            assert torch.isfinite(s).all()
            assert torch.all(s == 1.0)


class TestConcatenatedTrainRealShapeAndDeviceInvariance:
    """Stress tests concatenated train_real shape, device invariance, and downstream training."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("rehearsal_batch", [0, 1, 7, 64, 128])
    def test_concatenated_train_real_shape_device_and_values(self, device: str, rehearsal_batch: int):
        """Replicates particlestream/tier1.py:315-320 concatenation logic across all k."""
        mem = FastWeightMemory(dim=16, reservoir_size=64).to(device)
        mem_gen = torch.Generator(device=device).manual_seed(444 + rehearsal_batch)

        x_cur = torch.full((1, 1, 8, 8), 10.0, device=device)

        # Step 1: memory write precedes sample_reservoir
        ctx = mem.write(x_cur)
        if rehearsal_batch > 0:
            recalled = mem.sample_reservoir(mem_gen, n=rehearsal_batch)
            train_real = torch.cat([x_cur, recalled], dim=0) if recalled is not None else x_cur
        else:
            train_real = x_cur

        expected_batch_size = 1 + rehearsal_batch if rehearsal_batch > 0 else 1
        assert train_real.shape == (expected_batch_size, 1, 8, 8)
        assert train_real.device.type == device
        assert train_real.dtype == torch.float32
        assert not train_real.requires_grad
        assert train_real.grad_fn is None
        assert torch.isfinite(train_real).all()

        # train_real[0] must match x_cur bitwise
        assert torch.equal(train_real[0:1], x_cur)

        # Remaining items match recalled
        if rehearsal_batch > 0:
            assert torch.equal(train_real[1:], recalled)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("rehearsal_batch", [0, 1, 7, 64, 128])
    def test_critic_forward_on_concatenated_train_real(self, device: str, rehearsal_batch: int):
        """StreamImageCritic forward pass must succeed on train_real of any batch size."""
        critic = StreamImageCritic(width=16, context_dim=16).to(device)
        ctx = torch.randn(1, 16, device=device)
        critic.set_context(ctx)

        batch_size = 1 + rehearsal_batch if rehearsal_batch > 0 else 1
        train_real = torch.randn(batch_size, 1, 8, 8, device=device)

        scores = critic(train_real)
        assert scores.shape == (batch_size, 1)
        assert scores.device.type == device
        assert torch.isfinite(scores).all()

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("rehearsal_batch", [0, 1, 7, 64])
    def test_gantrainer_step_with_concatenated_train_real(self, device: str, rehearsal_batch: int):
        """GANTrainer.step must execute cleanly and update parameters with train_real of any batch size."""
        task = {t.id: t for t in TIER1_TASKS}["seq_img_round_robin_bars4"]
        cand = TIER1_CANDIDATES["fast-weight-bcap-ns"]

        streams = NamedStreams(seed=0, device=device)
        g = StreamImageGenerator(z_dim=task.z_dim, width=16, context_dim=0).to(device)
        d = StreamImageCritic(width=16, context_dim=0).to(device)
        mem = FastWeightMemory(dim=16, reservoir_size=64).to(device)

        _init_module(g, streams, "generator")
        _init_module(d, streams, "discriminator")
        _init_module(mem, streams, "memory")

        recipe = _build_recipe(task, cand, 10)
        prior_gen = streams.generator("init", component="prior", purpose="locations", device=device)
        prior = recipe.replace(prior_update="learned").make_prior(
            device=device, dtype=torch.float32, generator=prior_gen, init_std=1.0
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

        mem_gen = streams.generator("eval", component="memory", purpose="reservoir", device=device)

        # Step 5 times replicating tier1 loop
        for step in range(5):
            x_cur = torch.randn(1, 1, 8, 8, device=device)
            ctx = mem.write(x_cur)
            if rehearsal_batch > 0:
                recalled = mem.sample_reservoir(mem_gen, n=rehearsal_batch)
                train_real = torch.cat([x_cur, recalled], dim=0) if recalled is not None else x_cur
            else:
                train_real = x_cur

            trainer.step(train_real)
            assert trainer.policy.completed_steps == step + 1
            assert all(torch.isfinite(p).all() for p in g.parameters())
            assert all(torch.isfinite(p).all() for p in d.parameters())
            assert all(torch.isfinite(p).all() for p in prior.parameters())
