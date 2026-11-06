"""Frozen tiny task laws, semantic scorers, and analytic controls.

These helpers contain no trainer, optimizer, learned parameters, or data-dependent
selection. Every constructed training frame has batch size one. Scoring may
inspect several independently served frames; it never changes training state.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Sequence

import torch

MODES = ("dense4", "tokens4", "kernel2x2", "addition2bit")
INVALID = -1
# Frozen once from SHA256('particlestream/minimal-dense4/task-definition/seed-0/{i}').
DENSE_MASKS = ("6f4f3d65790f6149", "d9caad5360e1bf69", "ab9a9dd905a5d0fd", "74add64d3d80d2cf")
KERNEL = (0.1, 0.2, 0.3, 0.4)
KERNEL_HELDOUT = (3, 5, 10, 12)
ADDITION_HELDOUT = ((0, 1), (1, 3), (2, 0), (3, 2))
# Each slot is a 2x2 block: bit/categorical values are repeated within a slot.
SLOTS = ((1, 1), (1, 5), (5, 1), (5, 5))


def _spec(task_id, description, steps, tier, bounds):
    return MappingProxyType(dict(id=task_id, description=description, steps=steps,
                                 tier=tier, bounds=tuple(bounds)))


PROBE_SPECS = MappingProxyType({
    "dense4": _spec("seq_minimal_dense4", "Acquire four arbitrary dense 8x8 patterns without replay.", 96, 1,
        (("valid_fraction", ">=", .90), ("modes", "==", 4.), ("mass_tv", "<=", .15))),
    "tokens4": _spec("seq_minimal_tokens4", "Serve four integer token IDs with strict raw one-hot confidence.", 96, 1,
        (("valid_fraction", "==", 1.), ("modes", "==", 4.), ("mass_tv", "<=", .10))),
    "kernel2x2": _spec("seq_minimal_kernel2x2", "Learn one local kernel response on twelve patches and evaluate four unseen patches.", 192, 2,
        tuple((f"{split}_{metric}", "<=", value) for split in ("seen", "heldout")
              for metric, value in (("response_rmse", .03), ("max_abs_error", .05)))),
    "addition2bit": _spec("seq_minimal_addition2bit", "Add two-bit operands on held-out operand combinations.", 192, 2,
        tuple((f"{split}_{metric}", op, value) for split in ("seen", "heldout")
              for metric, op, value in (("exact_accuracy", "==", 1.), ("min_bit_confidence", ">=", .90)))),
})


@dataclass(frozen=True)
class ProbeCase:
    key: int | tuple[int, int]
    source: torch.Tensor | None
    target: torch.Tensor


def _blank(device="cpu"):
    return torch.zeros((1, 1, 8, 8), dtype=torch.float32, device=device)


def _slots(values, device="cpu"):
    frame = _blank(device)
    for value, (row, col) in zip(values, SLOTS):
        frame[:, :, row:row + 2, col:col + 2] = float(value)
    return frame.detach()


def dense_templates(device="cpu"):
    return torch.tensor([[(int(mask, 16) >> bit) & 1 for bit in range(64)]
                         for mask in DENSE_MASKS], dtype=torch.float32,
                        device=device).reshape(4, 1, 8, 8).detach()


def token_templates(device="cpu"):
    return torch.cat([_slots([int(i == token) for i in range(4)], device)
                      for token in range(4)])


def encode_operands(a: int, b: int, device="cpu"):
    if not (0 <= a < 4 and 0 <= b < 4):
        raise ValueError("operands must be integers in 0..3")
    return _slots((a & 1, (a >> 1) & 1, b & 1, (b >> 1) & 1), device)


def encode_sum(total: int, device="cpu"):
    if not 0 <= total <= 6:
        raise ValueError("sum must be an integer in 0..6")
    return _slots(((total >> bit) & 1 for bit in range(3)), device)


def kernel_source(mask: int, device="cpu"):
    if not 0 <= mask < 16:
        raise ValueError("patch mask must be in 0..15")
    frame = _blank(device)
    frame[:, :, 3:5, 3:5] = torch.tensor([(mask >> bit) & 1 for bit in range(4)],
                                        dtype=frame.dtype, device=device).reshape(2, 2)
    return frame.detach()


def kernel_response(mask: int, coefficients=KERNEL):
    return sum(((mask >> bit) & 1) * weight for bit, weight in enumerate(coefficients))


def paired_cases(mode: str, split="seen", device="cpu") -> tuple[ProbeCase, ...]:
    """Return independent frames; held-out sources never enter training_cases."""
    if split not in {"seen", "heldout", "all"}:
        raise ValueError("split must be seen, heldout, or all")
    if mode == "kernel2x2":
        keys = [mask for mask in range(16)
                if split == "all" or ((mask in KERNEL_HELDOUT) == (split == "heldout"))]
        return tuple(ProbeCase(mask, kernel_source(mask, device),
                              torch.full((1, 1, 8, 8), kernel_response(mask), device=device))
                     for mask in keys)
    if mode == "addition2bit":
        keys = [(a, b) for a in range(4) for b in range(4)
                if split == "all" or (((a, b) in ADDITION_HELDOUT) == (split == "heldout"))]
        return tuple(ProbeCase((a, b), encode_operands(a, b, device), encode_sum(a + b, device))
                     for a, b in keys)
    raise ValueError(f"{mode!r} is not a paired task")


def training_cases(mode: str, device="cpu") -> tuple[ProbeCase, ...]:
    """Balanced chronological cycle; caller takes one case per optimizer update."""
    if mode in {"dense4", "tokens4"}:
        bank = dense_templates(device) if mode == "dense4" else token_templates(device)
        return tuple(ProbeCase(i, None, bank[i:i + 1].clone()) for i in range(4))
    return paired_cases(mode, "seen", device)


def target_bank(mode: str, device="cpu"):
    if mode == "dense4":
        return dense_templates(device)
    if mode == "tokens4":
        return token_templates(device)
    if mode == "addition2bit":
        return torch.cat([encode_sum(total, device) for total in range(7)])
    if mode == "kernel2x2":
        return torch.cat([case.target for case in paired_cases(mode, "all", device)])
    raise ValueError(f"unknown minimal mode {mode!r}")


def _check_frames(samples):
    if not isinstance(samples, torch.Tensor) or samples.ndim != 4 or tuple(samples.shape[1:]) != (1, 8, 8):
        raise ValueError("expected served frames shaped (N,1,8,8)")
    if len(samples) == 0:
        raise ValueError("at least one served frame is required")
    return samples.detach()


def decode_tokens(samples: torch.Tensor) -> torch.Tensor:
    """Frozen integer/INVALID interface; ambiguous soft codes are never projected."""
    samples = _check_frames(samples)
    bank = token_templates(samples.device)
    errors = (samples[:, None] - bank[None]).abs().flatten(2).amax(2)
    finite = torch.isfinite(samples).flatten(1).all(1)
    errors = torch.nan_to_num(errors, nan=1e6, posinf=1e6, neginf=1e6)
    best_error, token = errors.min(1)
    return torch.where(finite & (best_error <= .100001), token, torch.full_like(token, INVALID))


def decode_token(frame: torch.Tensor) -> int:
    """Serve exactly one token ID; malformed/nonconfident output returns INVALID."""
    if tuple(frame.shape) != (1, 1, 8, 8):
        raise ValueError("decode_token requires one frame shaped (1,1,8,8)")
    return int(decode_tokens(frame)[0].item())


def score_distribution(mode: str, samples: torch.Tensor) -> dict[str, float]:
    samples = _check_frames(samples)
    if mode == "tokens4":
        labels = decode_tokens(samples)
    elif mode == "dense4":
        errors = (samples[:, None] - dense_templates(samples.device)[None]).abs().flatten(2).amax(2)
        errors = torch.nan_to_num(errors, nan=1e6, posinf=1e6, neginf=1e6)
        best_error, labels = errors.min(1)
        finite = torch.isfinite(samples).flatten(1).all(1)
        labels = torch.where(finite & (best_error <= .100001), labels, torch.full_like(labels, INVALID))
    else:
        raise ValueError(f"{mode!r} is not a distribution task")
    counts = torch.tensor([(labels == index).sum().item() for index in range(4)]
                          + [(labels == INVALID).sum().item()], dtype=torch.float64)
    probabilities = counts / len(samples)
    target = torch.tensor((.25, .25, .25, .25, 0.), dtype=torch.float64)
    return {"valid_fraction": float((labels != INVALID).float().mean().item()),
            "invalid_fraction": float(probabilities[-1]),
            "modes": float((counts[:4] > 0).sum().item()),
            "mass_tv": float((probabilities - target).abs().sum().item() / 2)}


def _paired_tensor(predictions, cases):
    if not isinstance(predictions, torch.Tensor):
        raise ValueError("paired predictions must be a tensor")
    if predictions.ndim == 4:
        predictions = predictions[:, None]
    if (predictions.ndim != 5 or tuple(predictions.shape[2:]) != (1, 8, 8)
            or predictions.shape[0] != len(cases) or not len(cases) or not predictions.shape[1]):
        raise ValueError("expected (cases,centers,1,8,8) paired predictions")
    return predictions.detach()


def score_paired(mode: str, predictions: torch.Tensor, cases: Sequence[ProbeCase]) -> dict[str, float]:
    predictions = _paired_tensor(predictions, cases)
    targets = torch.cat([case.target.to(predictions.device) for case in cases])[:, None]
    finite = torch.isfinite(predictions).flatten(2).all(2)
    safe = torch.nan_to_num(predictions, nan=1e6, posinf=1e6, neginf=-1e6)
    if mode == "kernel2x2":
        errors = safe - targets
        return {"response_rmse": float(errors.square().mean().sqrt().item()),
                "max_abs_error": float(errors.abs().max().item())}
    if mode != "addition2bit":
        raise ValueError(f"{mode!r} is not a paired task")
    bits, confidences = [], []
    for bit, (row, col) in enumerate(SLOTS[:3]):
        block = safe[..., row:row + 2, col:col + 2]
        target_bit = targets[..., row:row + 2, col:col + 2]
        bits.append(block.mean(dim=(-1, -2, -3)) >= .5)
        confidence = torch.where(target_bit >= .5, block, 1 - block)
        confidences.append(confidence.flatten(2).amin(2))
    sums = sum(bit.to(torch.int64) * (1 << index) for index, bit in enumerate(bits))
    actual = torch.tensor([sum(case.key) for case in cases], device=predictions.device)[:, None]
    correct = (sums == actual) & finite
    carrying = actual[:, 0] >= 4
    in_range = ((safe >= 0) & (safe <= 1)).flatten(2).all(2)
    confidence = torch.stack(confidences).amin(0)
    confidence = torch.where(finite & in_range, confidence, torch.zeros_like(confidence))
    return {"exact_accuracy": float(correct.float().mean().item()),
            "min_bit_confidence": float(confidence.min().item()),
            "carry_accuracy": float(correct[carrying].float().mean().item()) if bool(carrying.any()) else 1.,
            "invalid_fraction": float((~finite | ~in_range).float().mean().item())}


def score_paired_splits(mode: str, seen_predictions: torch.Tensor,
                        heldout_predictions: torch.Tensor) -> dict[str, float]:
    metrics = {}
    for split, predictions in (("seen", seen_predictions), ("heldout", heldout_predictions)):
        cases = paired_cases(mode, split, predictions.device)
        metrics.update({f"{split}_{name}": value for name, value in score_paired(mode, predictions, cases).items()})
    return metrics


def bounds_pass(mode: str, metrics: dict[str, float]) -> bool:
    operations = {">=": lambda value, bound: value >= bound,
                  "<=": lambda value, bound: value <= bound,
                  "==": lambda value, bound: value == bound}
    return all(name in metrics and operations[op](metrics[name], bound)
               for name, op, bound in PROBE_SPECS[mode]["bounds"])


def control_scores(mode: str, device="cpu") -> dict[str, dict[str, float]]:
    """Score oracle and cheap destructive outputs through the production scorers."""
    if mode in {"dense4", "tokens4"}:
        bank = target_bank(mode, device)
        outputs = {"oracle": bank.repeat_interleave(8, dim=0),
                   "collapsed_mode": bank[:1].repeat(32, 1, 1, 1),
                   "blurred_mean" if mode == "dense4" else "soft_one_hot": bank.mean(0, keepdim=True).repeat(32, 1, 1, 1)}
        return {name: score_distribution(mode, frames) for name, frames in outputs.items()}
    scorings = {}
    controls = ("oracle", "zero", "copied_input_pixel", "horizontal_flip", "vertical_flip") if mode == "kernel2x2" else ("oracle", "carry_removed", "ignored_operand", "shuffled_condition")
    for control in controls:
        predictions = []
        for split in ("seen", "heldout"):
            cases = paired_cases(mode, split, device)
            frames = []
            for index, case in enumerate(cases):
                if control == "oracle":
                    frame = case.target
                elif mode == "kernel2x2":
                    if control == "zero": value = 0.
                    elif control == "copied_input_pixel": value = float(case.key & 1)
                    else: value = kernel_response(case.key, (.2, .1, .4, .3) if control == "horizontal_flip" else (.3, .4, .1, .2))
                    frame = torch.full_like(case.target, value)
                else:
                    a, b = case.key
                    value = (a + b) & 3 if control == "carry_removed" else a
                    if control == "shuffled_condition":
                        value = sum(cases[(index + 1) % len(cases)].key)
                    frame = encode_sum(value, device)
                frames.append(frame)
            predictions.append(torch.cat(frames)[:, None])
        scorings[control] = score_paired_splits(mode, *predictions)
    return scorings


def probe_manifest(mode: str) -> dict:
    """JSON-compatible immutable-law description for fingerprints and reports."""
    spec = PROBE_SPECS[mode]
    manifest = {"schema_version": 1, "mode": mode, "task_id": spec["id"],
                "steps": spec["steps"], "batch_size": 1,
                "bounds": [list(bound) for bound in spec["bounds"]],
                "frame_shape": [1, 1, 8, 8], "chronological_order": "ascending_key_round_robin"}
    if mode == "dense4":
        manifest.update(bitmaps_hex=list(DENSE_MASKS), bit_order="least_significant_first_row_major",
                        task_definition_stream="minimal-dense4/task-definition/seed-0", pixel_values=[0., 1.],
                        quality_max_abs_error=.10, target_mass=[.25] * 4)
    elif mode == "tokens4":
        manifest.update(slots=[list(slot) for slot in SLOTS], slot_shape=[2, 2],
                        encoding="one_hot_blocks_zero_padding", decoder="whole_code_max_abs_error_le_0.10_or_INVALID",
                        invalid_id=INVALID, target_mass=[.25] * 4)
    elif mode == "kernel2x2":
        manifest.update(kernel=list(KERNEL), patch_shape=[2, 2], patch_origin=[3, 3],
                        bit_order="bit_0_first_row_major", heldout_masks=list(KERNEL_HELDOUT),
                        response_encoding="uniform_scalar_8x8_all_pixels_scored")
    elif mode == "addition2bit":
        manifest.update(heldout_pairs=[list(pair) for pair in ADDITION_HELDOUT],
                        slots=[list(slot) for slot in SLOTS], slot_shape=[2, 2],
                        source_bits=["a_lsb", "a_msb", "b_lsb", "b_msb"],
                        target_bits=["sum_lsb", "sum_bit_1", "sum_bit_2"], pixel_values=[0., 1.])
    else:
        raise ValueError(f"unknown minimal mode {mode!r}")
    return manifest
