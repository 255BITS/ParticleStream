"""Semantic controls and fixed splits for the minimal acquisition/rule probes."""
import json
import math

import pytest
import torch

from particlestream.minimal_tasks import (
    ADDITION_HELDOUT, DENSE_MASKS, INVALID, KERNEL_HELDOUT, MODES, PROBE_SPECS,
    bounds_pass, control_scores, decode_token, decode_tokens, dense_templates,
    encode_sum, paired_cases, probe_manifest, score_distribution, score_paired,
    target_bank, training_cases,
)


@pytest.mark.parametrize("mode", MODES)
def test_same_declared_scorers_pass_oracle_and_fail_every_destructive_control(mode):
    controls = control_scores(mode)
    assert bounds_pass(mode, controls["oracle"])
    for name, metrics in controls.items():
        assert all(math.isfinite(value) for value in metrics.values()), name
        if name != "oracle":
            assert not bounds_pass(mode, metrics), name
            # Each shortcut must fail on held-out inputs in its own right.
            held_bounds = [bound for bound in PROBE_SPECS[mode]["bounds"] if bound[0].startswith("heldout_")]
            if held_bounds:
                comparisons = {"<=": lambda x, y: x <= y, ">=": lambda x, y: x >= y, "==": lambda x, y: x == y}
                assert not all(comparisons[op](metrics[key], value) for key, op, value in held_bounds), name


@pytest.mark.parametrize("mode", ("kernel2x2", "addition2bit"))
def test_frozen_heldout_inputs_are_never_emitted_for_training(mode):
    training = training_cases(mode)
    heldout = paired_cases(mode, "heldout")
    assert len(training) == 12 and len(heldout) == 4
    assert not {case.key for case in training} & {case.key for case in heldout}
    assert {case.key for case in training} | {case.key for case in heldout} == {case.key for case in paired_cases(mode, "all")}
    assert all(tuple(case.source.shape) == tuple(case.target.shape) == (1, 1, 8, 8) for case in training + heldout)
    assert all(not case.source.requires_grad and not case.target.requires_grad for case in training + heldout)
    if mode == "addition2bit":
        assert tuple(case.key for case in heldout) == ADDITION_HELDOUT
        assert {a for a, b in (case.key for case in training)} == set(range(4))
        assert {b for a, b in (case.key for case in training)} == set(range(4))
        assert {sum(case.key) for case in training} == set(range(7))
    else:
        assert tuple(case.key for case in heldout) == KERNEL_HELDOUT


def test_dense_patterns_are_unrelated_and_one_wrong_pixel_is_not_hidden_by_rmse():
    bank = dense_templates()
    assert len(set(DENSE_MASKS)) == 4
    for i in range(4):
        for j in range(i):
            assert int((bank[i] != bank[j]).sum()) >= 16
    samples = bank.repeat_interleave(8, dim=0)
    original = samples.clone()
    samples[0, 0, 0, 0] = 1 - samples[0, 0, 0, 0]
    assert float((samples[0] - original[0]).square().mean().sqrt()) == .125
    result = score_distribution("dense4", samples)
    assert result["invalid_fraction"] == 1 / 32
    assert result["mass_tv"] == 1 / 32


def test_token_decoder_serves_integers_and_rejects_ambiguous_or_corrupt_codes():
    bank = target_bank("tokens4")
    assert decode_tokens(bank).tolist() == [0, 1, 2, 3]
    assert [decode_token(frame[None]) for frame in bank] == [0, 1, 2, 3]
    assert decode_token(bank.mean(0, keepdim=True)) == INVALID
    padding_corrupt = bank[:1].clone()
    padding_corrupt[0, 0, 0, 0] = .2
    assert decode_token(padding_corrupt) == INVALID
    nonfinite = bank[:1].clone()
    nonfinite[0, 0, 0, 0] = float("nan")
    assert decode_token(nonfinite) == INVALID


@pytest.mark.parametrize("mode", ("dense4", "tokens4"))
@pytest.mark.parametrize("bad", (float("nan"), float("inf"), -float("inf")))
def test_invalid_distribution_outputs_stay_in_the_mass_denominator(mode, bad):
    samples = torch.full((32, 1, 8, 8), bad)
    metrics = score_distribution(mode, samples)
    assert metrics["valid_fraction"] == 0
    assert metrics["invalid_fraction"] == metrics["mass_tv"] == 1
    assert metrics["modes"] == 0
    assert not bounds_pass(mode, metrics)


def test_addition_requires_correct_bits_at_every_declared_center():
    cases = paired_cases("addition2bit", "heldout")
    perfect = torch.cat([case.target for case in cases])[:, None].repeat(1, 4, 1, 1, 1)
    assert score_paired("addition2bit", perfect, cases)["exact_accuracy"] == 1
    # One center is wrong on one held-out pair; averaging must not grant a pass.
    perfect[0, 3] = encode_sum(0)[0]
    metrics = score_paired("addition2bit", perfect, cases)
    assert metrics["exact_accuracy"] == 15 / 16
    assert metrics["min_bit_confidence"] == 0
    assert metrics["carry_accuracy"] == 1
    perfect[0, 3] = float("nan")
    metrics = score_paired("addition2bit", perfect, cases)
    assert metrics["invalid_fraction"] == 1 / 16
    assert metrics["min_bit_confidence"] == 0


def test_kernel_scores_every_response_pixel_and_never_crops_away_an_error():
    cases = paired_cases("kernel2x2", "heldout")
    outputs = torch.cat([case.target for case in cases])[:, None]
    outputs[0, 0, 0, 7, 7] += .1
    metrics = score_paired("kernel2x2", outputs, cases)
    assert metrics["max_abs_error"] > .05
    assert metrics["response_rmse"] < .03


@pytest.mark.parametrize("mode", MODES)
def test_task_manifest_and_new_case_construction_are_stable(mode):
    before = json.dumps(probe_manifest(mode), sort_keys=True)
    cases = training_cases(mode)
    cases[0].target.fill_(.123)
    assert not torch.all(training_cases(mode)[0].target == .123)
    assert json.dumps(probe_manifest(mode), sort_keys=True) == before
    with pytest.raises(TypeError):
        PROBE_SPECS[mode]["steps"] = 1
