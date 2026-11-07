# BCAP

[← Technique leaderboard](../technique-inventory.md)

BCAP penalizes critic input-gradient norms above a fixed cap on real and generated examples. The cap is a soft term in the critic loss. The generator and trainable prior still learn through the adversarial objective. Adam versus the preset optimizer, learning rates, annealing, prior updates and reconstruction are recipe choices, shown separately below.

[Implementation](../../../particlestream/tier1.py) · [Protocol](../../../EXPERIMENTATION.md) · [Terminology](../../../RESEARCH_GLOSSARY.md)

Each section describes one exact global recipe. Configuration and parent deltas come from the declarations; results come from Forge's current source-bound receipts. Required and stretch evidence stay separate. A final metric can meet its gate while the sustained verdict fails: every one of the last five checks must pass.

The installed engine is pinned to ParticleGAN/Forge 0.9.7. Older version or replay wording in declaration history describes provenance; the effective configuration below controls execution.

<a id="online-bcap-balanced"></a>
## `online-bcap-balanced`

[Exact declaration](../../../configs/forge/ideas/online-bcap-balanced.json)

Parent: [`online-bcap-transport-causal`](bcap.md#online-bcap-transport-causal).

Declared changes:

- Context-local balanced transport: at most 64 detached conditioning centroids; per-context equal-mass row dual updates at rate 0.05, clip 1; global usage penalty removed

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.transport.context_capacity` | _unset_ | `64` |
| `memory_policy.transport.context_clip` | _unset_ | `8.0` |
| `memory_policy.transport.context_tolerance` | _unset_ | `0.25` |
| `memory_policy.transport.dual_clip` | _unset_ | `1.0` |
| `memory_policy.transport.dual_rate` | _unset_ | `0.05` |
| `memory_policy.transport.router` | _unset_ | `"context_balanced"` |
| `memory_policy.transport.usage_penalty` | `0.01` | `0.0` |

### Effective declaration

```json
{
  "memory_policy": {
    "age_horizon": 8.0,
    "association_weight": 0.125,
    "change_threshold": 0.04,
    "context_scale": 3.0,
    "contrast_threshold": 0.03,
    "decay": 0.9,
    "dim": 16,
    "kind": "fast_weight",
    "reader": "causal_state",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "transport": {
      "context_capacity": 64,
      "context_clip": 8.0,
      "context_tolerance": 0.25,
      "dual_clip": 1.0,
      "dual_rate": 0.05,
      "reconstruction_weight": 10.0,
      "router": "context_balanced",
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.0
    },
    "write_lr": 0.35
  },
  "recipe_overrides": {
    "betas": [
      0.0,
      0.999
    ],
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05
  },
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **FAIL**, 2/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `a9019a75be0f291361108d7a1e4214b46d8179113a21deab0e6c7c5bcbf06aef`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-bcap-balanced-required"></a>
### Required evidence

<a id="online-bcap-balanced-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.09375` | [receipt](../receipts/cuda/online-bcap-balanced/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bcap-balanced_seq_img_round_robin_bars4.gif) | 2026-11-07T16:22:10.268+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0.78125`; `modes=4`; `finite_template_tv=0.21875`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-balanced/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-balanced_seq_img_dwell_bars4.gif) | 2026-11-07T16:22:20.466+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.24144`; `direction_acc=0.6875`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-balanced/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bcap-balanced_seq_img_translating_bar_next.gif) | 2026-11-07T16:22:32.325+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=0.8125`; `within_state_modes=2`; `branch_tv=0.1875`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap-balanced/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bcap-balanced_seq_img_fork_rollout.gif) | 2026-11-07T16:22:52.017+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0.59375`; `modes=2`; `finite_template_tv=0.5625`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-balanced/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-balanced_seq_img_variable_dwell_bars4.gif) | 2026-11-07T16:23:09.928+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-bcap-balanced/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bcap-balanced_seq_img_rare_mode_bars4.gif) | 2026-11-07T16:23:27.933+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.95312`; `prediction_rmse=0.21865`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-balanced/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bcap-balanced_seq_img_delayed_cue.gif) | 2026-11-07T16:23:47.913+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0.46875`; `within_state_modes=2`; `branch_tv=0.53125`; `trajectory_consistency=0.34375`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap-balanced/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bcap-balanced_seq_img_persistent_fork.gif) | 2026-11-07T16:24:21.487+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.087605`; `xy_rmse=0.19863`; `step_accuracy=0.67969`; `orbit_completion=0.3125`; `waypoint_modes=8`; failed terminal bounds: orbit_completion >= 0.9; step_accuracy >= 0.9; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-balanced/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bcap-balanced_seq_img_circle_orbit.gif) | 2026-11-07T16:24:57.841+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.099236`; `s2c_rmse=0.10336`; `c2s_rmse=0.095106`; `xy_rmse=0.32679`; `step_accuracy=0.39844`; `cycle_completion=0.03125`; `waypoint_modes=8`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-balanced/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bcap-balanced_seq_img_square_circle_cycle.gif) | 2026-11-07T16:26:07.810+00:00 |

<a id="online-bcap-balanced-stretch"></a>
### Stretch evidence

<a id="online-bcap-balanced-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bcap-balanced/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bcap-balanced_seq_minimal_dense4.gif) | 2026-11-07T16:26:12.777+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0.1875`; `modes=3`; `mass_tv=0.8125`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bcap-balanced/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bcap-balanced_seq_minimal_tokens4.gif) | 2026-11-07T16:26:17.965+00:00 |

<a id="online-bcap-balanced-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.41735`; `seen_max_abs_error=1`; `heldout_response_rmse=0.37518`; `heldout_max_abs_error=0.7`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bcap-balanced/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bcap-balanced_seq_minimal_kernel2x2.gif) | 2026-11-07T16:26:27.089+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.3125`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0.125`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bcap-balanced/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bcap-balanced_seq_minimal_addition2bit.gif) | 2026-11-07T16:26:36.280+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011162`; `relative_mse=0.44333`; `moving_direction_accuracy=0.84375`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-bcap-balanced/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bcap-balanced_seq_robotics_fetch_reach.gif) | 2026-11-07T16:27:01.612+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011523`; `relative_mse=0.88472`; `moving_direction_accuracy=0.67442`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-balanced/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bcap-balanced_seq_robotics_fetch_push.gif) | 2026-11-07T16:27:27.129+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010111`; `relative_mse=0.62224`; `moving_direction_accuracy=0.75`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-balanced/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bcap-balanced_seq_robotics_fetch_slide.gif) | 2026-11-07T16:27:52.783+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0049915`; `relative_mse=0.15873`; `moving_direction_accuracy=0.67105`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-bcap-balanced/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bcap-balanced_seq_robotics_fetch_pick_and_place.gif) | 2026-11-07T16:28:18.752+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0085216`; `relative_mse=1.0741`; `moving_direction_accuracy=0.57827`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-balanced/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bcap-balanced_seq_robotics_hand_reach.gif) | 2026-11-07T16:28:44.739+00:00 |

<a id="online-bcap-balanced-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.76406`; `interpolation_speed_error=0.62868`; `interpolation_raw_rmse=0.073745`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.5`; `extrapolation_centroid_rmse=0.91925`; `extrapolation_speed_error=0.76872`; `extrapolation_raw_rmse=0.070728`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.39583`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bcap-balanced/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bcap-balanced_seq_minimal_velocity_extrapolation.gif) | 2026-11-07T16:29:09.632+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bcap-balanced --device cuda`.

<a id="online-bcap"></a>
## `online-bcap`

[Exact declaration](../../../configs/forge/ideas/online-bcap.json)

Parent: [`stream-bcap`](bcap.md#stream-bcap).

Declared changes:

- Strict online baseline: explicitly disable every replay path; preserve the complete historical optimization recipe
- One chronological observation per update; real/fake batch_size=1; causal serve-before-arrival prediction

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.rehearsal_batch` | `7` | `0` |
| `memory_policy.replay_policy` | _unset_ | `"none"` |

### Effective declaration

```json
{
  "memory_policy": {
    "decay": 0.9,
    "dim": 16,
    "kind": "fast_weight",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "write_lr": 0.35
  },
  "recipe_overrides": {
    "loss": "non_saturating",
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05
  },
  "recipe_preset": "bcap"
}
```

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `7f1a73501855c2ea32f71334e689559ed197390e7be4c9963c7db0b1c113b151`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-bcap-required"></a>
### Required evidence

<a id="online-bcap-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9; modes == 4.0 | [receipt](../receipts/cuda/online-bcap/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bcap_seq_img_round_robin_bars4.gif) | 2026-11-07T16:02:47.434+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bcap_seq_img_dwell_bars4.gif) | 2026-11-07T16:03:21.276+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.35472`; `direction_acc=0.5`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bcap_seq_img_translating_bar_next.gif) | 2026-11-07T16:03:58.024+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bcap_seq_img_fork_rollout.gif) | 2026-11-07T16:04:35.742+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bcap_seq_img_variable_dwell_bars4.gif) | 2026-11-07T16:05:12.563+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.40625`; `finite_template_tv=0.6875`; `support_modes=1`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-bcap/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bcap_seq_img_rare_mode_bars4.gif) | 2026-11-07T16:05:48.349+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.5`; `prediction_rmse=0.29146`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bcap_seq_img_delayed_cue.gif) | 2026-11-07T16:07:06.795+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bcap_seq_img_persistent_fork.gif) | 2026-11-07T16:07:53.744+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.2041`; `xy_rmse=1.0326`; `step_accuracy=0`; `orbit_completion=0`; `waypoint_modes=0`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bcap_seq_img_circle_orbit.gif) | 2026-11-07T16:08:37.960+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.20174`; `s2c_rmse=0.20584`; `c2s_rmse=0.19764`; `xy_rmse=1.1395`; `step_accuracy=0`; `cycle_completion=0`; `waypoint_modes=0`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bcap_seq_img_square_circle_cycle.gif) | 2026-11-07T16:10:04.548+00:00 |

<a id="online-bcap-stretch"></a>
### Stretch evidence

<a id="online-bcap-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bcap/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bcap_seq_minimal_dense4.gif) | 2026-11-07T16:10:13.123+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bcap/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bcap_seq_minimal_tokens4.gif) | 2026-11-07T16:10:21.770+00:00 |

<a id="online-bcap-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.3108`; `seen_max_abs_error=0.74771`; `heldout_response_rmse=0.17301`; `heldout_max_abs_error=0.44771`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bcap/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bcap_seq_minimal_kernel2x2.gif) | 2026-11-07T16:10:37.117+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.083333`; `seen_min_bit_confidence=1.946e-05`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=1.946e-05`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bcap/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bcap_seq_minimal_addition2bit.gif) | 2026-11-07T16:10:52.450+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.016884`; `relative_mse=1.0144`; `moving_direction_accuracy=0.60938`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bcap_seq_robotics_fetch_reach.gif) | 2026-11-07T16:12:06.492+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.017304`; `relative_mse=1.9951`; `moving_direction_accuracy=0.34884`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bcap_seq_robotics_fetch_push.gif) | 2026-11-07T16:12:45.256+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.019216`; `relative_mse=2.2474`; `moving_direction_accuracy=0.80392`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bcap_seq_robotics_fetch_slide.gif) | 2026-11-07T16:13:23.540+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.016364`; `relative_mse=1.7061`; `moving_direction_accuracy=0.62719`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bcap_seq_robotics_fetch_pick_and_place.gif) | 2026-11-07T16:14:04.528+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012815`; `relative_mse=2.429`; `moving_direction_accuracy=0.47664`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bcap_seq_robotics_hand_reach.gif) | 2026-11-07T16:14:42.630+00:00 |

<a id="online-bcap-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.4927`; `interpolation_speed_error=0.40942`; `interpolation_raw_rmse=0.073798`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.5`; `extrapolation_centroid_rmse=0.64694`; `extrapolation_speed_error=0.56666`; `extrapolation_raw_rmse=0.070742`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.5`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bcap/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bcap_seq_minimal_velocity_extrapolation.gif) | 2026-11-07T16:15:18.322+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bcap --device cuda`.

<a id="online-bcap-adam"></a>
## `online-bcap-adam`

[Exact declaration](../../../configs/forge/ideas/online-bcap-adam.json)

Parent: [`stream-bcap-adam`](bcap.md#stream-bcap-adam).

Declared changes:

- Explicit replay_policy=none: train every arriving observation once, batch_size=1
- Predict before arrival; learn next-frame targets from the preceding detached context

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.rehearsal_batch` | `7` | `0` |
| `memory_policy.replay_policy` | _unset_ | `"none"` |

### Effective declaration

```json
{
  "memory_policy": {
    "decay": 0.9,
    "dim": 16,
    "kind": "fast_weight",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "write_lr": 0.35
  },
  "recipe_overrides": {
    "betas": [
      0.0,
      0.999
    ],
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05
  },
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `a930dd8c9d1e350dc8d902131a9404c7e2ee29e64aea283699dd657089331b3c`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-bcap-adam-required"></a>
### Required evidence

<a id="online-bcap-adam-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.78125`; `modes=4`; `finite_template_tv=0.25`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9 | [receipt](../receipts/cuda/online-bcap-adam/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bcap-adam_seq_img_round_robin_bars4.gif) | 2026-11-07T16:15:29.138+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-adam/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-adam_seq_img_dwell_bars4.gif) | 2026-11-07T16:15:39.015+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.33884`; `direction_acc=0.5`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-adam/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bcap-adam_seq_img_translating_bar_next.gif) | 2026-11-07T16:15:49.472+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap-adam/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bcap-adam_seq_img_fork_rollout.gif) | 2026-11-07T16:15:59.741+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-adam/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-adam_seq_img_variable_dwell_bars4.gif) | 2026-11-07T16:16:09.111+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.53125`; `finite_template_tv=0.46875`; `support_modes=4`; `rare_mass=0.03125`; `rare_mass=0.03125`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-bcap-adam/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bcap-adam_seq_img_rare_mode_bars4.gif) | 2026-11-07T16:16:18.351+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.51562`; `prediction_rmse=0.2986`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-adam/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bcap-adam_seq_img_delayed_cue.gif) | 2026-11-07T16:16:28.910+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0.03125`; `within_state_modes=1`; `branch_tv=0.96875`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap-adam/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bcap-adam_seq_img_persistent_fork.gif) | 2026-11-07T16:16:45.384+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.11022`; `xy_rmse=0.33159`; `step_accuracy=0.32031`; `orbit_completion=0`; `waypoint_modes=8`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-adam/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bcap-adam_seq_img_circle_orbit.gif) | 2026-11-07T16:17:04.049+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.16012`; `s2c_rmse=0.15918`; `c2s_rmse=0.16107`; `xy_rmse=0.74504`; `step_accuracy=0.015625`; `cycle_completion=0`; `waypoint_modes=1`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-adam/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bcap-adam_seq_img_square_circle_cycle.gif) | 2026-11-07T16:17:38.865+00:00 |

<a id="online-bcap-adam-stretch"></a>
### Stretch evidence

<a id="online-bcap-adam-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bcap-adam/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bcap-adam_seq_minimal_dense4.gif) | 2026-11-07T16:17:42.387+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bcap-adam/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bcap-adam_seq_minimal_tokens4.gif) | 2026-11-07T16:17:45.792+00:00 |

<a id="online-bcap-adam-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.39622`; `seen_max_abs_error=0.96251`; `heldout_response_rmse=0.30054`; `heldout_max_abs_error=0.66251`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bcap-adam/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bcap-adam_seq_minimal_kernel2x2.gif) | 2026-11-07T16:17:51.356+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.083333`; `seen_min_bit_confidence=1.9478e-09`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=1.9478e-09`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bcap-adam/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bcap-adam_seq_minimal_addition2bit.gif) | 2026-11-07T16:17:56.665+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.018141`; `relative_mse=1.171`; `moving_direction_accuracy=0.5`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-adam/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bcap-adam_seq_robotics_fetch_reach.gif) | 2026-11-07T16:18:10.692+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012828`; `relative_mse=1.0964`; `moving_direction_accuracy=0.5814`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-adam/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bcap-adam_seq_robotics_fetch_push.gif) | 2026-11-07T16:18:22.859+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.013395`; `relative_mse=1.0921`; `moving_direction_accuracy=0.4951`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-adam/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bcap-adam_seq_robotics_fetch_slide.gif) | 2026-11-07T16:18:32.429+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011575`; `relative_mse=0.85355`; `moving_direction_accuracy=0.58333`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-adam/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bcap-adam_seq_robotics_fetch_pick_and_place.gif) | 2026-11-07T16:18:41.844+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010953`; `relative_mse=1.7745`; `moving_direction_accuracy=0.51986`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-adam/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bcap-adam_seq_robotics_hand_reach.gif) | 2026-11-07T16:18:51.345+00:00 |

<a id="online-bcap-adam-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.77773`; `interpolation_speed_error=0.8295`; `interpolation_raw_rmse=0.073698`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.47917`; `extrapolation_centroid_rmse=0.91625`; `extrapolation_speed_error=0.7741`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.5625`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bcap-adam/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bcap-adam_seq_minimal_velocity_extrapolation.gif) | 2026-11-07T16:19:01.015+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bcap-adam --device cuda`.

<a id="online-bcap-adam-causal"></a>
## `online-bcap-adam-causal`

[Exact declaration](../../../configs/forge/ideas/online-bcap-adam-causal.json)

Parent: [`online-bcap-adam`](bcap.md#online-bcap-adam).

Declared changes:

- Causal state reader queries detached associative memory and carries current image, velocity, salient history and bounded repetition age in the existing 16-dimensional context

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.age_horizon` | _unset_ | `8.0` |
| `memory_policy.association_weight` | _unset_ | `0.125` |
| `memory_policy.change_threshold` | _unset_ | `0.04` |
| `memory_policy.context_scale` | _unset_ | `3.0` |
| `memory_policy.contrast_threshold` | _unset_ | `0.03` |
| `memory_policy.reader` | _unset_ | `"causal_state"` |

### Effective declaration

```json
{
  "memory_policy": {
    "age_horizon": 8.0,
    "association_weight": 0.125,
    "change_threshold": 0.04,
    "context_scale": 3.0,
    "contrast_threshold": 0.03,
    "decay": 0.9,
    "dim": 16,
    "kind": "fast_weight",
    "reader": "causal_state",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "write_lr": 0.35
  },
  "recipe_overrides": {
    "betas": [
      0.0,
      0.999
    ],
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05
  },
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `6cde8880c4e1141fe8c08e1b11834867a573b7a360c6fe5ee1c0834288fef9a3`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-bcap-adam-causal-required"></a>
### Required evidence

<a id="online-bcap-adam-causal-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.78125`; `modes=4`; `finite_template_tv=0.25`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_img_round_robin_bars4.gif) | 2026-11-07T16:19:08.325+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_img_dwell_bars4.gif) | 2026-11-07T16:19:14.713+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.28741`; `direction_acc=0.5`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_img_translating_bar_next.gif) | 2026-11-07T16:19:21.740+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_img_fork_rollout.gif) | 2026-11-07T16:19:28.796+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_img_variable_dwell_bars4.gif) | 2026-11-07T16:19:35.309+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.53125`; `finite_template_tv=0.46875`; `support_modes=4`; `rare_mass=0.03125`; `rare_mass=0.03125`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_img_rare_mode_bars4.gif) | 2026-11-07T16:19:41.673+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.85938`; `prediction_rmse=0.29285`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_img_delayed_cue.gif) | 2026-11-07T16:19:48.904+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0.0625`; `within_state_modes=1`; `branch_tv=0.9375`; `trajectory_consistency=0.03125`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_img_persistent_fork.gif) | 2026-11-07T16:20:02.024+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.15671`; `xy_rmse=0.68478`; `step_accuracy=0.25781`; `orbit_completion=0`; `waypoint_modes=7`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_img_circle_orbit.gif) | 2026-11-07T16:20:17.288+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.1352`; `s2c_rmse=0.13422`; `c2s_rmse=0.13617`; `xy_rmse=0.62913`; `step_accuracy=0.28125`; `cycle_completion=0`; `waypoint_modes=3`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_img_square_circle_cycle.gif) | 2026-11-07T16:20:44.088+00:00 |

<a id="online-bcap-adam-causal-stretch"></a>
### Stretch evidence

<a id="online-bcap-adam-causal-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_minimal_dense4.gif) | 2026-11-07T16:20:47.054+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_minimal_tokens4.gif) | 2026-11-07T16:20:50.229+00:00 |

<a id="online-bcap-adam-causal-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.36141`; `seen_max_abs_error=0.99966`; `heldout_response_rmse=0.26174`; `heldout_max_abs_error=0.69709`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_minimal_kernel2x2.gif) | 2026-11-07T16:20:54.685+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.1875`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0.1875`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_minimal_addition2bit.gif) | 2026-11-07T16:20:59.209+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012071`; `relative_mse=0.51845`; `moving_direction_accuracy=0.77083`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_robotics_fetch_reach.gif) | 2026-11-07T16:21:09.053+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.016559`; `relative_mse=1.8269`; `moving_direction_accuracy=0.60465`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_robotics_fetch_push.gif) | 2026-11-07T16:21:18.795+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0090863`; `relative_mse=0.50251`; `moving_direction_accuracy=0.81373`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_robotics_fetch_slide.gif) | 2026-11-07T16:21:28.918+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0046633`; `relative_mse=0.13855`; `moving_direction_accuracy=0.64474`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_robotics_fetch_pick_and_place.gif) | 2026-11-07T16:21:38.761+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.008933`; `relative_mse=1.1803`; `moving_direction_accuracy=0.51636`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_robotics_hand_reach.gif) | 2026-11-07T16:21:48.988+00:00 |

<a id="online-bcap-adam-causal-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.71599`; `interpolation_speed_error=0.62902`; `interpolation_raw_rmse=0.073704`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.4375`; `extrapolation_centroid_rmse=0.92307`; `extrapolation_speed_error=0.77254`; `extrapolation_raw_rmse=0.070682`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.375`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_minimal_velocity_extrapolation.gif) | 2026-11-07T16:21:59.396+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bcap-adam-causal --device cuda`.

<a id="online-bcap-balanced-bce"></a>
## `online-bcap-balanced-bce`

[Exact declaration](../../../configs/forge/ideas/online-bcap-balanced-bce.json)

Parent: [`online-bcap-balanced-frozen`](bcap.md#online-bcap-balanced-frozen).

Declared changes:

- Replace only image reconstruction MSE with binary cross entropy at weight 10; routing remains current-image MSE plus context-local duals

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.transport.reconstruction_loss` | _unset_ | `"binary_cross_entropy"` |

### Effective declaration

```json
{
  "memory_policy": {
    "age_horizon": 8.0,
    "association_weight": 0.0,
    "change_threshold": 0.04,
    "context_scale": 3.0,
    "contrast_threshold": 0.03,
    "decay": 0.9,
    "dim": 16,
    "kind": "fast_weight",
    "reader": "causal_state",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "transport": {
      "context_capacity": 64,
      "context_clip": 8.0,
      "context_tolerance": 0.25,
      "dual_clip": 1.0,
      "dual_rate": 0.05,
      "reconstruction_loss": "binary_cross_entropy",
      "reconstruction_weight": 10.0,
      "router": "context_balanced",
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.0
    },
    "write_lr": 0.35
  },
  "recipe_overrides": {
    "betas": [
      0.0,
      0.999
    ],
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "prior_update": "frozen"
  },
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **FAIL**, 2/10 required passes (4 measured), 0/10 stretch passes (0 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `c2ff05d739fde37b2c35b23d6a4df5d7c9a6110f4c05f9e258aae410fb94a507`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-bcap-balanced-bce-required"></a>
### Required evidence

<a id="online-bcap-balanced-bce-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_img_round_robin_bars4.gif) | 2026-11-07T16:29:27.797+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0.71875`; `modes=4`; `finite_template_tv=0.28125`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_img_dwell_bars4.gif) | 2026-11-07T16:29:45.579+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.079114`; `direction_acc=0.96875` | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_img_translating_bar_next.gif) | 2026-11-07T16:30:05.128+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 1/5 | `hq=0.875`; `within_state_modes=2`; `branch_tv=0.15625`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_img_fork_rollout.gif) | 2026-11-07T16:30:24.675+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bcap-balanced-bce-stretch"></a>
### Stretch evidence

<a id="online-bcap-balanced-bce-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bcap-balanced-bce-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bcap-balanced-bce-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | no receipt | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bcap-balanced-bce --device cuda`.

<a id="online-bcap-balanced-frozen"></a>
## `online-bcap-balanced-frozen`

[Exact declaration](../../../configs/forge/ideas/online-bcap-balanced-frozen.json)

Parent: [`online-bcap-balanced`](bcap.md#online-bcap-balanced).

Declared changes:

- Freeze initial prior and ablate associative-history contribution; retain current state, velocity, salient content and repetition age

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.association_weight` | `0.125` | `0.0` |
| `recipe_overrides.prior_update` | _unset_ | `"frozen"` |

### Effective declaration

```json
{
  "memory_policy": {
    "age_horizon": 8.0,
    "association_weight": 0.0,
    "change_threshold": 0.04,
    "context_scale": 3.0,
    "contrast_threshold": 0.03,
    "decay": 0.9,
    "dim": 16,
    "kind": "fast_weight",
    "reader": "causal_state",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "transport": {
      "context_capacity": 64,
      "context_clip": 8.0,
      "context_tolerance": 0.25,
      "dual_clip": 1.0,
      "dual_rate": 0.05,
      "reconstruction_weight": 10.0,
      "router": "context_balanced",
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.0
    },
    "write_lr": 0.35
  },
  "recipe_overrides": {
    "betas": [
      0.0,
      0.999
    ],
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "prior_update": "frozen"
  },
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-bcap-balanced-frozen-required"></a>
### Required evidence

<a id="online-bcap-balanced-frozen-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bcap-balanced-frozen-stretch"></a>
### Stretch evidence

<a id="online-bcap-balanced-frozen-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bcap-balanced-frozen-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bcap-balanced-frozen-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | no receipt | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bcap-balanced-frozen --device cuda`.

<a id="online-bcap-stateless"></a>
## `online-bcap-stateless`

[Exact declaration](../../../configs/forge/ideas/online-bcap-stateless.json)

Parent: [`bcap`](bcap.md#bcap).

Declared changes:

- Strict online baseline: explicitly disable every replay path; preserve the complete historical optimization recipe
- One chronological observation per update; real/fake batch_size=1; causal serve-before-arrival prediction

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.rehearsal_batch` | _unset_ | `0` |
| `memory_policy.replay_policy` | _unset_ | `"none"` |

### Effective declaration

```json
{
  "memory_policy": {
    "kind": "none",
    "rehearsal_batch": 0,
    "replay_policy": "none"
  },
  "recipe_overrides": {},
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-bcap-stateless-required"></a>
### Required evidence

<a id="online-bcap-stateless-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bcap-stateless-stretch"></a>
### Stretch evidence

<a id="online-bcap-stateless-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bcap-stateless-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bcap-stateless-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | no receipt | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bcap-stateless --device cuda`.

<a id="online-bcap-transport"></a>
## `online-bcap-transport`

[Exact declaration](../../../configs/forge/ideas/online-bcap-transport.json)

Parent: [`online-bcap-adam`](bcap.md#online-bcap-adam).

Declared changes:

- Public prior manipulator: uniform critic sampling; generator routes the current observation by sequential clean-center MSE plus 0.01 log row usage, adds weight-10 reconstruction on one generated image, and clears the target before serving

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.transport` | _unset_ | `{"reconstruction_weight": 10.0, "temperature": 0.0, "usage_decay": 1.0, "usage_penalty": 0.01}` |

### Effective declaration

```json
{
  "memory_policy": {
    "decay": 0.9,
    "dim": 16,
    "kind": "fast_weight",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "transport": {
      "reconstruction_weight": 10.0,
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.01
    },
    "write_lr": 0.35
  },
  "recipe_overrides": {
    "betas": [
      0.0,
      0.999
    ],
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05
  },
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-bcap-transport-required"></a>
### Required evidence

<a id="online-bcap-transport-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bcap-transport-stretch"></a>
### Stretch evidence

<a id="online-bcap-transport-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bcap-transport-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bcap-transport-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | no receipt | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bcap-transport --device cuda`.

<a id="online-bcap-transport-causal"></a>
## `online-bcap-transport-causal`

[Exact declaration](../../../configs/forge/ideas/online-bcap-transport-causal.json)

Parent: [`online-bcap-adam-causal`](bcap.md#online-bcap-adam-causal).

Declared changes:

- Public prior manipulator: uniform critic sampling; generator routes the current observation by sequential clean-center MSE plus 0.01 log row usage, adds weight-10 reconstruction on one generated image, and clears the target before serving

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.transport` | _unset_ | `{"reconstruction_weight": 10.0, "temperature": 0.0, "usage_decay": 1.0, "usage_penalty": 0.01}` |

### Effective declaration

```json
{
  "memory_policy": {
    "age_horizon": 8.0,
    "association_weight": 0.125,
    "change_threshold": 0.04,
    "context_scale": 3.0,
    "contrast_threshold": 0.03,
    "decay": 0.9,
    "dim": 16,
    "kind": "fast_weight",
    "reader": "causal_state",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "transport": {
      "reconstruction_weight": 10.0,
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.01
    },
    "write_lr": 0.35
  },
  "recipe_overrides": {
    "betas": [
      0.0,
      0.999
    ],
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05
  },
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-bcap-transport-causal-required"></a>
### Required evidence

<a id="online-bcap-transport-causal-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bcap-transport-causal-stretch"></a>
### Stretch evidence

<a id="online-bcap-transport-causal-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bcap-transport-causal-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bcap-transport-causal-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | no receipt | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bcap-transport-causal --device cuda`.

<a id="online-bcap-transport-fit"></a>
## `online-bcap-transport-fit`

[Exact declaration](../../../configs/forge/ideas/online-bcap-transport-fit.json)

Parent: [`online-bcap-transport-causal`](bcap.md#online-bcap-transport-causal).

Declared changes:

- Combined global fitting recipe: frozen initial prior; constant generator LR 0.01; critic multiplier 0.25; reconstruction weight 100; associative read weight zero to isolate retained current/velocity/cue/age context

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.association_weight` | `0.125` | `0.0` |
| `memory_policy.transport.reconstruction_weight` | `10.0` | `100.0` |
| `recipe_overrides.d_lr_mult` | _unset_ | `0.25` |
| `recipe_overrides.lr` | `0.005` | `0.01` |
| `recipe_overrides.lr_floor` | `0.05` | `1.0` |
| `recipe_overrides.lr_schedule` | `"cosine"` | `"constant"` |
| `recipe_overrides.network_lr_floor` | `0.05` | `1.0` |
| `recipe_overrides.prior_update` | _unset_ | `"frozen"` |

### Effective declaration

```json
{
  "memory_policy": {
    "age_horizon": 8.0,
    "association_weight": 0.0,
    "change_threshold": 0.04,
    "context_scale": 3.0,
    "contrast_threshold": 0.03,
    "decay": 0.9,
    "dim": 16,
    "kind": "fast_weight",
    "reader": "causal_state",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "transport": {
      "reconstruction_weight": 100.0,
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.01
    },
    "write_lr": 0.35
  },
  "recipe_overrides": {
    "betas": [
      0.0,
      0.999
    ],
    "d_lr_mult": 0.25,
    "loss": "non_saturating",
    "lr": 0.01,
    "lr_anneal_start": 0.5,
    "lr_floor": 1.0,
    "lr_schedule": "constant",
    "network_lr_floor": 1.0,
    "prior_update": "frozen"
  },
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-bcap-transport-fit-required"></a>
### Required evidence

<a id="online-bcap-transport-fit-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, engine_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bcap-transport-fit-stretch"></a>
### Stretch evidence

<a id="online-bcap-transport-fit-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bcap-transport-fit-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bcap-transport-fit-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | no receipt | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bcap-transport-fit --device cuda`.

<a id="bcap"></a>
## `bcap`

[Exact declaration](../../../configs/forge/ideas/bcap.json)

Declared changes:

- Stateless BCAP baseline on batch_size=1 sequential stream

### Effective declaration

```json
{
  "memory_policy": {
    "kind": "none"
  },
  "recipe_overrides": {},
  "recipe_preset": "bcap_adam"
}
```

**Historical, read-only.** This declaration does not satisfy explicit `replay_policy=none` with `rehearsal_batch=0`; the executor rejects it before touching its saved evidence. It has no current qualification score.

Strict online successors: [`online-bcap-stateless`](bcap.md#online-bcap-stateless).

[Preserved receipt directory](../receipts/cuda/bcap) · [Archived cohorts](../cohorts) · [Research history](../research-notes.md)

<a id="stream-bcap"></a>
## `stream-bcap`

[Exact declaration](../../../configs/forge/ideas/stream-bcap.json)

Declared changes:

- ParticleGAN 0.9.6 public bcap trainer preset
- Shared 64-frame FIFO memory, seven replay samples and velocity context
- Global non-saturating loss and cosine schedule (start=0.5, floor=0.05)

### Effective declaration

```json
{
  "memory_policy": {
    "decay": 0.9,
    "dim": 16,
    "kind": "fast_weight",
    "rehearsal_batch": 7,
    "reservoir_size": 64,
    "write_lr": 0.35
  },
  "recipe_overrides": {
    "loss": "non_saturating",
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05
  },
  "recipe_preset": "bcap"
}
```

**Historical, read-only.** This declaration does not satisfy explicit `replay_policy=none` with `rehearsal_batch=0`; the executor rejects it before touching its saved evidence. It has no current qualification score.

Strict online successors: [`online-bcap`](bcap.md#online-bcap).

[Preserved receipt directory](../receipts/cuda/stream-bcap) · [Archived cohorts](../cohorts) · [Research history](../research-notes.md)

<a id="stream-bcap-adam"></a>
## `stream-bcap-adam`

[Exact declaration](../../../configs/forge/ideas/stream-bcap-adam.json)

Declared changes:

- ParticleGAN 0.9.6 public bcap_adam trainer preset
- Shared 64-frame FIFO memory, seven replay samples and velocity context
- Global non-saturating loss and cosine schedule (start=0.5, floor=0.05)

### Effective declaration

```json
{
  "memory_policy": {
    "decay": 0.9,
    "dim": 16,
    "kind": "fast_weight",
    "rehearsal_batch": 7,
    "reservoir_size": 64,
    "write_lr": 0.35
  },
  "recipe_overrides": {
    "betas": [
      0.0,
      0.999
    ],
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05
  },
  "recipe_preset": "bcap_adam"
}
```

**Historical, read-only.** This declaration does not satisfy explicit `replay_policy=none` with `rehearsal_batch=0`; the executor rejects it before touching its saved evidence. It has no current qualification score.

Strict online successors: [`online-bcap-adam`](bcap.md#online-bcap-adam).

[Preserved receipt directory](../receipts/cuda/stream-bcap-adam) · [Archived cohorts](../cohorts) · [Research history](../research-notes.md)
