# Global-Stream-BCAP

[← Technique leaderboard](../technique-inventory.md)

Global-Stream-BCAP combines BCAP and bounded frame memory with one declared non-saturating loss and cosine learning-rate schedule across tasks. Its strict online descendant retains these optimizer settings while removing rehearsal. The historical recipe's rehearsal buffer must not be confused with the current one-arrival, one-update execution contract.

[Implementation](../../../particlestream/memory.py) · [Protocol](../../../EXPERIMENTATION.md) · [Terminology](../../../RESEARCH_GLOSSARY.md)

Each section describes one exact global recipe. Configuration and parent deltas come from the declarations; results come from Forge's current source-bound receipts. Required and stretch evidence stay separate. A final metric can meet its gate while the sustained verdict fails: every one of the last five checks must pass.

The installed engine is pinned to ParticleGAN/Forge 0.9.7. Older version or replay wording in declaration history describes provenance; the effective configuration below controls execution.

<a id="online-fast-weight-bcap-ns"></a>
## `online-fast-weight-bcap-ns`

[Exact declaration](../../../configs/forge/ideas/online-fast-weight-bcap-ns.json)

Parent: [`fast-weight-bcap-ns`](global-stream-bcap.md#fast-weight-bcap-ns).

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
    "betas": [
      0.0,
      0.999
    ],
    "d_lr_mult": 1.0,
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "prior_lr_mult": 2.0,
    "prior_regularizer": "none",
    "reg_arm": "b_cap",
    "reg_coeff": 1.0,
    "reg_kappa": 1.0
  },
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `2c9308c504e410afb7aa51abdf79b8adbe75ed4b2d9de22bae2b184663ad01b8`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-fast-weight-bcap-ns-required"></a>
### Required evidence

<a id="online-fast-weight-bcap-ns-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.78125`; `modes=4`; `finite_template_tv=0.25`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_img_round_robin_bars4.gif) | 2026-11-07T17:48:04.492+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_img_dwell_bars4.gif) | 2026-11-07T17:48:10.917+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.33884`; `direction_acc=0.5`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_img_translating_bar_next.gif) | 2026-11-07T17:48:18.030+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_img_fork_rollout.gif) | 2026-11-07T17:48:24.911+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_img_variable_dwell_bars4.gif) | 2026-11-07T17:48:31.108+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.53125`; `finite_template_tv=0.46875`; `support_modes=4`; `rare_mass=0.03125`; `rare_mass=0.03125`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_img_rare_mode_bars4.gif) | 2026-11-07T17:48:37.118+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.51562`; `prediction_rmse=0.2986`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_img_delayed_cue.gif) | 2026-11-07T17:48:44.145+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0.03125`; `within_state_modes=1`; `branch_tv=0.96875`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_img_persistent_fork.gif) | 2026-11-07T17:48:56.282+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.11022`; `xy_rmse=0.33159`; `step_accuracy=0.32031`; `orbit_completion=0`; `waypoint_modes=8`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_img_circle_orbit.gif) | 2026-11-07T17:49:10.632+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.16012`; `s2c_rmse=0.15918`; `c2s_rmse=0.16107`; `xy_rmse=0.74504`; `step_accuracy=0.015625`; `cycle_completion=0`; `waypoint_modes=1`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_img_square_circle_cycle.gif) | 2026-11-07T17:49:35.058+00:00 |

<a id="online-fast-weight-bcap-ns-stretch"></a>
### Stretch evidence

<a id="online-fast-weight-bcap-ns-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_minimal_dense4.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_minimal_dense4.gif) | 2026-11-07T17:49:38.182+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_minimal_tokens4.gif) | 2026-11-07T17:49:41.145+00:00 |

<a id="online-fast-weight-bcap-ns-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.39622`; `seen_max_abs_error=0.96251`; `heldout_response_rmse=0.30054`; `heldout_max_abs_error=0.66251`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_minimal_kernel2x2.gif) | 2026-11-07T17:49:45.790+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.083333`; `seen_min_bit_confidence=1.9478e-09`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=1.9478e-09`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_minimal_addition2bit.gif) | 2026-11-07T17:49:50.753+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.018141`; `relative_mse=1.171`; `moving_direction_accuracy=0.5`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_robotics_fetch_reach.gif) | 2026-11-07T17:50:01.462+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012828`; `relative_mse=1.0964`; `moving_direction_accuracy=0.5814`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_robotics_fetch_push.gif) | 2026-11-07T17:50:12.089+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.013395`; `relative_mse=1.0921`; `moving_direction_accuracy=0.4951`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_robotics_fetch_slide.gif) | 2026-11-07T17:50:22.777+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011575`; `relative_mse=0.85355`; `moving_direction_accuracy=0.58333`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_robotics_fetch_pick_and_place.gif) | 2026-11-07T17:50:33.060+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010953`; `relative_mse=1.7745`; `moving_direction_accuracy=0.51986`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_robotics_hand_reach.gif) | 2026-11-07T17:50:47.738+00:00 |

<a id="online-fast-weight-bcap-ns-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.77773`; `interpolation_speed_error=0.8295`; `interpolation_raw_rmse=0.073698`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.47917`; `extrapolation_centroid_rmse=0.91625`; `extrapolation_speed_error=0.7741`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.5625`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_minimal_velocity_extrapolation.gif) | 2026-11-07T17:51:02.731+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-fast-weight-bcap-ns --device cuda`.

<a id="fast-weight-bcap-ns"></a>
## `fast-weight-bcap-ns`

[Exact declaration](../../../configs/forge/ideas/fast-weight-bcap-ns.json)

Parent: [`fast-weight-bcap`](fastweight-bcap.md#fast-weight-bcap).

Declared changes:

- Non-saturating GAN loss for batch_size=1 stability
- Cosine learning rate annealing (lr=0.005, betas=(0.0, 0.999), lr_schedule=cosine, lr_anneal_start=0.5, lr_floor=0.05)
- 64-frame rehearsal reservoir spanning full dwell=16 4-mode cycle
- Temporal velocity direction memory with static noise gating (||diff|| > 0.35)
- Joint non-linear conditional critic architecture
- Dwell-aware rehearsal replay and task-isolated context conditioning

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.rehearsal_batch` | _unset_ | `7` |
| `recipe_overrides.betas` | _unset_ | `[0.0, 0.999]` |
| `recipe_overrides.d_lr_mult` | _unset_ | `1.0` |
| `recipe_overrides.loss` | _unset_ | `"non_saturating"` |
| `recipe_overrides.lr` | _unset_ | `0.005` |
| `recipe_overrides.lr_anneal_start` | _unset_ | `0.5` |
| `recipe_overrides.lr_floor` | _unset_ | `0.05` |
| `recipe_overrides.lr_schedule` | _unset_ | `"cosine"` |
| `recipe_overrides.network_lr_floor` | _unset_ | `0.05` |
| `recipe_overrides.prior_lr_mult` | _unset_ | `2.0` |
| `recipe_overrides.prior_regularizer` | _unset_ | `"none"` |
| `recipe_overrides.reg_arm` | _unset_ | `"b_cap"` |
| `recipe_overrides.reg_coeff` | _unset_ | `1.0` |
| `recipe_overrides.reg_kappa` | _unset_ | `1.0` |

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
    "d_lr_mult": 1.0,
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "prior_lr_mult": 2.0,
    "prior_regularizer": "none",
    "reg_arm": "b_cap",
    "reg_coeff": 1.0,
    "reg_kappa": 1.0
  },
  "recipe_preset": "bcap_adam"
}
```

**Historical, read-only.** This declaration does not satisfy explicit `replay_policy=none` with `rehearsal_batch=0`; the executor rejects it before touching its saved evidence. It has no current qualification score.

Strict online successors: [`online-fast-weight-bcap-ns`](global-stream-bcap.md#online-fast-weight-bcap-ns).

[Preserved receipt directory](../receipts/cuda/fast-weight-bcap-ns) · [Archived cohorts](../cohorts) · [Research history](../research-notes.md)
