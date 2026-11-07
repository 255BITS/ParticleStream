# KA2

[← Technique leaderboard](../technique-inventory.md)

KA2 uses a critic-local controller: input-gradient penalties and an averaged critic-gradient anchor are governed by penalty-call history and optimizer moment surprise. Noise, averaging for serving, readiness and damping remain configuration choices. In this short online suite, the ready recipes explicitly change critic beta2 so moment readiness can engage within 480 updates; that does not change a task's budget or gates.

[Implementation](../../../particlestream/tier1.py) · [Protocol](../../../EXPERIMENTATION.md) · [Terminology](../../../RESEARCH_GLOSSARY.md)

Each section describes one exact global recipe. Configuration and parent deltas come from the declarations; results come from Forge's current source-bound receipts. Required and stretch evidence stay separate. A final metric can meet its gate while the sustained verdict fails: every one of the last five checks must pass.

The installed engine is pinned to ParticleGAN/Forge 0.9.7. Older version or replay wording in declaration history describes provenance; the effective configuration below controls execution.

<a id="online-ka2"></a>
## `online-ka2`

[Exact declaration](../../../configs/forge/ideas/online-ka2.json)

Parent: [`stream-ka2`](ka2.md#stream-ka2).

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
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05
  },
  "recipe_preset": "ka2"
}
```

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `0d327da8115c91e5d323c0162229bb735c39aba5181ec0d92a6a2508f1799e11`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-ka2-required"></a>
### Required evidence

<a id="online-ka2-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.1875`; `modes=3`; `finite_template_tv=0.8125`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9; modes == 4.0 | [receipt](../receipts/cuda/online-ka2/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-ka2_seq_img_round_robin_bars4.gif) | 2026-11-07T17:55:21.175+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0.03125`; `modes=1`; `finite_template_tv=0.96875`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-ka2_seq_img_dwell_bars4.gif) | 2026-11-07T17:55:28.520+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.5004`; `direction_acc=0.28125`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-ka2_seq_img_translating_bar_next.gif) | 2026-11-07T17:55:36.617+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-ka2_seq_img_fork_rollout.gif) | 2026-11-07T17:55:44.885+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0.0625`; `modes=2`; `finite_template_tv=0.9375`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-ka2_seq_img_variable_dwell_bars4.gif) | 2026-11-07T17:55:52.179+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.84375`; `finite_template_tv=0.1875`; `support_modes=3`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-ka2/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-ka2_seq_img_rare_mode_bars4.gif) | 2026-11-07T17:55:59.585+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.5`; `prediction_rmse=0.28821`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-ka2_seq_img_delayed_cue.gif) | 2026-11-07T17:56:08.007+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-ka2_seq_img_persistent_fork.gif) | 2026-11-07T17:56:21.469+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.20061`; `xy_rmse=1.1256`; `step_accuracy=0.24219`; `orbit_completion=0`; `waypoint_modes=2`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-ka2_seq_img_circle_orbit.gif) | 2026-11-07T17:56:36.736+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.20458`; `s2c_rmse=0.20166`; `c2s_rmse=0.20751`; `xy_rmse=0.9925`; `step_accuracy=0`; `cycle_completion=0`; `waypoint_modes=0`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-ka2_seq_img_square_circle_cycle.gif) | 2026-11-07T17:57:03.644+00:00 |

<a id="online-ka2-stretch"></a>
### Stretch evidence

<a id="online-ka2-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-ka2/seq_minimal_dense4.json) · [goal.gif](../gifs/online-ka2_seq_minimal_dense4.gif) | 2026-11-07T17:57:06.806+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-ka2/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-ka2_seq_minimal_tokens4.gif) | 2026-11-07T17:57:10.189+00:00 |

<a id="online-ka2-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.49227`; `seen_max_abs_error=0.99603`; `heldout_response_rmse=0.41912`; `heldout_max_abs_error=0.69603`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-ka2/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-ka2_seq_minimal_kernel2x2.gif) | 2026-11-07T17:57:15.477+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.10417`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0.125`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-ka2/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-ka2_seq_minimal_addition2bit.gif) | 2026-11-07T17:57:21.043+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0164`; `relative_mse=0.95707`; `moving_direction_accuracy=0.60938`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-ka2_seq_robotics_fetch_reach.gif) | 2026-11-07T17:57:32.136+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.01433`; `relative_mse=1.3682`; `moving_direction_accuracy=0.51744`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-ka2_seq_robotics_fetch_push.gif) | 2026-11-07T17:57:44.884+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012317`; `relative_mse=0.92339`; `moving_direction_accuracy=0.68137`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-ka2_seq_robotics_fetch_slide.gif) | 2026-11-07T17:57:58.275+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012233`; `relative_mse=0.95348`; `moving_direction_accuracy=0.66667`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-ka2_seq_robotics_fetch_pick_and_place.gif) | 2026-11-07T17:58:11.038+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0092127`; `relative_mse=1.2554`; `moving_direction_accuracy=0.50117`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-ka2_seq_robotics_hand_reach.gif) | 2026-11-07T17:58:23.642+00:00 |

<a id="online-ka2-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.83971`; `interpolation_speed_error=0.89875`; `interpolation_raw_rmse=0.073697`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.5`; `extrapolation_centroid_rmse=0.97613`; `extrapolation_speed_error=0.90967`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.5625`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-ka2/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-ka2_seq_minimal_velocity_extrapolation.gif) | 2026-11-07T17:58:36.330+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-ka2 --device cuda`.

<a id="online-ka2-clean"></a>
## `online-ka2-clean`

[Exact declaration](../../../configs/forge/ideas/online-ka2-clean.json)

Parent: [`stream-ka2-clean`](ka2.md#stream-ka2-clean).

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
    "d_guard_ratio": 0.0,
    "direct_particle_gain": false,
    "ema_decay": 0.0,
    "input_noise_std": 0.0,
    "latent_damping_max_rate": 0.0,
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "output_noise_std": 0.0,
    "output_noise_warmup": 0.0
  },
  "recipe_preset": "ka2"
}
```

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `e7d672470cb865075a0e31920abecaea78bd0611c2d25e92fc23ceca937b4c13`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-ka2-clean-required"></a>
### Required evidence

<a id="online-ka2-clean-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.375`; `modes=2`; `finite_template_tv=0.625`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-clean/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-ka2-clean_seq_img_round_robin_bars4.gif) | 2026-11-07T17:58:45.413+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-clean/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-ka2-clean_seq_img_dwell_bars4.gif) | 2026-11-07T17:58:53.383+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.26554`; `direction_acc=0.65625`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2-clean/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-ka2-clean_seq_img_translating_bar_next.gif) | 2026-11-07T17:59:02.372+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2-clean/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-ka2-clean_seq_img_fork_rollout.gif) | 2026-11-07T17:59:11.511+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-clean/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-ka2-clean_seq_img_variable_dwell_bars4.gif) | 2026-11-07T17:59:19.525+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.71875`; `finite_template_tv=0.28125`; `support_modes=3`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-ka2-clean/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-ka2-clean_seq_img_rare_mode_bars4.gif) | 2026-11-07T17:59:27.725+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.51562`; `prediction_rmse=0.29736`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2-clean/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-ka2-clean_seq_img_delayed_cue.gif) | 2026-11-07T17:59:37.085+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2-clean/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-ka2-clean_seq_img_persistent_fork.gif) | 2026-11-07T17:59:52.149+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.22958`; `xy_rmse=1.059`; `step_accuracy=0.015625`; `orbit_completion=0`; `waypoint_modes=3`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2-clean/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-ka2-clean_seq_img_circle_orbit.gif) | 2026-11-07T18:00:09.531+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.21195`; `s2c_rmse=0.19485`; `c2s_rmse=0.22906`; `xy_rmse=1.1484`; `step_accuracy=0`; `cycle_completion=0`; `waypoint_modes=0`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2-clean/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-ka2-clean_seq_img_square_circle_cycle.gif) | 2026-11-07T18:00:39.362+00:00 |

<a id="online-ka2-clean-stretch"></a>
### Stretch evidence

<a id="online-ka2-clean-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-ka2-clean/seq_minimal_dense4.json) · [goal.gif](../gifs/online-ka2-clean_seq_minimal_dense4.gif) | 2026-11-07T18:00:43.224+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-ka2-clean/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-ka2-clean_seq_minimal_tokens4.gif) | 2026-11-07T18:00:46.924+00:00 |

<a id="online-ka2-clean-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.54018`; `seen_max_abs_error=1`; `heldout_response_rmse=0.47447`; `heldout_max_abs_error=0.7`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-ka2-clean/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-ka2-clean_seq_minimal_kernel2x2.gif) | 2026-11-07T18:00:52.481+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.083333`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-ka2-clean/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-ka2-clean_seq_minimal_addition2bit.gif) | 2026-11-07T18:00:57.866+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.11946`; `relative_mse=50.784`; `moving_direction_accuracy=0`; `finite_fraction=0`; failed terminal bounds: finite_fraction == 1.0; moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-clean/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-ka2-clean_seq_robotics_fetch_reach.gif) | 2026-11-07T18:01:09.531+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.013094`; `relative_mse=1.1424`; `moving_direction_accuracy=0.62791`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-clean/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-ka2-clean_seq_robotics_fetch_push.gif) | 2026-11-07T18:01:21.330+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012093`; `relative_mse=0.89015`; `moving_direction_accuracy=0.69608`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-clean/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-ka2-clean_seq_robotics_fetch_slide.gif) | 2026-11-07T18:01:33.138+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.013169`; `relative_mse=1.1049`; `moving_direction_accuracy=0.71491`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-clean/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-ka2-clean_seq_robotics_fetch_pick_and_place.gif) | 2026-11-07T18:01:45.066+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0091477`; `relative_mse=1.2378`; `moving_direction_accuracy=0.5514`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-clean/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-ka2-clean_seq_robotics_hand_reach.gif) | 2026-11-07T18:01:56.696+00:00 |

<a id="online-ka2-clean-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.79998`; `interpolation_speed_error=1.021`; `interpolation_raw_rmse=0.073697`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.47917`; `extrapolation_centroid_rmse=0.82188`; `extrapolation_speed_error=0.7749`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.5`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-ka2-clean/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-ka2-clean_seq_minimal_velocity_extrapolation.gif) | 2026-11-07T18:02:08.543+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-ka2-clean --device cuda`.

<a id="online-ka2-light-anchor"></a>
## `online-ka2-light-anchor`

[Exact declaration](../../../configs/forge/ideas/online-ka2-light-anchor.json)

Parent: [`stream-ka2-light-anchor`](ka2.md#stream-ka2-light-anchor).

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
    "d_guard_ratio": 0.0,
    "direct_particle_gain": false,
    "ema_decay": 0.0,
    "input_noise_std": 0.0,
    "latent_damping_max_rate": 0.0,
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "output_noise_std": 0.0,
    "output_noise_warmup": 0.0,
    "reg_anchor_weight": 0.1
  },
  "recipe_preset": "ka2"
}
```

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `a4e55cc1635ce80f042db8b4ba21457497dd5d9966ab25098e99f27d55ab502b`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-ka2-light-anchor-required"></a>
### Required evidence

<a id="online-ka2-light-anchor-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.375`; `modes=2`; `finite_template_tv=0.625`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_img_round_robin_bars4.gif) | 2026-11-07T18:02:17.410+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_img_dwell_bars4.gif) | 2026-11-07T18:02:25.346+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.26554`; `direction_acc=0.65625`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_img_translating_bar_next.gif) | 2026-11-07T18:02:34.302+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_img_fork_rollout.gif) | 2026-11-07T18:02:43.011+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_img_variable_dwell_bars4.gif) | 2026-11-07T18:02:50.978+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.71875`; `finite_template_tv=0.28125`; `support_modes=3`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_img_rare_mode_bars4.gif) | 2026-11-07T18:02:58.777+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.51562`; `prediction_rmse=0.29736`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_img_delayed_cue.gif) | 2026-11-07T18:03:07.868+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_img_persistent_fork.gif) | 2026-11-07T18:03:22.796+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.22958`; `xy_rmse=1.059`; `step_accuracy=0.015625`; `orbit_completion=0`; `waypoint_modes=3`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_img_circle_orbit.gif) | 2026-11-07T18:03:39.884+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.21195`; `s2c_rmse=0.19485`; `c2s_rmse=0.22906`; `xy_rmse=1.1484`; `step_accuracy=0`; `cycle_completion=0`; `waypoint_modes=0`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_img_square_circle_cycle.gif) | 2026-11-07T18:04:09.633+00:00 |

<a id="online-ka2-light-anchor-stretch"></a>
### Stretch evidence

<a id="online-ka2-light-anchor-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_minimal_dense4.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_minimal_dense4.gif) | 2026-11-07T18:04:13.351+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_minimal_tokens4.gif) | 2026-11-07T18:04:17.174+00:00 |

<a id="online-ka2-light-anchor-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.54018`; `seen_max_abs_error=1`; `heldout_response_rmse=0.47447`; `heldout_max_abs_error=0.7`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_minimal_kernel2x2.gif) | 2026-11-07T18:04:22.800+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.083333`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_minimal_addition2bit.gif) | 2026-11-07T18:04:28.420+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.11946`; `relative_mse=50.784`; `moving_direction_accuracy=0`; `finite_fraction=0`; failed terminal bounds: finite_fraction == 1.0; moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_robotics_fetch_reach.gif) | 2026-11-07T18:04:40.481+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.013094`; `relative_mse=1.1424`; `moving_direction_accuracy=0.62791`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_robotics_fetch_push.gif) | 2026-11-07T18:04:52.801+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012093`; `relative_mse=0.89015`; `moving_direction_accuracy=0.69608`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_robotics_fetch_slide.gif) | 2026-11-07T18:05:04.802+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.013169`; `relative_mse=1.1049`; `moving_direction_accuracy=0.71491`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_robotics_fetch_pick_and_place.gif) | 2026-11-07T18:05:16.846+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0091477`; `relative_mse=1.2378`; `moving_direction_accuracy=0.5514`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_robotics_hand_reach.gif) | 2026-11-07T18:05:28.483+00:00 |

<a id="online-ka2-light-anchor-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.79998`; `interpolation_speed_error=1.021`; `interpolation_raw_rmse=0.073697`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.47917`; `extrapolation_centroid_rmse=0.82188`; `extrapolation_speed_error=0.7749`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.5`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_minimal_velocity_extrapolation.gif) | 2026-11-07T18:05:40.386+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-ka2-light-anchor --device cuda`.

<a id="online-ka2-ready"></a>
## `online-ka2-ready`

[Exact declaration](../../../configs/forge/ideas/online-ka2-ready.json)

Parent: [`stream-ka2-ready`](ka2.md#stream-ka2-ready).

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
    "d_betas": [
      0.0,
      0.99
    ],
    "d_guard_ratio": 0.0,
    "direct_particle_gain": false,
    "ema_decay": 0.0,
    "input_noise_std": 0.0,
    "latent_damping_max_rate": 0.0,
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "output_noise_std": 0.0,
    "output_noise_warmup": 0.0
  },
  "recipe_preset": "ka2"
}
```

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `231f209dd7dbdd3d067596f4c0d0842ed36a2a1135190e924445869f8fab17fa`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-ka2-ready-required"></a>
### Required evidence

<a id="online-ka2-ready-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.71875`; `modes=3`; `finite_template_tv=0.34375`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-ready/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-ka2-ready_seq_img_round_robin_bars4.gif) | 2026-11-07T18:05:49.709+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0.09375`; `modes=1`; `finite_template_tv=0.90625`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-ready/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-ka2-ready_seq_img_dwell_bars4.gif) | 2026-11-07T18:05:58.037+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.16925`; `direction_acc=0.96875`; failed terminal bounds: next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2-ready/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-ka2-ready_seq_img_translating_bar_next.gif) | 2026-11-07T18:06:07.296+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2-ready/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-ka2-ready_seq_img_fork_rollout.gif) | 2026-11-07T18:06:16.796+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0.09375`; `modes=1`; `finite_template_tv=0.90625`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-ready/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-ka2-ready_seq_img_variable_dwell_bars4.gif) | 2026-11-07T18:06:25.092+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.75`; `finite_template_tv=0.25`; `support_modes=3`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-ka2-ready/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-ka2-ready_seq_img_rare_mode_bars4.gif) | 2026-11-07T18:06:33.545+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.51562`; `prediction_rmse=0.29875`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2-ready/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-ka2-ready_seq_img_delayed_cue.gif) | 2026-11-07T18:06:42.888+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0.0625`; `within_state_modes=1`; `branch_tv=0.9375`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2-ready/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-ka2-ready_seq_img_persistent_fork.gif) | 2026-11-07T18:06:58.295+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.22018`; `xy_rmse=0.92888`; `step_accuracy=0.089844`; `orbit_completion=0`; `waypoint_modes=4`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2-ready/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-ka2-ready_seq_img_circle_orbit.gif) | 2026-11-07T18:07:15.641+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.18624`; `s2c_rmse=0.1815`; `c2s_rmse=0.19097`; `xy_rmse=0.92833`; `step_accuracy=0.042969`; `cycle_completion=0`; `waypoint_modes=1`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2-ready/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-ka2-ready_seq_img_square_circle_cycle.gif) | 2026-11-07T18:07:46.440+00:00 |

<a id="online-ka2-ready-stretch"></a>
### Stretch evidence

<a id="online-ka2-ready-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-ka2-ready/seq_minimal_dense4.json) · [goal.gif](../gifs/online-ka2-ready_seq_minimal_dense4.gif) | 2026-11-07T18:07:50.024+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-ka2-ready/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-ka2-ready_seq_minimal_tokens4.gif) | 2026-11-07T18:07:53.912+00:00 |

<a id="online-ka2-ready-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.51551`; `seen_max_abs_error=1`; `heldout_response_rmse=0.44619`; `heldout_max_abs_error=0.7`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-ka2-ready/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-ka2-ready_seq_minimal_kernel2x2.gif) | 2026-11-07T18:07:59.476+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.083333`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-ka2-ready/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-ka2-ready_seq_minimal_addition2bit.gif) | 2026-11-07T18:08:05.106+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.016269`; `relative_mse=0.94183`; `moving_direction_accuracy=0.57292`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-ready/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-ka2-ready_seq_robotics_fetch_reach.gif) | 2026-11-07T18:08:17.080+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.013558`; `relative_mse=1.2247`; `moving_direction_accuracy=0.40698`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-ready/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-ka2-ready_seq_robotics_fetch_push.gif) | 2026-11-07T18:08:29.583+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.014074`; `relative_mse=1.2055`; `moving_direction_accuracy=0.70588`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-ready/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-ka2-ready_seq_robotics_fetch_slide.gif) | 2026-11-07T18:08:41.949+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011935`; `relative_mse=0.90754`; `moving_direction_accuracy=0.74123`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-ready/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-ka2-ready_seq_robotics_fetch_pick_and_place.gif) | 2026-11-07T18:08:54.356+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012402`; `relative_mse=2.2752`; `moving_direction_accuracy=0.55257`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-ready/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-ka2-ready_seq_robotics_hand_reach.gif) | 2026-11-07T18:09:06.304+00:00 |

<a id="online-ka2-ready-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.78891`; `interpolation_speed_error=0.98427`; `interpolation_raw_rmse=0.073698`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.5`; `extrapolation_centroid_rmse=0.90545`; `extrapolation_speed_error=0.77463`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.52083`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-ka2-ready/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-ka2-ready_seq_minimal_velocity_extrapolation.gif) | 2026-11-07T18:09:18.719+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-ka2-ready --device cuda`.

<a id="online-ka2-ready-causal"></a>
## `online-ka2-ready-causal`

[Exact declaration](../../../configs/forge/ideas/online-ka2-ready-causal.json)

Parent: [`online-ka2-ready`](ka2.md#online-ka2-ready).

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
    "d_betas": [
      0.0,
      0.99
    ],
    "d_guard_ratio": 0.0,
    "direct_particle_gain": false,
    "ema_decay": 0.0,
    "input_noise_std": 0.0,
    "latent_damping_max_rate": 0.0,
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "output_noise_std": 0.0,
    "output_noise_warmup": 0.0
  },
  "recipe_preset": "ka2"
}
```

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (4 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `0701ee63be1d12ebd448ee4654f76d102e21cb1c75894f255aa8c85f2115f5d9`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-ka2-ready-causal-required"></a>
### Required evidence

<a id="online-ka2-ready-causal-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.71875`; `modes=3`; `finite_template_tv=0.34375`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_img_round_robin_bars4.gif) | 2026-11-07T18:09:28.175+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0.09375`; `modes=1`; `finite_template_tv=0.90625`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_img_dwell_bars4.gif) | 2026-11-07T18:09:36.813+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.28302`; `direction_acc=0.5`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_img_translating_bar_next.gif) | 2026-11-07T18:09:46.386+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_img_fork_rollout.gif) | 2026-11-07T18:09:56.189+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0.09375`; `modes=1`; `finite_template_tv=0.90625`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_img_variable_dwell_bars4.gif) | 2026-11-07T18:10:04.601+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.75`; `finite_template_tv=0.25`; `support_modes=3`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_img_rare_mode_bars4.gif) | 2026-11-07T18:10:13.239+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.8125`; `prediction_rmse=0.23001`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_img_delayed_cue.gif) | 2026-11-07T18:10:22.556+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; `trajectory_consistency=1`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_img_persistent_fork.gif) | 2026-11-07T18:10:39.425+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.21281`; `xy_rmse=1.0293`; `step_accuracy=0.039062`; `orbit_completion=0`; `waypoint_modes=2`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_img_circle_orbit.gif) | 2026-11-07T18:10:58.127+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.19866`; `s2c_rmse=0.20204`; `c2s_rmse=0.19527`; `xy_rmse=0.99382`; `step_accuracy=0.011719`; `cycle_completion=0`; `waypoint_modes=1`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_img_square_circle_cycle.gif) | 2026-11-07T18:11:29.533+00:00 |

<a id="online-ka2-ready-causal-stretch"></a>
### Stretch evidence

<a id="online-ka2-ready-causal-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_minimal_dense4.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_minimal_dense4.gif) | 2026-11-07T18:11:33.244+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_minimal_tokens4.gif) | 2026-11-07T18:11:36.886+00:00 |

<a id="online-ka2-ready-causal-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.32884`; `seen_max_abs_error=0.98452`; `heldout_response_rmse=0.2448`; `heldout_max_abs_error=0.65409`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_minimal_kernel2x2.gif) | 2026-11-07T18:11:42.207+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.083333`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_minimal_addition2bit.gif) | 2026-11-07T18:11:47.744+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-ka2-ready-causal-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | no receipt | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-ka2-ready-causal --device cuda`.

<a id="online-ka2-stateless"></a>
## `online-ka2-stateless`

[Exact declaration](../../../configs/forge/ideas/online-ka2-stateless.json)

Parent: [`ka2`](ka2.md#ka2).

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
  "recipe_preset": "ka2"
}
```

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-ka2-stateless-required"></a>
### Required evidence

<a id="online-ka2-stateless-tier-1-required"></a>
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

<a id="online-ka2-stateless-stretch"></a>
### Stretch evidence

<a id="online-ka2-stateless-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-ka2-stateless-tier-2-stretch"></a>
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

<a id="online-ka2-stateless-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | no receipt | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-ka2-stateless --device cuda`.

<a id="online-ka2-transport-causal"></a>
## `online-ka2-transport-causal`

[Exact declaration](../../../configs/forge/ideas/online-ka2-transport-causal.json)

Parent: [`online-ka2-ready-causal`](ka2.md#online-ka2-ready-causal).

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
    "d_betas": [
      0.0,
      0.99
    ],
    "d_guard_ratio": 0.0,
    "direct_particle_gain": false,
    "ema_decay": 0.0,
    "input_noise_std": 0.0,
    "latent_damping_max_rate": 0.0,
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "output_noise_std": 0.0,
    "output_noise_warmup": 0.0
  },
  "recipe_preset": "ka2"
}
```

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-ka2-transport-causal-required"></a>
### Required evidence

<a id="online-ka2-transport-causal-tier-1-required"></a>
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

<a id="online-ka2-transport-causal-stretch"></a>
### Stretch evidence

<a id="online-ka2-transport-causal-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-ka2-transport-causal-tier-2-stretch"></a>
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

<a id="online-ka2-transport-causal-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | no receipt | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-ka2-transport-causal --device cuda`.

<a id="ka2"></a>
## `ka2`

[Exact declaration](../../../configs/forge/ideas/ka2.json)

Declared changes:

- KA2 critic EMA anchor on batch_size=1 sequential stream

### Effective declaration

```json
{
  "memory_policy": {
    "kind": "none"
  },
  "recipe_overrides": {},
  "recipe_preset": "ka2"
}
```

**Historical, read-only.** This declaration does not satisfy explicit `replay_policy=none` with `rehearsal_batch=0`; the executor rejects it before touching its saved evidence. It has no current qualification score.

Strict online successors: [`online-ka2-stateless`](ka2.md#online-ka2-stateless).

[Preserved receipt directory](../receipts/cuda/ka2) · [Archived cohorts](../cohorts) · [Research history](../research-notes.md)

<a id="stream-ka2"></a>
## `stream-ka2`

[Exact declaration](../../../configs/forge/ideas/stream-ka2.json)

Declared changes:

- ParticleGAN 0.9.6 public ka2 trainer preset
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
  "recipe_preset": "ka2"
}
```

**Historical, read-only.** This declaration does not satisfy explicit `replay_policy=none` with `rehearsal_batch=0`; the executor rejects it before touching its saved evidence. It has no current qualification score.

Strict online successors: [`online-ka2`](ka2.md#online-ka2).

[Preserved receipt directory](../receipts/cuda/stream-ka2) · [Archived cohorts](../cohorts) · [Research history](../research-notes.md)

<a id="stream-ka2-clean"></a>
## `stream-ka2-clean`

[Exact declaration](../../../configs/forge/ideas/stream-ka2-clean.json)

Parent: [`stream-ka2`](ka2.md#stream-ka2).

Declared changes:

- Disable auxiliary noise, EMA serving, critic guard and latent damping; retain KA2 critic EMA anchor weight=1

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `recipe_overrides.d_guard_ratio` | _unset_ | `0.0` |
| `recipe_overrides.direct_particle_gain` | _unset_ | `false` |
| `recipe_overrides.ema_decay` | _unset_ | `0.0` |
| `recipe_overrides.input_noise_std` | _unset_ | `0.0` |
| `recipe_overrides.latent_damping_max_rate` | _unset_ | `0.0` |
| `recipe_overrides.output_noise_std` | _unset_ | `0.0` |
| `recipe_overrides.output_noise_warmup` | _unset_ | `0.0` |

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
    "d_guard_ratio": 0.0,
    "direct_particle_gain": false,
    "ema_decay": 0.0,
    "input_noise_std": 0.0,
    "latent_damping_max_rate": 0.0,
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "output_noise_std": 0.0,
    "output_noise_warmup": 0.0
  },
  "recipe_preset": "ka2"
}
```

**Historical, read-only.** This declaration does not satisfy explicit `replay_policy=none` with `rehearsal_batch=0`; the executor rejects it before touching its saved evidence. It has no current qualification score.

Strict online successors: [`online-ka2-clean`](ka2.md#online-ka2-clean).

[Preserved receipt directory](../receipts/cuda/stream-ka2-clean) · [Archived cohorts](../cohorts) · [Research history](../research-notes.md)

<a id="stream-ka2-light-anchor"></a>
## `stream-ka2-light-anchor`

[Exact declaration](../../../configs/forge/ideas/stream-ka2-light-anchor.json)

Parent: [`stream-ka2-clean`](ka2.md#stream-ka2-clean).

Declared changes:

- Reduce KA2 EMA anchor weight from 1 to 0.1 globally; retain the KA2 penalty and formulation optimizer

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `recipe_overrides.reg_anchor_weight` | _unset_ | `0.1` |

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
    "d_guard_ratio": 0.0,
    "direct_particle_gain": false,
    "ema_decay": 0.0,
    "input_noise_std": 0.0,
    "latent_damping_max_rate": 0.0,
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "output_noise_std": 0.0,
    "output_noise_warmup": 0.0,
    "reg_anchor_weight": 0.1
  },
  "recipe_preset": "ka2"
}
```

**Historical, read-only.** This declaration does not satisfy explicit `replay_policy=none` with `rehearsal_batch=0`; the executor rejects it before touching its saved evidence. It has no current qualification score.

Strict online successors: [`online-ka2-light-anchor`](ka2.md#online-ka2-light-anchor).

[Preserved receipt directory](../receipts/cuda/stream-ka2-light-anchor) · [Archived cohorts](../cohorts) · [Research history](../research-notes.md)

<a id="stream-ka2-ready"></a>
## `stream-ka2-ready`

[Exact declaration](../../../configs/forge/ideas/stream-ka2-ready.json)

Parent: [`stream-ka2-clean`](ka2.md#stream-ka2-clean).

Declared changes:

- Critic beta2=0.99 (other optimizer betas unchanged) engages KA2 readiness around step 80 within the fixed 480-step budget

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `recipe_overrides.d_betas` | _unset_ | `[0.0, 0.99]` |

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
    "d_betas": [
      0.0,
      0.99
    ],
    "d_guard_ratio": 0.0,
    "direct_particle_gain": false,
    "ema_decay": 0.0,
    "input_noise_std": 0.0,
    "latent_damping_max_rate": 0.0,
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "output_noise_std": 0.0,
    "output_noise_warmup": 0.0
  },
  "recipe_preset": "ka2"
}
```

**Historical, read-only.** This declaration does not satisfy explicit `replay_policy=none` with `rehearsal_batch=0`; the executor rejects it before touching its saved evidence. It has no current qualification score.

Strict online successors: [`online-ka2-ready`](ka2.md#online-ka2-ready).

[Preserved receipt directory](../receipts/cuda/stream-ka2-ready) · [Archived cohorts](../cohorts) · [Research history](../research-notes.md)
