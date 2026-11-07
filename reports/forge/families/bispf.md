# BISPF

[← Technique leaderboard](../technique-inventory.md)

Born-Infeld Sinkhorn Particle Flow combines an elastic critic-gradient penalty on real, fake and transport-chord inputs with a zero-mean logit gauge. Its public presets also provide particle-prior regularization and thermal policies. ParticleStream's declared recipes override some preset choices: inspect each exact configuration rather than assuming every BISPF option is enabled. With a real and fake batch of one, a Sinkhorn coupling has only one pair.

[Implementation](../../../particlestream/tier1.py) · [Protocol](../../../EXPERIMENTATION.md) · [Terminology](../../../RESEARCH_GLOSSARY.md)

Each section describes one exact global recipe. Configuration and parent deltas come from the declarations; results come from Forge's current source-bound receipts. Required and stretch evidence stay separate. A final metric can meet its gate while the sustained verdict fails: every one of the last five checks must pass.

The installed engine is pinned to ParticleGAN/Forge 0.9.7. Older version or replay wording in declaration history describes provenance; the effective configuration below controls execution.

<a id="online-bispf-frozen"></a>
## `online-bispf-frozen`

[Exact declaration](../../../configs/forge/ideas/online-bispf-frozen.json)

Parent: [`online-bispf-early`](bispf.md#online-bispf-early).

Declared changes:

- Freeze the initial prior table to remove sparse-row optimizer drift; all other settings unchanged

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `recipe_overrides.prior_update` | _unset_ | `"frozen"` |

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
    "lr_anneal_start": 0.25,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "prior_reg": 0.0,
    "prior_regularizer": "none",
    "prior_update": "frozen",
    "training_precision": "float32"
  },
  "recipe_preset": "bispf"
}
```

Current `cuda`, seed 0: **FAIL**, 1/10 required passes (10 measured), 0/10 stretch passes (2 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `02f098fcb214f4f7874f82ef83f6629ce8cfc85c2c3f3ac28b84ec45b7803676`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-bispf-frozen-required"></a>
### Required evidence

<a id="online-bispf-frozen-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.34375`; `modes=3`; `finite_template_tv=0.65625`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-frozen/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bispf-frozen_seq_img_round_robin_bars4.gif) | 2026-11-07T17:14:01.053+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0.0625`; `modes=1`; `finite_template_tv=0.9375`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-frozen/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-frozen_seq_img_dwell_bars4.gif) | 2026-11-07T17:14:08.208+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.071489`; `direction_acc=1` | [receipt](../receipts/cuda/online-bispf-frozen/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bispf-frozen_seq_img_translating_bar_next.gif) | 2026-11-07T17:14:16.121+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-frozen/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bispf-frozen_seq_img_fork_rollout.gif) | 2026-11-07T17:14:23.905+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-frozen/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-frozen_seq_img_variable_dwell_bars4.gif) | 2026-11-07T17:14:31.254+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.375`; `finite_template_tv=0.625`; `support_modes=3`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-bispf-frozen/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bispf-frozen_seq_img_rare_mode_bars4.gif) | 2026-11-07T17:14:38.121+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.48438`; `prediction_rmse=0.29732`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf-frozen/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bispf-frozen_seq_img_delayed_cue.gif) | 2026-11-07T17:14:46.218+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0.28125`; `within_state_modes=1`; `branch_tv=0.71875`; `trajectory_consistency=0.21875`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-frozen/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bispf-frozen_seq_img_persistent_fork.gif) | 2026-11-07T17:14:59.039+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.19472`; `xy_rmse=0.87879`; `step_accuracy=0.011719`; `orbit_completion=0`; `waypoint_modes=2`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-frozen/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bispf-frozen_seq_img_circle_orbit.gif) | 2026-11-07T17:15:14.549+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.22245`; `s2c_rmse=0.2304`; `c2s_rmse=0.21449`; `xy_rmse=1.1552`; `step_accuracy=0`; `cycle_completion=0`; `waypoint_modes=0`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-frozen/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bispf-frozen_seq_img_square_circle_cycle.gif) | 2026-11-07T17:15:40.480+00:00 |

<a id="online-bispf-frozen-stretch"></a>
### Stretch evidence

<a id="online-bispf-frozen-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bispf-frozen/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bispf-frozen_seq_minimal_dense4.gif) | 2026-11-07T17:15:43.826+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bispf-frozen/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bispf-frozen_seq_minimal_tokens4.gif) | 2026-11-07T17:15:47.057+00:00 |

<a id="online-bispf-frozen-tier-2-stretch"></a>
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

<a id="online-bispf-frozen-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | no receipt | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bispf-frozen --device cuda`.

<a id="online-bispf"></a>
## `online-bispf`

[Exact declaration](../../../configs/forge/ideas/online-bispf.json)

Parent: [`stream-bispf`](bispf.md#stream-bispf).

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
    "network_lr_floor": 0.05,
    "prior_reg": 0.0,
    "prior_regularizer": "none",
    "training_precision": "float32"
  },
  "recipe_preset": "bispf"
}
```

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `57f1ef4a06e343a80d329d3c6f09a59cdf1feabb50aa4fd4c757e9a99314686e`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-bispf-required"></a>
### Required evidence

<a id="online-bispf-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.875`; `modes=4`; `finite_template_tv=0.15625`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9 | [receipt](../receipts/cuda/online-bispf/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bispf_seq_img_round_robin_bars4.gif) | 2026-11-07T17:03:53.675+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bispf_seq_img_dwell_bars4.gif) | 2026-11-07T17:04:00.556+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.13298`; `direction_acc=0.96875`; failed terminal bounds: next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bispf_seq_img_translating_bar_next.gif) | 2026-11-07T17:04:08.126+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bispf_seq_img_fork_rollout.gif) | 2026-11-07T17:04:15.809+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bispf_seq_img_variable_dwell_bars4.gif) | 2026-11-07T17:04:23.006+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.8125`; `finite_template_tv=0.1875`; `support_modes=3`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-bispf/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bispf_seq_img_rare_mode_bars4.gif) | 2026-11-07T17:04:30.136+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.5`; `prediction_rmse=0.30088`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bispf_seq_img_delayed_cue.gif) | 2026-11-07T17:04:37.914+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0.03125`; `within_state_modes=1`; `branch_tv=0.96875`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bispf_seq_img_persistent_fork.gif) | 2026-11-07T17:04:50.449+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.22307`; `xy_rmse=1.0586`; `step_accuracy=0`; `orbit_completion=0`; `waypoint_modes=0`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bispf_seq_img_circle_orbit.gif) | 2026-11-07T17:05:05.581+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.20666`; `s2c_rmse=0.20285`; `c2s_rmse=0.21048`; `xy_rmse=1.0581`; `step_accuracy=0`; `cycle_completion=0`; `waypoint_modes=0`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bispf_seq_img_square_circle_cycle.gif) | 2026-11-07T17:05:31.209+00:00 |

<a id="online-bispf-stretch"></a>
### Stretch evidence

<a id="online-bispf-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bispf/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bispf_seq_minimal_dense4.gif) | 2026-11-07T17:05:34.474+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bispf/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bispf_seq_minimal_tokens4.gif) | 2026-11-07T17:05:37.564+00:00 |

<a id="online-bispf-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.49246`; `seen_max_abs_error=0.99973`; `heldout_response_rmse=0.41934`; `heldout_max_abs_error=0.69973`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bispf/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bispf_seq_minimal_kernel2x2.gif) | 2026-11-07T17:05:42.324+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.125`; `seen_min_bit_confidence=2.0158e-20`; `heldout_exact_accuracy=0.125`; `heldout_min_bit_confidence=2.0158e-20`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bispf/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bispf_seq_minimal_addition2bit.gif) | 2026-11-07T17:05:46.919+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.01715`; `relative_mse=1.0466`; `moving_direction_accuracy=0.60417`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bispf_seq_robotics_fetch_reach.gif) | 2026-11-07T17:05:57.077+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.01476`; `relative_mse=1.4516`; `moving_direction_accuracy=0.43605`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bispf_seq_robotics_fetch_push.gif) | 2026-11-07T17:06:07.244+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.013371`; `relative_mse=1.0882`; `moving_direction_accuracy=0.71569`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bispf_seq_robotics_fetch_slide.gif) | 2026-11-07T17:06:17.510+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011956`; `relative_mse=0.91078`; `moving_direction_accuracy=0.58333`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bispf_seq_robotics_fetch_pick_and_place.gif) | 2026-11-07T17:06:27.689+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0091335`; `relative_mse=1.2339`; `moving_direction_accuracy=0.49065`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bispf_seq_robotics_hand_reach.gif) | 2026-11-07T17:06:37.894+00:00 |

<a id="online-bispf-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.80453`; `interpolation_speed_error=0.65749`; `interpolation_raw_rmse=0.073697`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.45833`; `extrapolation_centroid_rmse=0.9529`; `extrapolation_speed_error=0.86176`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.45833`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bispf/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bispf_seq_minimal_velocity_extrapolation.gif) | 2026-11-07T17:06:47.931+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bispf --device cuda`.

<a id="online-bispf-early"></a>
## `online-bispf-early`

[Exact declaration](../../../configs/forge/ideas/online-bispf-early.json)

Parent: [`stream-bispf-early`](bispf.md#stream-bispf-early).

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
    "lr_anneal_start": 0.25,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "prior_reg": 0.0,
    "prior_regularizer": "none",
    "training_precision": "float32"
  },
  "recipe_preset": "bispf"
}
```

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `5667f5aea18b98d039c0ad9bbbfaa4d2b5efd920472d78b7481c10f5ffb70a03`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-bispf-early-required"></a>
### Required evidence

<a id="online-bispf-early-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 1/5 | `hq=0.90625`; `modes=4`; `finite_template_tv=0.09375`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9 | [receipt](../receipts/cuda/online-bispf-early/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bispf-early_seq_img_round_robin_bars4.gif) | 2026-11-07T17:06:55.806+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-early/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-early_seq_img_dwell_bars4.gif) | 2026-11-07T17:07:02.670+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.18288`; `direction_acc=0.96875`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf-early/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bispf-early_seq_img_translating_bar_next.gif) | 2026-11-07T17:07:10.484+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-early/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bispf-early_seq_img_fork_rollout.gif) | 2026-11-07T17:07:18.354+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-early/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-early_seq_img_variable_dwell_bars4.gif) | 2026-11-07T17:07:25.476+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.71875`; `finite_template_tv=0.28125`; `support_modes=3`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-bispf-early/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bispf-early_seq_img_rare_mode_bars4.gif) | 2026-11-07T17:07:32.563+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.5`; `prediction_rmse=0.29866`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf-early/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bispf-early_seq_img_delayed_cue.gif) | 2026-11-07T17:07:40.708+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-early/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bispf-early_seq_img_persistent_fork.gif) | 2026-11-07T17:07:53.420+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.23511`; `xy_rmse=1.0814`; `step_accuracy=0`; `orbit_completion=0`; `waypoint_modes=0`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-early/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bispf-early_seq_img_circle_orbit.gif) | 2026-11-07T17:08:08.393+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.20879`; `s2c_rmse=0.20123`; `c2s_rmse=0.21635`; `xy_rmse=1.0508`; `step_accuracy=0`; `cycle_completion=0`; `waypoint_modes=0`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-early/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bispf-early_seq_img_square_circle_cycle.gif) | 2026-11-07T17:08:34.438+00:00 |

<a id="online-bispf-early-stretch"></a>
### Stretch evidence

<a id="online-bispf-early-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bispf-early/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bispf-early_seq_minimal_dense4.gif) | 2026-11-07T17:08:37.621+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bispf-early/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bispf-early_seq_minimal_tokens4.gif) | 2026-11-07T17:08:41.041+00:00 |

<a id="online-bispf-early-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.48345`; `seen_max_abs_error=0.99634`; `heldout_response_rmse=0.40872`; `heldout_max_abs_error=0.69634`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bispf-early/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bispf-early_seq_minimal_kernel2x2.gif) | 2026-11-07T17:08:45.736+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.125`; `seen_min_bit_confidence=8.8454e-26`; `heldout_exact_accuracy=0.125`; `heldout_min_bit_confidence=8.8454e-26`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bispf-early/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bispf-early_seq_minimal_addition2bit.gif) | 2026-11-07T17:08:50.692+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.01634`; `relative_mse=0.95011`; `moving_direction_accuracy=0.5625`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-early/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bispf-early_seq_robotics_fetch_reach.gif) | 2026-11-07T17:09:00.667+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.017007`; `relative_mse=1.9272`; `moving_direction_accuracy=0.65116`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-early/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bispf-early_seq_robotics_fetch_push.gif) | 2026-11-07T17:09:10.762+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.041838`; `relative_mse=10.654`; `moving_direction_accuracy=0.80392`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-early/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bispf-early_seq_robotics_fetch_slide.gif) | 2026-11-07T17:09:21.634+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.01046`; `relative_mse=0.69706`; `moving_direction_accuracy=0.64912`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-bispf-early/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bispf-early_seq_robotics_fetch_pick_and_place.gif) | 2026-11-07T17:09:32.807+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0096267`; `relative_mse=1.3708`; `moving_direction_accuracy=0.52921`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-early/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bispf-early_seq_robotics_hand_reach.gif) | 2026-11-07T17:09:43.767+00:00 |

<a id="online-bispf-early-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.79642`; `interpolation_speed_error=0.65746`; `interpolation_raw_rmse=0.073697`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.45833`; `extrapolation_centroid_rmse=0.94778`; `extrapolation_speed_error=0.85925`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.47917`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bispf-early/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bispf-early_seq_minimal_velocity_extrapolation.gif) | 2026-11-07T17:09:54.387+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bispf-early --device cuda`.

<a id="online-bispf-early-causal"></a>
## `online-bispf-early-causal`

[Exact declaration](../../../configs/forge/ideas/online-bispf-early-causal.json)

Parent: [`online-bispf-early`](bispf.md#online-bispf-early).

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
    "lr_anneal_start": 0.25,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "prior_reg": 0.0,
    "prior_regularizer": "none",
    "training_precision": "float32"
  },
  "recipe_preset": "bispf"
}
```

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `77255100d934a00e74e80b5ae5d85021794aebad6ebbb840f7aa79b3222bc9b0`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-bispf-early-causal-required"></a>
### Required evidence

<a id="online-bispf-early-causal-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 1/5 | `hq=0.90625`; `modes=4`; `finite_template_tv=0.09375`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_img_round_robin_bars4.gif) | 2026-11-07T17:10:02.892+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_img_dwell_bars4.gif) | 2026-11-07T17:10:13.711+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.30693`; `direction_acc=0.5`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_img_translating_bar_next.gif) | 2026-11-07T17:10:25.865+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_img_fork_rollout.gif) | 2026-11-07T17:10:38.257+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_img_variable_dwell_bars4.gif) | 2026-11-07T17:10:48.533+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.71875`; `finite_template_tv=0.28125`; `support_modes=3`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_img_rare_mode_bars4.gif) | 2026-11-07T17:11:01.610+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.65625`; `prediction_rmse=0.30451`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_img_delayed_cue.gif) | 2026-11-07T17:11:15.067+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_img_persistent_fork.gif) | 2026-11-07T17:11:37.382+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.2136`; `xy_rmse=0.99377`; `step_accuracy=0.011719`; `orbit_completion=0`; `waypoint_modes=1`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_img_circle_orbit.gif) | 2026-11-07T17:11:59.681+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.10563`; `s2c_rmse=0.10199`; `c2s_rmse=0.10928`; `xy_rmse=0.43294`; `step_accuracy=0.25781`; `cycle_completion=0`; `waypoint_modes=4`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_img_square_circle_cycle.gif) | 2026-11-07T17:12:29.012+00:00 |

<a id="online-bispf-early-causal-stretch"></a>
### Stretch evidence

<a id="online-bispf-early-causal-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_minimal_dense4.gif) | 2026-11-07T17:12:32.434+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_minimal_tokens4.gif) | 2026-11-07T17:12:35.752+00:00 |

<a id="online-bispf-early-causal-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.45419`; `seen_max_abs_error=1`; `heldout_response_rmse=0.37517`; `heldout_max_abs_error=0.7`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_minimal_kernel2x2.gif) | 2026-11-07T17:12:40.889+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.20833`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0.0625`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_minimal_addition2bit.gif) | 2026-11-07T17:12:45.985+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011406`; `relative_mse=0.46292`; `moving_direction_accuracy=0.83854`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_robotics_fetch_reach.gif) | 2026-11-07T17:12:56.991+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.013763`; `relative_mse=1.262`; `moving_direction_accuracy=0.65698`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_robotics_fetch_push.gif) | 2026-11-07T17:13:07.915+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011325`; `relative_mse=0.78063`; `moving_direction_accuracy=0.82353`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_robotics_fetch_slide.gif) | 2026-11-07T17:13:19.252+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0094148`; `relative_mse=0.56472`; `moving_direction_accuracy=0.65351`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_robotics_fetch_pick_and_place.gif) | 2026-11-07T17:13:30.283+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0089819`; `relative_mse=1.1933`; `moving_direction_accuracy=0.55491`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_robotics_hand_reach.gif) | 2026-11-07T17:13:41.505+00:00 |

<a id="online-bispf-early-causal-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.75006`; `interpolation_speed_error=0.63178`; `interpolation_raw_rmse=0.073697`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.45833`; `extrapolation_centroid_rmse=0.92752`; `extrapolation_speed_error=0.7742`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.39583`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_minimal_velocity_extrapolation.gif) | 2026-11-07T17:13:52.994+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bispf-early-causal --device cuda`.

<a id="online-bispf-momentum"></a>
## `online-bispf-momentum`

[Exact declaration](../../../configs/forge/ideas/online-bispf-momentum.json)

Parent: [`online-bispf-early`](bispf.md#online-bispf-early).

Declared changes:

- Critic first-moment beta 0.95 carries optimizer evidence across correlated observations; one update per arrival

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `recipe_overrides.d_betas` | _unset_ | `[0.95, 0.999]` |

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
      0.95,
      0.999
    ],
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.25,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "prior_reg": 0.0,
    "prior_regularizer": "none",
    "training_precision": "float32"
  },
  "recipe_preset": "bispf"
}
```

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-bispf-momentum-required"></a>
### Required evidence

<a id="online-bispf-momentum-tier-1-required"></a>
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

<a id="online-bispf-momentum-stretch"></a>
### Stretch evidence

<a id="online-bispf-momentum-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bispf-momentum-tier-2-stretch"></a>
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

<a id="online-bispf-momentum-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | no receipt | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bispf-momentum --device cuda`.

<a id="online-bispf-optimized"></a>
## `online-bispf-optimized`

[Exact declaration](../../../configs/forge/ideas/online-bispf-optimized.json)

Parent: [`stream-bispf-optimized`](bispf.md#stream-bispf-optimized).

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
    "network_lr_floor": 0.05,
    "prior_reg": 0.0,
    "prior_regularizer": "none",
    "training_precision": "float32"
  },
  "recipe_preset": "bispf_optimized"
}
```

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-bispf-optimized-required"></a>
### Required evidence

<a id="online-bispf-optimized-tier-1-required"></a>
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

<a id="online-bispf-optimized-stretch"></a>
### Stretch evidence

<a id="online-bispf-optimized-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bispf-optimized-tier-2-stretch"></a>
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

<a id="online-bispf-optimized-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | no receipt | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bispf-optimized --device cuda`.

<a id="online-bispf-optimized-cool"></a>
## `online-bispf-optimized-cool`

[Exact declaration](../../../configs/forge/ideas/online-bispf-optimized-cool.json)

Parent: [`stream-bispf-optimized-cool`](bispf.md#stream-bispf-optimized-cool).

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
    "d_lr_mult": 1.0,
    "loss": "non_saturating",
    "lr": 0.003,
    "lr_anneal_start": 0.25,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "prior_lr_mult": 2.0,
    "prior_reg": 0.0,
    "prior_regularizer": "none",
    "training_precision": "float32"
  },
  "recipe_preset": "bispf_optimized"
}
```

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-bispf-optimized-cool-required"></a>
### Required evidence

<a id="online-bispf-optimized-cool-tier-1-required"></a>
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

<a id="online-bispf-optimized-cool-stretch"></a>
### Stretch evidence

<a id="online-bispf-optimized-cool-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bispf-optimized-cool-tier-2-stretch"></a>
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

<a id="online-bispf-optimized-cool-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | no receipt | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bispf-optimized-cool --device cuda`.

<a id="online-bispf-slow-critic"></a>
## `online-bispf-slow-critic`

[Exact declaration](../../../configs/forge/ideas/online-bispf-slow-critic.json)

Parent: [`online-bispf-early`](bispf.md#online-bispf-early).

Declared changes:

- Reduce critic learning-rate multiplier from 1 to 0.25 to limit chasing one observation

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `recipe_overrides.d_lr_mult` | _unset_ | `0.25` |

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
    "d_lr_mult": 0.25,
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.25,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "prior_reg": 0.0,
    "prior_regularizer": "none",
    "training_precision": "float32"
  },
  "recipe_preset": "bispf"
}
```

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-bispf-slow-critic-required"></a>
### Required evidence

<a id="online-bispf-slow-critic-tier-1-required"></a>
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

<a id="online-bispf-slow-critic-stretch"></a>
### Stretch evidence

<a id="online-bispf-slow-critic-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bispf-slow-critic-tier-2-stretch"></a>
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

<a id="online-bispf-slow-critic-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | no receipt | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bispf-slow-critic --device cuda`.

<a id="online-bispf-slow-prior"></a>
## `online-bispf-slow-prior`

[Exact declaration](../../../configs/forge/ideas/online-bispf-slow-prior.json)

Parent: [`online-bispf-early`](bispf.md#online-bispf-early).

Declared changes:

- Reduce prior learning-rate multiplier from 2 to 0.25 for sparse batch-size-1 row visits

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `recipe_overrides.prior_lr_mult` | _unset_ | `0.25` |

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
    "lr_anneal_start": 0.25,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "prior_lr_mult": 0.25,
    "prior_reg": 0.0,
    "prior_regularizer": "none",
    "training_precision": "float32"
  },
  "recipe_preset": "bispf"
}
```

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-bispf-slow-prior-required"></a>
### Required evidence

<a id="online-bispf-slow-prior-tier-1-required"></a>
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

<a id="online-bispf-slow-prior-stretch"></a>
### Stretch evidence

<a id="online-bispf-slow-prior-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bispf-slow-prior-tier-2-stretch"></a>
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

<a id="online-bispf-slow-prior-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | no receipt | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bispf-slow-prior --device cuda`.

<a id="online-bispf-transport-causal"></a>
## `online-bispf-transport-causal`

[Exact declaration](../../../configs/forge/ideas/online-bispf-transport-causal.json)

Parent: [`online-bispf-early-causal`](bispf.md#online-bispf-early-causal).

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
    "lr_anneal_start": 0.25,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "prior_reg": 0.0,
    "prior_regularizer": "none",
    "training_precision": "float32"
  },
  "recipe_preset": "bispf"
}
```

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-bispf-transport-causal-required"></a>
### Required evidence

<a id="online-bispf-transport-causal-tier-1-required"></a>
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

<a id="online-bispf-transport-causal-stretch"></a>
### Stretch evidence

<a id="online-bispf-transport-causal-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-bispf-transport-causal-tier-2-stretch"></a>
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

<a id="online-bispf-transport-causal-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | no receipt | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bispf-transport-causal --device cuda`.

<a id="stream-bispf"></a>
## `stream-bispf`

[Exact declaration](../../../configs/forge/ideas/stream-bispf.json)

Declared changes:

- ParticleGAN 0.9.6 public bispf trainer preset
- Shared 64-frame FIFO memory, seven replay samples and velocity context
- Global non-saturating loss and cosine schedule (start=0.5, floor=0.05)
- SBNP disabled for discrete target modes; FP32 training

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
    "network_lr_floor": 0.05,
    "prior_reg": 0.0,
    "prior_regularizer": "none",
    "training_precision": "float32"
  },
  "recipe_preset": "bispf"
}
```

**Historical, read-only.** This declaration does not satisfy explicit `replay_policy=none` with `rehearsal_batch=0`; the executor rejects it before touching its saved evidence. It has no current qualification score.

Strict online successors: [`online-bispf`](bispf.md#online-bispf).

[Preserved receipt directory](../receipts/cuda/stream-bispf) · [Archived cohorts](../cohorts) · [Research history](../research-notes.md)

<a id="stream-bispf-early"></a>
## `stream-bispf-early`

[Exact declaration](../../../configs/forge/ideas/stream-bispf-early.json)

Parent: [`stream-bispf`](bispf.md#stream-bispf).

Declared changes:

- Anneal from step 120 (0.25 of budget) to settle before the sustained terminal window

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `recipe_overrides.lr_anneal_start` | `0.5` | `0.25` |

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
    "lr_anneal_start": 0.25,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "prior_reg": 0.0,
    "prior_regularizer": "none",
    "training_precision": "float32"
  },
  "recipe_preset": "bispf"
}
```

**Historical, read-only.** This declaration does not satisfy explicit `replay_policy=none` with `rehearsal_batch=0`; the executor rejects it before touching its saved evidence. It has no current qualification score.

Strict online successors: [`online-bispf-early`](bispf.md#online-bispf-early).

[Preserved receipt directory](../receipts/cuda/stream-bispf-early) · [Archived cohorts](../cohorts) · [Research history](../research-notes.md)

<a id="stream-bispf-optimized"></a>
## `stream-bispf-optimized`

[Exact declaration](../../../configs/forge/ideas/stream-bispf-optimized.json)

Declared changes:

- ParticleGAN 0.9.6 public bispf_optimized trainer preset
- Shared 64-frame FIFO memory, seven replay samples and velocity context
- Global non-saturating loss and cosine schedule (start=0.5, floor=0.05)
- SBNP disabled for discrete target modes; FP32 training

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
    "network_lr_floor": 0.05,
    "prior_reg": 0.0,
    "prior_regularizer": "none",
    "training_precision": "float32"
  },
  "recipe_preset": "bispf_optimized"
}
```

**Historical, read-only.** This declaration does not satisfy explicit `replay_policy=none` with `rehearsal_batch=0`; the executor rejects it before touching its saved evidence. It has no current qualification score.

Strict online successors: [`online-bispf-optimized`](bispf.md#online-bispf-optimized).

[Preserved receipt directory](../receipts/cuda/stream-bispf-optimized) · [Archived cohorts](../cohorts) · [Research history](../research-notes.md)

<a id="stream-bispf-optimized-cool"></a>
## `stream-bispf-optimized-cool`

[Exact declaration](../../../configs/forge/ideas/stream-bispf-optimized-cool.json)

Parent: [`stream-bispf-optimized`](bispf.md#stream-bispf-optimized).

Declared changes:

- Conservative global DualNorm rates: lr=0.003, D multiplier=1, prior multiplier=2; annealing starts at 0.25

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `recipe_overrides.d_lr_mult` | _unset_ | `1.0` |
| `recipe_overrides.lr` | _unset_ | `0.003` |
| `recipe_overrides.lr_anneal_start` | `0.5` | `0.25` |
| `recipe_overrides.prior_lr_mult` | _unset_ | `2.0` |

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
    "d_lr_mult": 1.0,
    "loss": "non_saturating",
    "lr": 0.003,
    "lr_anneal_start": 0.25,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "prior_lr_mult": 2.0,
    "prior_reg": 0.0,
    "prior_regularizer": "none",
    "training_precision": "float32"
  },
  "recipe_preset": "bispf_optimized"
}
```

**Historical, read-only.** This declaration does not satisfy explicit `replay_policy=none` with `rehearsal_batch=0`; the executor rejects it before touching its saved evidence. It has no current qualification score.

Strict online successors: [`online-bispf-optimized-cool`](bispf.md#online-bispf-optimized-cool).

[Preserved receipt directory](../receipts/cuda/stream-bispf-optimized-cool) · [Archived cohorts](../cohorts) · [Research history](../research-notes.md)
