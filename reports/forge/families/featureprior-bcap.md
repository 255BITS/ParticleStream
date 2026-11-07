# FeaturePrior-BCAP

[← Technique leaderboard](../technique-inventory.md)

FeaturePrior-BCAP adds a learned conditional prior to BCAP and a causal memory reader. A fixed named orthogonal 8×64 projection encodes each arriving image. At most 32 feature means and 64 context/outcome summaries accumulate after successful updates. Serving draws empirical feature quantiles near the observed context while retaining every row's MoG perturbation. The storage prior is frozen; a weight-10 BCE current-observation posterior trains the image decoder. Serving receives neither the current target nor stored images, and this policy uses no generator-center probes.

[Implementation](../../../particlestream/feature_prior.py) · [Protocol](../../../EXPERIMENTATION.md) · [Terminology](../../../RESEARCH_GLOSSARY.md)

Each section describes one exact global recipe. Configuration and parent deltas come from the declarations; results come from Forge's current source-bound receipts. Required and stretch evidence stay separate. A final metric can meet its gate while the sustained verdict fails: every one of the last five checks must pass.

The installed engine is pinned to ParticleGAN/Forge 0.9.7. Older version or replay wording in declaration history describes provenance; the effective configuration below controls execution.

<a id="online-feature-prior-bce"></a>
## `online-feature-prior-bce`

[Exact declaration](../../../configs/forge/ideas/online-feature-prior-bce.json)

Parent: [`online-bcap-adam-causal`](bcap.md#online-bcap-adam-causal).

Declared changes:

- New observed-feature conditional prior: fixed named deterministic orthogonal 8x64 encoder; at most 32 latent feature means and 64 observed-context outcome counts; learn statistics only after successful current-observation updates
- Freeze storage prior; learned additive conditional transform preserves each sampled MoG perturbation and task capacity, samples empirical feature quantiles at nearest past context; no image output bypass or stored-frame replay
- G-phase posterior encodes only the current arrived frame; BCE reconstruction weight 10 trains the unchanged image decoder; D and serving use learned conditional prior without current target or generator probes
- Ablate associative-history context term; retain appearance, velocity, salient content and repetition age

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.association_weight` | `0.125` | `0.0` |
| `memory_policy.transport` | _unset_ | `{"context_capacity": 64, "context_clip": 8.0, "context_tolerance": 0.25, "feature_scale": 2.0, "feature_tolerance": 0.15, "image_center": 0.5, "reconstruction_loss": "binary_cross_entropy", "reconstruction_weight": 10.0, "router": "feature_conditional", "temperature": 0.0, "usage_decay": 1.0, "usage_penalty": 0.0}` |
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
      "feature_scale": 2.0,
      "feature_tolerance": 0.15,
      "image_center": 0.5,
      "reconstruction_loss": "binary_cross_entropy",
      "reconstruction_weight": 10.0,
      "router": "feature_conditional",
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

Current `cuda`, seed 0: **PASS**, 10/10 required passes (10 measured), 1/10 stretch passes (10 measured); qualified tier 1.

Receipt identity: candidate_fingerprint `4f8ec6b6bf449a96ce37bc21303963338f80dc20d6dcf3e72e59f9777f8ddd41`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-feature-prior-bce-required"></a>
### Required evidence

<a id="online-feature-prior-bce-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-feature-prior-bce/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_img_round_robin_bars4.gif) | 2026-11-06T17:13:01.861+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-feature-prior-bce/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_img_dwell_bars4.gif) | 2026-11-06T17:13:08.749+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.008454`; `direction_acc=1` | [receipt](../receipts/cuda/online-feature-prior-bce/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_img_translating_bar_next.gif) | 2026-11-06T17:13:16.356+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-feature-prior-bce/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_img_fork_rollout.gif) | 2026-11-06T17:13:24.113+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-feature-prior-bce/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_img_variable_dwell_bars4.gif) | 2026-11-06T17:13:30.890+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-feature-prior-bce/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_img_rare_mode_bars4.gif) | 2026-11-06T17:13:37.882+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.024176` | [receipt](../receipts/cuda/online-feature-prior-bce/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_img_delayed_cue.gif) | 2026-11-06T17:13:45.629+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.03125`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-feature-prior-bce/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_img_persistent_fork.gif) | 2026-11-06T17:14:00.952+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.012616`; `xy_rmse=0.003612`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-feature-prior-bce/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_img_circle_orbit.gif) | 2026-11-06T17:14:18.851+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.012665`; `s2c_rmse=0.011734`; `c2s_rmse=0.013596`; `xy_rmse=0.003817`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-feature-prior-bce/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_img_square_circle_cycle.gif) | 2026-11-06T17:14:50.801+00:00 |

<a id="online-feature-prior-bce-stretch"></a>
### Stretch evidence

<a id="online-feature-prior-bce-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-feature-prior-bce/seq_minimal_dense4.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_minimal_dense4.gif) | 2026-11-06T22:49:02.095+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-feature-prior-bce/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_minimal_tokens4.gif) | 2026-11-06T22:49:08.950+00:00 |

<a id="online-feature-prior-bce-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.042742`; `seen_max_abs_error=0.16605`; `heldout_response_rmse=0.26142`; `heldout_max_abs_error=0.43249`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-feature-prior-bce/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_minimal_kernel2x2.gif) | 2026-11-06T22:49:17.656+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.97912`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=6.7075e-16`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-feature-prior-bce/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_minimal_addition2bit.gif) | 2026-11-06T22:49:27.701+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0055261`; `relative_mse=0.10866`; `moving_direction_accuracy=0.95833`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-feature-prior-bce/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_robotics_fetch_reach.gif) | 2026-11-06T23:23:03.809+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010941`; `relative_mse=0.79759`; `moving_direction_accuracy=0.74419`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-feature-prior-bce/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_robotics_fetch_push.gif) | 2026-11-06T23:23:37.908+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0076109`; `relative_mse=0.35257`; `moving_direction_accuracy=0.77451`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-feature-prior-bce/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_robotics_fetch_slide.gif) | 2026-11-06T23:24:11.533+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 4/5 | `position_delta_rmse_m=0.0026655`; `relative_mse=0.045265`; `moving_direction_accuracy=0.84211`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75 | [receipt](../receipts/cuda/online-feature-prior-bce/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_robotics_fetch_pick_and_place.gif) | 2026-11-06T23:24:43.966+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.008179`; `relative_mse=0.98949`; `moving_direction_accuracy=0.58879`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-feature-prior-bce/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_robotics_hand_reach.gif) | 2026-11-06T23:25:15.333+00:00 |

<a id="online-feature-prior-bce-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.029967`; `interpolation_speed_error=0.039319`; `interpolation_raw_rmse=0.026119`; `interpolation_point_validity=0.83333`; `interpolation_direction_accuracy=0.91667`; `extrapolation_centroid_rmse=0.083345`; `extrapolation_speed_error=0.074005`; `extrapolation_raw_rmse=0.053561`; `extrapolation_point_validity=0.58333`; `extrapolation_direction_accuracy=1`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-feature-prior-bce/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_minimal_velocity_extrapolation.gif) | 2026-11-06T22:49:47.206+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-feature-prior-bce --device cuda`.
