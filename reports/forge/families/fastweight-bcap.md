<!-- Generated Forge family report -->

# FastWeight-BCAP

[← Technique leaderboard](../technique-inventory.md)

<a id="technique-overview"></a>
## Technique overview

FastWeight-BCAP extends the preset `bcap_adam` trainer with a detached Hebbian fast-weight memory module (`FastWeightMemory`, $d_c=16$) that updates only on real arrivals and conditions both generator and critic without backpropagating through temporal history. Its strict online declaration (`online-fast-weight-bcap`) preserves the complete `bcap_adam` relativistic + VICReg optimization recipe while disabling frame replay.

[Implementation](../../../particlestream/memory.py) · [Mathematical formulation](fastweight-bcap.md#mathematical-formulation) · [Simplified pseudocode](fastweight-bcap.md#simplified-pseudocode) · [Training details](fastweight-bcap.md#training-details) · [Protocol](../../../EXPERIMENTATION.md) · [Terminology](../../../RESEARCH_GLOSSARY.md)

<a id="mathematical-formulation"></a>
## Mathematical formulation

$G$ is the generator, $D$ the critic, and $P$ the trainable $K$-particle MoG prior ($K=32$, $d_z=8$, $\sigma=0.025$). At step $t$, $x_t\in[0,1]^{1\times 8\times 8}$ is the arriving real frame with flattened vector $\bar x_t=\operatorname{vec}(x_t)\in\mathbb R^{64}$. $W_k,W_v\in\mathbb R^{d_c\times 64}$ ($d_c=16$) are fixed projections inside `FastWeightMemory`, $M_t\in\mathbb R^{d_c\times d_c}$ is the detached Hebbian fast-weight matrix with decay $\gamma=0.90$ and write rate $\eta_w=0.35$, and $c_t\in\mathbb R^{d_c}$ is the detached velocity context passed to step $t+1$. $\Delta(x,y)=D(x,c_{t-1})-D(y,c_{t-1})$ is the paired relativistic score difference and $R_{\mathrm{VICReg}}(P)$ is the preset prior regularizer.

**Detached Hebbian Writer and Thresholded Velocity Reader**

$$
\begin{aligned}k_t&=\frac{W_k\bar x_t}{\lVert W_k\bar x_t\rVert_2},\qquad v_t=\frac{W_v\bar x_t}{\lVert W_v\bar x_t\rVert_2},\qquad M_t=\gamma M_{t-1}+\eta_w\,v_t k_{t-1}^\top,\\c_t&=\begin{cases}3.0\,\dfrac{k_t-k_{t-1}}{\lVert k_t-k_{t-1}\rVert_2},&\lVert k_{t-1}\rVert_2>0\land\lVert k_t-k_{t-1}\rVert_2>0.35,\\[6pt]\mathbf 0_{d_c},&\text{otherwise}.\end{cases}\end{aligned}
$$

Only real observations mutate $(M_t,k_{t-1})$ inside `torch.no_grad()`. Because the original `velocity` reader emits the normalized key difference $k_t-k_{t-1}$ without querying $M_t$, it cleanly disambiguates 1-step motion (`translating_bar_next`) but collapses to $\mathbf 0_{d_c}$ across repeated distractors ($k_t=k_{t-1}$).

**Single-Arrival Relativistic BCAP and VICReg Objectives**

$$
\begin{aligned}\mathcal L_D&=\operatorname{softplus}\big(-\Delta(x_t,\hat x_D)\big)+\frac{\lambda}{2}\Big[(\lVert g_r\rVert_2-\kappa)_+^2+(\lVert g_f\rVert_2-\kappa)_+^2\Big],\\\mathcal L_G&=\operatorname{softplus}\big(\Delta(x_t,\hat x_G)\big)+\lambda_{\mathrm{prior}}R_{\mathrm{VICReg}}(P).\end{aligned}
$$

With empty `recipe_overrides`, `online-fast-weight-bcap` inherits `bcap_adam`'s constant-LR relativistic pairing and repulsive `vicreg` prior regularizer over uniform `batch_size=1` draws $\hat x_D,\hat x_G$.

<a id="simplified-pseudocode"></a>
## Simplified pseudocode

```text
Initialize G, D, trainable MoG prior P and detached FastWeightMemory (d_c = 16).
For each chronological single-arrival step t:
  On predictive streams at arrival 0, warm-start c_0 = FastWeightMemory.write(x_0) without stepping G or D.
  Serve pre-arrival prediction y_hat = G(z_{i_serve}, c_{t-1}) under isolated eval RNG.
  Draw uniform i_D and compute detached fake x_hat_D = stop_gradient(G(z_{i_D}, c_{t-1})).
  Backpropagate relativistic critic loss softplus(D(x_hat_D, c_{t-1}) - D(x_t, c_{t-1})) + BCAP cap; step D.
  With D fixed, draw uniform i_G, compute x_hat_G = G(z_{i_G}, c_{t-1}), and evaluate relativistic G loss + VICReg prior loss.
  Step G and trainable prior P at constant preset learning rates.
  Update detached M_t and emit thresholded velocity context c_t = FastWeightMemory.write(x_t) (zero replay).
```

<a id="training-details"></a>
## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Preset paired relativistic logistic loss (`loss="relativistic"`) on raw critic scores. |
| Optimizer | Native Adam (`bcap_adam` preset: $\beta_1=0$, $\beta_2=0.999$). |
| Learning rates and annealing | Constant unannealed `bcap_adam` rates (`lr=0.00425`, `critic_lr_mult=1.0`, `prior_lr_mult=2.0`, `lr_floor=1.0`). |
| Parameter-gradient clipping | None (`critic_grad_guard=False`). |
| Critic penalties and anchors | Fixed real/fake one-sided squared $L_2$ BCAP cap ($\lambda=1.0$, $\kappa=1.0$). |
| Damping and update guards | None (`latent_damping_max_rate=0.0`); `prior_regularizer="vicreg"` spreads trainable latent centers. |
| Online memory, prior and routing | Detached `FastWeightMemory` (`dim=16`, `decay=0.9`, `write_lr=0.35`, `reader="velocity"`), uniform particle sampling, `replay_policy="none"`, `rehearsal_batch=0`. |
| Training and sampling noise | Zero scheduled critic-input or generator-output additive noise; task-declared MoG $\sigma=0.025$ is preserved. |
| Parameter averaging and serving | Live unaveraged weights (`ema_decay=0.0`, `serve_average=False`). |

<a id="configuration-differences"></a>
## Configuration differences

- `online-fast-weight-bcap` passes `seq_img_translating_bar_next` (`1/10` required passes), proving that detached key-difference conditioning alone resolves 1-step interior motion ambiguity without replay.
- On unconditional multi-mode streams (`bars4`, `dwell_bars4`), pairing `batch_size=1` relativistic score subtraction with unannealed updates and repulsive `vicreg` prior regularization prevents stable 4-mode convergence.
- The historical parent `fast-weight-bcap` omitted explicit `replay_policy="none"` fields and remains preserved as read-only provenance.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [particlestream/models.py](../../../particlestream/models.py)
- [particlestream/memory.py](../../../particlestream/memory.py)
- [particlestream/online.py](../../../particlestream/online.py)
- [particlestream/tier1.py](../../../particlestream/tier1.py)
- [configs/forge/ideas/online-fast-weight-bcap.json](../../../configs/forge/ideas/online-fast-weight-bcap.json)

</details>

Each section below describes one exact global recipe. Configuration and parent deltas come from the declarations; results come from Forge's current source-bound receipts. Required and stretch evidence stay separate. A final metric can meet its gate while the sustained verdict fails: every one of the last five checks must pass.

The installed engine is pinned to ParticleGAN/Forge 0.9.7. Older version or replay wording in declaration history describes provenance; the effective configuration below controls execution.

<a id="online-fast-weight-bcap"></a>
## `online-fast-weight-bcap`

[Exact declaration](../../../configs/forge/ideas/online-fast-weight-bcap.json)

Parent: [`fast-weight-bcap`](fastweight-bcap.md#fast-weight-bcap).

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
    "decay": 0.9,
    "dim": 16,
    "kind": "fast_weight",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "write_lr": 0.35
  },
  "recipe_overrides": {},
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **FAIL**, 1/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `0e1fe83f484a595f9d373fa5e618e0e845c3326748058ba973309bf034c1be16`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-fast-weight-bcap-required"></a>
### Required evidence

<a id="online-fast-weight-bcap-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.75`; `modes=4`; `finite_template_tv=0.25`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9 | [receipt](../receipts/cuda/online-fast-weight-bcap/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-fast-weight-bcap_seq_img_round_robin_bars4.gif) | 2026-11-12T17:02:01.378+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-fast-weight-bcap/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-fast-weight-bcap_seq_img_dwell_bars4.gif) | 2026-11-12T17:02:08.079+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.067943`; `direction_acc=1` | [receipt](../receipts/cuda/online-fast-weight-bcap/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-fast-weight-bcap_seq_img_translating_bar_next.gif) | 2026-11-12T17:02:15.921+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=0.78125`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-fast-weight-bcap/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-fast-weight-bcap_seq_img_fork_rollout.gif) | 2026-11-12T17:02:23.639+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-fast-weight-bcap/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-fast-weight-bcap_seq_img_variable_dwell_bars4.gif) | 2026-11-12T17:02:30.759+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.78125`; `finite_template_tv=0.21875`; `support_modes=3`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-fast-weight-bcap/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-fast-weight-bcap_seq_img_rare_mode_bars4.gif) | 2026-11-12T17:02:37.541+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.48438`; `prediction_rmse=0.30656`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-fast-weight-bcap/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-fast-weight-bcap_seq_img_delayed_cue.gif) | 2026-11-12T17:02:45.439+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-fast-weight-bcap/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-fast-weight-bcap_seq_img_persistent_fork.gif) | 2026-11-12T17:02:58.523+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.21021`; `xy_rmse=0.93395`; `step_accuracy=0.035156`; `orbit_completion=0`; `waypoint_modes=3`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-fast-weight-bcap/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-fast-weight-bcap_seq_img_circle_orbit.gif) | 2026-11-12T17:03:13.515+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.21069`; `s2c_rmse=0.23547`; `c2s_rmse=0.18591`; `xy_rmse=0.89113`; `step_accuracy=0.011719`; `cycle_completion=0`; `waypoint_modes=1`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-fast-weight-bcap/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-fast-weight-bcap_seq_img_square_circle_cycle.gif) | 2026-11-12T17:03:39.866+00:00 |

<a id="online-fast-weight-bcap-stretch"></a>
### Stretch evidence

<a id="online-fast-weight-bcap-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-fast-weight-bcap/seq_minimal_dense4.json) · [goal.gif](../gifs/online-fast-weight-bcap_seq_minimal_dense4.gif) | 2026-11-12T17:03:43.292+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0.1875`; `modes=3`; `mass_tv=0.8125`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-fast-weight-bcap/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-fast-weight-bcap_seq_minimal_tokens4.gif) | 2026-11-12T17:03:46.556+00:00 |

<a id="online-fast-weight-bcap-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.39936`; `seen_max_abs_error=0.9825`; `heldout_response_rmse=0.30467`; `heldout_max_abs_error=0.6825`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-fast-weight-bcap/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-fast-weight-bcap_seq_minimal_kernel2x2.gif) | 2026-11-12T17:03:51.546+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.1875`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0.1875`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-fast-weight-bcap/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-fast-weight-bcap_seq_minimal_addition2bit.gif) | 2026-11-12T17:03:56.540+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.016261`; `relative_mse=0.94086`; `moving_direction_accuracy=0.61979`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-fast-weight-bcap/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-fast-weight-bcap_seq_robotics_fetch_reach.gif) | 2026-11-12T17:04:07.140+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.015482`; `relative_mse=1.597`; `moving_direction_accuracy=0.37209`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-fast-weight-bcap/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-fast-weight-bcap_seq_robotics_fetch_push.gif) | 2026-11-12T17:04:20.843+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.013804`; `relative_mse=1.1597`; `moving_direction_accuracy=0.70588`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-fast-weight-bcap/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-fast-weight-bcap_seq_robotics_fetch_slide.gif) | 2026-11-12T17:04:34.236+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.017896`; `relative_mse=2.0404`; `moving_direction_accuracy=0.71491`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-fast-weight-bcap/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-fast-weight-bcap_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T17:04:48.187+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0093366`; `relative_mse=1.2894`; `moving_direction_accuracy=0.46379`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-fast-weight-bcap/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-fast-weight-bcap_seq_robotics_hand_reach.gif) | 2026-11-12T17:05:01.031+00:00 |

<a id="online-fast-weight-bcap-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.76409`; `interpolation_speed_error=0.6699`; `interpolation_raw_rmse=0.073698`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.4375`; `extrapolation_centroid_rmse=0.8861`; `extrapolation_speed_error=0.77405`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.5`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-fast-weight-bcap/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-fast-weight-bcap_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T17:05:13.798+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-fast-weight-bcap --device cuda`.

<a id="fast-weight-bcap"></a>
## `fast-weight-bcap`

[Exact declaration](../../../configs/forge/ideas/fast-weight-bcap.json)

Parent: [`bcap`](bcap.md#bcap).

Declared changes:

- Detached FastWeightMemory writer/reader context + bounded 64-frame reservoir replay on batch_size=1 stream

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.decay` | _unset_ | `0.9` |
| `memory_policy.dim` | _unset_ | `16` |
| `memory_policy.kind` | `"none"` | `"fast_weight"` |
| `memory_policy.reservoir_size` | _unset_ | `64` |
| `memory_policy.write_lr` | _unset_ | `0.35` |

### Effective declaration

```json
{
  "memory_policy": {
    "decay": 0.9,
    "dim": 16,
    "kind": "fast_weight",
    "reservoir_size": 64,
    "write_lr": 0.35
  },
  "recipe_overrides": {},
  "recipe_preset": "bcap_adam"
}
```

**Historical, read-only.** This declaration does not satisfy explicit `replay_policy=none` with `rehearsal_batch=0`; the executor rejects it before touching its saved evidence. It has no current qualification score.

Strict online successors: [`online-fast-weight-bcap`](fastweight-bcap.md#online-fast-weight-bcap).

[Preserved receipt directory](../receipts/cuda/fast-weight-bcap) · [Archived cohorts](../cohorts) · [Research history](../research-notes.md)

<a id="refresh"></a>
## Refresh

```sh
particlestream compile --device cuda
```

This page is generated alongside the leaderboard from committed source-bound receipts and declarations; editing a generated page cannot change a verdict or earn qualification.

<a id="references"></a>
## References

- [Using Fast Weights to Attend to the Recent Past](https://arxiv.org/abs/1610.06258). Detached fast-weight associative state updated online from arriving activations.
- [The relativistic discriminator: a key element missing from standard GAN](https://arxiv.org/abs/1807.00734). Paired relativistic logistic objective inherited from the `bcap_adam` preset.
- [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906). Variance-covariance prior regularizer inherited from the `bcap_adam` preset.
