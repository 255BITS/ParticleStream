<!-- Generated Forge family report -->

# Global-Stream-BCAP

[← Technique leaderboard](../technique-inventory.md)

<a id="technique-overview"></a>
## Technique overview

Global-Stream-BCAP combines `FastWeightMemory` ($d_c=16$ velocity reader) and the fixed BCAP critic cap with one global `batch_size=1` optimization recipe: decoupled non-saturating logistic loss, an unregularized trainable prior (`prior_regularizer="none"`), faster Adam second-moment tracking ($\beta_2=0.99$), and cosine learning-rate annealing. Its strict online declaration (`online-fast-weight-bcap-ns`) removes the 7-frame reservoir rehearsal batch used by the historical `fast-weight-bcap-ns` recipe.

[Implementation](../../../particlestream/memory.py) · [Mathematical formulation](global-stream-bcap.md#mathematical-formulation) · [Simplified pseudocode](global-stream-bcap.md#simplified-pseudocode) · [Training details](global-stream-bcap.md#training-details) · [Protocol](../../../EXPERIMENTATION.md) · [Terminology](../../../RESEARCH_GLOSSARY.md)

<a id="mathematical-formulation"></a>
## Mathematical formulation

$G$ is the generator, $D$ the critic, and $P$ the trainable $K$-particle MoG prior ($K=32$, $d_z=8$, $\sigma=0.025$) with centers $\{\mu_i\}_{i=0}^{K-1}$. $x_t$ is the single real arrival at step $t\in\{1,\dots,T\}$, $c_{t-1}\in\mathbb R^{16}$ is the detached velocity context from `FastWeightMemory`, $\hat x_D=G(\mu_{i_D}+\sigma\epsilon_D,c_{t-1})$ is the detached critic fake, and $\hat x_G=G(\mu_{i_G}+\sigma\epsilon_G,c_{t-1})$ is the differentiable generator fake. $\eta_0=0.005$ is the base learning rate, $u_t=t/T$ is normalized progress, $u_0=0.5$ is the cosine anneal start, and $f=0.05$ is the LR floor.

**Decoupled Non-Saturating BCAP Objective with Unregularized Prior**

$$
\begin{aligned}\mathcal L_D&=\operatorname{softplus}(-D(x_t,c_{t-1}))+\operatorname{softplus}(D(\hat x_D,c_{t-1}))+\frac{\lambda}{2}\Big[(\lVert g_r\rVert_2-\kappa)_+^2+(\lVert g_f\rVert_2-\kappa)_+^2\Big],\\\mathcal L_G&=\operatorname{softplus}(-D(\hat x_G,c_{t-1})).\end{aligned}
$$

Replacing relativistic pairing with independent real/fake logistic terms avoids single-arrival gradient inversion when $x_t$ and $\hat x_G$ belong to different valid modes, while `prior_regularizer="none"` lets the 32 MoG centers cluster onto discrete templates without repulsive spreading.

**Delayed Cosine Learning-Rate Schedule**

$$
\eta(t)=\begin{cases}\eta_0,&u_t\le u_0,\\[6pt]\eta_0\left(f+\dfrac{1-f}{2}\left[1+\cos\!\left(\pi\dfrac{u_t-u_0}{1-u_0}\right)\right]\right),&u_t>u_0.\end{cases}
$$

Network groups step at $\eta(t)$ and trainable prior centers step at $2\eta(t)$ (`prior_lr_mult=2.0`), holding full plasticity through the first half of the stream before cooling to $5\%$ of $\eta_0$.

<a id="simplified-pseudocode"></a>
## Simplified pseudocode

```text
Initialize G, D, unregularized trainable MoG prior P and detached FastWeightMemory (d_c = 16).
For each chronological single-arrival step t of T:
  Update network LR eta(t) and prior LR 2 * eta(t) from the delayed cosine schedule (u_0 = 0.5, floor = 0.05).
  Serve pre-arrival prediction y_hat = G(z_{i_serve}, c_{t-1}) under isolated eval RNG when predictive.
  Draw uniform i_D and compute detached fake x_hat_D = stop_gradient(G(z_{i_D}, c_{t-1})).
  Backpropagate non-saturating critic loss + fixed BCAP cap (lambda = 1, kappa = 1) on x_t alone; step D.
  With D fixed, draw uniform i_G, compute x_hat_G = G(z_{i_G}, c_{t-1}), and backpropagate softplus(-D(x_hat_G, c_{t-1})).
  Step G and trainable prior row mu_{i_G} without prior repulsion or generator probes.
  Advance detached velocity context c_t = FastWeightMemory.write(x_t) with rehearsal_batch = 0.
```

<a id="training-details"></a>
## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Decoupled non-saturating logistic loss (`loss="non_saturating"`) without auxiliary reconstruction. |
| Optimizer | Native Adam (`bcap_adam`: $\beta_1=0$, $\beta_2=0.99$, `prior_betas=(0, 0.99)`). |
| Learning rates and annealing | Base LR $0.005$, `critic_lr_mult=1.0`, `prior_lr_mult=2.0`, holding until `lr_anneal_start=0.5` then cosine-decaying to `lr_floor=0.05` (`prior_lr_floor=0.05`). |
| Parameter-gradient clipping | None (`critic_grad_guard=False`). |
| Critic penalties and anchors | Fixed real/fake one-sided squared $L_2$ BCAP cap ($\lambda=1.0$, $\kappa=1.0$). |
| Damping and update guards | None (`latent_damping_max_rate=0.0`, `prior_regularizer="none"`). |
| Online memory, prior and routing | Detached `FastWeightMemory` (`dim=16`, `reader="velocity"`), uniform unrouted particle draws. `online-fast-weight-bcap-ns` enforces `replay_policy="none"` and `rehearsal_batch=0` (removing historical `fast-weight-bcap-ns`'s `rehearsal_batch=7`). |
| Training and sampling noise | Zero scheduled critic-input or generator-output additive noise (`0.0`); task MoG $\sigma=0.025$ is preserved. |
| Parameter averaging and serving | Live unaveraged weights (`ema_decay=0.0`, `serve_average=False`). |

<a id="configuration-differences"></a>
## Configuration differences

- Global-Stream-BCAP isolates the exact boundary between replay-assisted streaming and true single-stream online learning: removing the 7-frame FIFO rehearsal batch drops `fast-weight-bcap-ns` to `0/10` required passes (`online-fast-weight-bcap-ns`).
- Without rehearsal, `dwell=16` identical-frame blocks overwrite the critic and pull unrouted trainable particles toward whichever mode currently occupies the stream (`seq_img_dwell_bars4` collapses to 2/4 modes), motivating the subsequent causal-state, context-balanced, and observed-feature prior families.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [particlestream/models.py](../../../particlestream/models.py)
- [particlestream/memory.py](../../../particlestream/memory.py)
- [particlestream/online.py](../../../particlestream/online.py)
- [particlestream/tier1.py](../../../particlestream/tier1.py)
- [configs/forge/ideas/online-fast-weight-bcap-ns.json](../../../configs/forge/ideas/online-fast-weight-bcap-ns.json)
- [configs/forge/ideas/fast-weight-bcap-ns.json](../../../configs/forge/ideas/fast-weight-bcap-ns.json)

</details>

Each section below describes one exact global recipe. Configuration and parent deltas come from the declarations; results come from Forge's current source-bound receipts. Required and stretch evidence stay separate. A final metric can meet its gate while the sustained verdict fails: every one of the last five checks must pass.

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
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.78125`; `modes=4`; `finite_template_tv=0.25`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_img_round_robin_bars4.gif) | 2026-11-14T02:53:48.614+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_img_dwell_bars4.gif) | 2026-11-14T02:53:56.113+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.33884`; `direction_acc=0.5`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_img_translating_bar_next.gif) | 2026-11-14T02:54:05.171+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_img_fork_rollout.gif) | 2026-11-14T02:54:15.915+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_img_variable_dwell_bars4.gif) | 2026-11-14T02:54:25.451+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.53125`; `finite_template_tv=0.46875`; `support_modes=4`; `rare_mass=0.03125`; `rare_mass=0.03125`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_img_rare_mode_bars4.gif) | 2026-11-14T02:54:35.166+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.51562`; `prediction_rmse=0.2986`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_img_delayed_cue.gif) | 2026-11-14T02:54:45.920+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0.03125`; `within_state_modes=1`; `branch_tv=0.96875`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_img_persistent_fork.gif) | 2026-11-14T02:55:01.823+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.11022`; `xy_rmse=0.33159`; `step_accuracy=0.32031`; `orbit_completion=0`; `waypoint_modes=8`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_img_circle_orbit.gif) | 2026-11-14T02:55:20.984+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.16012`; `s2c_rmse=0.15918`; `c2s_rmse=0.16107`; `xy_rmse=0.74504`; `step_accuracy=0.015625`; `cycle_completion=0`; `waypoint_modes=1`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_img_square_circle_cycle.gif) | 2026-11-14T02:55:55.594+00:00 |

<a id="online-fast-weight-bcap-ns-stretch"></a>
### Stretch evidence

<a id="online-fast-weight-bcap-ns-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_minimal_dense4.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_minimal_dense4.gif) | 2026-11-14T02:55:59.249+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_minimal_tokens4.gif) | 2026-11-14T02:56:02.691+00:00 |

<a id="online-fast-weight-bcap-ns-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.39622`; `seen_max_abs_error=0.96251`; `heldout_response_rmse=0.30054`; `heldout_max_abs_error=0.66251`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_minimal_kernel2x2.gif) | 2026-11-14T02:56:08.367+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.083333`; `seen_min_bit_confidence=1.9478e-09`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=1.9478e-09`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_minimal_addition2bit.gif) | 2026-11-14T02:56:13.820+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.018141`; `relative_mse=1.171`; `moving_direction_accuracy=0.5`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_robotics_fetch_reach.gif) | 2026-11-14T02:56:27.664+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012828`; `relative_mse=1.0964`; `moving_direction_accuracy=0.5814`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_robotics_fetch_push.gif) | 2026-11-14T02:56:40.630+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.013395`; `relative_mse=1.0921`; `moving_direction_accuracy=0.4951`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_robotics_fetch_slide.gif) | 2026-11-14T02:56:54.983+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011575`; `relative_mse=0.85355`; `moving_direction_accuracy=0.58333`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_robotics_fetch_pick_and_place.gif) | 2026-11-14T02:57:09.200+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010953`; `relative_mse=1.7745`; `moving_direction_accuracy=0.51986`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_robotics_hand_reach.gif) | 2026-11-14T02:57:23.496+00:00 |

<a id="online-fast-weight-bcap-ns-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.77773`; `interpolation_speed_error=0.8295`; `interpolation_raw_rmse=0.073698`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.47917`; `extrapolation_centroid_rmse=0.91625`; `extrapolation_speed_error=0.7741`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.5625`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-fast-weight-bcap-ns/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-fast-weight-bcap-ns_seq_minimal_velocity_extrapolation.gif) | 2026-11-14T02:57:34.917+00:00 |

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

<a id="refresh"></a>
## Refresh

```sh
particlestream compile --device cuda
```

This page is generated alongside the leaderboard from committed source-bound receipts and declarations; editing a generated page cannot change a verdict or earn qualification.

<a id="references"></a>
## References

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661). Non-saturating logistic adversarial objective replacing relativistic score pairing.
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983). Cosine learning-rate annealing schedule applied over the second half of the stream horizon.
- [Using Fast Weights to Attend to the Recent Past](https://arxiv.org/abs/1610.06258). Detached fast-weight memory state maintained without backpropagating through time.
