<!-- Generated Forge family report -->

# BISPF

[← Technique leaderboard](../technique-inventory.md)

<a id="technique-overview"></a>
## Technique overview

Born-Infeld Sinkhorn Particle Flow (BISPF) is a critic-regularized ParticleGAN formulation without an EMA critic copy. It couples real and generated batches via scale-free entropic Optimal Transport (Sinkhorn-Knopp), bounds critic slopes with a two-regime Elastic Born-Infeld potential and a zero-mean logit gauge, and provides optional Stratified Blue-Noise Particle (SBNP) prior regularization and Harmonic Mode Capture thermal cooling. Under ParticleStream's `batch_size=1` sequential contract, the Sinkhorn coupling reduces to a single $(1,1)$ pair while retaining EMA residual/scale cap adaptation.

[Implementation](../../../particlestream/tier1.py) · [Mathematical formulation](bispf.md#mathematical-formulation) · [Simplified pseudocode](bispf.md#simplified-pseudocode) · [Training details](bispf.md#training-details) · [Protocol](../../../EXPERIMENTATION.md) · [Terminology](../../../RESEARCH_GLOSSARY.md)

<a id="mathematical-formulation"></a>
## Mathematical formulation

$G$ is the generator, $D$ the raw-score critic, $P$ the $K$-particle task prior ($K=32$, $d_z=8$, $\sigma=0.025$), $x_t$ the single real arrival, $c_{t-1}$ the detached memory context, and $y_t=G(z_i,c_{t-1})$ the generated sample. $\Pi_\varepsilon$ is the Sinkhorn coupling at entropic scale $\varepsilon=0.15$ ($\Pi_\varepsilon=[1]$ when $B_r=B_f=1$), with Sinkhorn marginal-residual EMA $\bar r_t\in[0,1]$ and cloud-scale EMA $\bar s_t>0$ (decay $\gamma_{\mathrm{th}}=0.95$). $g_r=\nabla_x D(x_t,c_{t-1})$, $g_f=\nabla_y D(y_t,c_{t-1})$, and $g_c=\nabla_{x_{\mathrm{chord}}} D(x_{\mathrm{chord}},c_{t-1})$ are critic input gradients; $d=64$ counts $1\times 8\times 8$ image coordinates, $\kappa$ is the nominal cap, $\phi_\beta(u)=\beta^{-1}\log(1+e^{\beta u})$ ($\beta=30$) is the smooth Born-Infeld wall, $\lambda_{\mathrm{floor}}$ is the harmonic floor weight, $\lambda_{\mathrm{cap}}$ the elastic cap weight, $\alpha_c$ the chord blend, and $\gamma_{\mathrm{gauge}}$ the zero-mean logit gauge weight.

**Elastic Born-Infeld Regularizer and Zero-Mean Logit Gauge**

$$
R_{\mathrm{BISPF}}=\frac{\lambda_{\mathrm{floor}}}{2 m_g d}\,\mathbb E\!\left[\sum_{g\in\mathcal G}\lVert g\rVert_2^2\right]+\frac{\lambda_{\mathrm{cap}}}{2}\,\mathbb E\!\left[\phi_\beta(\lVert g_r\rVert_2-\kappa_{\mathrm{eff}})^2+(1-\alpha_c)\phi_\beta(\lVert g_f\rVert_2-\kappa_{\mathrm{eff}})^2+\alpha_c\phi_\beta(\lVert g_c\rVert_2-\kappa_{\mathrm{eff}})^2\right]+\frac{\gamma_{\mathrm{gauge}}}{2}\bar D^2.
$$

Here $\bar D=\frac12(\mathbb E[D(x_t,c_{t-1})]+\mathbb E[D(y_t,c_{t-1})])$ pins the translational logit gauge without an EMA critic network. On 4D image tensors at default $\alpha_c=0.02\le 0.10$, the BISPF image gate bypasses pixel-chord interpolation ($\mathcal G=\{g_r,g_f\}$, $m_g=2$, effectively $\alpha_c=0$ in the cap) so discrete spatial modes are not blended across pixel space, using $\lambda_{\mathrm{floor}}^{\mathrm{img}}=5\times 10^{-4}$ and $\gamma_{\mathrm{gauge}}^{\mathrm{img}}=5\times 10^{-3}$.

**Scale- and Residual-Adapted Image Cap**

$$
s_{\mathrm{img}}=\operatorname{clip}\!\left(\frac{\bar s_t}{s_{\mathrm{ref}}},\,0.15,\,1.0\right),\qquad \kappa_{\mathrm{eff}}=\frac{\kappa}{s_{\mathrm{img}}}\Big(0.6+1.1\big(1-\operatorname{clip}(\bar r_t,0,1)\big)\Big).
$$

With $s_{\mathrm{ref}}=4.899$, the effective slope wall $\kappa_{\mathrm{eff}}$ adapts smoothly from the observed Sinkhorn scale EMA $\bar s_t$ and residual EMA $\bar r_t$ even when each online arrival has `batch_size=1`.

**Stratified Blue-Noise Particle (SBNP) Prior Regularizer**

$$
R_{\mathrm{SBNP}}(P)=\lambda_P\left(\mathcal T_{\mathrm{moment}}(P)+w_{\mathrm{repel}}\,\mathbb I(K>2k_{\mathrm{nn}}+d_z)\,\frac{1}{K k_{\mathrm{nn}}}\sum_{i=1}^K\sum_{m=1}^{k_{\mathrm{nn}}}\log\!\left(1+\frac{\bar r_{\mathrm{knn}}^2}{\lVert \mu_i-\mu_{\mathcal N_m(i)}\rVert_2^2+\epsilon}\right)\right).
$$

The `bispf` and `bispf_optimized` presets enable SBNP (`prior_regularizer="sbnp"`), whereas ParticleStream's tuned online BISPF recipes override `prior_regularizer="none"` (and `prior_update="frozen"` in `online-bispf-frozen`) so 32 particles can represent 4 discrete visual modes without repulsive blue-noise spreading.

<a id="simplified-pseudocode"></a>
## Simplified pseudocode

```text
Choose optimizer (DualNorm in bispf_optimized or BISPFCriticAdam in bispf), rates and prior policy.
For each chronological single-arrival step t:
  Serve pre-arrival prediction under isolated eval RNG when task requires causal prediction.
  Draw particle row i_D and compute detached fake y_D = stop_gradient(G(z_{i_D}, c_{t-1})).
  Evaluate single-pair Sinkhorn coupling Pi = [[1]] between x_t and y_D; update residual and scale EMAs.
  Compute EBI harmonic floor + smooth softplus_beta cap at kappa_eff on input gradients plus zero-mean logit gauge.
  Backpropagate critic non-saturating logistic loss + BISPF penalty and step D.
  With D fixed, draw particle row i_G (uniform or routed), recompute differentiable fake y_G = G(z_{i_G}, c_{t-1}),
    and backpropagate generator adversarial loss (+ declared prior or reconstruction terms).
  Step G (and trainable prior P when unfrozen) and advance detached memory c_t = memory.write(x_t).
```

<a id="training-details"></a>
## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Decoupled non-saturating logistic loss (`loss="non_saturating"`) across declared ParticleStream BISPF candidates; routed causal variant (`online-bispf-transport-causal`) adds MSE reconstruction. |
| Optimizer | `bispf` preset uses `BISPFCriticAdam` / `K3PGeneratorAdam` ($\beta_1=0$, $\beta_2=0.99$ in tuned candidates); `bispf_optimized` uses DualNorm (`optimizer_svd_backend="batched"`). |
| Learning rates and annealing | Preset `bispf` / `bispf_optimized` run constant high rates (`0.008` / `0.012`, `critic_lr_mult=1.5`, `prior_lr_mult=2.5`); tuned online BISPF candidates use `lr=0.005` cosine-annealed to `lr_floor=0.05` (with `lr_anneal_start=0.25` in `online-bispf-early*` and `online-bispf-frozen`). |
| Parameter-gradient clipping | None (`critic_grad_guard=False`). Bounding acts through the smooth $\beta=30$ Elastic Born-Infeld potential. |
| Critic penalties and anchors | Elastic Born-Infeld critic regularizer ($\kappa=1.0$, $\beta=30$, image floor $5\times 10^{-4}$) plus zero-mean logit gauge ($5\times 10^{-3}$ on images). Zero EMA critic anchor (`reg_anchor_weight=0.0`). |
| Damping and update guards | Harmonic Mode Capture thermal cooler (`thermal_cooling=False` in `bispf`, enabled or `bispf_cooler_freeze_2d` in `bispf_optimized*`). No A2 row damping. |
| Online memory, prior and routing | `FastWeightMemory` velocity reader on most BISPF candidates; `CausalStateMemory` + competitive transport on `online-bispf-transport-causal`. All active candidates enforce `replay_policy="none"` and `rehearsal_batch=0`. |
| Training and sampling noise | Zero scheduled critic-input or generator-output noise (`0.0`); task-declared MoG $\sigma=0.025$ is preserved. |
| Parameter averaging and serving | Live unaveraged weights (`ema_decay=0.0`, `serve_average=False`). |

<a id="configuration-differences"></a>
## Configuration differences

- At `batch_size=1`, Sinkhorn coupling between one real and one generated sample is always the $1\times 1$ matrix $[1]$; BISPF therefore acts purely through its scale-adapted EBI gradient wall and zero-mean logit gauge rather than multi-sample OT pairing.
- `online-bispf-frozen` is the strongest measured BISPF recipe (`1/10` required passes, passing `seq_img_round_robin_bars4`), confirming that freezing the 32-row MoG table prevents sparse-row drift on i.i.d. cycling.
- Preset `online-bispf-optimized*` recipes inherit `lr=0.012`, `critic_lr_mult=1.5`, and repulsive `sbnp`, which destabilize single-sample $8\times 8$ streams (`DEAD_ENDS.md`).

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [particlestream/tier1.py](../../../particlestream/tier1.py)
- [particlestream/transport.py](../../../particlestream/transport.py)
- [particlestream/memory.py](../../../particlestream/memory.py)
- [configs/forge/ideas/online-bispf-frozen.json](../../../configs/forge/ideas/online-bispf-frozen.json)
- [configs/forge/ideas/online-bispf-early.json](../../../configs/forge/ideas/online-bispf-early.json)
- [configs/forge/ideas/online-bispf-transport-causal.json](../../../configs/forge/ideas/online-bispf-transport-causal.json)

</details>

Each section below describes one exact global recipe. Configuration and parent deltas come from the declarations; results come from Forge's current source-bound receipts. Required and stretch evidence stay separate. A final metric can meet its gate while the sustained verdict fails: every one of the last five checks must pass.

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

Current `cuda`, seed 0: **FAIL**, 1/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `02f098fcb214f4f7874f82ef83f6629ce8cfc85c2c3f3ac28b84ec45b7803676`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-bispf-frozen-required"></a>
### Required evidence

<a id="online-bispf-frozen-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.34375`; `modes=3`; `finite_template_tv=0.65625`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-frozen/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bispf-frozen_seq_img_round_robin_bars4.gif) | 2026-11-12T06:15:55.867+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0.0625`; `modes=1`; `finite_template_tv=0.9375`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-frozen/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-frozen_seq_img_dwell_bars4.gif) | 2026-11-12T06:16:11.305+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.071489`; `direction_acc=1` | [receipt](../receipts/cuda/online-bispf-frozen/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bispf-frozen_seq_img_translating_bar_next.gif) | 2026-11-12T06:16:26.659+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-frozen/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bispf-frozen_seq_img_fork_rollout.gif) | 2026-11-12T06:16:42.715+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-frozen/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-frozen_seq_img_variable_dwell_bars4.gif) | 2026-11-12T06:16:57.827+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.375`; `finite_template_tv=0.625`; `support_modes=3`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-bispf-frozen/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bispf-frozen_seq_img_rare_mode_bars4.gif) | 2026-11-12T06:17:12.988+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.48438`; `prediction_rmse=0.29732`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf-frozen/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bispf-frozen_seq_img_delayed_cue.gif) | 2026-11-12T06:17:29.383+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0.28125`; `within_state_modes=1`; `branch_tv=0.71875`; `trajectory_consistency=0.21875`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-frozen/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bispf-frozen_seq_img_persistent_fork.gif) | 2026-11-12T06:17:57.725+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.19472`; `xy_rmse=0.87879`; `step_accuracy=0.011719`; `orbit_completion=0`; `waypoint_modes=2`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-frozen/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bispf-frozen_seq_img_circle_orbit.gif) | 2026-11-12T06:18:28.651+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.22245`; `s2c_rmse=0.2304`; `c2s_rmse=0.21449`; `xy_rmse=1.1552`; `step_accuracy=0`; `cycle_completion=0`; `waypoint_modes=0`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-frozen/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bispf-frozen_seq_img_square_circle_cycle.gif) | 2026-11-12T06:19:25.794+00:00 |

<a id="online-bispf-frozen-stretch"></a>
### Stretch evidence

<a id="online-bispf-frozen-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bispf-frozen/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bispf-frozen_seq_minimal_dense4.gif) | 2026-11-12T06:19:30.707+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bispf-frozen/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bispf-frozen_seq_minimal_tokens4.gif) | 2026-11-12T06:19:35.325+00:00 |

<a id="online-bispf-frozen-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.40169`; `seen_max_abs_error=0.97481`; `heldout_response_rmse=0.30771`; `heldout_max_abs_error=0.67481`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bispf-frozen/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bispf-frozen_seq_minimal_kernel2x2.gif) | 2026-11-12T06:19:44.269+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.14583`; `seen_min_bit_confidence=3.9127e-16`; `heldout_exact_accuracy=0.1875`; `heldout_min_bit_confidence=3.9127e-16`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bispf-frozen/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bispf-frozen_seq_minimal_addition2bit.gif) | 2026-11-12T06:19:53.039+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.017424`; `relative_mse=1.0803`; `moving_direction_accuracy=0.45312`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-frozen/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bispf-frozen_seq_robotics_fetch_reach.gif) | 2026-11-12T06:20:12.391+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.014819`; `relative_mse=1.4632`; `moving_direction_accuracy=0.44767`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-frozen/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bispf-frozen_seq_robotics_fetch_push.gif) | 2026-11-12T06:20:29.492+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011903`; `relative_mse=0.86238`; `moving_direction_accuracy=0.67157`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-frozen/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bispf-frozen_seq_robotics_fetch_slide.gif) | 2026-11-12T06:20:40.212+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010651`; `relative_mse=0.72274`; `moving_direction_accuracy=0.69298`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-frozen/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bispf-frozen_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T06:20:50.520+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0098418`; `relative_mse=1.4327`; `moving_direction_accuracy=0.51285`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-frozen/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bispf-frozen_seq_robotics_hand_reach.gif) | 2026-11-12T06:21:00.958+00:00 |

<a id="online-bispf-frozen-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.81049`; `interpolation_speed_error=0.7724`; `interpolation_raw_rmse=0.073697`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.47917`; `extrapolation_centroid_rmse=0.95761`; `extrapolation_speed_error=0.873`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.5`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bispf-frozen/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bispf-frozen_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T06:21:11.078+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bispf-frozen --device cuda`.

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

Current `cuda`, seed 0: **FAIL**, 1/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `6c0e647f7c7b5bada184cc4b79970e9268f953b097611ea7753753dd07c58bbd`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-bispf-slow-prior-required"></a>
### Required evidence

<a id="online-bispf-slow-prior-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.40625`; `modes=3`; `finite_template_tv=0.59375`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-slow-prior/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bispf-slow-prior_seq_img_round_robin_bars4.gif) | 2026-11-12T06:37:20.575+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-slow-prior/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-slow-prior_seq_img_dwell_bars4.gif) | 2026-11-12T06:37:28.405+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.096812`; `direction_acc=1` | [receipt](../receipts/cuda/online-bispf-slow-prior/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bispf-slow-prior_seq_img_translating_bar_next.gif) | 2026-11-12T06:37:37.137+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-slow-prior/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bispf-slow-prior_seq_img_fork_rollout.gif) | 2026-11-12T06:37:45.466+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-slow-prior/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-slow-prior_seq_img_variable_dwell_bars4.gif) | 2026-11-12T06:37:52.866+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.65625`; `finite_template_tv=0.34375`; `support_modes=3`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-bispf-slow-prior/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bispf-slow-prior_seq_img_rare_mode_bars4.gif) | 2026-11-12T06:38:00.648+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.51562`; `prediction_rmse=0.29737`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf-slow-prior/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bispf-slow-prior_seq_img_delayed_cue.gif) | 2026-11-12T06:38:09.393+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-slow-prior/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bispf-slow-prior_seq_img_persistent_fork.gif) | 2026-11-12T06:38:22.887+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.20569`; `xy_rmse=0.88729`; `step_accuracy=0.027344`; `orbit_completion=0`; `waypoint_modes=1`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-slow-prior/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bispf-slow-prior_seq_img_circle_orbit.gif) | 2026-11-12T06:38:38.982+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.20728`; `s2c_rmse=0.20702`; `c2s_rmse=0.20754`; `xy_rmse=1.1045`; `step_accuracy=0`; `cycle_completion=0`; `waypoint_modes=0`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-slow-prior/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bispf-slow-prior_seq_img_square_circle_cycle.gif) | 2026-11-12T06:39:07.318+00:00 |

<a id="online-bispf-slow-prior-stretch"></a>
### Stretch evidence

<a id="online-bispf-slow-prior-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bispf-slow-prior/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bispf-slow-prior_seq_minimal_dense4.gif) | 2026-11-12T06:39:11.127+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bispf-slow-prior/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bispf-slow-prior_seq_minimal_tokens4.gif) | 2026-11-12T06:39:15.562+00:00 |

<a id="online-bispf-slow-prior-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.39521`; `seen_max_abs_error=0.9609`; `heldout_response_rmse=0.2992`; `heldout_max_abs_error=0.6609`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bispf-slow-prior/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bispf-slow-prior_seq_minimal_kernel2x2.gif) | 2026-11-12T06:39:20.879+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.14583`; `seen_min_bit_confidence=4.847e-15`; `heldout_exact_accuracy=0.1875`; `heldout_min_bit_confidence=4.847e-15`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bispf-slow-prior/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bispf-slow-prior_seq_minimal_addition2bit.gif) | 2026-11-12T06:39:25.953+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.01953`; `relative_mse=1.3573`; `moving_direction_accuracy=0.56771`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-slow-prior/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bispf-slow-prior_seq_robotics_fetch_reach.gif) | 2026-11-12T06:39:40.311+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.01427`; `relative_mse=1.3568`; `moving_direction_accuracy=0.48837`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-slow-prior/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bispf-slow-prior_seq_robotics_fetch_push.gif) | 2026-11-12T06:39:52.157+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012774`; `relative_mse=0.99319`; `moving_direction_accuracy=0.56373`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-slow-prior/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bispf-slow-prior_seq_robotics_fetch_slide.gif) | 2026-11-12T06:40:05.975+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011044`; `relative_mse=0.77712`; `moving_direction_accuracy=0.68421`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-slow-prior/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bispf-slow-prior_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T06:40:23.139+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.015077`; `relative_mse=3.3623`; `moving_direction_accuracy=0.53738`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-slow-prior/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bispf-slow-prior_seq_robotics_hand_reach.gif) | 2026-11-12T06:40:36.329+00:00 |

<a id="online-bispf-slow-prior-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.81296`; `interpolation_speed_error=0.7737`; `interpolation_raw_rmse=0.073697`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.45833`; `extrapolation_centroid_rmse=0.95906`; `extrapolation_speed_error=0.87107`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.52083`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bispf-slow-prior/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bispf-slow-prior_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T06:40:48.438+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bispf-slow-prior --device cuda`.

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
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.875`; `modes=4`; `finite_template_tv=0.15625`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9 | [receipt](../receipts/cuda/online-bispf/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bispf_seq_img_round_robin_bars4.gif) | 2026-11-12T06:00:35.983+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bispf_seq_img_dwell_bars4.gif) | 2026-11-12T06:00:43.829+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.13298`; `direction_acc=0.96875`; failed terminal bounds: next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bispf_seq_img_translating_bar_next.gif) | 2026-11-12T06:00:52.466+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bispf_seq_img_fork_rollout.gif) | 2026-11-12T06:01:04.112+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bispf_seq_img_variable_dwell_bars4.gif) | 2026-11-12T06:01:14.629+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.8125`; `finite_template_tv=0.1875`; `support_modes=3`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-bispf/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bispf_seq_img_rare_mode_bars4.gif) | 2026-11-12T06:01:25.251+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.5`; `prediction_rmse=0.30088`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bispf_seq_img_delayed_cue.gif) | 2026-11-12T06:01:37.078+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0.03125`; `within_state_modes=1`; `branch_tv=0.96875`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bispf_seq_img_persistent_fork.gif) | 2026-11-12T06:01:53.644+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.22307`; `xy_rmse=1.0586`; `step_accuracy=0`; `orbit_completion=0`; `waypoint_modes=0`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bispf_seq_img_circle_orbit.gif) | 2026-11-12T06:02:11.485+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.20666`; `s2c_rmse=0.20285`; `c2s_rmse=0.21048`; `xy_rmse=1.0581`; `step_accuracy=0`; `cycle_completion=0`; `waypoint_modes=0`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bispf_seq_img_square_circle_cycle.gif) | 2026-11-12T06:02:47.244+00:00 |

<a id="online-bispf-stretch"></a>
### Stretch evidence

<a id="online-bispf-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bispf/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bispf_seq_minimal_dense4.gif) | 2026-11-12T06:02:51.347+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bispf/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bispf_seq_minimal_tokens4.gif) | 2026-11-12T06:02:55.092+00:00 |

<a id="online-bispf-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.49246`; `seen_max_abs_error=0.99973`; `heldout_response_rmse=0.41934`; `heldout_max_abs_error=0.69973`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bispf/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bispf_seq_minimal_kernel2x2.gif) | 2026-11-12T06:03:01.542+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.125`; `seen_min_bit_confidence=2.0158e-20`; `heldout_exact_accuracy=0.125`; `heldout_min_bit_confidence=2.0158e-20`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bispf/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bispf_seq_minimal_addition2bit.gif) | 2026-11-12T06:03:07.508+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.01715`; `relative_mse=1.0466`; `moving_direction_accuracy=0.60417`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bispf_seq_robotics_fetch_reach.gif) | 2026-11-12T06:03:21.518+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.01476`; `relative_mse=1.4516`; `moving_direction_accuracy=0.43605`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bispf_seq_robotics_fetch_push.gif) | 2026-11-12T06:03:34.570+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.013371`; `relative_mse=1.0882`; `moving_direction_accuracy=0.71569`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bispf_seq_robotics_fetch_slide.gif) | 2026-11-12T06:03:46.348+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011956`; `relative_mse=0.91078`; `moving_direction_accuracy=0.58333`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bispf_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T06:04:01.195+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0091335`; `relative_mse=1.2339`; `moving_direction_accuracy=0.49065`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bispf_seq_robotics_hand_reach.gif) | 2026-11-12T06:04:15.804+00:00 |

<a id="online-bispf-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.80453`; `interpolation_speed_error=0.65749`; `interpolation_raw_rmse=0.073697`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.45833`; `extrapolation_centroid_rmse=0.9529`; `extrapolation_speed_error=0.86176`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.45833`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bispf/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bispf_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T06:04:29.170+00:00 |

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
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 1/5 | `hq=0.90625`; `modes=4`; `finite_template_tv=0.09375`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9 | [receipt](../receipts/cuda/online-bispf-early/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bispf-early_seq_img_round_robin_bars4.gif) | 2026-11-12T06:04:40.190+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-early/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-early_seq_img_dwell_bars4.gif) | 2026-11-12T06:04:47.768+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.18288`; `direction_acc=0.96875`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf-early/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bispf-early_seq_img_translating_bar_next.gif) | 2026-11-12T06:04:56.296+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-early/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bispf-early_seq_img_fork_rollout.gif) | 2026-11-12T06:05:07.268+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-early/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-early_seq_img_variable_dwell_bars4.gif) | 2026-11-12T06:05:17.725+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.71875`; `finite_template_tv=0.28125`; `support_modes=3`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-bispf-early/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bispf-early_seq_img_rare_mode_bars4.gif) | 2026-11-12T06:05:28.192+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.5`; `prediction_rmse=0.29866`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf-early/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bispf-early_seq_img_delayed_cue.gif) | 2026-11-12T06:05:36.833+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-early/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bispf-early_seq_img_persistent_fork.gif) | 2026-11-12T06:05:52.613+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.23511`; `xy_rmse=1.0814`; `step_accuracy=0`; `orbit_completion=0`; `waypoint_modes=0`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-early/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bispf-early_seq_img_circle_orbit.gif) | 2026-11-12T06:06:13.128+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.20879`; `s2c_rmse=0.20123`; `c2s_rmse=0.21635`; `xy_rmse=1.0508`; `step_accuracy=0`; `cycle_completion=0`; `waypoint_modes=0`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-early/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bispf-early_seq_img_square_circle_cycle.gif) | 2026-11-12T06:06:44.315+00:00 |

<a id="online-bispf-early-stretch"></a>
### Stretch evidence

<a id="online-bispf-early-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bispf-early/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bispf-early_seq_minimal_dense4.gif) | 2026-11-12T06:06:48.252+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bispf-early/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bispf-early_seq_minimal_tokens4.gif) | 2026-11-12T06:06:51.999+00:00 |

<a id="online-bispf-early-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.48345`; `seen_max_abs_error=0.99634`; `heldout_response_rmse=0.40872`; `heldout_max_abs_error=0.69634`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bispf-early/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bispf-early_seq_minimal_kernel2x2.gif) | 2026-11-12T06:06:58.217+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.125`; `seen_min_bit_confidence=8.8454e-26`; `heldout_exact_accuracy=0.125`; `heldout_min_bit_confidence=8.8454e-26`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bispf-early/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bispf-early_seq_minimal_addition2bit.gif) | 2026-11-12T06:07:04.131+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.01634`; `relative_mse=0.95011`; `moving_direction_accuracy=0.5625`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-early/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bispf-early_seq_robotics_fetch_reach.gif) | 2026-11-12T06:07:19.058+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.017007`; `relative_mse=1.9272`; `moving_direction_accuracy=0.65116`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-early/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bispf-early_seq_robotics_fetch_push.gif) | 2026-11-12T06:07:30.356+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.041838`; `relative_mse=10.654`; `moving_direction_accuracy=0.80392`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-early/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bispf-early_seq_robotics_fetch_slide.gif) | 2026-11-12T06:07:49.430+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.01046`; `relative_mse=0.69706`; `moving_direction_accuracy=0.64912`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-bispf-early/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bispf-early_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T06:08:09.456+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0096267`; `relative_mse=1.3708`; `moving_direction_accuracy=0.52921`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-early/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bispf-early_seq_robotics_hand_reach.gif) | 2026-11-12T06:08:29.729+00:00 |

<a id="online-bispf-early-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.79642`; `interpolation_speed_error=0.65746`; `interpolation_raw_rmse=0.073697`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.45833`; `extrapolation_centroid_rmse=0.94778`; `extrapolation_speed_error=0.85925`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.47917`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bispf-early/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bispf-early_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T06:08:52.136+00:00 |

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
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 1/5 | `hq=0.90625`; `modes=4`; `finite_template_tv=0.09375`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_img_round_robin_bars4.gif) | 2026-11-12T06:09:08.364+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_img_dwell_bars4.gif) | 2026-11-12T06:09:23.610+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.30693`; `direction_acc=0.5`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_img_translating_bar_next.gif) | 2026-11-12T06:09:40.401+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_img_fork_rollout.gif) | 2026-11-12T06:09:56.278+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_img_variable_dwell_bars4.gif) | 2026-11-12T06:10:11.071+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.71875`; `finite_template_tv=0.28125`; `support_modes=3`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_img_rare_mode_bars4.gif) | 2026-11-12T06:10:26.344+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.65625`; `prediction_rmse=0.30451`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_img_delayed_cue.gif) | 2026-11-12T06:10:42.314+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_img_persistent_fork.gif) | 2026-11-12T06:11:17.503+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.2136`; `xy_rmse=0.99377`; `step_accuracy=0.011719`; `orbit_completion=0`; `waypoint_modes=1`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_img_circle_orbit.gif) | 2026-11-12T06:11:55.982+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.10563`; `s2c_rmse=0.10199`; `c2s_rmse=0.10928`; `xy_rmse=0.43294`; `step_accuracy=0.25781`; `cycle_completion=0`; `waypoint_modes=4`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_img_square_circle_cycle.gif) | 2026-11-12T06:13:07.448+00:00 |

<a id="online-bispf-early-causal-stretch"></a>
### Stretch evidence

<a id="online-bispf-early-causal-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_minimal_dense4.gif) | 2026-11-12T06:13:12.282+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_minimal_tokens4.gif) | 2026-11-12T06:13:17.123+00:00 |

<a id="online-bispf-early-causal-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.45419`; `seen_max_abs_error=1`; `heldout_response_rmse=0.37517`; `heldout_max_abs_error=0.7`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_minimal_kernel2x2.gif) | 2026-11-12T06:13:26.357+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.20833`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0.0625`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_minimal_addition2bit.gif) | 2026-11-12T06:13:35.377+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011406`; `relative_mse=0.46292`; `moving_direction_accuracy=0.83854`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_robotics_fetch_reach.gif) | 2026-11-12T06:13:52.664+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.013763`; `relative_mse=1.262`; `moving_direction_accuracy=0.65698`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_robotics_fetch_push.gif) | 2026-11-12T06:14:04.244+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011325`; `relative_mse=0.78063`; `moving_direction_accuracy=0.82353`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_robotics_fetch_slide.gif) | 2026-11-12T06:14:25.589+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0094148`; `relative_mse=0.56472`; `moving_direction_accuracy=0.65351`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T06:14:48.820+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0089819`; `relative_mse=1.1933`; `moving_direction_accuracy=0.55491`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_robotics_hand_reach.gif) | 2026-11-12T06:15:12.046+00:00 |

<a id="online-bispf-early-causal-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.75006`; `interpolation_speed_error=0.63178`; `interpolation_raw_rmse=0.073697`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.45833`; `extrapolation_centroid_rmse=0.92752`; `extrapolation_speed_error=0.7742`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.39583`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bispf-early-causal/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bispf-early-causal_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T06:15:39.350+00:00 |

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

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `1a2ec8c64991ff95e4fb43d5927803bf7f41d673a316c993448b1fb7cc4316cf`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-bispf-momentum-required"></a>
### Required evidence

<a id="online-bispf-momentum-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-momentum/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bispf-momentum_seq_img_round_robin_bars4.gif) | 2026-11-12T06:21:22.409+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-momentum/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-momentum_seq_img_dwell_bars4.gif) | 2026-11-12T06:21:31.254+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.46831`; `direction_acc=0.34375`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf-momentum/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bispf-momentum_seq_img_translating_bar_next.gif) | 2026-11-12T06:21:39.539+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-momentum/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bispf-momentum_seq_img_fork_rollout.gif) | 2026-11-12T06:21:47.465+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-momentum/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-momentum_seq_img_variable_dwell_bars4.gif) | 2026-11-12T06:21:54.876+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.21875`; `finite_template_tv=0.78125`; `support_modes=1`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-bispf-momentum/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bispf-momentum_seq_img_rare_mode_bars4.gif) | 2026-11-12T06:22:02.180+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.5`; `prediction_rmse=0.31415`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf-momentum/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bispf-momentum_seq_img_delayed_cue.gif) | 2026-11-12T06:22:10.409+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-momentum/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bispf-momentum_seq_img_persistent_fork.gif) | 2026-11-12T06:22:23.678+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.25972`; `xy_rmse=1.1539`; `step_accuracy=0`; `orbit_completion=0`; `waypoint_modes=0`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-momentum/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bispf-momentum_seq_img_circle_orbit.gif) | 2026-11-12T06:22:39.325+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.30796`; `s2c_rmse=0.36515`; `c2s_rmse=0.25078`; `xy_rmse=1.1958`; `step_accuracy=0`; `cycle_completion=0`; `waypoint_modes=0`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-momentum/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bispf-momentum_seq_img_square_circle_cycle.gif) | 2026-11-12T06:23:05.606+00:00 |

<a id="online-bispf-momentum-stretch"></a>
### Stretch evidence

<a id="online-bispf-momentum-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bispf-momentum/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bispf-momentum_seq_minimal_dense4.gif) | 2026-11-12T06:23:10.725+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bispf-momentum/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bispf-momentum_seq_minimal_tokens4.gif) | 2026-11-12T06:23:15.285+00:00 |

<a id="online-bispf-momentum-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.51937`; `seen_max_abs_error=0.99996`; `heldout_response_rmse=0.45064`; `heldout_max_abs_error=0.69996`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bispf-momentum/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bispf-momentum_seq_minimal_kernel2x2.gif) | 2026-11-12T06:23:21.397+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.083333`; `seen_min_bit_confidence=1.8248e-24`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=1.8248e-24`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bispf-momentum/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bispf-momentum_seq_minimal_addition2bit.gif) | 2026-11-12T06:23:26.312+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.033919`; `relative_mse=4.0939`; `moving_direction_accuracy=0.43229`; `finite_fraction=0.96875`; failed terminal bounds: finite_fraction == 1.0; moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-momentum/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bispf-momentum_seq_robotics_fetch_reach.gif) | 2026-11-12T06:23:36.764+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.018285`; `relative_mse=2.2276`; `moving_direction_accuracy=0.52326`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-momentum/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bispf-momentum_seq_robotics_fetch_push.gif) | 2026-11-12T06:23:47.361+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.01565`; `relative_mse=1.4908`; `moving_direction_accuracy=0.64706`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-momentum/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bispf-momentum_seq_robotics_fetch_slide.gif) | 2026-11-12T06:23:57.746+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.020614`; `relative_mse=2.7074`; `moving_direction_accuracy=0.35088`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-momentum/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bispf-momentum_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T06:24:08.243+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.7428`; `relative_mse=8161.1`; `moving_direction_accuracy=0.48715`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-momentum/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bispf-momentum_seq_robotics_hand_reach.gif) | 2026-11-12T06:24:18.638+00:00 |

<a id="online-bispf-momentum-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.80599`; `interpolation_speed_error=0.98757`; `interpolation_raw_rmse=0.073697`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.45833`; `extrapolation_centroid_rmse=0.85714`; `extrapolation_speed_error=0.84952`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.5625`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bispf-momentum/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bispf-momentum_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T06:24:29.133+00:00 |

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

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `669bad93145e590d98311168b023cb673093dc947fc52cda0208b9c060b88f7b`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-bispf-optimized-required"></a>
### Required evidence

<a id="online-bispf-optimized-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-optimized/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bispf-optimized_seq_img_round_robin_bars4.gif) | 2026-11-12T06:24:41.798+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-optimized/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-optimized_seq_img_dwell_bars4.gif) | 2026-11-12T06:24:52.156+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.35677`; `direction_acc=0.5`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf-optimized/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bispf-optimized_seq_img_translating_bar_next.gif) | 2026-11-12T06:25:04.000+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-optimized/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bispf-optimized_seq_img_fork_rollout.gif) | 2026-11-12T06:25:15.817+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-optimized/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-optimized_seq_img_variable_dwell_bars4.gif) | 2026-11-12T06:25:25.691+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.34375`; `finite_template_tv=0.6875`; `support_modes=1`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-bispf-optimized/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bispf-optimized_seq_img_rare_mode_bars4.gif) | 2026-11-12T06:25:35.945+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.5`; `prediction_rmse=0.29708`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf-optimized/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bispf-optimized_seq_img_delayed_cue.gif) | 2026-11-12T06:25:47.574+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-optimized/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bispf-optimized_seq_img_persistent_fork.gif) | 2026-11-12T06:26:04.095+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.19917`; `xy_rmse=1.0319`; `step_accuracy=0`; `orbit_completion=0`; `waypoint_modes=0`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-optimized/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bispf-optimized_seq_img_circle_orbit.gif) | 2026-11-12T06:26:22.874+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.20082`; `s2c_rmse=0.20145`; `c2s_rmse=0.20019`; `xy_rmse=1.1539`; `step_accuracy=0`; `cycle_completion=0`; `waypoint_modes=0`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-optimized/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bispf-optimized_seq_img_square_circle_cycle.gif) | 2026-11-12T06:26:57.112+00:00 |

<a id="online-bispf-optimized-stretch"></a>
### Stretch evidence

<a id="online-bispf-optimized-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bispf-optimized/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bispf-optimized_seq_minimal_dense4.gif) | 2026-11-12T06:27:00.953+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bispf-optimized/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bispf-optimized_seq_minimal_tokens4.gif) | 2026-11-12T06:27:05.057+00:00 |

<a id="online-bispf-optimized-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.31016`; `seen_max_abs_error=0.74491`; `heldout_response_rmse=0.17185`; `heldout_max_abs_error=0.44491`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bispf-optimized/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bispf-optimized_seq_minimal_kernel2x2.gif) | 2026-11-12T06:27:11.372+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.083333`; `seen_min_bit_confidence=1.5962e-05`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=1.5962e-05`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bispf-optimized/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bispf-optimized_seq_minimal_addition2bit.gif) | 2026-11-12T06:27:18.040+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.016563`; `relative_mse=0.97614`; `moving_direction_accuracy=0.56771`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-optimized/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bispf-optimized_seq_robotics_fetch_reach.gif) | 2026-11-12T06:27:32.423+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.013553`; `relative_mse=1.2239`; `moving_direction_accuracy=0.42442`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-optimized/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bispf-optimized_seq_robotics_fetch_push.gif) | 2026-11-12T06:27:46.291+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.017166`; `relative_mse=1.7935`; `moving_direction_accuracy=0.80392`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-optimized/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bispf-optimized_seq_robotics_fetch_slide.gif) | 2026-11-12T06:28:00.537+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.01042`; `relative_mse=0.6917`; `moving_direction_accuracy=0.72807`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-optimized/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bispf-optimized_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T06:28:14.292+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012176`; `relative_mse=2.1928`; `moving_direction_accuracy=0.45327`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-optimized/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bispf-optimized_seq_robotics_hand_reach.gif) | 2026-11-12T06:28:28.076+00:00 |

<a id="online-bispf-optimized-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.51843`; `interpolation_speed_error=0.42722`; `interpolation_raw_rmse=0.073785`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.4375`; `extrapolation_centroid_rmse=0.67044`; `extrapolation_speed_error=0.57839`; `extrapolation_raw_rmse=0.070734`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.45833`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bispf-optimized/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bispf-optimized_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T06:28:41.350+00:00 |

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

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `0f3b7cd014e65bed1cd9b6cd61d5fc1940a188ef4aa37780ebe26d40fa30b573`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-bispf-optimized-cool-required"></a>
### Required evidence

<a id="online-bispf-optimized-cool-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-optimized-cool/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bispf-optimized-cool_seq_img_round_robin_bars4.gif) | 2026-11-12T06:28:51.926+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-optimized-cool/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-optimized-cool_seq_img_dwell_bars4.gif) | 2026-11-12T06:29:02.133+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.36004`; `direction_acc=0.51562`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf-optimized-cool/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bispf-optimized-cool_seq_img_translating_bar_next.gif) | 2026-11-12T06:29:13.743+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-optimized-cool/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bispf-optimized-cool_seq_img_fork_rollout.gif) | 2026-11-12T06:29:24.939+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-optimized-cool/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-optimized-cool_seq_img_variable_dwell_bars4.gif) | 2026-11-12T06:29:34.465+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0`; `finite_template_tv=1`; `support_modes=0`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-bispf-optimized-cool/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bispf-optimized-cool_seq_img_rare_mode_bars4.gif) | 2026-11-12T06:29:44.256+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.5`; `prediction_rmse=0.33203`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf-optimized-cool/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bispf-optimized-cool_seq_img_delayed_cue.gif) | 2026-11-12T06:29:55.648+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-optimized-cool/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bispf-optimized-cool_seq_img_persistent_fork.gif) | 2026-11-12T06:30:12.559+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.22832`; `xy_rmse=1.4609`; `step_accuracy=0`; `orbit_completion=0`; `waypoint_modes=0`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-optimized-cool/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bispf-optimized-cool_seq_img_circle_orbit.gif) | 2026-11-12T06:30:32.002+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.21999`; `s2c_rmse=0.22778`; `c2s_rmse=0.21219`; `xy_rmse=1.4198`; `step_accuracy=0`; `cycle_completion=0`; `waypoint_modes=0`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-optimized-cool/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bispf-optimized-cool_seq_img_square_circle_cycle.gif) | 2026-11-12T06:31:05.185+00:00 |

<a id="online-bispf-optimized-cool-stretch"></a>
### Stretch evidence

<a id="online-bispf-optimized-cool-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bispf-optimized-cool/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bispf-optimized-cool_seq_minimal_dense4.gif) | 2026-11-12T06:31:09.033+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bispf-optimized-cool/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bispf-optimized-cool_seq_minimal_tokens4.gif) | 2026-11-12T06:31:12.741+00:00 |

<a id="online-bispf-optimized-cool-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.30519`; `seen_max_abs_error=0.61867`; `heldout_response_rmse=0.16272`; `heldout_max_abs_error=0.31867`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bispf-optimized-cool/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bispf-optimized-cool_seq_minimal_kernel2x2.gif) | 2026-11-12T06:31:18.955+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.083333`; `seen_min_bit_confidence=0.083985`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=0.083985`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bispf-optimized-cool/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bispf-optimized-cool_seq_minimal_addition2bit.gif) | 2026-11-12T06:31:24.908+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.022396`; `relative_mse=1.7848`; `moving_direction_accuracy=0.39583`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-optimized-cool/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bispf-optimized-cool_seq_robotics_fetch_reach.gif) | 2026-11-12T06:31:38.615+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.016165`; `relative_mse=1.7412`; `moving_direction_accuracy=0.64535`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-optimized-cool/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bispf-optimized-cool_seq_robotics_fetch_push.gif) | 2026-11-12T06:31:52.378+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.014359`; `relative_mse=1.2548`; `moving_direction_accuracy=0.63235`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-optimized-cool/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bispf-optimized-cool_seq_robotics_fetch_slide.gif) | 2026-11-12T06:32:06.472+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012573`; `relative_mse=1.0071`; `moving_direction_accuracy=0.57456`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-optimized-cool/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bispf-optimized-cool_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T06:32:20.736+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.01864`; `relative_mse=5.1394`; `moving_direction_accuracy=0.45093`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-optimized-cool/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bispf-optimized-cool_seq_robotics_hand_reach.gif) | 2026-11-12T06:32:36.775+00:00 |

<a id="online-bispf-optimized-cool-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.32928`; `interpolation_speed_error=0.27337`; `interpolation_raw_rmse=0.1047`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.39583`; `extrapolation_centroid_rmse=0.48252`; `extrapolation_speed_error=0.41967`; `extrapolation_raw_rmse=0.10136`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.375`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bispf-optimized-cool/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bispf-optimized-cool_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T06:32:55.512+00:00 |

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

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `8b6e4eafb340ab7d5111c2bc5ce21c9b7850e7db740549748cf5e0a4e9fb9f70`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-bispf-slow-critic-required"></a>
### Required evidence

<a id="online-bispf-slow-critic-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-slow-critic/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bispf-slow-critic_seq_img_round_robin_bars4.gif) | 2026-11-12T06:33:08.359+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0.03125`; `modes=1`; `finite_template_tv=0.96875`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-slow-critic/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-slow-critic_seq_img_dwell_bars4.gif) | 2026-11-12T06:33:19.705+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.20794`; `direction_acc=0.90625`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf-slow-critic/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bispf-slow-critic_seq_img_translating_bar_next.gif) | 2026-11-12T06:33:32.623+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-slow-critic/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bispf-slow-critic_seq_img_fork_rollout.gif) | 2026-11-12T06:33:45.348+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-slow-critic/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-slow-critic_seq_img_variable_dwell_bars4.gif) | 2026-11-12T06:33:57.134+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.09375`; `finite_template_tv=0.90625`; `support_modes=1`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-bispf-slow-critic/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bispf-slow-critic_seq_img_rare_mode_bars4.gif) | 2026-11-12T06:34:08.619+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.5`; `prediction_rmse=0.29558`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf-slow-critic/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bispf-slow-critic_seq_img_delayed_cue.gif) | 2026-11-12T06:34:22.105+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-slow-critic/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bispf-slow-critic_seq_img_persistent_fork.gif) | 2026-11-12T06:34:42.328+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.24962`; `xy_rmse=1.0098`; `step_accuracy=0`; `orbit_completion=0`; `waypoint_modes=0`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-slow-critic/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bispf-slow-critic_seq_img_circle_orbit.gif) | 2026-11-12T06:35:05.177+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.18198`; `s2c_rmse=0.18061`; `c2s_rmse=0.18336`; `xy_rmse=0.85512`; `step_accuracy=0`; `cycle_completion=0`; `waypoint_modes=0`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-slow-critic/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bispf-slow-critic_seq_img_square_circle_cycle.gif) | 2026-11-12T06:35:45.978+00:00 |

<a id="online-bispf-slow-critic-stretch"></a>
### Stretch evidence

<a id="online-bispf-slow-critic-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bispf-slow-critic/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bispf-slow-critic_seq_minimal_dense4.gif) | 2026-11-12T06:35:50.077+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bispf-slow-critic/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bispf-slow-critic_seq_minimal_tokens4.gif) | 2026-11-12T06:35:54.002+00:00 |

<a id="online-bispf-slow-critic-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.32258`; `seen_max_abs_error=0.84718`; `heldout_response_rmse=0.19337`; `heldout_max_abs_error=0.54718`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bispf-slow-critic/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bispf-slow-critic_seq_minimal_kernel2x2.gif) | 2026-11-12T06:35:59.312+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.14583`; `seen_min_bit_confidence=3.6836e-05`; `heldout_exact_accuracy=0.125`; `heldout_min_bit_confidence=3.6836e-05`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bispf-slow-critic/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bispf-slow-critic_seq_minimal_addition2bit.gif) | 2026-11-12T06:36:04.737+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.017517`; `relative_mse=1.0918`; `moving_direction_accuracy=0.53125`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-slow-critic/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bispf-slow-critic_seq_robotics_fetch_reach.gif) | 2026-11-12T06:36:15.850+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012996`; `relative_mse=1.1253`; `moving_direction_accuracy=0.34302`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-slow-critic/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bispf-slow-critic_seq_robotics_fetch_push.gif) | 2026-11-12T06:36:27.372+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012197`; `relative_mse=0.90546`; `moving_direction_accuracy=0.76471`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-slow-critic/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bispf-slow-critic_seq_robotics_fetch_slide.gif) | 2026-11-12T06:36:38.756+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0097118`; `relative_mse=0.6009`; `moving_direction_accuracy=0.67982`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-bispf-slow-critic/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bispf-slow-critic_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T06:36:49.887+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.009632`; `relative_mse=1.3723`; `moving_direction_accuracy=0.52921`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-slow-critic/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bispf-slow-critic_seq_robotics_hand_reach.gif) | 2026-11-12T06:37:00.715+00:00 |

<a id="online-bispf-slow-critic-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.85186`; `interpolation_speed_error=0.66184`; `interpolation_raw_rmse=0.23182`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.4375`; `extrapolation_centroid_rmse=0.85454`; `extrapolation_speed_error=0.675`; `extrapolation_raw_rmse=0.2526`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.39583`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_raw_rmse <= 0.12; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_raw_rmse <= 0.12; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bispf-slow-critic/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bispf-slow-critic_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T06:37:11.904+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bispf-slow-critic --device cuda`.

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

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `eff6330e8d0bedde23727e598c9f12a05fdc152fa0cb0db9ced818e95526dfa9`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-bispf-transport-causal-required"></a>
### Required evidence

<a id="online-bispf-transport-causal-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.65625`; `modes=4`; `finite_template_tv=0.34375`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9 | [receipt](../receipts/cuda/online-bispf-transport-causal/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bispf-transport-causal_seq_img_round_robin_bars4.gif) | 2026-11-12T06:41:07.019+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0.40625`; `modes=4`; `finite_template_tv=0.59375`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-transport-causal/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-transport-causal_seq_img_dwell_bars4.gif) | 2026-11-12T06:41:22.994+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.33069`; `direction_acc=0.57812`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf-transport-causal/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bispf-transport-causal_seq_img_translating_bar_next.gif) | 2026-11-12T06:41:55.875+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=0.625`; `within_state_modes=2`; `branch_tv=0.375`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85 | [receipt](../receipts/cuda/online-bispf-transport-causal/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bispf-transport-causal_seq_img_fork_rollout.gif) | 2026-11-12T06:42:38.088+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0.625`; `modes=4`; `finite_template_tv=0.375`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bispf-transport-causal/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bispf-transport-causal_seq_img_variable_dwell_bars4.gif) | 2026-11-12T06:43:17.450+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.75`; `finite_template_tv=0.28125`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85 | [receipt](../receipts/cuda/online-bispf-transport-causal/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bispf-transport-causal_seq_img_rare_mode_bars4.gif) | 2026-11-12T06:44:08.036+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.70312`; `prediction_rmse=0.27281`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bispf-transport-causal/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bispf-transport-causal_seq_img_delayed_cue.gif) | 2026-11-12T06:44:43.806+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0.375`; `within_state_modes=1`; `branch_tv=0.625`; `trajectory_consistency=0.21875`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bispf-transport-causal/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bispf-transport-causal_seq_img_persistent_fork.gif) | 2026-11-12T06:45:42.176+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.16985`; `xy_rmse=0.7605`; `step_accuracy=0.20312`; `orbit_completion=0`; `waypoint_modes=8`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-transport-causal/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bispf-transport-causal_seq_img_circle_orbit.gif) | 2026-11-12T06:46:15.391+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.15823`; `s2c_rmse=0.15019`; `c2s_rmse=0.16628`; `xy_rmse=0.78034`; `step_accuracy=0.070312`; `cycle_completion=0`; `waypoint_modes=2`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bispf-transport-causal/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bispf-transport-causal_seq_img_square_circle_cycle.gif) | 2026-11-12T06:47:09.464+00:00 |

<a id="online-bispf-transport-causal-stretch"></a>
### Stretch evidence

<a id="online-bispf-transport-causal-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bispf-transport-causal/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bispf-transport-causal_seq_minimal_dense4.gif) | 2026-11-12T06:47:14.167+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0.25`; `modes=4`; `mass_tv=0.75`; failed terminal bounds: mass_tv <= 0.1; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bispf-transport-causal/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bispf-transport-causal_seq_minimal_tokens4.gif) | 2026-11-12T06:47:18.835+00:00 |

<a id="online-bispf-transport-causal-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.38473`; `seen_max_abs_error=0.93527`; `heldout_response_rmse=0.29789`; `heldout_max_abs_error=0.63822`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bispf-transport-causal/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bispf-transport-causal_seq_minimal_kernel2x2.gif) | 2026-11-12T06:47:27.552+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.10417`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0.125`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bispf-transport-causal/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bispf-transport-causal_seq_minimal_addition2bit.gif) | 2026-11-12T06:47:35.952+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0085994`; `relative_mse=0.26314`; `moving_direction_accuracy=0.86458`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-bispf-transport-causal/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bispf-transport-causal_seq_robotics_fetch_reach.gif) | 2026-11-12T06:47:53.959+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011208`; `relative_mse=0.83692`; `moving_direction_accuracy=0.66279`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-transport-causal/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bispf-transport-causal_seq_robotics_fetch_push.gif) | 2026-11-12T06:48:11.569+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0073404`; `relative_mse=0.32795`; `moving_direction_accuracy=0.76471`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-bispf-transport-causal/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bispf-transport-causal_seq_robotics_fetch_slide.gif) | 2026-11-12T06:48:31.957+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 1/5 | `position_delta_rmse_m=0.003573`; `relative_mse=0.081337`; `moving_direction_accuracy=0.74123`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75 | [receipt](../receipts/cuda/online-bispf-transport-causal/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bispf-transport-causal_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T06:48:50.785+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.008864`; `relative_mse=1.1622`; `moving_direction_accuracy=0.54907`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bispf-transport-causal/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bispf-transport-causal_seq_robotics_hand_reach.gif) | 2026-11-12T06:49:12.266+00:00 |

<a id="online-bispf-transport-causal-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.81989`; `interpolation_speed_error=0.64751`; `interpolation_raw_rmse=0.073698`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.4375`; `extrapolation_centroid_rmse=0.96564`; `extrapolation_speed_error=0.77089`; `extrapolation_raw_rmse=0.070666`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.41667`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bispf-transport-causal/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bispf-transport-causal_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T06:49:28.453+00:00 |

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

<a id="refresh"></a>
## Refresh

```sh
particlestream compile --device cuda
```

This page is generated alongside the leaderboard from committed source-bound receipts and declarations; editing a generated page cannot change a verdict or earn qualification.

<a id="references"></a>
## References

- [Sinkhorn Distances: Lightspeed Computation of Optimal Transport](https://arxiv.org/abs/1306.0895). Entropic Optimal Transport Sinkhorn-Knopp coupling and marginal-residual statistics.
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). First-order moment optimizer used in `bispf` recipes alongside DualNorm in `bispf_optimized`.
- [The relativistic discriminator: a key element missing from standard GAN](https://arxiv.org/abs/1807.00734). Relativistic pairing foundation supported by the upstream BISPF preset family.
