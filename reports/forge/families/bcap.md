<!-- Generated Forge family report -->

# BCAP

[← Technique leaderboard](../technique-inventory.md)

<a id="technique-overview"></a>
## Technique overview

BCAP regularizes the critic with a fixed one-sided squared hinge penalty whenever per-sample real or generated input-gradient norms exceed cap threshold $\kappa$. Within ParticleStream, this critic-cap formulation spans DualNorm and Adam optimizers, unrouted uniform particle sampling, competitive 32-center MSE routing, and context-balanced Kantorovich dual routing paired with detached velocity or causal-state memory under `batch_size=1` online execution.

[Implementation](../../../particlestream/tier1.py) · [Mathematical formulation](bcap.md#mathematical-formulation) · [Simplified pseudocode](bcap.md#simplified-pseudocode) · [Training details](bcap.md#training-details) · [Protocol](../../../EXPERIMENTATION.md) · [Terminology](../../../RESEARCH_GLOSSARY.md)

<a id="mathematical-formulation"></a>
## Mathematical formulation

$G$ is the generator, $D$ the raw-score critic, and $P$ the $K$-component task prior ($K=32$, $d_z=8$, $\sigma=0.025$) with centers $\{\mu_j\}_{j=0}^{K-1}$. At step $t$, $x_t$ is the single real arrival, $c_{t-1}$ is the optional detached memory context, $\hat x_D=G(z_{i_D},c_{t-1})$ is the detached critic fake, and $\hat x_G=G(z_{i_G},c_{t-1})$ is the differentiable generator fake. $g_r=\nabla_x D(x_t,c_{t-1})$ and $g_f=\nabla_{\hat x} D(\hat x_D,c_{t-1})$ are per-sample input gradients; $\lambda$ is penalty strength, $\kappa$ is the gradient cap, and $(a)_+=\max(a,0)$. For routed variants, $\hat x_j^{(0)}=G(\mu_j,c_{t-1})$ denotes a detached eval-mode clean-center probe, $m_j$ is global assignment mass ($\gamma_u$ decay, $\lambda_u$ usage weight), and $u_b\in\mathbb R^K$ is the zero-sum clipped Kantorovich potential vector at matched context bucket $b$.

**Fixed Capped Input-Gradient Penalty**

$$
R_{\mathrm{BCAP}}(D;x_t,\hat x_D)=\frac{\lambda}{2}\Big[(\lVert g_r\rVert_2-\kappa)_+^2+(\lVert g_f\rVert_2-\kappa)_+^2\Big].
$$

BCAP penalizes critic input-gradient norms above $\kappa$ without dividing by input dimension $d=64$ and without hard-clipping critic parameter gradients. Canonical ParticleStream candidates use $\lambda=1$ and $\kappa=1$.

**Single-Arrival Critic and Generator Objectives**

$$
\begin{aligned}\mathcal L_D&=\ell_D\big(D(x_t,c_{t-1}),\,D(\hat x_D,c_{t-1})\big)+R_{\mathrm{BCAP}}(D;x_t,\hat x_D),\\\mathcal L_G&=\ell_G\big(D(\hat x_G,c_{t-1}),\,D(x_t,c_{t-1})\big)+R_{\mathrm{prior}}(P)+\lambda_{\mathrm{rec}}\,\mathcal L_{\mathrm{rec}}(\hat x_G,x_t).\end{aligned}
$$

Preset candidates inherit paired relativistic logistic $\ell_D,\ell_G$, whereas tuned online candidates switch to decoupled non-saturating logistic loss so single-sample `batch_size=1` mode mismatches do not invert real-vs-fake score differences.

**Competitive and Context-Balanced Particle Routing**

$$
\begin{aligned}s_j^{\mathrm{comp}}&=\frac{1}{64}\lVert \hat x_j^{(0)}-x_t\rVert_F^2+\lambda_u\log(1+m_j),\qquad s_j^{\mathrm{bal}}=\frac{1}{64}\lVert \hat x_j^{(0)}-x_t\rVert_F^2+u_{b,j},\\\tilde u_b&=u_b+\eta_{\mathrm{dual}}\!\left(e_{i_G}-\frac{1}{K}\mathbf 1_K\right),\qquad u_b\leftarrow\big(\tilde u_b-\operatorname{mean}(\tilde u_b)\mathbf 1_K\big)\cdot\min\!\left(1,\,\frac{u_{\mathrm{clip}}}{\lVert \tilde u_b-\operatorname{mean}(\tilde u_b)\mathbf 1_K\rVert_\infty}\right).\end{aligned}
$$

Critic steps always sample $i_D\sim\mathcal U\{0,\dots,K-1\}$ uniformly so serving matches the uniform prior. Generator steps either sample uniformly (`router=None`) or probe all $K=32$ clean centers in eval mode and select $i_G=\arg\min_j s_j$ (or softmax at temperature $T>0$), maintaining either global load-balancing mass $m$ or per-context zero-sum duals $u_b$.

<a id="simplified-pseudocode"></a>
## Simplified pseudocode

```text
Choose optimizer (DualNorm or Adam), adversarial loss, memory reader and optional particle router.
For each chronological single-arrival step t:
  Serve pre-arrival prediction from uniform prior row under isolated eval RNG when task is predictive.
  Sample uniform critic index i_D ~ Uniform({0, ..., K - 1}) and form detached fake x_hat_D = G(z_{i_D}, c_{t-1}).
  Compute input gradients g_r = grad_x D(x_t, c_{t-1}) and g_f = grad_x D(x_hat_D, c_{t-1}).
  Backpropagate ell_D + fixed BCAP cap penalty and step D.
  If a router is configured, evaluate K = 32 detached clean-center probes G(mu_j, c_{t-1}) against x_t
    and select i_G using MSE + global usage penalty or context-local Kantorovich duals u_b; else draw uniform i_G.
  Recompute differentiable fake x_hat_G = G(z_{i_G}, c_{t-1}) and evaluate ell_G + R_prior + lambda_rec L_rec(x_hat_G, x_t).
  Step G (and trainable prior P unless prior_update = 'frozen'); update dual/usage state and detached memory c_t.
```

<a id="training-details"></a>
## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Paired relativistic logistic in preset baselines (`online-bcap`, `online-bcap-adam`); decoupled non-saturating logistic in tuned online recipes, optionally paired with MSE or weight-10 BCE reconstruction on routed generator steps. |
| Optimizer | `bcap` preset uses DualNorm (`online-bcap`); `bcap_adam` recipes use Adam ($\beta_1=0$, $\beta_2=0.999$ default or $0.99$ in tuned online recipes). |
| Learning rates and annealing | Preset baselines run constant LRs (`0.012` DualNorm, `0.00425` Adam); tuned online recipes use base LR `0.005` cosine-annealed from $50\%$ of budget to `lr_floor=0.05`. |
| Parameter-gradient clipping | None (`critic_grad_guard=False`). Critic regularity comes solely from the soft BCAP input-gradient hinge. |
| Critic penalties and anchors | Fixed real/fake one-sided squared $L_2$ cap ($\lambda=1.0$, $\kappa=1.0$) every critic step; no EMA critic anchor. |
| Damping and update guards | No A2 sparse-row damping (`latent_damping_max_rate=0.0`). Frozen-prior variants (`online-bcap-balanced-frozen`, `online-bcap-balanced-bce`) eliminate sparse-row Adam drift altogether. |
| Online memory, prior and routing | Supports `kind="none"`, `FastWeightMemory` (`reader="velocity"`), or `CausalStateMemory` (`reader="causal_state"`), combined with uniform, `competitive`, or `context_balanced` routing (`replay_policy="none"`, `rehearsal_batch=0` on all active recipes). |
| Training and sampling noise | Zero scheduled critic-input or generator-output noise in tuned online recipes; task-owned MoG $\sigma=0.025$ is preserved. |
| Parameter averaging and serving | Live weights (`ema_decay=0.0`, `serve_average=False`) on tuned online candidates. |

<a id="configuration-differences"></a>
## Configuration differences

- The fixed BCAP critic input-gradient cap defines the family; optimizer choice, loss pairing, prior freeze, memory reader, and generator routing are explicit recipe deltas.
- Among BCAP recipes, `online-bcap-balanced-bce` is the family representative (`3/10` required passes, `1/10` stretch pass on `seq_minimal_tokens4`), demonstrating that context-local zero-sum duals + BCE reconstruction capture stationary modes and 1-step motion (`bars4`, `dwell_bars4`, `translating_bar_next`) but still diffuse uniform conditional capacity across contexts.
- Historical `bcap`, `stream-bcap`, and `stream-bcap-adam` declarations predate the explicit `replay_policy="none"` schema and remain read-only provenance records.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [particlestream/tier1.py](../../../particlestream/tier1.py)
- [particlestream/transport.py](../../../particlestream/transport.py)
- [particlestream/balanced_transport.py](../../../particlestream/balanced_transport.py)
- [particlestream/memory.py](../../../particlestream/memory.py)
- [configs/forge/ideas/online-bcap-balanced-bce.json](../../../configs/forge/ideas/online-bcap-balanced-bce.json)
- [configs/forge/ideas/online-bcap-adam-causal.json](../../../configs/forge/ideas/online-bcap-adam-causal.json)

</details>

Each section below describes one exact global recipe. Configuration and parent deltas come from the declarations; results come from Forge's current source-bound receipts. Required and stretch evidence stay separate. A final metric can meet its gate while the sustained verdict fails: every one of the last five checks must pass.

The installed engine is pinned to ParticleGAN/Forge 0.9.7. Older version or replay wording in declaration history describes provenance; the effective configuration below controls execution.

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

Current `cuda`, seed 0: **FAIL**, 3/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `c2ff05d739fde37b2c35b23d6a4df5d7c9a6110f4c05f9e258aae410fb94a507`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-bcap-balanced-bce-required"></a>
### Required evidence

<a id="online-bcap-balanced-bce-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_img_round_robin_bars4.gif) | 2026-11-13T21:08:20.028+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0.71875`; `modes=4`; `finite_template_tv=0.28125`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_img_dwell_bars4.gif) | 2026-11-13T21:08:30.686+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.079114`; `direction_acc=0.96875` | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_img_translating_bar_next.gif) | 2026-11-13T21:08:42.370+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 1/5 | `hq=0.875`; `within_state_modes=2`; `branch_tv=0.15625`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_img_fork_rollout.gif) | 2026-11-13T21:08:54.044+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0.4375`; `modes=2`; `finite_template_tv=0.59375`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_img_variable_dwell_bars4.gif) | 2026-11-13T21:09:04.371+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=0.96875`; `finite_template_tv=0.0625`; `support_modes=4`; `rare_mass=0.03125`; `rare_mass=0.03125` | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_img_rare_mode_bars4.gif) | 2026-11-13T21:09:14.712+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.92188`; `prediction_rmse=0.20725`; failed terminal bounds: prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_img_delayed_cue.gif) | 2026-11-13T21:09:26.446+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0.8125`; `within_state_modes=2`; `branch_tv=0.1875`; `trajectory_consistency=0.71875`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95 | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_img_persistent_fork.gif) | 2026-11-13T21:09:44.400+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.066614`; `xy_rmse=0.14638`; `step_accuracy=0.70312`; `orbit_completion=0.375`; `waypoint_modes=8`; failed terminal bounds: orbit_completion >= 0.9; step_accuracy >= 0.9; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_img_circle_orbit.gif) | 2026-11-13T21:10:04.557+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.079456`; `s2c_rmse=0.08313`; `c2s_rmse=0.075783`; `xy_rmse=0.20207`; `step_accuracy=0.46094`; `cycle_completion=0.0625`; `waypoint_modes=8`; failed terminal bounds: cycle_completion >= 0.9; s2c_rmse <= 0.12; step_accuracy >= 0.9; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_img_square_circle_cycle.gif) | 2026-11-13T21:10:41.327+00:00 |

<a id="online-bcap-balanced-bce-stretch"></a>
### Stretch evidence

<a id="online-bcap-balanced-bce-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_minimal_dense4.gif) | 2026-11-13T21:10:45.203+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0.09375`; `modes=3`; `mass_tv=0.90625`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_minimal_tokens4.gif) | 2026-11-13T21:10:49.424+00:00 |

<a id="online-bcap-balanced-bce-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.45941`; `seen_max_abs_error=0.98179`; `heldout_response_rmse=0.45059`; `heldout_max_abs_error=0.69969`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_minimal_kernel2x2.gif) | 2026-11-13T21:10:55.906+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.22917`; `seen_min_bit_confidence=2.9723e-14`; `heldout_exact_accuracy=0.125`; `heldout_min_bit_confidence=4.1739e-14`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_minimal_addition2bit.gif) | 2026-11-13T21:11:02.615+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012411`; `relative_mse=0.54812`; `moving_direction_accuracy=0.75521`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_robotics_fetch_reach.gif) | 2026-11-13T21:11:16.974+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011895`; `relative_mse=0.9427`; `moving_direction_accuracy=0.6686`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_robotics_fetch_push.gif) | 2026-11-13T21:11:31.541+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010818`; `relative_mse=0.71236`; `moving_direction_accuracy=0.69608`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_robotics_fetch_slide.gif) | 2026-11-13T21:11:46.122+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0054411`; `relative_mse=0.18862`; `moving_direction_accuracy=0.69298`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_robotics_fetch_pick_and_place.gif) | 2026-11-13T21:12:01.010+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0090099`; `relative_mse=1.2007`; `moving_direction_accuracy=0.55724`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_robotics_hand_reach.gif) | 2026-11-13T21:12:15.506+00:00 |

<a id="online-bcap-balanced-bce-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.42615`; `interpolation_speed_error=0.63747`; `interpolation_raw_rmse=0.10049`; `interpolation_point_validity=0.020833`; `interpolation_direction_accuracy=0.77083`; `extrapolation_centroid_rmse=0.84137`; `extrapolation_speed_error=0.73494`; `extrapolation_raw_rmse=0.11515`; `extrapolation_point_validity=0.041667`; `extrapolation_direction_accuracy=0.60417`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_raw_rmse <= 0.12; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_raw_rmse <= 0.12; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bcap-balanced-bce/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bcap-balanced-bce_seq_minimal_velocity_extrapolation.gif) | 2026-11-13T21:12:30.309+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bcap-balanced-bce --device cuda`.

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
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.09375` | [receipt](../receipts/cuda/online-bcap-balanced/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bcap-balanced_seq_img_round_robin_bars4.gif) | 2026-11-13T21:03:52.726+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0.78125`; `modes=4`; `finite_template_tv=0.21875`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-balanced/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-balanced_seq_img_dwell_bars4.gif) | 2026-11-13T21:04:03.452+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.24144`; `direction_acc=0.6875`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-balanced/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bcap-balanced_seq_img_translating_bar_next.gif) | 2026-11-13T21:04:15.424+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=0.8125`; `within_state_modes=2`; `branch_tv=0.1875`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap-balanced/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bcap-balanced_seq_img_fork_rollout.gif) | 2026-11-13T21:04:27.219+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0.59375`; `modes=2`; `finite_template_tv=0.5625`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-balanced/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-balanced_seq_img_variable_dwell_bars4.gif) | 2026-11-13T21:04:38.220+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-bcap-balanced/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bcap-balanced_seq_img_rare_mode_bars4.gif) | 2026-11-13T21:04:48.944+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.95312`; `prediction_rmse=0.21865`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-balanced/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bcap-balanced_seq_img_delayed_cue.gif) | 2026-11-13T21:05:00.903+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0.46875`; `within_state_modes=2`; `branch_tv=0.53125`; `trajectory_consistency=0.34375`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap-balanced/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bcap-balanced_seq_img_persistent_fork.gif) | 2026-11-13T21:05:19.397+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.087605`; `xy_rmse=0.19863`; `step_accuracy=0.67969`; `orbit_completion=0.3125`; `waypoint_modes=8`; failed terminal bounds: orbit_completion >= 0.9; step_accuracy >= 0.9; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-balanced/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bcap-balanced_seq_img_circle_orbit.gif) | 2026-11-13T21:05:40.066+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.099236`; `s2c_rmse=0.10336`; `c2s_rmse=0.095106`; `xy_rmse=0.32679`; `step_accuracy=0.39844`; `cycle_completion=0.03125`; `waypoint_modes=8`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-balanced/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bcap-balanced_seq_img_square_circle_cycle.gif) | 2026-11-13T21:06:17.183+00:00 |

<a id="online-bcap-balanced-stretch"></a>
### Stretch evidence

<a id="online-bcap-balanced-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bcap-balanced/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bcap-balanced_seq_minimal_dense4.gif) | 2026-11-13T21:06:21.228+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0.1875`; `modes=3`; `mass_tv=0.8125`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bcap-balanced/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bcap-balanced_seq_minimal_tokens4.gif) | 2026-11-13T21:06:25.213+00:00 |

<a id="online-bcap-balanced-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.41735`; `seen_max_abs_error=1`; `heldout_response_rmse=0.37518`; `heldout_max_abs_error=0.7`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bcap-balanced/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bcap-balanced_seq_minimal_kernel2x2.gif) | 2026-11-13T21:06:31.823+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.3125`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0.125`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bcap-balanced/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bcap-balanced_seq_minimal_addition2bit.gif) | 2026-11-13T21:06:38.294+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011162`; `relative_mse=0.44333`; `moving_direction_accuracy=0.84375`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-bcap-balanced/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bcap-balanced_seq_robotics_fetch_reach.gif) | 2026-11-13T21:06:53.268+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011523`; `relative_mse=0.88472`; `moving_direction_accuracy=0.67442`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-balanced/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bcap-balanced_seq_robotics_fetch_push.gif) | 2026-11-13T21:07:08.116+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010111`; `relative_mse=0.62224`; `moving_direction_accuracy=0.75`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-balanced/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bcap-balanced_seq_robotics_fetch_slide.gif) | 2026-11-13T21:07:23.290+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0049915`; `relative_mse=0.15873`; `moving_direction_accuracy=0.67105`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-bcap-balanced/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bcap-balanced_seq_robotics_fetch_pick_and_place.gif) | 2026-11-13T21:07:38.423+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0085216`; `relative_mse=1.0741`; `moving_direction_accuracy=0.57827`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-balanced/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bcap-balanced_seq_robotics_hand_reach.gif) | 2026-11-13T21:07:53.348+00:00 |

<a id="online-bcap-balanced-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.76406`; `interpolation_speed_error=0.62868`; `interpolation_raw_rmse=0.073745`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.5`; `extrapolation_centroid_rmse=0.91925`; `extrapolation_speed_error=0.76872`; `extrapolation_raw_rmse=0.070728`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.39583`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bcap-balanced/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bcap-balanced_seq_minimal_velocity_extrapolation.gif) | 2026-11-13T21:08:08.356+00:00 |

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
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9; modes == 4.0 | [receipt](../receipts/cuda/online-bcap/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bcap_seq_img_round_robin_bars4.gif) | 2026-11-13T20:51:51.613+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bcap_seq_img_dwell_bars4.gif) | 2026-11-13T20:52:08.071+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.35472`; `direction_acc=0.5`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bcap_seq_img_translating_bar_next.gif) | 2026-11-13T20:52:26.072+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bcap_seq_img_fork_rollout.gif) | 2026-11-13T20:52:43.934+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bcap_seq_img_variable_dwell_bars4.gif) | 2026-11-13T20:53:00.349+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.40625`; `finite_template_tv=0.6875`; `support_modes=1`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-bcap/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bcap_seq_img_rare_mode_bars4.gif) | 2026-11-13T20:53:16.961+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.5`; `prediction_rmse=0.29146`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bcap_seq_img_delayed_cue.gif) | 2026-11-13T20:53:34.965+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bcap_seq_img_persistent_fork.gif) | 2026-11-13T20:53:57.707+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.2041`; `xy_rmse=1.0326`; `step_accuracy=0`; `orbit_completion=0`; `waypoint_modes=0`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bcap_seq_img_circle_orbit.gif) | 2026-11-13T20:54:22.956+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.20174`; `s2c_rmse=0.20584`; `c2s_rmse=0.19764`; `xy_rmse=1.1395`; `step_accuracy=0`; `cycle_completion=0`; `waypoint_modes=0`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bcap_seq_img_square_circle_cycle.gif) | 2026-11-13T20:55:09.281+00:00 |

<a id="online-bcap-stretch"></a>
### Stretch evidence

<a id="online-bcap-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bcap/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bcap_seq_minimal_dense4.gif) | 2026-11-13T20:55:14.461+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bcap/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bcap_seq_minimal_tokens4.gif) | 2026-11-13T20:55:19.477+00:00 |

<a id="online-bcap-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.3108`; `seen_max_abs_error=0.74771`; `heldout_response_rmse=0.17301`; `heldout_max_abs_error=0.44771`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bcap/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bcap_seq_minimal_kernel2x2.gif) | 2026-11-13T20:55:28.343+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.083333`; `seen_min_bit_confidence=1.946e-05`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=1.946e-05`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bcap/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bcap_seq_minimal_addition2bit.gif) | 2026-11-13T20:55:36.993+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.016884`; `relative_mse=1.0144`; `moving_direction_accuracy=0.60938`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bcap_seq_robotics_fetch_reach.gif) | 2026-11-13T20:55:57.594+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.017304`; `relative_mse=1.9951`; `moving_direction_accuracy=0.34884`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bcap_seq_robotics_fetch_push.gif) | 2026-11-13T20:56:18.162+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.019216`; `relative_mse=2.2474`; `moving_direction_accuracy=0.80392`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bcap_seq_robotics_fetch_slide.gif) | 2026-11-13T20:56:38.939+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.016364`; `relative_mse=1.7061`; `moving_direction_accuracy=0.62719`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bcap_seq_robotics_fetch_pick_and_place.gif) | 2026-11-13T20:56:59.571+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012815`; `relative_mse=2.429`; `moving_direction_accuracy=0.47664`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bcap_seq_robotics_hand_reach.gif) | 2026-11-13T20:57:20.179+00:00 |

<a id="online-bcap-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.4927`; `interpolation_speed_error=0.40942`; `interpolation_raw_rmse=0.073798`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.5`; `extrapolation_centroid_rmse=0.64694`; `extrapolation_speed_error=0.56666`; `extrapolation_raw_rmse=0.070742`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.5`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bcap/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bcap_seq_minimal_velocity_extrapolation.gif) | 2026-11-13T20:57:38.710+00:00 |

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
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.78125`; `modes=4`; `finite_template_tv=0.25`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9 | [receipt](../receipts/cuda/online-bcap-adam/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bcap-adam_seq_img_round_robin_bars4.gif) | 2026-11-13T20:57:46.517+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-adam/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-adam_seq_img_dwell_bars4.gif) | 2026-11-13T20:57:53.060+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.33884`; `direction_acc=0.5`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-adam/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bcap-adam_seq_img_translating_bar_next.gif) | 2026-11-13T20:58:00.428+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap-adam/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bcap-adam_seq_img_fork_rollout.gif) | 2026-11-13T20:58:07.710+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-adam/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-adam_seq_img_variable_dwell_bars4.gif) | 2026-11-13T20:58:14.312+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.53125`; `finite_template_tv=0.46875`; `support_modes=4`; `rare_mass=0.03125`; `rare_mass=0.03125`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-bcap-adam/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bcap-adam_seq_img_rare_mode_bars4.gif) | 2026-11-13T20:58:20.746+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.51562`; `prediction_rmse=0.2986`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-adam/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bcap-adam_seq_img_delayed_cue.gif) | 2026-11-13T20:58:28.173+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0.03125`; `within_state_modes=1`; `branch_tv=0.96875`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap-adam/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bcap-adam_seq_img_persistent_fork.gif) | 2026-11-13T20:58:40.589+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.11022`; `xy_rmse=0.33159`; `step_accuracy=0.32031`; `orbit_completion=0`; `waypoint_modes=8`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-adam/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bcap-adam_seq_img_circle_orbit.gif) | 2026-11-13T20:58:54.992+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.16012`; `s2c_rmse=0.15918`; `c2s_rmse=0.16107`; `xy_rmse=0.74504`; `step_accuracy=0.015625`; `cycle_completion=0`; `waypoint_modes=1`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-adam/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bcap-adam_seq_img_square_circle_cycle.gif) | 2026-11-13T20:59:20.091+00:00 |

<a id="online-bcap-adam-stretch"></a>
### Stretch evidence

<a id="online-bcap-adam-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bcap-adam/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bcap-adam_seq_minimal_dense4.gif) | 2026-11-13T20:59:23.262+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bcap-adam/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bcap-adam_seq_minimal_tokens4.gif) | 2026-11-13T20:59:26.253+00:00 |

<a id="online-bcap-adam-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.39622`; `seen_max_abs_error=0.96251`; `heldout_response_rmse=0.30054`; `heldout_max_abs_error=0.66251`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bcap-adam/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bcap-adam_seq_minimal_kernel2x2.gif) | 2026-11-13T20:59:30.934+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.083333`; `seen_min_bit_confidence=1.9478e-09`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=1.9478e-09`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bcap-adam/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bcap-adam_seq_minimal_addition2bit.gif) | 2026-11-13T20:59:35.477+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.018141`; `relative_mse=1.171`; `moving_direction_accuracy=0.5`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-adam/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bcap-adam_seq_robotics_fetch_reach.gif) | 2026-11-13T20:59:45.420+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012828`; `relative_mse=1.0964`; `moving_direction_accuracy=0.5814`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-adam/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bcap-adam_seq_robotics_fetch_push.gif) | 2026-11-13T20:59:55.162+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.013395`; `relative_mse=1.0921`; `moving_direction_accuracy=0.4951`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-adam/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bcap-adam_seq_robotics_fetch_slide.gif) | 2026-11-13T21:00:05.094+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011575`; `relative_mse=0.85355`; `moving_direction_accuracy=0.58333`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-adam/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bcap-adam_seq_robotics_fetch_pick_and_place.gif) | 2026-11-13T21:00:14.906+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010953`; `relative_mse=1.7745`; `moving_direction_accuracy=0.51986`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-adam/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bcap-adam_seq_robotics_hand_reach.gif) | 2026-11-13T21:00:24.723+00:00 |

<a id="online-bcap-adam-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.77773`; `interpolation_speed_error=0.8295`; `interpolation_raw_rmse=0.073698`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.47917`; `extrapolation_centroid_rmse=0.91625`; `extrapolation_speed_error=0.7741`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.5625`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bcap-adam/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bcap-adam_seq_minimal_velocity_extrapolation.gif) | 2026-11-13T21:00:34.816+00:00 |

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
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.78125`; `modes=4`; `finite_template_tv=0.25`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_img_round_robin_bars4.gif) | 2026-11-13T21:00:42.616+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_img_dwell_bars4.gif) | 2026-11-13T21:00:49.119+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.28741`; `direction_acc=0.5`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_img_translating_bar_next.gif) | 2026-11-13T21:00:56.592+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_img_fork_rollout.gif) | 2026-11-13T21:01:04.068+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_img_variable_dwell_bars4.gif) | 2026-11-13T21:01:10.722+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.53125`; `finite_template_tv=0.46875`; `support_modes=4`; `rare_mass=0.03125`; `rare_mass=0.03125`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_img_rare_mode_bars4.gif) | 2026-11-13T21:01:17.496+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.85938`; `prediction_rmse=0.29285`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_img_delayed_cue.gif) | 2026-11-13T21:01:24.817+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0.0625`; `within_state_modes=1`; `branch_tv=0.9375`; `trajectory_consistency=0.03125`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_img_persistent_fork.gif) | 2026-11-13T21:01:38.750+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.15671`; `xy_rmse=0.68478`; `step_accuracy=0.25781`; `orbit_completion=0`; `waypoint_modes=7`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_img_circle_orbit.gif) | 2026-11-13T21:01:54.585+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.1352`; `s2c_rmse=0.13422`; `c2s_rmse=0.13617`; `xy_rmse=0.62913`; `step_accuracy=0.28125`; `cycle_completion=0`; `waypoint_modes=3`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_img_square_circle_cycle.gif) | 2026-11-13T21:02:22.740+00:00 |

<a id="online-bcap-adam-causal-stretch"></a>
### Stretch evidence

<a id="online-bcap-adam-causal-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_minimal_dense4.gif) | 2026-11-13T21:02:25.823+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_minimal_tokens4.gif) | 2026-11-13T21:02:29.079+00:00 |

<a id="online-bcap-adam-causal-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.36141`; `seen_max_abs_error=0.99966`; `heldout_response_rmse=0.26174`; `heldout_max_abs_error=0.69709`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_minimal_kernel2x2.gif) | 2026-11-13T21:02:33.688+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.1875`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0.1875`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_minimal_addition2bit.gif) | 2026-11-13T21:02:38.543+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012071`; `relative_mse=0.51845`; `moving_direction_accuracy=0.77083`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_robotics_fetch_reach.gif) | 2026-11-13T21:02:48.690+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.016559`; `relative_mse=1.8269`; `moving_direction_accuracy=0.60465`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_robotics_fetch_push.gif) | 2026-11-13T21:02:58.866+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0090863`; `relative_mse=0.50251`; `moving_direction_accuracy=0.81373`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_robotics_fetch_slide.gif) | 2026-11-13T21:03:09.071+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0046633`; `relative_mse=0.13855`; `moving_direction_accuracy=0.64474`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_robotics_fetch_pick_and_place.gif) | 2026-11-13T21:03:19.469+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.008933`; `relative_mse=1.1803`; `moving_direction_accuracy=0.51636`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_robotics_hand_reach.gif) | 2026-11-13T21:03:29.566+00:00 |

<a id="online-bcap-adam-causal-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.71599`; `interpolation_speed_error=0.62902`; `interpolation_raw_rmse=0.073704`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.4375`; `extrapolation_centroid_rmse=0.92307`; `extrapolation_speed_error=0.77254`; `extrapolation_raw_rmse=0.070682`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.375`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bcap-adam-causal/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bcap-adam-causal_seq_minimal_velocity_extrapolation.gif) | 2026-11-13T21:03:40.588+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bcap-adam-causal --device cuda`.

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

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `60b8fbbaa988816bbd67085c1d69baa0604e5d2a498624b5cf1c0f78e8014bf3`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-bcap-balanced-frozen-required"></a>
### Required evidence

<a id="online-bcap-balanced-frozen-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 4/5 | `hq=0.96875`; `modes=4`; `finite_template_tv=0.0625`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-balanced-frozen/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bcap-balanced-frozen_seq_img_round_robin_bars4.gif) | 2026-11-13T21:12:41.912+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0.3125`; `modes=2`; `finite_template_tv=0.6875`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-balanced-frozen/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-balanced-frozen_seq_img_dwell_bars4.gif) | 2026-11-13T21:12:52.400+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.21219`; `direction_acc=0.75`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-balanced-frozen/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bcap-balanced-frozen_seq_img_translating_bar_next.gif) | 2026-11-13T21:13:03.924+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=0.78125`; `within_state_modes=2`; `branch_tv=0.21875`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85 | [receipt](../receipts/cuda/online-bcap-balanced-frozen/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bcap-balanced-frozen_seq_img_fork_rollout.gif) | 2026-11-13T21:13:15.447+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0.4375`; `modes=3`; `finite_template_tv=0.5625`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-balanced-frozen/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-balanced-frozen_seq_img_variable_dwell_bars4.gif) | 2026-11-13T21:13:25.831+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 4/5 | `hq=0.9375`; `finite_template_tv=0.0625`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85 | [receipt](../receipts/cuda/online-bcap-balanced-frozen/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bcap-balanced-frozen_seq_img_rare_mode_bars4.gif) | 2026-11-13T21:13:36.314+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.9375`; `prediction_rmse=0.24255`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-balanced-frozen/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bcap-balanced-frozen_seq_img_delayed_cue.gif) | 2026-11-13T21:13:47.808+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0.6875`; `within_state_modes=2`; `branch_tv=0.3125`; `trajectory_consistency=0.59375`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95 | [receipt](../receipts/cuda/online-bcap-balanced-frozen/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bcap-balanced-frozen_seq_img_persistent_fork.gif) | 2026-11-13T21:14:06.049+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.07791`; `xy_rmse=0.17035`; `step_accuracy=0.63672`; `orbit_completion=0.1875`; `waypoint_modes=8`; failed terminal bounds: orbit_completion >= 0.9; step_accuracy >= 0.9; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-balanced-frozen/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bcap-balanced-frozen_seq_img_circle_orbit.gif) | 2026-11-13T21:14:26.094+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.10371`; `s2c_rmse=0.10414`; `c2s_rmse=0.10329`; `xy_rmse=0.3601`; `step_accuracy=0.375`; `cycle_completion=0.0625`; `waypoint_modes=8`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; step_accuracy >= 0.9; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-balanced-frozen/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bcap-balanced-frozen_seq_img_square_circle_cycle.gif) | 2026-11-13T21:15:02.615+00:00 |

<a id="online-bcap-balanced-frozen-stretch"></a>
### Stretch evidence

<a id="online-bcap-balanced-frozen-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bcap-balanced-frozen/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bcap-balanced-frozen_seq_minimal_dense4.gif) | 2026-11-13T21:15:06.471+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0.3125`; `modes=3`; `mass_tv=0.6875`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bcap-balanced-frozen/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bcap-balanced-frozen_seq_minimal_tokens4.gif) | 2026-11-13T21:15:10.599+00:00 |

<a id="online-bcap-balanced-frozen-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.33211`; `seen_max_abs_error=0.92382`; `heldout_response_rmse=0.2867`; `heldout_max_abs_error=0.60838`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bcap-balanced-frozen/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bcap-balanced-frozen_seq_minimal_kernel2x2.gif) | 2026-11-13T21:15:17.023+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.20833`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0.1875`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bcap-balanced-frozen/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bcap-balanced-frozen_seq_minimal_addition2bit.gif) | 2026-11-13T21:15:23.479+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.014011`; `relative_mse=0.69853`; `moving_direction_accuracy=0.77604`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-bcap-balanced-frozen/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bcap-balanced-frozen_seq_robotics_fetch_reach.gif) | 2026-11-13T21:15:37.836+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011437`; `relative_mse=0.87156`; `moving_direction_accuracy=0.61628`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-balanced-frozen/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bcap-balanced-frozen_seq_robotics_fetch_push.gif) | 2026-11-13T21:15:52.391+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011533`; `relative_mse=0.80957`; `moving_direction_accuracy=0.67157`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-balanced-frozen/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bcap-balanced-frozen_seq_robotics_fetch_slide.gif) | 2026-11-13T21:16:06.996+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0068278`; `relative_mse=0.29701`; `moving_direction_accuracy=0.73246`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-bcap-balanced-frozen/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bcap-balanced-frozen_seq_robotics_fetch_pick_and_place.gif) | 2026-11-13T21:16:21.645+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.008694`; `relative_mse=1.118`; `moving_direction_accuracy=0.53855`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-balanced-frozen/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bcap-balanced-frozen_seq_robotics_hand_reach.gif) | 2026-11-13T21:16:36.111+00:00 |

<a id="online-bcap-balanced-frozen-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.68349`; `interpolation_speed_error=0.59612`; `interpolation_raw_rmse=0.073855`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.52083`; `extrapolation_centroid_rmse=0.93322`; `extrapolation_speed_error=0.7573`; `extrapolation_raw_rmse=0.070854`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.45833`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bcap-balanced-frozen/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bcap-balanced-frozen_seq_minimal_velocity_extrapolation.gif) | 2026-11-13T21:16:50.653+00:00 |

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

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `d9ae959b3ccfbaca0fa40a0867b434a69892dbd185b804ccaf4b99e3bc51e714`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-bcap-stateless-required"></a>
### Required evidence

<a id="online-bcap-stateless-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.75`; `modes=4`; `finite_template_tv=0.25`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9 | [receipt](../receipts/cuda/online-bcap-stateless/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bcap-stateless_seq_img_round_robin_bars4.gif) | 2026-11-13T21:16:58.168+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-stateless/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-stateless_seq_img_dwell_bars4.gif) | 2026-11-13T21:17:04.453+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.43106`; `direction_acc=0.5`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-stateless/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bcap-stateless_seq_img_translating_bar_next.gif) | 2026-11-13T21:17:11.472+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=0.34375`; `within_state_modes=2`; `branch_tv=0.65625`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap-stateless/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bcap-stateless_seq_img_fork_rollout.gif) | 2026-11-13T21:17:18.265+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-stateless/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-stateless_seq_img_variable_dwell_bars4.gif) | 2026-11-13T21:17:24.610+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.78125`; `finite_template_tv=0.21875`; `support_modes=3`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-bcap-stateless/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bcap-stateless_seq_img_rare_mode_bars4.gif) | 2026-11-13T21:17:30.730+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.48438`; `prediction_rmse=0.29576`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-stateless/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bcap-stateless_seq_img_delayed_cue.gif) | 2026-11-13T21:17:37.768+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0.0625`; `within_state_modes=2`; `branch_tv=0.9375`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap-stateless/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bcap-stateless_seq_img_persistent_fork.gif) | 2026-11-13T21:17:46.548+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.24728`; `xy_rmse=1.1249`; `step_accuracy=0.027344`; `orbit_completion=0`; `waypoint_modes=3`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-stateless/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bcap-stateless_seq_img_circle_orbit.gif) | 2026-11-13T21:17:57.151+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.24344`; `s2c_rmse=0.24669`; `c2s_rmse=0.24019`; `xy_rmse=1.2256`; `step_accuracy=0`; `cycle_completion=0`; `waypoint_modes=0`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-stateless/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bcap-stateless_seq_img_square_circle_cycle.gif) | 2026-11-13T21:18:14.781+00:00 |

<a id="online-bcap-stateless-stretch"></a>
### Stretch evidence

<a id="online-bcap-stateless-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bcap-stateless/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bcap-stateless_seq_minimal_dense4.gif) | 2026-11-13T21:18:17.906+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0.1875`; `modes=3`; `mass_tv=0.8125`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bcap-stateless/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bcap-stateless_seq_minimal_tokens4.gif) | 2026-11-13T21:18:20.979+00:00 |

<a id="online-bcap-stateless-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.39936`; `seen_max_abs_error=0.9825`; `heldout_response_rmse=0.30467`; `heldout_max_abs_error=0.6825`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bcap-stateless/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bcap-stateless_seq_minimal_kernel2x2.gif) | 2026-11-13T21:18:25.538+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.1875`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0.1875`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bcap-stateless/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bcap-stateless_seq_minimal_addition2bit.gif) | 2026-11-13T21:18:29.989+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.016915`; `relative_mse=1.0181`; `moving_direction_accuracy=0.52083`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-stateless/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bcap-stateless_seq_robotics_fetch_reach.gif) | 2026-11-13T21:18:38.773+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.01645`; `relative_mse=1.8029`; `moving_direction_accuracy=0.38372`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-stateless/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bcap-stateless_seq_robotics_fetch_push.gif) | 2026-11-13T21:18:47.715+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011496`; `relative_mse=0.80439`; `moving_direction_accuracy=0.67157`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-stateless/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bcap-stateless_seq_robotics_fetch_slide.gif) | 2026-11-13T21:18:56.645+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.014172`; `relative_mse=1.2796`; `moving_direction_accuracy=0.64912`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-stateless/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bcap-stateless_seq_robotics_fetch_pick_and_place.gif) | 2026-11-13T21:19:05.562+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011694`; `relative_mse=2.0227`; `moving_direction_accuracy=0.51869`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-stateless/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bcap-stateless_seq_robotics_hand_reach.gif) | 2026-11-13T21:19:14.501+00:00 |

<a id="online-bcap-stateless-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.71672`; `interpolation_speed_error=0.56229`; `interpolation_raw_rmse=0.073698`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.14583`; `extrapolation_centroid_rmse=0.84737`; `extrapolation_speed_error=0.67479`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.125`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bcap-stateless/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bcap-stateless_seq_minimal_velocity_extrapolation.gif) | 2026-11-13T21:19:22.208+00:00 |

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

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `40feedd54a83b5ffb2352e474b69984092e9e7ae3b1eac85684a0d2cfaf51f6d`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-bcap-transport-required"></a>
### Required evidence

<a id="online-bcap-transport-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.625`; `modes=4`; `finite_template_tv=0.375`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9 | [receipt](../receipts/cuda/online-bcap-transport/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bcap-transport_seq_img_round_robin_bars4.gif) | 2026-11-13T21:19:33.827+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0.46875`; `modes=3`; `finite_template_tv=0.53125`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-transport/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-transport_seq_img_dwell_bars4.gif) | 2026-11-13T21:19:44.398+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.32305`; `direction_acc=0.60938`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-transport/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bcap-transport_seq_img_translating_bar_next.gif) | 2026-11-13T21:19:55.643+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=0.4375`; `within_state_modes=2`; `branch_tv=0.5625`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85 | [receipt](../receipts/cuda/online-bcap-transport/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bcap-transport_seq_img_fork_rollout.gif) | 2026-11-13T21:20:07.322+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0.625`; `modes=2`; `finite_template_tv=0.5`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-transport/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-transport_seq_img_variable_dwell_bars4.gif) | 2026-11-13T21:20:17.822+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.59375`; `finite_template_tv=0.40625`; `support_modes=4`; `rare_mass=0.03125`; `rare_mass=0.03125`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85 | [receipt](../receipts/cuda/online-bcap-transport/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bcap-transport_seq_img_rare_mode_bars4.gif) | 2026-11-13T21:20:28.419+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.51562`; `prediction_rmse=0.28648`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-transport/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bcap-transport_seq_img_delayed_cue.gif) | 2026-11-13T21:20:40.093+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0.6875`; `within_state_modes=2`; `branch_tv=0.3125`; `trajectory_consistency=0.5`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95 | [receipt](../receipts/cuda/online-bcap-transport/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bcap-transport_seq_img_persistent_fork.gif) | 2026-11-13T21:20:56.780+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.10948`; `xy_rmse=0.38788`; `step_accuracy=0.25781`; `orbit_completion=0`; `waypoint_modes=8`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-transport/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bcap-transport_seq_img_circle_orbit.gif) | 2026-11-13T21:21:15.299+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.11614`; `s2c_rmse=0.10802`; `c2s_rmse=0.12426`; `xy_rmse=0.46055`; `step_accuracy=0.11328`; `cycle_completion=0`; `waypoint_modes=7`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-transport/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bcap-transport_seq_img_square_circle_cycle.gif) | 2026-11-13T21:21:48.670+00:00 |

<a id="online-bcap-transport-stretch"></a>
### Stretch evidence

<a id="online-bcap-transport-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bcap-transport/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bcap-transport_seq_minimal_dense4.gif) | 2026-11-13T21:21:52.513+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0.375`; `modes=4`; `mass_tv=0.625`; failed terminal bounds: mass_tv <= 0.1; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bcap-transport/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bcap-transport_seq_minimal_tokens4.gif) | 2026-11-13T21:21:56.518+00:00 |

<a id="online-bcap-transport-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.35165`; `seen_max_abs_error=0.83801`; `heldout_response_rmse=0.23873`; `heldout_max_abs_error=0.53801`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bcap-transport/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bcap-transport_seq_minimal_kernel2x2.gif) | 2026-11-13T21:22:02.773+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.125`; `seen_min_bit_confidence=6.1013e-16`; `heldout_exact_accuracy=0.125`; `heldout_min_bit_confidence=6.1013e-16`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bcap-transport/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bcap-transport_seq_minimal_addition2bit.gif) | 2026-11-13T21:22:09.259+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.021302`; `relative_mse=1.6147`; `moving_direction_accuracy=0.46875`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-transport/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bcap-transport_seq_robotics_fetch_reach.gif) | 2026-11-13T21:22:23.054+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012875`; `relative_mse=1.1045`; `moving_direction_accuracy=0.54651`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-transport/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bcap-transport_seq_robotics_fetch_push.gif) | 2026-11-13T21:22:37.107+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012399`; `relative_mse=0.93564`; `moving_direction_accuracy=0.65686`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-transport/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bcap-transport_seq_robotics_fetch_slide.gif) | 2026-11-13T21:22:51.334+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0099639`; `relative_mse=0.63252`; `moving_direction_accuracy=0.64912`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-bcap-transport/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bcap-transport_seq_robotics_fetch_pick_and_place.gif) | 2026-11-13T21:23:05.627+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0092675`; `relative_mse=1.2704`; `moving_direction_accuracy=0.48832`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-transport/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bcap-transport_seq_robotics_hand_reach.gif) | 2026-11-13T21:23:19.689+00:00 |

<a id="online-bcap-transport-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.66626`; `interpolation_speed_error=0.85267`; `interpolation_raw_rmse=0.0737`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.47917`; `extrapolation_centroid_rmse=0.76143`; `extrapolation_speed_error=0.8437`; `extrapolation_raw_rmse=0.070669`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.47917`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bcap-transport/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bcap-transport_seq_minimal_velocity_extrapolation.gif) | 2026-11-13T21:23:33.252+00:00 |

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

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `2cd4dc852fdf263ec9c37a9f691768fe8dd93cdfc5ce1abb7c877dcee9a03f77`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-bcap-transport-causal-required"></a>
### Required evidence

<a id="online-bcap-transport-causal-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.625`; `modes=4`; `finite_template_tv=0.375`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9 | [receipt](../receipts/cuda/online-bcap-transport-causal/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bcap-transport-causal_seq_img_round_robin_bars4.gif) | 2026-11-13T21:23:45.151+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0.46875`; `modes=3`; `finite_template_tv=0.53125`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-transport-causal/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-transport-causal_seq_img_dwell_bars4.gif) | 2026-11-13T21:23:55.799+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.29835`; `direction_acc=0.64062`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-transport-causal/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bcap-transport-causal_seq_img_translating_bar_next.gif) | 2026-11-13T21:24:07.505+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=0.46875`; `within_state_modes=2`; `branch_tv=0.53125`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85 | [receipt](../receipts/cuda/online-bcap-transport-causal/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bcap-transport-causal_seq_img_fork_rollout.gif) | 2026-11-13T21:24:19.294+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0.625`; `modes=2`; `finite_template_tv=0.5`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-transport-causal/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-transport-causal_seq_img_variable_dwell_bars4.gif) | 2026-11-13T21:24:29.892+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.59375`; `finite_template_tv=0.40625`; `support_modes=4`; `rare_mass=0.03125`; `rare_mass=0.03125`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85 | [receipt](../receipts/cuda/online-bcap-transport-causal/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bcap-transport-causal_seq_img_rare_mode_bars4.gif) | 2026-11-13T21:24:40.674+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.76562`; `prediction_rmse=0.26538`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-transport-causal/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bcap-transport-causal_seq_img_delayed_cue.gif) | 2026-11-13T21:24:52.425+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0.6875`; `within_state_modes=1`; `branch_tv=0.3125`; `trajectory_consistency=0.53125`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap-transport-causal/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bcap-transport-causal_seq_img_persistent_fork.gif) | 2026-11-13T21:25:10.830+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.12236`; `xy_rmse=0.48696`; `step_accuracy=0.35938`; `orbit_completion=0.0625`; `waypoint_modes=8`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-transport-causal/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bcap-transport-causal_seq_img_circle_orbit.gif) | 2026-11-13T21:25:31.007+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.12325`; `s2c_rmse=0.11433`; `c2s_rmse=0.13216`; `xy_rmse=0.49374`; `step_accuracy=0.22656`; `cycle_completion=0`; `waypoint_modes=6`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-transport-causal/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bcap-transport-causal_seq_img_square_circle_cycle.gif) | 2026-11-13T21:26:07.643+00:00 |

<a id="online-bcap-transport-causal-stretch"></a>
### Stretch evidence

<a id="online-bcap-transport-causal-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bcap-transport-causal/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bcap-transport-causal_seq_minimal_dense4.gif) | 2026-11-13T21:26:11.671+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0.375`; `modes=4`; `mass_tv=0.625`; failed terminal bounds: mass_tv <= 0.1; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bcap-transport-causal/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bcap-transport-causal_seq_minimal_tokens4.gif) | 2026-11-13T21:26:15.563+00:00 |

<a id="online-bcap-transport-causal-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.40151`; `seen_max_abs_error=0.97615`; `heldout_response_rmse=0.31227`; `heldout_max_abs_error=0.67819`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bcap-transport-causal/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bcap-transport-causal_seq_minimal_kernel2x2.gif) | 2026-11-13T21:26:21.924+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.14583`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0.1875`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bcap-transport-causal/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bcap-transport-causal_seq_minimal_addition2bit.gif) | 2026-11-13T21:26:28.416+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.008503`; `relative_mse=0.25727`; `moving_direction_accuracy=0.90104`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-bcap-transport-causal/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bcap-transport-causal_seq_robotics_fetch_reach.gif) | 2026-11-13T21:26:42.823+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011089`; `relative_mse=0.81938`; `moving_direction_accuracy=0.67442`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-transport-causal/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bcap-transport-causal_seq_robotics_fetch_push.gif) | 2026-11-13T21:26:57.478+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010062`; `relative_mse=0.61623`; `moving_direction_accuracy=0.72059`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-transport-causal/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bcap-transport-causal_seq_robotics_fetch_slide.gif) | 2026-11-13T21:27:11.992+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0043865`; `relative_mse=0.12259`; `moving_direction_accuracy=0.69298`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75 | [receipt](../receipts/cuda/online-bcap-transport-causal/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bcap-transport-causal_seq_robotics_fetch_pick_and_place.gif) | 2026-11-13T21:27:26.721+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0084701`; `relative_mse=1.0612`; `moving_direction_accuracy=0.5035`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-transport-causal/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bcap-transport-causal_seq_robotics_hand_reach.gif) | 2026-11-13T21:27:41.248+00:00 |

<a id="online-bcap-transport-causal-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.79884`; `interpolation_speed_error=0.64256`; `interpolation_raw_rmse=0.073712`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.47917`; `extrapolation_centroid_rmse=0.94375`; `extrapolation_speed_error=0.76789`; `extrapolation_raw_rmse=0.070706`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.41667`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bcap-transport-causal/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bcap-transport-causal_seq_minimal_velocity_extrapolation.gif) | 2026-11-13T21:27:55.855+00:00 |

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

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `ee49d9d54f1b5a70b3842a9314e2439f50a451b2008005e0d93579d7d357cc5a`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-bcap-transport-fit-required"></a>
### Required evidence

<a id="online-bcap-transport-fit-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.625`; `modes=3`; `finite_template_tv=0.46875`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-transport-fit/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-bcap-transport-fit_seq_img_round_robin_bars4.gif) | 2026-11-13T21:28:07.478+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0.15625`; `modes=2`; `finite_template_tv=0.84375`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-transport-fit/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-transport-fit_seq_img_dwell_bars4.gif) | 2026-11-13T21:28:17.668+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.25474`; `direction_acc=0.67188`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-transport-fit/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-bcap-transport-fit_seq_img_translating_bar_next.gif) | 2026-11-13T21:28:29.078+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=0.59375`; `within_state_modes=2`; `branch_tv=0.40625`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap-transport-fit/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-bcap-transport-fit_seq_img_fork_rollout.gif) | 2026-11-13T21:28:40.102+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0.5625`; `modes=4`; `finite_template_tv=0.4375`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-bcap-transport-fit/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-bcap-transport-fit_seq_img_variable_dwell_bars4.gif) | 2026-11-13T21:28:50.405+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.3125`; `finite_template_tv=0.6875`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-bcap-transport-fit/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-bcap-transport-fit_seq_img_rare_mode_bars4.gif) | 2026-11-13T21:29:00.588+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.078125`; `prediction_rmse=0.34803`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-bcap-transport-fit/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-bcap-transport-fit_seq_img_delayed_cue.gif) | 2026-11-13T21:29:11.845+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-bcap-transport-fit/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-bcap-transport-fit_seq_img_persistent_fork.gif) | 2026-11-13T21:29:29.559+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.68253`; `xy_rmse=0.9571`; `step_accuracy=0`; `orbit_completion=0`; `waypoint_modes=0`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-transport-fit/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-bcap-transport-fit_seq_img_circle_orbit.gif) | 2026-11-13T21:29:49.257+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.31403`; `s2c_rmse=0.29772`; `c2s_rmse=0.33034`; `xy_rmse=1.1588`; `step_accuracy=0`; `cycle_completion=0`; `waypoint_modes=0`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-bcap-transport-fit/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-bcap-transport-fit_seq_img_square_circle_cycle.gif) | 2026-11-13T21:30:24.806+00:00 |

<a id="online-bcap-transport-fit-stretch"></a>
### Stretch evidence

<a id="online-bcap-transport-fit-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-bcap-transport-fit/seq_minimal_dense4.json) · [goal.gif](../gifs/online-bcap-transport-fit_seq_minimal_dense4.gif) | 2026-11-13T21:30:28.850+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0.5`; `modes=4`; `mass_tv=0.5`; failed terminal bounds: mass_tv <= 0.1; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-bcap-transport-fit/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-bcap-transport-fit_seq_minimal_tokens4.gif) | 2026-11-13T21:30:32.736+00:00 |

<a id="online-bcap-transport-fit-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.43706`; `seen_max_abs_error=1`; `heldout_response_rmse=0.40833`; `heldout_max_abs_error=0.7`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-bcap-transport-fit/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-bcap-transport-fit_seq_minimal_kernel2x2.gif) | 2026-11-13T21:30:39.354+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.14583`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0.0625`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-bcap-transport-fit/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-bcap-transport-fit_seq_minimal_addition2bit.gif) | 2026-11-13T21:30:45.570+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.03311`; `relative_mse=3.9009`; `moving_direction_accuracy=0.61458`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-transport-fit/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-bcap-transport-fit_seq_robotics_fetch_reach.gif) | 2026-11-13T21:30:59.820+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=380.65`; `relative_mse=9.6541e+08`; `moving_direction_accuracy=0.33721`; `finite_fraction=0.75`; failed terminal bounds: finite_fraction == 1.0; moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-transport-fit/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-bcap-transport-fit_seq_robotics_fetch_push.gif) | 2026-11-13T21:31:13.942+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0091139`; `relative_mse=0.50556`; `moving_direction_accuracy=0.73529`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-transport-fit/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-bcap-transport-fit_seq_robotics_fetch_slide.gif) | 2026-11-13T21:31:28.284+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.027733`; `relative_mse=4.9`; `moving_direction_accuracy=0.68421`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-transport-fit/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-bcap-transport-fit_seq_robotics_fetch_pick_and_place.gif) | 2026-11-13T21:31:42.490+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=1.3551e+15`; `relative_mse=2.7163e+34`; `moving_direction_accuracy=0`; `finite_fraction=0`; failed terminal bounds: finite_fraction == 1.0; moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-bcap-transport-fit/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-bcap-transport-fit_seq_robotics_hand_reach.gif) | 2026-11-13T21:31:56.699+00:00 |

<a id="online-bcap-transport-fit-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.71881`; `interpolation_speed_error=0.5701`; `interpolation_raw_rmse=0.073723`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.45833`; `extrapolation_centroid_rmse=0.94037`; `extrapolation_speed_error=0.76463`; `extrapolation_raw_rmse=0.070687`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.45833`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-bcap-transport-fit/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-bcap-transport-fit_seq_minimal_velocity_extrapolation.gif) | 2026-11-13T21:32:10.822+00:00 |

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

<a id="refresh"></a>
## Refresh

```sh
particlestream compile --device cuda
```

This page is generated alongside the leaderboard from committed source-bound receipts and declarations; editing a generated page cannot change a verdict or earn qualification.

<a id="references"></a>
## References

- [On the regularization of Wasserstein GANs](https://arxiv.org/abs/1709.08894). Related one-sided squared hinge gradient penalty; BCAP applies the cap directly at real and fake samples.
- [The relativistic discriminator: a key element missing from standard GAN](https://arxiv.org/abs/1807.00734). Paired relativistic logistic loss used in preset `bcap` and `bcap_adam` baselines.
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661). Non-saturating logistic adversarial loss used in tuned `batch_size=1` online BCAP recipes.
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). Base optimizer in `bcap_adam` recipes.
