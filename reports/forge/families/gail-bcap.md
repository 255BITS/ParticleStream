<!-- Generated Forge family report -->

# GAIL-BCAP

[← Technique leaderboard](../technique-inventory.md)

<a id="technique-overview"></a>
## Technique overview

GAIL-BCAP translates stabilized generative adversarial imitation control into ParticleStream's strict `batch_size=1` online world-modeling contract. It pairs a Hinge margin critic and fixed BCAP input-gradient cap with a prioritized survival-plus-style generator objective: unbounded linear Hinge generator slopes are canceled in favor of pixelwise BCE reconstruction minus a bounded sigmoid critic style bonus modulated by a causal discounted return advantage baseline. Serving draws entropy-regularized categorical quantiles over active observed feature prototypes, optionally blended beyond discrete context tolerance with a detached Sherman-Morrison recursive associative fast-weight operator.

[Implementation](../../../particlestream/gail_prior.py) · [Mathematical formulation](gail-bcap.md#mathematical-formulation) · [Simplified pseudocode](gail-bcap.md#simplified-pseudocode) · [Training details](gail-bcap.md#training-details) · [Protocol](../../../EXPERIMENTATION.md) · [Terminology](../../../RESEARCH_GLOSSARY.md)

<a id="mathematical-formulation"></a>
## Mathematical formulation

$G$ is the task-owned residual generator, $D$ the raw-score convolutional critic, and $P$ the frozen $K$-row `MoGParticlePrior` ($K=32$, $d_z=8$, intrinsic scale $\sigma=0.025$). At chronological arrival $t$, $x_t\in[0,1]^{1\times 8\times 8}$ is the single target frame, $c_{t-1}\in\mathbb R^{d_c}$ ($d_c=16$) is the detached causal-state context (`write_lr` $\eta_w=0.5$), and $\phi(x_t)=s_f W_{\mathrm{enc}}(\bar x_t-c_0\mathbf 1_{64})$ is the deterministic orthogonal latent feature code ($s_f=2$, $c_0=0.5$). $\lambda_{\mathrm{rec}}=10$ is the survival reconstruction weight, $w_{\mathrm{style}}=0.1$ the bounded style bonus weight, $\sigma_{\mathrm{sig}}(u)=(1+e^{-u})^{-1}$ the logistic sigmoid, $\gamma_R=0.99$ the causal return discount, $R_t$ and $(b_t,v_t)$ the post-commit discounted return trace and running baseline/variance moments, and $\alpha_A=0.25$ the advantage modulation scale. $\eta_H=0.01$ is the active-prototype policy entropy bonus over observed bucket counts $N_{b^*,m}$, and $(M_t\in\mathbb R^{d_z\times(d_c+1)}, P_t\in\mathbb R^{(d_c+1)\times(d_c+1)})$ is the detached Sherman-Morrison recursive ridge fast-weight operator ($\lambda_{\mathrm{ridge}}=0.05$, $\tau_c=0.25$, $s_{\mathrm{blend}}=0.25$, $(a)_+=\max(a,0)$).

**Hinge-BCAP Critic and Prioritized Survival + Bounded Style Bonus Generator**

$$
\begin{aligned}\mathcal L_D&=\big(1-D(x_t,c_{t-1})\big)_++\big(1+D(\hat x_D,c_{t-1})\big)_++\frac{\lambda}{2}\Big[(\lVert g_r\rVert_2-\kappa)_+^2+(\lVert g_f\rVert_2-\kappa)_+^2\Big],\\g_{t-1}&=1+\alpha_A\tanh\!\left(\frac{R_{t-1}-b_{t-1}}{\sqrt{v_{t-1}+10^{-6}}}\right),\\\mathcal L_G&=\lambda_{\mathrm{rec}}\Big[\operatorname{BCE}(\hat x_G,x_t)-w_{\mathrm{style}}\,g_{t-1}\,\sigma_{\mathrm{sig}}\big(D(\hat x_G,c_{t-1})\big)\Big].\end{aligned}
$$

Canceling the default unbounded Hinge generator term $-D(\hat x_G,c_{t-1})$ prevents single-sample critic spikes from destabilizing the shared decoder. After successful updates, the scalar imitation reward $r_t=-\operatorname{BCE}(\hat x_G,x_t)+w_{\mathrm{style}}\sigma_{\mathrm{sig}}(D(\hat x_G,c_{t-1}))$ updates $R_t=\gamma_R R_{t-1}+(1-\gamma_R)r_t$ and $(b_t,v_t)$ strictly for step $t+1$.

**Entropy-Regularized Active-Prototype Quantile Policy**

$$
\begin{aligned}\mathcal A_{b^*}&=\big\{m\in\{0,\dots,M_{\mathrm{occ}}-1\}:N_{b^*,m}>0\big\},\\\tilde p_{b^*,m}&=(1-\eta_H)\frac{N_{b^*,m}}{\sum_{j\in\mathcal A_{b^*}}N_{b^*,j}}+\eta_H\frac{\mathbb I(m\in\mathcal A_{b^*})}{|\mathcal A_{b^*}|},\\m^*(i)&=\min\!\left(m_{\max}(\mathcal A_{b^*}),\;\inf\left\{m\ge 0:\sum_{j=0}^m \tilde p_{b^*,j}>\frac{i+0.5}{K}\right\}\right).\end{aligned}
$$

Mixing $\eta_H=0.01$ uniform probability strictly over active prototypes $\mathcal A_{b^*}$ at the nearest context bucket $b^*$ prevents early finite-sample lock-out on rare branches (`5:5:5:1`) without assigning probability mass to unobserved prototypes.

**Detached Sherman-Morrison Associative Fast-Weight Extrapolation**

$$
\begin{aligned}\tilde c&=\begin{bmatrix}\operatorname{clip}(c_{t-1},-c_{\max},c_{\max})/s_f\\ 1\end{bmatrix},\qquad P_t=P_{t-1}-\frac{P_{t-1}\tilde c\,\tilde c^\top P_{t-1}}{1+\tilde c^\top P_{t-1}\tilde c},\\M_t&=M_{t-1}+\big(\phi(x_t)-M_{t-1}\tilde c\big)\frac{\tilde c^\top P_{t-1}}{1+\tilde c^\top P_{t-1}\tilde c},\\\alpha(c_{t-1})&=\alpha_{\max}\frac{(d_{\min}-\tau_c)_+}{(d_{\min}-\tau_c)_++s_{\mathrm{blend}}},\qquad z_i^{\mathrm{cond}}=(1-\alpha)\hat\mu_{m^*(i)}+\alpha\,\Pi_{4 s_f}(M_{t-1}\tilde c)+\epsilon_i.\end{aligned}
$$

Within seen bucket tolerance ($d_{\min}\le\tau_c$), $\alpha(c_{t-1})=0$ preserves exact discrete multimodal quantiles $\hat\mu_{m^*(i)}+\epsilon_i$. For unseen continuous or compositional contexts ($d_{\min}>\tau_c$), $z_i^{\mathrm{cond}}$ smoothly interpolates toward the radially bounded linear associative prediction $\Pi_{4 s_f}(M_{t-1}\tilde c)$ initialized from $P_0=\lambda_{\mathrm{ridge}}^{-1}I$.

<a id="simplified-pseudocode"></a>
## Simplified pseudocode

```text
Initialize fixed orthogonal encoder W_enc, frozen MoG prior P, empty prototype/context tables, and P_0 = (1 / assoc_ridge) I.
For each chronological single-arrival step t:
  Before revealing x_t, serve y_hat = G(z_cond(i_serve, c_{t-1}), c_{t-1}) under an isolated eval RNG.
  Reveal single target frame x_t (batch_size = 1) and encode phi(x_t) = s_f W_enc (x_bar_t - c_0).
  Draw uniform row i_D; form z_D via entropy-regularized active quantiles blended with M_{t-1} c_tilde when d_min > tau_c.
  Evaluate Hinge critic loss max(0, 1 - D(x_t)) + max(0, 1 + D(G(z_D))) + fixed BCAP cap penalty; step D.
  With D fixed, draw uniform row i_G and form posterior code z_G = phi(x_t) + eps_{i_G}.
  Cancel default -D(x_hat_G) slope and backpropagate 10 * (BCE(x_hat_G, x_t) - 0.1 * g_{t-1} * sigmoid(D(x_hat_G))); step G.
  After both updates succeed, commit phi(x_t) and c_{t-1} to prototype/context counts, update return baseline (R_t, b_t, v_t), and update Sherman-Morrison fast weights (M_t, P_t).
  Advance detached causal state c_t = CausalStateMemory.write(x_t, write_lr = 0.5) with replay_policy = 'none'.
```

<a id="training-details"></a>
## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Hinge margin adversarial critic loss (`loss="hinge"`), paired on $G$ with canceled linear critic slope (`cancel_linear_gan=True`), prioritized pixelwise BCE survival anchor ($\lambda_{\mathrm{rec}}=10$), and causal advantage-weighted sigmoid style bonus ($w_{\mathrm{style}}=0.1$, $\gamma_R=0.99$, $\alpha_A=0.25$). |
| Optimizer | Native Adam via `bcap_adam` ($\beta_1=0, \beta_2=0.999$ in `online-gail-bcap-hinge` and `online-gail-bcap-bce`; $\beta_1=0.5, \beta_2=0.999$ in `online-gail-bcap-momentum`). Base MoG storage table is frozen (`prior_update="frozen"`). |
| Learning rates and annealing | Base LR $0.005$ (`critic_lr_mult=1.0`, `prior_lr_mult=1.0`), constant through $50\%$ of task budget (`lr_anneal_start=0.5`) then cosine-annealed to `lr_floor=0.05` (`network_lr_floor=0.05`). |
| Parameter-gradient clipping | None (`critic_grad_guard=False`). Unbounded generator critic gradients are eliminated structurally via the bounded sigmoid style bonus and BCAP input-gradient cap. |
| Critic penalties and anchors | Fixed one-sided squared $L_2$ BCAP cap on real and conditional-prior fake inputs ($\lambda=1.0$, $\kappa=1.0$) every critic step; no EMA critic anchor. |
| Damping and update guards | No A2 sparse-row damping (`latent_damping_max_rate=0.0`), zero generator-center probes (`probe_law="no_generator_probes_v1"`), and radial norm bound $\lVert\hat\phi_{\mathrm{assoc}}\rVert_2\le 4 s_f$ on associative latent extrapolation. |
| Online memory, prior and routing | Detached `CausalStateMemory` ($d_c=16$, `write_lr=0.5`, `association_weight=0.0`, `age_horizon=8`) paired with `GAILConditionalReconstruction` (`router="gail_conditional"`, $\le 32$ prototypes, $\le 64$ context buckets, `policy_entropy_bonus=0.01`, optional Sherman-Morrison `assoc_ridge=0.05`; `replay_policy="none"`, `rehearsal_batch=0`). |
| Training and sampling noise | Zero scheduled critic-input or generator-output additive noise (`0.0`). Intrinsic task MoG perturbation $\sigma=0.025$ is preserved on every conditional and posterior latent draw. |
| Parameter averaging and serving | Live unaveraged generator weights (`ema_decay=0.0`, `serve_average=False`). All prototype, histogram, return-baseline, and Sherman-Morrison commits occur strictly after each arrival's updates. |

<a id="configuration-differences"></a>
## Configuration differences

- GAIL-BCAP forms an explicit three-step single-factor ablation ladder over `online-feature-prior-bce`: `online-gail-bcap-hinge` isolates Hinge critic + prioritized survival/sigmoid style bonus + causal return advantage + active categorical entropy; `online-gail-bcap-bce` enables Sherman-Morrison associative fast-weight extrapolation (`assoc_ridge=0.05`); and `online-gail-bcap-momentum` adds `betas=[0.5, 0.999]` Adam first-moment smoothing.
- Keeping `CausalStateMemory.association_weight=0.0` avoids train-versus-evaluation context shift when evaluators reset stream memory on short prefixes, while placing persistent cross-episode fast weights $M_t$ inside `GAILConditionalReconstruction` cuts unseen `seq_minimal_kernel2x2` held-out error by ~3x (`0.261 -> 0.087`) and `seq_img_translating_bar_next` next-frame RMSE by ~25% (`0.0085 -> 0.0064`) without degrading discrete multimodal forks.
- Executed strictly with `batch_size=1` (real and fake), `seed=0`, `replay_policy="none"`, `rehearsal_batch=0`, and zero generator probes across all 20 tasks.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [particlestream/gail_prior.py](../../../particlestream/gail_prior.py)
- [particlestream/feature_prior.py](../../../particlestream/feature_prior.py)
- [particlestream/memory.py](../../../particlestream/memory.py)
- [particlestream/tier1.py](../../../particlestream/tier1.py)
- [configs/forge/ideas/online-gail-bcap-hinge.json](../../../configs/forge/ideas/online-gail-bcap-hinge.json)
- [configs/forge/ideas/online-gail-bcap-bce.json](../../../configs/forge/ideas/online-gail-bcap-bce.json)
- [configs/forge/ideas/online-gail-bcap-momentum.json](../../../configs/forge/ideas/online-gail-bcap-momentum.json)

</details>

Each section below describes one exact global recipe. Configuration and parent deltas come from the declarations; results come from Forge's current source-bound receipts. Required and stretch evidence stay separate. A final metric can meet its gate while the sustained verdict fails: every one of the last five checks must pass.

The installed engine is pinned to ParticleGAN/Forge 0.9.7. Older version or replay wording in declaration history describes provenance; the effective configuration below controls execution.

<a id="online-gail-bcap-bce"></a>
## `online-gail-bcap-bce`

[Exact declaration](../../../configs/forge/ideas/online-gail-bcap-bce.json)

Parent: [`online-gail-bcap-hinge`](gail-bcap.md#online-gail-bcap-hinge).

Declared changes:

- Enable detached Sherman-Morrison recursive associative fast-weight operator (assoc_ridge=0.05, assoc_forgetting=1.0, assoc_blend_scale=0.25, assoc_max_blend=1.0) to extrapolate latent feature expectations smoothly beyond discrete context tolerance

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.transport.assoc_max_blend` | `0.0` | `1.0` |
| `memory_policy.transport.assoc_ridge` | `0.0` | `0.05` |

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
      "advantage_scale": 0.25,
      "assoc_blend_scale": 0.25,
      "assoc_forgetting": 1.0,
      "assoc_max_blend": 1.0,
      "assoc_ridge": 0.05,
      "cancel_linear_gan": true,
      "context_capacity": 64,
      "context_clip": 8.0,
      "context_tolerance": 0.25,
      "feature_scale": 2.0,
      "feature_tolerance": 0.15,
      "image_center": 0.5,
      "policy_entropy_bonus": 0.01,
      "reconstruction_loss": "binary_cross_entropy",
      "reconstruction_weight": 10.0,
      "return_discount": 0.99,
      "router": "gail_conditional",
      "style_bonus_weight": 0.1,
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.0
    },
    "write_lr": 0.5
  },
  "recipe_overrides": {
    "betas": [
      0.0,
      0.999
    ],
    "loss": "hinge",
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

Receipt identity: candidate_fingerprint `010caed96830d115255257866a111e9acaa9a94c950a4fa54bfcf0510dc48592`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-gail-bcap-bce-required"></a>
### Required evidence

<a id="online-gail-bcap-bce-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_round_robin_bars4.gif) | 2026-11-12T07:00:41.555+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_dwell_bars4.gif) | 2026-11-12T07:00:49.546+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.006364`; `direction_acc=1` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_translating_bar_next.gif) | 2026-11-12T07:01:00.980+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_fork_rollout.gif) | 2026-11-12T07:01:10.161+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_variable_dwell_bars4.gif) | 2026-11-12T07:01:24.275+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_rare_mode_bars4.gif) | 2026-11-12T07:01:32.244+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.023147` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_delayed_cue.gif) | 2026-11-12T07:01:43.132+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.03125`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_persistent_fork.gif) | 2026-11-12T07:02:03.483+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.011924`; `xy_rmse=0.003893`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_circle_orbit.gif) | 2026-11-12T07:02:25.234+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.012503`; `s2c_rmse=0.011269`; `c2s_rmse=0.013737`; `xy_rmse=0.003717`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_square_circle_cycle.gif) | 2026-11-12T07:03:04.349+00:00 |

<a id="online-gail-bcap-bce-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-bce-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_minimal_dense4.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_minimal_dense4.gif) | 2026-11-12T07:03:08.264+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_minimal_tokens4.gif) | 2026-11-12T07:03:12.118+00:00 |

<a id="online-gail-bcap-bce-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.04189`; `seen_max_abs_error=0.15896`; `heldout_response_rmse=0.086803`; `heldout_max_abs_error=0.18952`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_minimal_kernel2x2.gif) | 2026-11-12T07:03:18.148+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.97076`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=1.1497e-08`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_minimal_addition2bit.gif) | 2026-11-12T07:03:24.347+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0053724`; `relative_mse=0.1027`; `moving_direction_accuracy=0.95833`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_robotics_fetch_reach.gif) | 2026-11-12T07:03:38.428+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011217`; `relative_mse=0.83837`; `moving_direction_accuracy=0.74419`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_robotics_fetch_push.gif) | 2026-11-12T07:03:52.215+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0077145`; `relative_mse=0.36224`; `moving_direction_accuracy=0.80392`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_robotics_fetch_slide.gif) | 2026-11-12T07:04:05.573+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 4/5 | `position_delta_rmse_m=0.0025834`; `relative_mse=0.04252`; `moving_direction_accuracy=0.80702`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T07:04:19.081+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0081327`; `relative_mse=0.97832`; `moving_direction_accuracy=0.58411`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_robotics_hand_reach.gif) | 2026-11-12T07:04:31.949+00:00 |

<a id="online-gail-bcap-bce-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.035795`; `interpolation_speed_error=0.04004`; `interpolation_raw_rmse=0.029099`; `interpolation_point_validity=0.75`; `interpolation_direction_accuracy=0.91667`; `extrapolation_centroid_rmse=0.077977`; `extrapolation_speed_error=0.060972`; `extrapolation_raw_rmse=0.050648`; `extrapolation_point_validity=0.5`; `extrapolation_direction_accuracy=1`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T07:04:46.394+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-gail-bcap-bce --device cuda`.

<a id="online-gail-bcap-hinge"></a>
## `online-gail-bcap-hinge`

[Exact declaration](../../../configs/forge/ideas/online-gail-bcap-hinge.json)

Parent: [`online-feature-prior-bce`](featureprior-bcap.md#online-feature-prior-bce).

Declared changes:

- Switch critic adversarial objective to Hinge loss paired with BCAP input-gradient cap and write_lr=0.5 causal-state fast-weight updates
- Stabilized GAIL-BCAP generator objective: cancel unbounded linear Hinge generator slope and optimize prioritized BCE survival anchor minus bounded sigmoid critic style bonus (weight 0.1) modulated by causal discounted return advantage (gamma=0.99, scale=0.25)
- Entropy-regularized categorical outcome quantiles (bonus=0.01) over active observed prototypes per context; associative fast-weight extrapolation disabled (assoc_ridge=0, assoc_max_blend=0)

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.transport.advantage_scale` | _unset_ | `0.25` |
| `memory_policy.transport.assoc_blend_scale` | _unset_ | `0.25` |
| `memory_policy.transport.assoc_forgetting` | _unset_ | `1.0` |
| `memory_policy.transport.assoc_max_blend` | _unset_ | `0.0` |
| `memory_policy.transport.assoc_ridge` | _unset_ | `0.0` |
| `memory_policy.transport.cancel_linear_gan` | _unset_ | `true` |
| `memory_policy.transport.policy_entropy_bonus` | _unset_ | `0.01` |
| `memory_policy.transport.return_discount` | _unset_ | `0.99` |
| `memory_policy.transport.router` | `"feature_conditional"` | `"gail_conditional"` |
| `memory_policy.transport.style_bonus_weight` | _unset_ | `0.1` |
| `memory_policy.write_lr` | `0.35` | `0.5` |
| `recipe_overrides.loss` | `"non_saturating"` | `"hinge"` |

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
      "advantage_scale": 0.25,
      "assoc_blend_scale": 0.25,
      "assoc_forgetting": 1.0,
      "assoc_max_blend": 0.0,
      "assoc_ridge": 0.0,
      "cancel_linear_gan": true,
      "context_capacity": 64,
      "context_clip": 8.0,
      "context_tolerance": 0.25,
      "feature_scale": 2.0,
      "feature_tolerance": 0.15,
      "image_center": 0.5,
      "policy_entropy_bonus": 0.01,
      "reconstruction_loss": "binary_cross_entropy",
      "reconstruction_weight": 10.0,
      "return_discount": 0.99,
      "router": "gail_conditional",
      "style_bonus_weight": 0.1,
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.0
    },
    "write_lr": 0.5
  },
  "recipe_overrides": {
    "betas": [
      0.0,
      0.999
    ],
    "loss": "hinge",
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

Receipt identity: candidate_fingerprint `1bce0bff0efab1534109b49a04efa41ef80284b4489bb6ab68d3037476cf84bf`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-gail-bcap-hinge-required"></a>
### Required evidence

<a id="online-gail-bcap-hinge-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_round_robin_bars4.gif) | 2026-11-12T07:04:55.065+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_dwell_bars4.gif) | 2026-11-12T07:05:03.034+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.007497`; `direction_acc=1` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_translating_bar_next.gif) | 2026-11-12T07:05:11.940+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_fork_rollout.gif) | 2026-11-12T07:05:21.029+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_variable_dwell_bars4.gif) | 2026-11-12T07:05:28.817+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_rare_mode_bars4.gif) | 2026-11-12T07:05:36.561+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.023308` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_delayed_cue.gif) | 2026-11-12T07:05:45.394+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.03125`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_persistent_fork.gif) | 2026-11-12T07:06:03.102+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.011981`; `xy_rmse=0.003659`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_circle_orbit.gif) | 2026-11-12T07:06:22.794+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.011578`; `s2c_rmse=0.01075`; `c2s_rmse=0.012407`; `xy_rmse=0.00359`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_square_circle_cycle.gif) | 2026-11-12T07:06:59.825+00:00 |

<a id="online-gail-bcap-hinge-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-hinge-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_minimal_dense4.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_minimal_dense4.gif) | 2026-11-12T07:07:03.754+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_minimal_tokens4.gif) | 2026-11-12T07:07:07.552+00:00 |

<a id="online-gail-bcap-hinge-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.041775`; `seen_max_abs_error=0.16378`; `heldout_response_rmse=0.26537`; `heldout_max_abs_error=0.43112`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_minimal_kernel2x2.gif) | 2026-11-12T07:07:13.673+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.96474`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=9.6964e-10`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_minimal_addition2bit.gif) | 2026-11-12T07:07:19.709+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.005476`; `relative_mse=0.1067`; `moving_direction_accuracy=0.95833`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_robotics_fetch_reach.gif) | 2026-11-12T07:07:33.356+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.01111`; `relative_mse=0.82236`; `moving_direction_accuracy=0.69767`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_robotics_fetch_push.gif) | 2026-11-12T07:07:46.722+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0075477`; `relative_mse=0.34674`; `moving_direction_accuracy=0.76471`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_robotics_fetch_slide.gif) | 2026-11-12T07:07:59.678+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 1/5 | `position_delta_rmse_m=0.0026558`; `relative_mse=0.044936`; `moving_direction_accuracy=0.73684`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T07:08:12.054+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0081538`; `relative_mse=0.9834`; `moving_direction_accuracy=0.58879`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_robotics_hand_reach.gif) | 2026-11-12T07:08:24.495+00:00 |

<a id="online-gail-bcap-hinge-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.031857`; `interpolation_speed_error=0.039085`; `interpolation_raw_rmse=0.026463`; `interpolation_point_validity=0.83333`; `interpolation_direction_accuracy=0.91667`; `extrapolation_centroid_rmse=0.079113`; `extrapolation_speed_error=0.066138`; `extrapolation_raw_rmse=0.05589`; `extrapolation_point_validity=0.5`; `extrapolation_direction_accuracy=1`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T07:08:37.916+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-gail-bcap-hinge --device cuda`.

<a id="online-gail-bcap-momentum"></a>
## `online-gail-bcap-momentum`

[Exact declaration](../../../configs/forge/ideas/online-gail-bcap-momentum.json)

Parent: [`online-gail-bcap-bce`](gail-bcap.md#online-gail-bcap-bce).

Declared changes:

- Enable nonzero Adam first-moment smoothing (betas=[0.5, 0.999]) across sequential batch_size=1 arrivals

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `recipe_overrides.betas` | `[0.0, 0.999]` | `[0.5, 0.999]` |

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
      "advantage_scale": 0.25,
      "assoc_blend_scale": 0.25,
      "assoc_forgetting": 1.0,
      "assoc_max_blend": 1.0,
      "assoc_ridge": 0.05,
      "cancel_linear_gan": true,
      "context_capacity": 64,
      "context_clip": 8.0,
      "context_tolerance": 0.25,
      "feature_scale": 2.0,
      "feature_tolerance": 0.15,
      "image_center": 0.5,
      "policy_entropy_bonus": 0.01,
      "reconstruction_loss": "binary_cross_entropy",
      "reconstruction_weight": 10.0,
      "return_discount": 0.99,
      "router": "gail_conditional",
      "style_bonus_weight": 0.1,
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.0
    },
    "write_lr": 0.5
  },
  "recipe_overrides": {
    "betas": [
      0.5,
      0.999
    ],
    "loss": "hinge",
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

Receipt identity: candidate_fingerprint `afe55ca27079ef913fe69606c46a3c962c6a9bdfd4a51da438d2be03a7426e0b`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-gail-bcap-momentum-required"></a>
### Required evidence

<a id="online-gail-bcap-momentum-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_round_robin_bars4.gif) | 2026-11-12T07:08:46.900+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_dwell_bars4.gif) | 2026-11-12T07:08:54.501+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.00797`; `direction_acc=1` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_translating_bar_next.gif) | 2026-11-12T07:09:03.252+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_fork_rollout.gif) | 2026-11-12T07:09:11.932+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_variable_dwell_bars4.gif) | 2026-11-12T07:09:19.860+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_rare_mode_bars4.gif) | 2026-11-12T07:09:27.514+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.023138` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_delayed_cue.gif) | 2026-11-12T07:09:36.345+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.03125`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_persistent_fork.gif) | 2026-11-12T07:09:54.671+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.013045`; `xy_rmse=0.003758`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_circle_orbit.gif) | 2026-11-12T07:10:16.510+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.012296`; `s2c_rmse=0.011309`; `c2s_rmse=0.013282`; `xy_rmse=0.00323`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_square_circle_cycle.gif) | 2026-11-12T07:10:55.411+00:00 |

<a id="online-gail-bcap-momentum-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-momentum-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_minimal_dense4.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_minimal_dense4.gif) | 2026-11-12T07:10:59.594+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_minimal_tokens4.gif) | 2026-11-12T07:11:03.551+00:00 |

<a id="online-gail-bcap-momentum-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.0353`; `seen_max_abs_error=0.1779`; `heldout_response_rmse=0.0899`; `heldout_max_abs_error=0.23582`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_minimal_kernel2x2.gif) | 2026-11-12T07:11:09.663+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.9756`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=8.2579e-08`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_minimal_addition2bit.gif) | 2026-11-12T07:11:15.787+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0054013`; `relative_mse=0.10381`; `moving_direction_accuracy=0.95833`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_robotics_fetch_reach.gif) | 2026-11-12T07:11:29.360+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010969`; `relative_mse=0.80171`; `moving_direction_accuracy=0.67442`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_robotics_fetch_push.gif) | 2026-11-12T07:11:42.956+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0074193`; `relative_mse=0.33504`; `moving_direction_accuracy=0.7451`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_robotics_fetch_slide.gif) | 2026-11-12T07:11:56.565+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 2/5 | `position_delta_rmse_m=0.0028069`; `relative_mse=0.050196`; `moving_direction_accuracy=0.80702`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T07:12:10.451+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0080597`; `relative_mse=0.96082`; `moving_direction_accuracy=0.6028`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_robotics_hand_reach.gif) | 2026-11-12T07:12:24.723+00:00 |

<a id="online-gail-bcap-momentum-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.031232`; `interpolation_speed_error=0.037349`; `interpolation_raw_rmse=0.025589`; `interpolation_point_validity=0.66667`; `interpolation_direction_accuracy=1`; `extrapolation_centroid_rmse=0.073497`; `extrapolation_speed_error=0.03789`; `extrapolation_raw_rmse=0.053371`; `extrapolation_point_validity=0.58333`; `extrapolation_direction_accuracy=1`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T07:12:40.249+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-gail-bcap-momentum --device cuda`.

<a id="refresh"></a>
## Refresh

```sh
particlestream compile --device cuda
```

This page is generated alongside the leaderboard from committed source-bound receipts and declarations; editing a generated page cannot change a verdict or earn qualification.

<a id="references"></a>
## References

- [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476). Imitation-style occupancy matching with causal return advantage and policy entropy regularization.
- [Geometric GAN](https://arxiv.org/abs/1705.02894). Hinge margin critic objective paired with the BCAP input-gradient cap.
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438). Discounted return baseline and variance-standardized advantage gain for online policy updates.
- [Using Fast Weights to Attend to the Recent Past](https://arxiv.org/abs/1610.06258). Detached associative fast-weight operator extrapolating unseen contexts alongside slow weights.
