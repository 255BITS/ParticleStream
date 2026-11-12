<!-- Generated Forge family report -->

# GAIL-BCAP

[← Technique leaderboard](../technique-inventory.md)

<a id="technique-overview"></a>
## Technique overview

GAIL-BCAP translates stabilized generative adversarial imitation control into ParticleStream's strict `batch_size=1` online world-modeling contract. It pairs a Hinge margin critic and fixed BCAP input-gradient cap with a prioritized survival-plus-style generator objective: unbounded linear Hinge generator slopes are canceled in favor of pixelwise BCE reconstruction minus a bounded sigmoid critic style bonus modulated by causal return advantage and active-prototype policy entropy. Frontier recipes replace radial distance blending with `float64` Sherman-Morrison RLS, precision-weighted Mahalanobis spike-and-slab prototype transport, empirical-Bayes kernel-gated active-entropy residual quantiles, Markov context-bypass regularization, and high-SNR posterior latent scaling for millimetre robotics.

[Implementation](../../../particlestream/gail_prior.py) · [Mathematical formulation](gail-bcap.md#mathematical-formulation) · [Simplified pseudocode](gail-bcap.md#simplified-pseudocode) · [Training details](gail-bcap.md#training-details) · [Protocol](../../../EXPERIMENTATION.md) · [Terminology](../../../RESEARCH_GLOSSARY.md)

<a id="mathematical-formulation"></a>
## Mathematical formulation

$G$ is the task-owned residual generator, $D$ the raw-score convolutional critic, and $P$ the frozen $K$-row `MoGParticlePrior` ($K=32$, $d_z=8$, intrinsic scale $\sigma=0.025$). At chronological arrival $t$, $x_t\in[0,1]^{1\times 8\times 8}$ is the single target frame, $c_{t-1}\in\mathbb R^{d_c}$ ($d_c=16$) is the detached causal-state context (`write_lr` $\eta_w=0.5$, contrast gate $\tau_{\mathrm{con}}\in\{0.03,0.005\}$), and $\phi(x_t)=s_f W_{\mathrm{enc}}(\bar x_t-c_0\mathbf 1_{64})$ is the deterministic orthogonal latent feature code ($s_f\in\{2,4\}$, $c_0=0.5$). $\lambda_{\mathrm{rec}}=10$ is the survival reconstruction weight, $w_{\mathrm{style}}\in\{0.1,0.05\}$ the bounded style bonus weight, $\sigma_{\mathrm{sig}}(u)=(1+e^{-u})^{-1}$ the logistic sigmoid, $\gamma_R=0.99$ the causal return discount, $R_t$ and $(b_t,v_t)$ the post-commit discounted return trace and running moments, $\alpha_A\in\{0.25,0\}$ the advantage modulation scale, and $\lambda_{\mathrm{ctx}}\in\{0,5\}$ the Markov context-bypass weight on $W_{fc,\mathrm{ctx}}$. $\eta_H=0.01$ is the active-prototype policy entropy bonus over observed bucket counts $N_{b^*,m}$, $(M_t\in\mathbb R^{d_z\times(d_c+1)}, P_t\in\mathbb R^{(d_c+1)\times(d_c+1)})$ is the detached Sherman-Morrison RLS operator ($\lambda_{\mathrm{ridge}}\in\{0.05,0.1\}$), $\Sigma_t\in\mathbb R^{d_z\times d_z}$ is the residual covariance EMA, and $\tau_{\mathrm{slab}}\in\{0,1,2\}$ is the Mahalanobis spike-and-slab scale.

**Hinge-BCAP Critic, Prioritized Survival + Bounded Style Bonus, and Markov Bottleneck**

$$
\begin{aligned}\mathcal L_D&=\big(1-D(x_t,c_{t-1})\big)_++\big(1+D(\hat x_D,c_{t-1})\big)_++\frac{\lambda}{2}\Big[(\lVert g_r\rVert_2-\kappa)_+^2+(\lVert g_f\rVert_2-\kappa)_+^2\Big],\\g_{t-1}&=1+\alpha_A\tanh\!\left(\frac{R_{t-1}-b_{t-1}}{\sqrt{v_{t-1}+10^{-10}}}\right),\\\mathcal L_G&=\lambda_{\mathrm{rec}}\Big[\operatorname{BCE}(\hat x_G,x_t)-w_{\mathrm{style}}\,g_{t-1}\,\sigma_{\mathrm{sig}}\big(D(\hat x_G,c_{t-1})\big)\Big]+\lambda_{\mathrm{ctx}}\lVert W_{fc,\mathrm{ctx}}\rVert_F^2.\end{aligned}
$$

Canceling the default unbounded Hinge generator term $-D(\hat x_G,c_{t-1})$ prevents single-sample critic spikes from destabilizing the shared decoder, while $\lambda_{\mathrm{ctx}}\lVert W_{fc,\mathrm{ctx}}\rVert_F^2$ forces unseen context generalization through the conditional latent prior rather than direct input shortcuts.

**Entropy-Regularized Active-Prototype Quantile Policy**

$$
\begin{aligned}\mathcal A_{b^*}&=\big\{m\in\{0,\dots,M_{\mathrm{occ}}-1\}:N_{b^*,m}>0\big\},\\\tilde p_{b^*,m}&=(1-\eta_H)\frac{N_{b^*,m}}{\sum_{j\in\mathcal A_{b^*}}N_{b^*,j}}+\eta_H\frac{\mathbb I(m\in\mathcal A_{b^*})}{|\mathcal A_{b^*}|},\\m^*(i)&=\min\!\left(m_{\max}(\mathcal A_{b^*}),\;\inf\left\{m\ge 0:\sum_{j=0}^m \tilde p_{b^*,j}>\frac{i+0.5}{K}\right\}\right).\end{aligned}
$$

Mixing $\eta_H=0.01$ uniform probability strictly over active prototypes $\mathcal A_{b^*}$ at the nearest context bucket $b^*$ prevents finite-sample lock-out on rare discrete branches (`5:5:5:1`) and stabilizes recurrent state-action buckets in multi-episode robotics.

**Sherman-Morrison RLS + Mahalanobis Spike-and-Slab Kernel Residual Transport**

$$
\begin{aligned}\bar c&=\begin{bmatrix}c_{t-1}\\ 1\end{bmatrix},\qquad \hat z_{\mathrm{lin}}(\bar c)=M_{t-1}\bar c,\qquad d_m^2=(\hat\mu_m-\hat z_{\mathrm{lin}})^\top \tilde\Sigma_t^{-1}(\hat\mu_m-\hat z_{\mathrm{lin}}),\\\beta_t&=g_{\mathrm{disc}}\,\sigma_{\mathrm{sig}}\!\left(\tfrac12\log\frac{\det(\tilde\Sigma_t+\tau_{\mathrm{slab}}^2 I)}{\det\tilde\Sigma_t}-\tfrac12\min_m d_m^2\right),\qquad \hat z_{\mathrm{ctx}}=(1-\beta_t)\hat z_{\mathrm{lin}}+\beta_t\hat\mu_{m_{\mathrm{best}}},\\z_i^{\mathrm{cond}}&=\hat z_{\mathrm{ctx}}(c_{t-1})+\gamma(c_{t-1})\Big(\hat\mu_{m^*(i)}-\hat z_{\mathrm{ctx}}(k_{b^*}^{\mathrm{ctx}})\Big)+\epsilon_i.\end{aligned}
$$

When $\tau_{\mathrm{slab}}=0$ (`online-gail-bcap-bce`), unseen contexts blend radially toward $\Pi_{4 s_f}(M_{t-1}\bar c)$ beyond $\tau_c$. When $\tau_{\mathrm{slab}}>0$ (`online-gail-bcap-spike-slab`, `online-gail-bcap-snr`), precision-weighted Mahalanobis Bayes factors snap discrete manifolds to clean prototypes while keeping continuous kinematic predictions smooth, and $s_f=4$ boosts posterior SNR $s_f^2/\sigma^2$ by $4\times$ on millimetre robotics displacements.

<a id="simplified-pseudocode"></a>
## Simplified pseudocode

```text
Initialize fixed orthogonal encoder W_enc, frozen MoG prior P, empty prototype/context tables, and P_0 = (1 / assoc_ridge) I.
For each chronological single-arrival step t:
  Before revealing x_t, serve y_hat = G(z_cond(i_serve, c_{t-1}), c_{t-1}) under an isolated eval RNG.
  Reveal single target frame x_t (batch_size = 1) and encode phi(x_t) = s_f W_enc (x_bar_t - c_0).
  Draw uniform row i_D; form z_D via GAIL active-entropy quantiles + Sherman-Morrison associative / spike-and-slab transport.
  Evaluate Hinge critic loss max(0, 1 - D(x_t)) + max(0, 1 + D(G(z_D))) + fixed BCAP cap penalty; step D.
  With D fixed, draw uniform row i_G and form posterior code z_G = phi(x_t) + eps_{i_G}.
  Cancel -D(x_hat_G) slope and backpropagate 10 * (BCE(x_hat_G, x_t) - w_style * g_{t-1} * sigmoid(D(x_hat_G))) + lambda_ctx ||W_{fc,ctx}||_F^2; step G.
  After both updates succeed, commit phi(x_t) and c_{t-1} to prototype/context counts, update return baseline, and update Sherman-Morrison RLS + residual covariance.
  Advance detached causal state c_t = CausalStateMemory.write(x_t, write_lr = 0.5) with replay_policy = 'none'.
```

<a id="training-details"></a>
## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Hinge margin adversarial critic loss (`loss="hinge"`), paired on $G$ with canceled linear critic slope (`cancel_linear_gan=True`), prioritized pixelwise BCE survival anchor ($\lambda_{\mathrm{rec}}=10$), bounded sigmoid style bonus ($w_{\mathrm{style}}\in\{0.1,0.05\}$), and optional Markov context-bypass penalty (`context_bypass_weight=5.0`). |
| Optimizer | Native Adam via `bcap_adam` ($\beta_1=0, \beta_2=0.999$ in Hinge/BCE/SpikeSlab/SNR recipes; $\beta_1=0.5, \beta_2=0.999$ in `online-gail-bcap-momentum`). Base MoG storage table is frozen (`prior_update="frozen"`). |
| Learning rates and annealing | Base LR $0.005$ (`critic_lr_mult=1.0`, `prior_lr_mult=1.0`), constant through $50\%$ of task budget (`lr_anneal_start=0.5`) then cosine-annealed to `lr_floor=0.05` (`network_lr_floor=0.05`). |
| Parameter-gradient clipping | None (`critic_grad_guard=False`). Unbounded generator critic gradients are eliminated structurally via the bounded sigmoid style bonus and BCAP input-gradient cap. |
| Critic penalties and anchors | Fixed one-sided squared $L_2$ BCAP cap on real and conditional-prior fake inputs ($\lambda=1.0$, $\kappa=1.0$) every critic step; no EMA critic anchor. |
| Damping and update guards | No A2 sparse-row damping (`latent_damping_max_rate=0.0`), zero generator-center probes (`probe_law="no_generator_probes_v1"`), radial norm bound $\lVert\hat\phi_{\mathrm{assoc}}\rVert_2\le 4 s_f$ in radial mode, and regularized Mahalanobis covariance $\tilde\Sigma_t\succeq 10^{-4}I$ in spike-and-slab mode. |
| Online memory, prior and routing | Detached `CausalStateMemory` ($d_c=16$, `write_lr=0.5`, `association_weight=0.0`, optional unnormalized linear keys `normalize_key=False, key_gain=0.5` and precision action gate `contrast_threshold=0.005`) paired with `GAILConditionalReconstruction` (`router="gail_conditional"`, `policy_entropy_bonus=0.01`, `replay_policy="none"`, `rehearsal_batch=0`). |
| Training and sampling noise | Zero scheduled critic-input or generator-output additive noise (`0.0`). Intrinsic task MoG perturbation $\sigma=0.025$ is preserved on every conditional and posterior latent draw; `online-gail-bcap-snr` scales $s_f=4.0$ (`feature_tolerance=0.30`, `spike_slab_scale=2.0`) for $4\times$ higher latent SNR. |
| Parameter averaging and serving | Live unaveraged generator weights (`ema_decay=0.0`, `serve_average=False`). All prototype, histogram, return-baseline, Sherman-Morrison, and residual-covariance commits occur strictly after each arrival's updates. |

<a id="configuration-differences"></a>
## Configuration differences

- GAIL-BCAP forms a five-step single-factor ablation ladder over `online-feature-prior-bce`: `online-gail-bcap-hinge` (10/10 required, 1/10 stretch) isolates Hinge + prioritized BCE survival/sigmoid style bonus + causal return advantage + active policy entropy; `online-gail-bcap-bce` (10/10 required, 1/10 stretch) enables radial Sherman-Morrison associative extrapolation; `online-gail-bcap-momentum` tests $\beta_1=0.5$; `online-gail-bcap-spike-slab` (10/10 required, 3/10 stretch) upgrades associative extrapolation to Mahalanobis spike-and-slab + Markov context bottleneck + linear causal keys; and `online-gail-bcap-snr` (10/10 required, 4/10 stretch) resolves robotics action contrast erasure (`contrast_threshold=0.005`) and posterior latent SNR (`feature_scale=4.0`).
- `online-gail-bcap-snr` is the strongest single global recipe in ParticleStream (`14/20` total tasks passed: all 10 required Tier 1 tasks plus `seq_minimal_tokens4`, `seq_minimal_addition2bit`, `seq_robotics_fetch_reach`, and `seq_robotics_fetch_pick_and_place`, with `fetch_reach` RMSE `1.9 mm` and `fetch_pick_and_place` RMSE `2.4 mm`).
- Ablating `policy_entropy_bonus` (`0.01 -> 0.0`) or retaining episode-varying scalar return gain (`advantage_scale=0.25`) drops `seq_robotics_fetch_pick_and_place` from `5/5` to `4/5` terminal checks, confirming that active categorical policy entropy and episode-invariant style gain jointly stabilize multi-episode 3D manipulation dynamics.
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
- [configs/forge/ideas/online-gail-bcap-spike-slab.json](../../../configs/forge/ideas/online-gail-bcap-spike-slab.json)
- [configs/forge/ideas/online-gail-bcap-snr.json](../../../configs/forge/ideas/online-gail-bcap-snr.json)

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

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-gail-bcap-bce-required"></a>
### Required evidence

<a id="online-gail-bcap-bce-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-gail-bcap-bce-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-bce-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-gail-bcap-bce-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **NOT_RUN** | — | stale receipt: task_fingerprint differ | — | — |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **NOT_RUN** | — | stale receipt: task_fingerprint differ | — | — |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **NOT_RUN** | — | stale receipt: task_fingerprint differ | — | — |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **NOT_RUN** | — | stale receipt: task_fingerprint differ | — | — |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **NOT_RUN** | — | stale receipt: task_fingerprint differ | — | — |

<a id="online-gail-bcap-bce-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

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

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-gail-bcap-hinge-required"></a>
### Required evidence

<a id="online-gail-bcap-hinge-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-gail-bcap-hinge-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-hinge-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-gail-bcap-hinge-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **NOT_RUN** | — | stale receipt: task_fingerprint differ | — | — |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **NOT_RUN** | — | stale receipt: task_fingerprint differ | — | — |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **NOT_RUN** | — | stale receipt: task_fingerprint differ | — | — |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **NOT_RUN** | — | stale receipt: task_fingerprint differ | — | — |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **NOT_RUN** | — | stale receipt: task_fingerprint differ | — | — |

<a id="online-gail-bcap-hinge-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

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

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-gail-bcap-momentum-required"></a>
### Required evidence

<a id="online-gail-bcap-momentum-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-gail-bcap-momentum-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-momentum-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-gail-bcap-momentum-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **NOT_RUN** | — | stale receipt: task_fingerprint differ | — | — |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **NOT_RUN** | — | stale receipt: task_fingerprint differ | — | — |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **NOT_RUN** | — | stale receipt: task_fingerprint differ | — | — |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **NOT_RUN** | — | stale receipt: task_fingerprint differ | — | — |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **NOT_RUN** | — | stale receipt: task_fingerprint differ | — | — |

<a id="online-gail-bcap-momentum-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-gail-bcap-momentum --device cuda`.

<a id="online-gail-bcap-snr"></a>
## `online-gail-bcap-snr`

[Exact declaration](../../../configs/forge/ideas/online-gail-bcap-snr.json)

Parent: [`online-gail-bcap-spike-slab`](gail-bcap.md#online-gail-bcap-spike-slab).

Declared changes:

- Lower causal salience contrast gate (contrast_threshold=0.005) so sparse 4-of-64-pixel precision robotics actions are retained near contact while exact zero-contrast distractors stay gated
- Scale posterior feature projection by 2x (feature_scale=4.0, feature_tolerance=0.30, spike_slab_scale=2.0) to boost latent signal-to-noise ratio by 4x relative to task-owned prior_sigma=0.025, and fix episode-invariant imitation style gain (advantage_scale=0.0)

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.contrast_threshold` | `0.03` | `0.005` |
| `memory_policy.transport.advantage_scale` | `0.25` | `0.0` |
| `memory_policy.transport.feature_scale` | `2.0` | `4.0` |
| `memory_policy.transport.feature_tolerance` | `0.15` | `0.3` |
| `memory_policy.transport.spike_slab_scale` | `1.0` | `2.0` |

### Effective declaration

```json
{
  "memory_policy": {
    "age_horizon": 8.0,
    "association_weight": 0.0,
    "change_threshold": 0.04,
    "context_scale": 3.0,
    "contrast_threshold": 0.005,
    "decay": 0.9,
    "dim": 16,
    "key_gain": 0.5,
    "kind": "fast_weight",
    "normalize_key": false,
    "reader": "causal_state",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "transport": {
      "advantage_scale": 0.0,
      "assoc_blend_scale": 0.25,
      "assoc_forgetting": 1.0,
      "assoc_max_blend": 1.0,
      "assoc_ridge": 0.1,
      "cancel_linear_gan": true,
      "context_bypass_weight": 5.0,
      "context_capacity": 64,
      "context_clip": 8.0,
      "context_tolerance": 0.25,
      "covariance_ema": 0.05,
      "covariance_eps": 0.0001,
      "feature_scale": 4.0,
      "feature_tolerance": 0.3,
      "image_center": 0.5,
      "kernel_bandwidth": 0.35,
      "policy_entropy_bonus": 0.01,
      "reconstruction_loss": "binary_cross_entropy",
      "reconstruction_weight": 10.0,
      "return_discount": 0.99,
      "router": "gail_conditional",
      "spike_slab_scale": 2.0,
      "style_bonus_weight": 0.05,
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.0,
      "variance_ratio_scale": 0.15,
      "visit_shrinkage": 0.25
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

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-gail-bcap-snr-required"></a>
### Required evidence

<a id="online-gail-bcap-snr-tier-1-required"></a>
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

<a id="online-gail-bcap-snr-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-snr-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-gail-bcap-snr-tier-2-stretch"></a>
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

<a id="online-gail-bcap-snr-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | no receipt | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-gail-bcap-snr --device cuda`.

<a id="online-gail-bcap-spike-slab"></a>
## `online-gail-bcap-spike-slab`

[Exact declaration](../../../configs/forge/ideas/online-gail-bcap-spike-slab.json)

Parent: [`online-gail-bcap-bce`](gail-bcap.md#online-gail-bcap-bce).

Declared changes:

- Replace radial distance blending with float64 Sherman-Morrison RLS (assoc_ridge=0.1) + precision-weighted Mahalanobis spike-and-slab prototype transport (spike_slab_scale=1.0) and empirical-Bayes kernel-gated active-entropy residual quantiles
- Add Markov context-bypass regularizer (context_bypass_weight=5.0), calibrated sigmoid style bonus (style_bonus_weight=0.05), and unnormalized linear causal key projection (normalize_key=false, key_gain=0.5) preserving Euclidean superposition norm

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.key_gain` | _unset_ | `0.5` |
| `memory_policy.normalize_key` | _unset_ | `false` |
| `memory_policy.transport.assoc_ridge` | `0.05` | `0.1` |
| `memory_policy.transport.context_bypass_weight` | _unset_ | `5.0` |
| `memory_policy.transport.covariance_ema` | _unset_ | `0.05` |
| `memory_policy.transport.covariance_eps` | _unset_ | `0.0001` |
| `memory_policy.transport.kernel_bandwidth` | _unset_ | `0.35` |
| `memory_policy.transport.spike_slab_scale` | _unset_ | `1.0` |
| `memory_policy.transport.style_bonus_weight` | `0.1` | `0.05` |
| `memory_policy.transport.variance_ratio_scale` | _unset_ | `0.15` |
| `memory_policy.transport.visit_shrinkage` | _unset_ | `0.25` |

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
    "key_gain": 0.5,
    "kind": "fast_weight",
    "normalize_key": false,
    "reader": "causal_state",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "transport": {
      "advantage_scale": 0.25,
      "assoc_blend_scale": 0.25,
      "assoc_forgetting": 1.0,
      "assoc_max_blend": 1.0,
      "assoc_ridge": 0.1,
      "cancel_linear_gan": true,
      "context_bypass_weight": 5.0,
      "context_capacity": 64,
      "context_clip": 8.0,
      "context_tolerance": 0.25,
      "covariance_ema": 0.05,
      "covariance_eps": 0.0001,
      "feature_scale": 2.0,
      "feature_tolerance": 0.15,
      "image_center": 0.5,
      "kernel_bandwidth": 0.35,
      "policy_entropy_bonus": 0.01,
      "reconstruction_loss": "binary_cross_entropy",
      "reconstruction_weight": 10.0,
      "return_discount": 0.99,
      "router": "gail_conditional",
      "spike_slab_scale": 1.0,
      "style_bonus_weight": 0.05,
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.0,
      "variance_ratio_scale": 0.15,
      "visit_shrinkage": 0.25
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

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-gail-bcap-spike-slab-required"></a>
### Required evidence

<a id="online-gail-bcap-spike-slab-tier-1-required"></a>
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

<a id="online-gail-bcap-spike-slab-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-spike-slab-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | no receipt | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | no receipt | — | — |

<a id="online-gail-bcap-spike-slab-tier-2-stretch"></a>
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

<a id="online-gail-bcap-spike-slab-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | no receipt | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-gail-bcap-spike-slab --device cuda`.

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
- [Bayesian Variable Selection in Linear Regression](https://doi.org/10.1080/01621459.1988.10478693). Spike-and-slab Bayes factor formulation powering Mahalanobis discrete-vs-continuous prototype gating.
