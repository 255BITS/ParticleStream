<!-- Generated Forge family report -->

# ResidualPrior-BCAP

[← Technique leaderboard](../technique-inventory.md)

<a id="technique-overview"></a>
## Technique overview

ResidualPrior-BCAP extends FeaturePrior-BCAP by replacing nearest-neighbor context lookup on unseen states with an online parametric residual conditional prior (`ResidualFeatureConditionalReconstruction`). A detached `float64` Sherman-Morrison Recursive Least Squares (RLS) affine predictor maintains global second-order sufficient statistics across all sequential arrivals without replay, paired with a Mahalanobis precision-weighted spike-and-slab prototype posterior that snaps along invariant subspaces on discrete compositional manifolds while staying linear on continuous physical manifolds. An empirical-Bayes kernel gate restores exact multimodal empirical quantiles at recurrent contexts, and a Markov bottleneck penalty prevents direct generator context-shortcut weights from bypassing the latent prior.

[Implementation](../../../particlestream/residual_prior.py) · [Mathematical formulation](residualprior-bcap.md#mathematical-formulation) · [Simplified pseudocode](residualprior-bcap.md#simplified-pseudocode) · [Training details](residualprior-bcap.md#training-details) · [Protocol](../../../EXPERIMENTATION.md) · [Terminology](../../../RESEARCH_GLOSSARY.md)

<a id="mathematical-formulation"></a>
## Mathematical formulation

$G$ is the task-owned residual generator with input layer $W_{\mathrm{fc}}=[W_z,\,W_{\mathrm{ctx}}]$, $D$ the raw-score convolutional critic, and $P$ the frozen $K$-row `MoGParticlePrior` ($K=32$, $d_z=8$, $\sigma=0.025$) with base centers $\mu_i^{\mathrm{base}}\in\mathbb R^{d_z}$. At step $t$, $x_t\in[0,1]^{1\times 8\times 8}$ is the single arriving target frame, $\phi(x_t)=s_f W_{\mathrm{enc}}(\bar x_t-c_0\mathbf 1_{64})\in\mathbb R^{d_z}$ its fixed orthogonal latent code ($s_f=2$, $c_0=0.5$), and $\bar c_{t-1}=[\tilde c_{t-1};\,1]\in\mathbb R^{d_c+1}$ the clipped preceding causal context augmented with bias ($d_c=16$). $W_t\in\mathbb R^{d_z\times(d_c+1)}$ and $P_t\in\mathbb R^{(d_c+1)\times(d_c+1)}$ are the online RLS weights and inverse-Gram matrix ($\lambda_{\mathrm{rls}}=0.1$), $S_t\in\mathbb R^{d_z\times d_z}$ and $q_t\in\mathbb R_{\ge 0}$ are exponentially weighted ($\alpha_S=0.05$) post-fit RLS residual covariance and nearest-prototype quantization variance ($\varepsilon_S=10^{-4}$), $\hat\mu_m\in\mathbb R^{d_z}$ are online prototypes ($M_t\le 32$), and $(k_b^{\mathrm{ctx}},V_b,M_{b,m})$ are context keys, visit counts, and outcome masses ($C_{\max}=64$, $\kappa_{\mathrm{visit}}=0.25$, $h_k=0.35$).

**Causal Salient Key Projection (Normalized vs. Linear-Key Ablation)**

$$
\begin{aligned}k_t^{\mathrm{norm}}&=s_t\frac{W_k\tilde x_t}{\lVert W_k\tilde x_t\rVert_2},\qquad k_t^{\mathrm{lin}}=s_t\,g_k W_k\tilde x_t,\qquad f_t=k_t+\tfrac12(k_t-k_{t-1})+(1-s_t)r_{t-1}.\end{aligned}
$$

`online-residual-prior-bce` inherits unit-sphere normalization $k_t^{\mathrm{norm}}$ (`normalize_key=True`), whereas `online-residual-prior-linear-key` uses unnormalized salient projection $k_t^{\mathrm{lin}}$ (`normalize_key=False`, $g_k=0.5$) so disjoint pixel patches and continuous physical state-action pairs preserve exact Euclidean superposition in context space while low-contrast arrivals ($s_t=0$) remain gated.

**Online Sherman-Morrison RLS Predictor and Residual Covariance**

$$
\begin{aligned}g_t&=\frac{P_{t-1}\bar c_{t-1}}{1+\bar c_{t-1}^\top P_{t-1}\bar c_{t-1}},\qquad W_t=W_{t-1}+\big(\phi(x_t)-W_{t-1}\bar c_{t-1}\big)g_t^\top,\qquad P_t=P_{t-1}-g_t\bar c_{t-1}^\top P_{t-1},\\e_t^{\mathrm{post}}&=\phi(x_t)-W_t\bar c_{t-1},\qquad S_t=(1-\alpha_S)S_{t-1}+\alpha_S\,e_t^{\mathrm{post}}(e_t^{\mathrm{post}})^\top,\qquad q_t=(1-\alpha_S)q_{t-1}+\alpha_S\min_m\lVert\phi(x_t)-\hat\mu_m\rVert_2^2.\end{aligned}
$$

Rank-1 Sherman-Morrison updates accumulate exact ridge-regularized linear sufficient statistics across correlated `batch_size=1` streams without forgetting earlier episodes. After warm-up ($t>d_c$), $S_t$ captures directional post-fit residual covariance while $q_t$ tracks continuous vs. discrete target dispersion.

**Mahalanobis Spike-and-Slab Prototype Posterior and Kernel-Gated Residual Quantiles**

$$
\begin{aligned}\hat z_{\mathrm{lin}}(c)&=W_t\bar c,\qquad \tilde S_t=\tfrac12(S_t+S_t^\top)+\varepsilon_S I,\qquad m^*(c)=\arg\min_{0\le m<M_t}(\hat\mu_m-\hat z_{\mathrm{lin}}(c))^\top\tilde S_t^{-1}(\hat\mu_m-\hat z_{\mathrm{lin}}(c)),\\\beta_t(c)&=\exp\!\left(-\left(\frac{q_t}{\rho_v\operatorname{tr}(S_t)+10^{-5}}\right)^2\right)\sigma\!\left(\tfrac12\log\frac{\det(\tilde S_t+\sigma_{\mathrm{slab}}^2 I)}{\det(\tilde S_t)}-\tfrac12 d_{\tilde S_t}^2(\hat\mu_{m^*(c)},\hat z_{\mathrm{lin}}(c))\right),\\\hat z_{\mathrm{ctx}}(c)&=(1-\beta_t(c))\hat z_{\mathrm{lin}}(c)+\beta_t(c)\hat\mu_{m^*(c)},\qquad z_i^{\mathrm{cond}}(c)=\hat z_{\mathrm{ctx}}(c)+\gamma_t(c)\big(\hat\mu_{q(i,b^*)}-\hat z_{\mathrm{ctx}}(k_{b^*}^{\mathrm{ctx}})\big)+\epsilon_i.\end{aligned}
$$

Along exact linear invariants where $\tilde S_t$ has near-zero variance, $\tilde S_t^{-1}$ enforces the invariant while down-weighting nonlinear carry/parity directions ($\rho_v=0.15$, $\sigma_{\mathrm{slab}}=1.0$). Using the shared discreteness gate $g_{\mathrm{disc},t}=\exp(-(q_t/(\rho_v\operatorname{tr}(S_t)+10^{-5}))^2)$, continuous manifolds ($q_t/\operatorname{tr}(S_t)\gg\rho_v$) satisfy $\beta_t(c)\to 0$ and $\gamma_t(c)\to 0$ for smooth extrapolation, while recurrent multimodal contexts ($V_{b^*}>1$) recover exact branch quantiles via $\gamma_t(c)=\frac{V_{b^*}-1}{V_{b^*}-1+\kappa_{\mathrm{visit}}}g_{\mathrm{disc},t}\exp(-\lVert c-k_{b^*}^{\mathrm{ctx}}\rVert_2^2/h_k^2)$.

**Markov-Bottlenecked Generator and Critic Objectives**

$$
\begin{aligned}\mathcal L_D&=\operatorname{softplus}(-D(x_t,c_{t-1}))+\operatorname{softplus}(D(\hat x_D,c_{t-1}))+\frac{\lambda}{2}\Big[(\lVert g_r\rVert_2-\kappa)_+^2+(\lVert g_f\rVert_2-\kappa)_+^2\Big],\\\mathcal L_G&=\operatorname{softplus}(-D(\hat x_G,c_{t-1}))+\lambda_{\mathrm{rec}}\,\operatorname{BCE}(\hat x_G,x_t)+\lambda_{\mathrm{ctx}}\,\lVert W_{\mathrm{ctx}}\rVert_F^2.\end{aligned}
$$

Because $G$ trains on current-observation posterior codes $z_{i_G}^{\mathrm{post}}=\phi(x_t)+\epsilon_{i_G}$, penalizing the direct context-input block $W_{\mathrm{ctx}}$ ($\lambda_{\mathrm{ctx}}=5$, $\lambda_{\mathrm{rec}}=10$) prevents $G$ from memorizing seen contexts around $z$ and ensures held-out states decode cleanly through $z_i^{\mathrm{cond}}(c)$.

<a id="simplified-pseudocode"></a>
## Simplified pseudocode

```text
Initialize fixed orthogonal encoder W_enc, float64 RLS (W_0 = 0, P_0 = I / lambda_rls), S_0 = 1e-2 I, q_0 = 0, and empty prototype/context tables.
For each chronological single-arrival step t (batch_size = 1, replay = none):
  Before revealing x_t, serve y_hat = G(z_cond(i_serve, c_{t-1}), c_{t-1}) under an isolated eval RNG.
  Reveal single target frame x_t and encode phi(x_t) = s_f W_enc (x_bar_t - c_0).
  Draw uniform row i_D; compute z_ctx(c_{t-1}) via RLS + Mahalanobis spike-and-slab and add kernel-gated live residual quantile to form z_D.
  Update critic D with non-saturating logistic loss on x_t and stop_gradient(G(z_D, c_{t-1})) plus BCAP input-gradient penalty.
  Draw uniform row i_G; form posterior code z_G = phi(x_t) + eps_{i_G}; update G with adversarial loss + 10 * BCE + 5 * ||W_ctx||_F^2.
  After both updates succeed, commit phi(x_t) to prototype table (<=32) and c_{t-1} to context table (<=64), then run rank-1 Sherman-Morrison RLS and EMA updates for (W_t, P_t, S_t, q_t).
  Advance detached causal state c_t = CausalStateMemory.write(x_t) without storing or replaying frames.
```

<a id="training-details"></a>
## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Decoupled non-saturating logistic adversarial loss for $D$ and $G$, paired on $G$ with pixelwise BCE posterior reconstruction ($\lambda_{\mathrm{rec}}=10$) and Frobenius regularization on $G$'s direct context-input weights ($\lambda_{\mathrm{ctx}}=5$). |
| Optimizer | Native Adam via `bcap_adam` ($\beta_1=0$, $\beta_2=0.99$, `prior_betas=(0, 0.99)`). Base `MoGParticlePrior` is frozen (`prior_update="frozen"`); RLS and covariance buffers update analytically in `float64` without optimizer state. |
| Learning rates and annealing | Base LR $0.005$ with `critic_lr_mult=1.0`, `prior_lr_mult=1.0`, holding constant through `lr_anneal_start=0.5` before cosine-decaying to `lr_floor=0.05`. |
| Parameter-gradient clipping | None (`critic_grad_guard=False`). Critic Lipschitz control acts solely through the BCAP input-gradient cap. |
| Critic penalties and anchors | Fixed one-sided squared $L_2$ BCAP cap on real and residual-prior fake inputs ($\lambda=1.0$, $\kappa=1.0$) every step; no critic EMA anchor. |
| Damping and update guards | No sparse-row latent damping (`latent_damping_max_rate=0.0`) and zero generator-center probes (`probe_law="no_generator_probes_v1"`). |
| Online memory, prior and routing | Detached `CausalStateMemory` ($d_c=16$, `association_weight=0.0`) plus `ResidualFeatureConditionalReconstruction` (`rls_ridge=0.1`, `covariance_ema=0.05`, `slab_scale=1.0`, `variance_ratio_scale=0.15`, `visit_shrinkage=0.25`, `kernel_bandwidth=0.35`, `replay_policy="none"`, `rehearsal_batch=0`). |
| Training and sampling noise | Zero scheduled critic-input or generator-output additive noise (`0.0`); intrinsic task MoG perturbation $\sigma=0.025$ is preserved on every transformed particle row. |
| Parameter averaging and serving | Live unaveraged generator weights (`ema_decay=0.0`, `serve_average=False`). Pre-arrival serving queries only committed sufficient statistics and preceding detached context $c_{t-1}$. |

<a id="configuration-differences"></a>
## Configuration differences

- `online-residual-prior-bce` isolates the prior/transport change relative to `online-feature-prior-bce`, keeping `CausalStateMemory` (`normalize_key=True`) and every optimizer/loss override unchanged while replacing `FeatureConditionalReconstruction` with `ResidualFeatureConditionalReconstruction`.
- `online-residual-prior-linear-key` is a single-factor child of `online-residual-prior-bce` setting `normalize_key=False` (`key_gain=0.5`) in `CausalStateMemory`, preserving linear Euclidean norm across multi-patch compositional frames and continuous state-action encodings.
- When $d_c=0$ (`seq_minimal_dense4`, `seq_minimal_tokens4`, `seq_img_bars4`, `seq_img_dwell_bars4`, `seq_img_variable_dwell`, `seq_img_rare_mode`), `ResidualFeatureConditionalReconstruction` delegates directly to `FeatureConditionalReconstruction` and preserves exact marginal empirical quantiles.
- Zero trainable `nn.Parameter` tensors or random weights are added by the residual prior; all updates are deterministic closed-form matrix operations committed strictly after `GANTrainer.step` succeeds.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [particlestream/residual_prior.py](../../../particlestream/residual_prior.py)
- [particlestream/feature_prior.py](../../../particlestream/feature_prior.py)
- [particlestream/memory.py](../../../particlestream/memory.py)
- [particlestream/transport.py](../../../particlestream/transport.py)
- [particlestream/tier1.py](../../../particlestream/tier1.py)
- [configs/forge/ideas/online-residual-prior-bce.json](../../../configs/forge/ideas/online-residual-prior-bce.json)
- [configs/forge/ideas/online-residual-prior-linear-key.json](../../../configs/forge/ideas/online-residual-prior-linear-key.json)

</details>

Each section below describes one exact global recipe. Configuration and parent deltas come from the declarations; results come from Forge's current source-bound receipts. Required and stretch evidence stay separate. A final metric can meet its gate while the sustained verdict fails: every one of the last five checks must pass.

The installed engine is pinned to ParticleGAN/Forge 0.9.7. Older version or replay wording in declaration history describes provenance; the effective configuration below controls execution.

<a id="online-residual-prior-linear-key"></a>
## `online-residual-prior-linear-key`

[Exact declaration](../../../configs/forge/ideas/online-residual-prior-linear-key.json)

Parent: [`online-residual-prior-bce`](residualprior-bcap.md#online-residual-prior-bce).

Declared changes:

- Unnormalized salient causal key projection (normalize_key=false, key_gain=0.5) preserves Euclidean superposition norm across compositional and continuous kinematic inputs while retaining contrast gating

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.key_gain` | _unset_ | `0.5` |
| `memory_policy.normalize_key` | _unset_ | `false` |

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
      "outcome_decay": 1.0,
      "reconstruction_loss": "binary_cross_entropy",
      "reconstruction_weight": 10.0,
      "rls_ridge": 0.1,
      "router": "residual_feature_conditional",
      "slab_scale": 1.0,
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.0,
      "variance_ratio_scale": 0.15,
      "visit_shrinkage": 0.25
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

Current `cuda`, seed 0: **PASS**, 10/10 required passes (10 measured), 3/10 stretch passes (10 measured); qualified tier 1.

Receipt identity: candidate_fingerprint `1dea3f5242caf231176371c83503b3439d69e06cbae9aedb8cbe237de17478f6`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-residual-prior-linear-key-required"></a>
### Required evidence

<a id="online-residual-prior-linear-key-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-residual-prior-linear-key/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-residual-prior-linear-key_seq_img_round_robin_bars4.gif) | 2026-11-13T23:32:01.296+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-residual-prior-linear-key/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-residual-prior-linear-key_seq_img_dwell_bars4.gif) | 2026-11-13T23:32:08.733+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.008856`; `direction_acc=1` | [receipt](../receipts/cuda/online-residual-prior-linear-key/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-residual-prior-linear-key_seq_img_translating_bar_next.gif) | 2026-11-13T23:32:18.844+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-residual-prior-linear-key/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-residual-prior-linear-key_seq_img_fork_rollout.gif) | 2026-11-13T23:32:29.010+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-residual-prior-linear-key/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-residual-prior-linear-key_seq_img_variable_dwell_bars4.gif) | 2026-11-13T23:32:36.269+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-residual-prior-linear-key/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-residual-prior-linear-key_seq_img_rare_mode_bars4.gif) | 2026-11-13T23:32:43.786+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.026427` | [receipt](../receipts/cuda/online-residual-prior-linear-key/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-residual-prior-linear-key_seq_img_delayed_cue.gif) | 2026-11-13T23:32:53.851+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.09375`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-residual-prior-linear-key/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-residual-prior-linear-key_seq_img_persistent_fork.gif) | 2026-11-13T23:33:19.740+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.020518`; `xy_rmse=0.003371`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-residual-prior-linear-key/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-residual-prior-linear-key_seq_img_circle_orbit.gif) | 2026-11-13T23:33:50.261+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.020386`; `s2c_rmse=0.020614`; `c2s_rmse=0.020157`; `xy_rmse=0.005701`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-residual-prior-linear-key/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-residual-prior-linear-key_seq_img_square_circle_cycle.gif) | 2026-11-13T23:34:47.260+00:00 |

<a id="online-residual-prior-linear-key-stretch"></a>
### Stretch evidence

<a id="online-residual-prior-linear-key-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-residual-prior-linear-key/seq_minimal_dense4.json) · [goal.gif](../gifs/online-residual-prior-linear-key_seq_minimal_dense4.gif) | 2026-11-13T23:34:50.698+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-residual-prior-linear-key/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-residual-prior-linear-key_seq_minimal_tokens4.gif) | 2026-11-13T23:34:54.403+00:00 |

<a id="online-residual-prior-linear-key-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.04305`; `seen_max_abs_error=0.17417`; `heldout_response_rmse=0.038166`; `heldout_max_abs_error=0.090952`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-residual-prior-linear-key/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-residual-prior-linear-key_seq_minimal_kernel2x2.gif) | 2026-11-13T23:35:01.441+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **PASS** | 5/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.96531`; `heldout_exact_accuracy=1`; `heldout_min_bit_confidence=0.96443` | [receipt](../receipts/cuda/online-residual-prior-linear-key/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-residual-prior-linear-key_seq_minimal_addition2bit.gif) | 2026-11-13T23:35:08.596+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **PASS** | 5/5 | `position_delta_rmse_m=0.0035343`; `relative_mse=0.044448`; `moving_direction_accuracy=0.95833`; `finite_fraction=1` | [receipt](../receipts/cuda/online-residual-prior-linear-key/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-residual-prior-linear-key_seq_robotics_fetch_reach.gif) | 2026-11-13T23:35:23.109+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010359`; `relative_mse=0.71504`; `moving_direction_accuracy=0.74419`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-residual-prior-linear-key/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-residual-prior-linear-key_seq_robotics_fetch_push.gif) | 2026-11-13T23:35:39.087+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0073618`; `relative_mse=0.32987`; `moving_direction_accuracy=0.72549`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-residual-prior-linear-key/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-residual-prior-linear-key_seq_robotics_fetch_slide.gif) | 2026-11-13T23:35:55.421+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 4/5 | `position_delta_rmse_m=0.002799`; `relative_mse=0.049913`; `moving_direction_accuracy=0.77193`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75 | [receipt](../receipts/cuda/online-residual-prior-linear-key/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-residual-prior-linear-key_seq_robotics_fetch_pick_and_place.gif) | 2026-11-13T23:36:10.576+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0081776`; `relative_mse=0.98914`; `moving_direction_accuracy=0.6215`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-residual-prior-linear-key/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-residual-prior-linear-key_seq_robotics_hand_reach.gif) | 2026-11-13T23:36:24.818+00:00 |

<a id="online-residual-prior-linear-key-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.034183`; `interpolation_speed_error=0.047889`; `interpolation_raw_rmse=0.048552`; `interpolation_point_validity=0.083333`; `interpolation_direction_accuracy=0.91667`; `extrapolation_centroid_rmse=0.17539`; `extrapolation_speed_error=0.16964`; `extrapolation_raw_rmse=0.098298`; `extrapolation_point_validity=0.16667`; `extrapolation_direction_accuracy=0.75`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_raw_rmse <= 0.12; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-residual-prior-linear-key/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-residual-prior-linear-key_seq_minimal_velocity_extrapolation.gif) | 2026-11-13T23:36:43.068+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-residual-prior-linear-key --device cuda`.

<a id="online-residual-prior-bce"></a>
## `online-residual-prior-bce`

[Exact declaration](../../../configs/forge/ideas/online-residual-prior-bce.json)

Parent: [`online-feature-prior-bce`](featureprior-bcap.md#online-feature-prior-bce).

Declared changes:

- Parametric residual conditional prior: online float64 Sherman-Morrison RLS affine predictor plus Mahalanobis precision spike-and-slab prototype transport and empirical-Bayes kernel-gated live residual quantiles replace 1-NN context lookup on unseen states
- Markov bottleneck regularizer (context_bypass_weight=5.0) penalizes direct generator input-context shortcut weights so held-out predictions decode through the conditional latent prior

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.transport.context_bypass_weight` | _unset_ | `5.0` |
| `memory_policy.transport.covariance_ema` | _unset_ | `0.05` |
| `memory_policy.transport.covariance_eps` | _unset_ | `0.0001` |
| `memory_policy.transport.kernel_bandwidth` | _unset_ | `0.35` |
| `memory_policy.transport.outcome_decay` | _unset_ | `1.0` |
| `memory_policy.transport.rls_ridge` | _unset_ | `0.1` |
| `memory_policy.transport.router` | `"feature_conditional"` | `"residual_feature_conditional"` |
| `memory_policy.transport.slab_scale` | _unset_ | `1.0` |
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
    "kind": "fast_weight",
    "reader": "causal_state",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "transport": {
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
      "outcome_decay": 1.0,
      "reconstruction_loss": "binary_cross_entropy",
      "reconstruction_weight": 10.0,
      "rls_ridge": 0.1,
      "router": "residual_feature_conditional",
      "slab_scale": 1.0,
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.0,
      "variance_ratio_scale": 0.15,
      "visit_shrinkage": 0.25
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

Receipt identity: candidate_fingerprint `8d524b72a95ff9f642df58edf11c3fa522f05b6e3c9a5653f266eedd29a47e28`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-residual-prior-bce-required"></a>
### Required evidence

<a id="online-residual-prior-bce-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-residual-prior-bce/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-residual-prior-bce_seq_img_round_robin_bars4.gif) | 2026-11-13T23:27:23.256+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-residual-prior-bce/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-residual-prior-bce_seq_img_dwell_bars4.gif) | 2026-11-13T23:27:30.672+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.009906`; `direction_acc=1` | [receipt](../receipts/cuda/online-residual-prior-bce/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-residual-prior-bce_seq_img_translating_bar_next.gif) | 2026-11-13T23:27:40.800+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-residual-prior-bce/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-residual-prior-bce_seq_img_fork_rollout.gif) | 2026-11-13T23:27:51.041+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-residual-prior-bce/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-residual-prior-bce_seq_img_variable_dwell_bars4.gif) | 2026-11-13T23:27:58.344+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-residual-prior-bce/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-residual-prior-bce_seq_img_rare_mode_bars4.gif) | 2026-11-13T23:28:05.820+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.029977` | [receipt](../receipts/cuda/online-residual-prior-bce/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-residual-prior-bce_seq_img_delayed_cue.gif) | 2026-11-13T23:28:15.966+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.03125`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-residual-prior-bce/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-residual-prior-bce_seq_img_persistent_fork.gif) | 2026-11-13T23:28:40.453+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.023236`; `xy_rmse=0.005273`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-residual-prior-bce/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-residual-prior-bce_seq_img_circle_orbit.gif) | 2026-11-13T23:29:08.642+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.020737`; `s2c_rmse=0.021127`; `c2s_rmse=0.020347`; `xy_rmse=0.006415`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-residual-prior-bce/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-residual-prior-bce_seq_img_square_circle_cycle.gif) | 2026-11-13T23:30:00.445+00:00 |

<a id="online-residual-prior-bce-stretch"></a>
### Stretch evidence

<a id="online-residual-prior-bce-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-residual-prior-bce/seq_minimal_dense4.json) · [goal.gif](../gifs/online-residual-prior-bce_seq_minimal_dense4.gif) | 2026-11-13T23:30:03.894+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-residual-prior-bce/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-residual-prior-bce_seq_minimal_tokens4.gif) | 2026-11-13T23:30:07.567+00:00 |

<a id="online-residual-prior-bce-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.047976`; `seen_max_abs_error=0.22047`; `heldout_response_rmse=0.066654`; `heldout_max_abs_error=0.14824`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-residual-prior-bce/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-residual-prior-bce_seq_minimal_kernel2x2.gif) | 2026-11-13T23:30:14.595+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.93636`; `heldout_exact_accuracy=0.25`; `heldout_min_bit_confidence=2.3467e-15`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-residual-prior-bce/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-residual-prior-bce_seq_minimal_addition2bit.gif) | 2026-11-13T23:30:21.589+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0050827`; `relative_mse=0.091926`; `moving_direction_accuracy=0.95833`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-residual-prior-bce/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-residual-prior-bce_seq_robotics_fetch_reach.gif) | 2026-11-13T23:30:35.874+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011576`; `relative_mse=0.89294`; `moving_direction_accuracy=0.67442`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-residual-prior-bce/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-residual-prior-bce_seq_robotics_fetch_push.gif) | 2026-11-13T23:30:51.116+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0082458`; `relative_mse=0.41384`; `moving_direction_accuracy=0.68627`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-residual-prior-bce/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-residual-prior-bce_seq_robotics_fetch_slide.gif) | 2026-11-13T23:31:06.458+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 3/5 | `position_delta_rmse_m=0.002891`; `relative_mse=0.053249`; `moving_direction_accuracy=0.77193`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75 | [receipt](../receipts/cuda/online-residual-prior-bce/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-residual-prior-bce_seq_robotics_fetch_pick_and_place.gif) | 2026-11-13T23:31:21.143+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0083971`; `relative_mse=1.043`; `moving_direction_accuracy=0.58879`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-residual-prior-bce/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-residual-prior-bce_seq_robotics_hand_reach.gif) | 2026-11-13T23:31:35.466+00:00 |

<a id="online-residual-prior-bce-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.031094`; `interpolation_speed_error=0.042178`; `interpolation_raw_rmse=0.037342`; `interpolation_point_validity=0.16667`; `interpolation_direction_accuracy=0.91667`; `extrapolation_centroid_rmse=0.1451`; `extrapolation_speed_error=0.093398`; `extrapolation_raw_rmse=0.086448`; `extrapolation_point_validity=0.25`; `extrapolation_direction_accuracy=0.66667`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-residual-prior-bce/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-residual-prior-bce_seq_minimal_velocity_extrapolation.gif) | 2026-11-13T23:31:52.526+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-residual-prior-bce --device cuda`.

<a id="refresh"></a>
## Refresh

```sh
particlestream compile --device cuda
```

This page is generated alongside the leaderboard from committed source-bound receipts and declarations; editing a generated page cannot change a verdict or earn qualification.

<a id="references"></a>
## References

- [Adjustment of an Inverse Matrix Corresponding to a Change in One Element of a Given Matrix](https://doi.org/10.1214/aoms/1177729893). Rank-1 Sherman-Morrison inverse-Gram recursion powering the exact online batch_size=1 RLS predictor.
- [Bayesian Variable Selection in Linear Regression](https://doi.org/10.1080/01621459.1988.10478693). Spike-and-slab Bayes factor formulation adapted for precision-weighted discrete prototype vs. continuous slab gating.
- [Deep Variational Information Bottleneck](https://arxiv.org/abs/1612.00410). Information-bottleneck motivation for regularizing direct generator context-shortcut weights so predictions route through the latent code.
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661). Non-saturating logistic adversarial objective paired with the residual conditional prior.
