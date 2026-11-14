<!-- Generated Forge family report -->

# SubspacePrior-BCAP

[← Technique leaderboard](../technique-inventory.md)

<a id="technique-overview"></a>
## Technique overview

SubspacePrior-BCAP extends ResidualPrior-BCAP (`SubspaceResidualConditionalReconstruction`) to eliminate two coupled bottlenecks of fixed-projection online priors under strict `batch_size=1` streaming (`replay_policy="none"`): (1) blind `8×64` random-projection nullspace loss and 1:1 MoG Wiener attenuation on millimetre physical displacements, solved by causally anchoring up to $K_{\mathrm{sub}}=6$ orthonormal two-pass Gram-Schmidt innovation directions with fixed discovery-time precision-whitening gains alongside a 2-row deterministic Johnson-Lindenstrauss tail; and (2) global hyperplane collapse on piecewise-nonlinear continuous dynamics, solved by recording exact unquantized per-bucket latent means and adding a smooth Nadaraya-Watson kernel-ridge residual gated by $(1-g_{\mathrm{disc},t})$.

[Implementation](../../../particlestream/subspace_prior.py) · [Mathematical formulation](subspaceprior-bcap.md#mathematical-formulation) · [Simplified pseudocode](subspaceprior-bcap.md#simplified-pseudocode) · [Training details](subspaceprior-bcap.md#training-details) · [Protocol](../../../EXPERIMENTATION.md) · [Terminology](../../../RESEARCH_GLOSSARY.md)

<a id="mathematical-formulation"></a>
## Mathematical formulation

$G$ is the task-owned residual generator with input layer $W_{\mathrm{fc}}=[W_z,\,W_{\mathrm{ctx}}]$, $D$ the raw-score convolutional critic, and $P$ the frozen $K$-row `MoGParticlePrior` ($K=32$, $d_z=8$, $\sigma=0.025$) with base centers $\mu_i^{\mathrm{base}}\in\mathbb R^{d_z}$. At step $t$, $x_t\in[0,1]^{1\times 8\times 8}$ is the single arriving frame, $v_t=\bar x_t-c_0\mathbf 1_{64}\in\mathbb R^{64}$ ($c_0=0.5$), $E_0\in\mathbb R^{d_z\times 64}$ is the deterministic orthogonal fallback matrix, $K_{\mathrm{sub}}=6$ is the anchored subspace capacity, $u_j\in\mathbb R^{64}$ and $g_j>0$ ($0\le j<k_t\le K_{\mathrm{sub}}$) are committed unit innovation basis vectors and fixed precision-whitening gains ($\tau_{\mathrm{abs}}=0.008$, $\tau_{\mathrm{rel}}=0.12$, $\nu_0=0.10$, $s_f=2.1$). $c_{t-1}\in\mathbb R^{d_c}$ ($d_c=16$) is the detached causal context ($\tau_{\mathrm{con}}\in\{0.03,0.002\}$), $(W_t,P_t,S_t,q_t)$ are online `float64` RLS and residual-covariance statistics ($\lambda_{\mathrm{rls}}=0.1$), $g_{\mathrm{disc},t}\in(0,1]$ is the empirical-Bayes discreteness gate, and $(k_b^{\mathrm{ctx}},V_b,\bar z_b)$ are context keys, visit counts, and exact unquantized latent means ($C_{\max}=64$, $\omega_{\mathrm{nw}}=1.0$, $h_{\mathrm{nw}}=0.65$, $\kappa_{\mathrm{nw}}=0.06$).

**Precision-Whitened Anchored Gram-Schmidt Innovation Subspace and JL Tail**

$$
\begin{aligned}r_t&=\Big(I-U_{k_{t-1}}U_{k_{t-1}}^\top\Big)^2 v_t,\qquad \mathcal I_t=\mathbb I\!\left[k_{t-1}<K_{\mathrm{sub}}\;\wedge\;\lVert r_t\rVert_2>\tau_{\mathrm{abs}}\;\wedge\;\frac{\lVert r_t\rVert_2}{\lVert v_t\rVert_2+10^{-3}}>\tau_{\mathrm{rel}}\right],\\u_{k_{t-1}}&=\frac{r_t}{\lVert r_t\rVert_2},\qquad g_{k_{t-1}}=\big(\lVert r_t\rVert_2^2+\nu_0^2\big)^{-1/2}\quad(\text{anchored once when }\mathcal I_t=1),\\\phi_t(x)&=s_f\begin{bmatrix}\operatorname{diag}(g_{0:k_t-1})U_{k_t}^\top\\0_{(K_{\mathrm{sub}}-k_t)\times 64}\\E_{0,\,K_{\mathrm{sub}}:d_z}\end{bmatrix}(\bar x-c_0\mathbf 1_{64}).\end{aligned}
$$

Two-pass classical Gram-Schmidt orthogonalization in `float64` captures exact active target subspaces while the dimensionless relative innovation gate $\tau_{\mathrm{rel}}=0.12$ rejects isotropic sensor noise ($\rho\approx 0.027$) without blocking 6-DoF gripper-and-object displacements ($\rho\ge 0.20$). Because $u_k\perp\operatorname{span}(u_0,\dots,u_{k-1})$ and $(u_k,g_k)$ never mutate once committed, earlier coordinates suffer zero retroactive drift, high-contrast images ($\lVert r\rVert_2\approx 4$) stay near norm $s_f\approx 2.1$, and millimetre displacements ($\lVert r\rVert_2\approx 0.08$) gain $\sim 6.25\times$ SNR over $\sigma=0.025$.

**Dual-Regime Semiparametric RLS, Spike-and-Slab, and Nadaraya-Watson Residual Prior**

$$
\begin{aligned}\omega_b(c)&=\min(V_b,4)\exp\!\left(-\frac{\lVert c-k_b^{\mathrm{ctx}}\rVert_2^2}{2h_{\mathrm{nw}}^2}\right),\qquad \hat r_{\mathrm{nw}}(c)=\frac{\sum_{b=0}^{B_t-1}\omega_b(c)\big(\bar z_b-W_t\bar k_b^{\mathrm{ctx}}\big)}{\kappa_{\mathrm{nw}}+\sum_{b=0}^{B_t-1}\omega_b(c)},\\\hat z_{\mathrm{base}}(c)&=\hat z_{\mathrm{ctx}}(c)+\omega_{\mathrm{nw}}(1-g_{\mathrm{disc},t})\hat r_{\mathrm{nw}}(c),\qquad z_i^{\mathrm{cond}}(c)=\hat z_{\mathrm{base}}(c)+\gamma_t(c)\big(\hat\mu_{q(i,b^*)}-\hat z_{\mathrm{ctx}}(k_{b^*}^{\mathrm{ctx}})\big)+\epsilon_i.\end{aligned}
$$

Each context bucket maintains its exact unquantized latent mean $\bar z_b\in\mathbb R^{d_z}$ alongside discrete prototype counts. On continuous manifolds ($g_{\mathrm{disc},t}\to 0$), $(1-g_{\mathrm{disc},t})\hat r_{\mathrm{nw}}(c)$ interpolates piecewise-nonlinear contact, grasp, and friction residuals near visited states while shrinking via $\kappa_{\mathrm{nw}}=0.06$ to global linear RLS far out-of-distribution; on discrete multimodal tasks ($g_{\mathrm{disc},t}\to 1$), exact empirical branch quantiles remain untouched.

**Fine-Salience Unnormalized Causal Reader**

$$
\begin{aligned}s_t&=\mathbb I\!\left[\operatorname{rms}(\bar x_t-\operatorname{mean}(\bar x_t)\mathbf 1_{64})\ge\tau_{\mathrm{con}}\right],\qquad k_t=s_t\,g_k W_k\tilde x_t,\qquad f_t=k_t+\tfrac12(k_t-k_{t-1})+(1-s_t)r_{t-1}.\end{aligned}
$$

Sparse 4-pixel-out-of-64 robotics action frames carry RMS contrast $\frac14\lVert 0.125\,a\rVert_2$, which falls below $\tau_{\mathrm{con}}=0.03$ whenever $\lVert a\rVert_2<0.96$ during precision grasp and hold phases. Lowering $\tau_{\mathrm{con}}$ from $0.03$ (`online-subspace-prior`) to $0.002$ (`online-subspace-prior-fine-salience`) preserves fine control actions down to $\lVert a\rVert_2\approx 0.064$ while still gating exact zero-contrast uniform distractors on delayed-cue memory streams.

**Critic-Calmed Single-Arrival Adversarial and Markov-Bottlenecked Posterior Objectives**

$$
\begin{aligned}\mathcal L_D&=\operatorname{softplus}(-D(x_t,c_{t-1}))+\operatorname{softplus}(D(\hat x_D,c_{t-1}))+\frac{\lambda}{2}\Big[(\lVert g_r\rVert_2-\kappa)_+^2+(\lVert g_f\rVert_2-\kappa)_+^2\Big],\\\mathcal L_G&=\operatorname{softplus}(-D(\hat x_G,c_{t-1}))+\lambda_{\mathrm{rec}}\,\operatorname{BCE}(\hat x_G,x_t)+\lambda_{\mathrm{ctx}}\,\lVert W_{\mathrm{ctx}}\rVert_F^2.\end{aligned}
$$

Pairing critic step-size calming (`d_lr_mult=0.02`), stronger BCE posterior anchoring ($\lambda_{\mathrm{rec}}=100$, $\lambda_{\mathrm{ctx}}=5$), and causal Adam first-moment smoothing ($\beta_1=0.4$, $\eta=0.007$) prevents single-arrival critic gradients and shared convolutional bias oscillations from perturbing previously acquired modes without replaying historical frames.

<a id="simplified-pseudocode"></a>
## Simplified pseudocode

```text
Initialize fallback orthogonal encoder E_0, empty Gram-Schmidt subspace (k_0 = 0), float64 RLS (W_0 = 0, P_0 = I / lambda_rls), and empty prototype/context tables.
For each chronological single-arrival step t (batch_size = 1, replay = none):
  Before revealing x_t, serve y_hat = G(z_cond(i_serve, c_{t-1}), c_{t-1}) under an isolated eval RNG.
  Reveal x_t; compute two-pass Gram-Schmidt residual r_t against U_{k_{t-1}} and stage (u_cand, g_cand) if ||r_t||_2 > tau_abs and ||r_t||_2 / (||v_t||_2 + 1e-3) > tau_rel.
  Encode current target phi_t(x_t) using committed rows, any staged innovation row, and trailing JL rows E_0[K_sub:d_z].
  Draw uniform row i_D; form z_D from RLS + spike-and-slab + (1 - g_disc) * r_nw(c_{t-1}) + gamma_t * quantile_residual + eps_{i_D}; update D at lr * 0.02 with BCAP cap.
  Draw uniform row i_G; form posterior z_G = phi_t(x_t) + eps_{i_G}; update G with non-saturating GAN + 100 * BCE + 5 * ||W_ctx||_F^2 using Adam (beta_1 = 0.4).
  After both optimizer steps succeed, commit any staged (u_cand, g_cand), update prototypes (<=32), context buckets and unquantized means z_bar_b (<=64), and rank-1 float64 RLS.
  Advance detached causal state c_t = CausalStateMemory.write(x_t) without storing or replaying frames.
```

<a id="training-details"></a>
## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Decoupled non-saturating logistic adversarial loss for $D$ and $G$, paired on $G$ with pixelwise BCE posterior reconstruction ($\lambda_{\mathrm{rec}}=100$) and Frobenius Markov-bottleneck regularization on direct generator context weights ($\lambda_{\mathrm{ctx}}=5$). |
| Optimizer | Native Adam via `bcap_adam` with causal first-moment smoothing ($\beta_1=0.4$, $\beta_2=0.999$). Base `MoGParticlePrior` is frozen (`prior_update="frozen"`); subspace, RLS, and Nadaraya-Watson buffers update analytically without optimizer state. |
| Learning rates and annealing | Generator base LR $0.007$ and calmed critic LR multiplier `d_lr_mult=0.02`, holding constant through `lr_anneal_start=0.55` before cosine-decaying to `lr_floor=0.06` (`network_lr_floor=0.06`). |
| Parameter-gradient clipping | None (`critic_grad_guard=False`). Critic Lipschitz control acts solely through the BCAP input-gradient cap. |
| Critic penalties and anchors | Fixed one-sided squared $L_2$ BCAP cap on real and subspace-prior fake inputs ($\lambda=1.0$, $\kappa=1.0$) every step; no critic EMA anchor. |
| Damping and update guards | No sparse-row latent damping (`latent_damping_max_rate=0.0`) and zero generator-center probes (`probe_law="no_generator_probes_v1"`). |
| Online memory, prior and routing | Detached `CausalStateMemory` ($d_c=16$, `normalize_key=False`, `key_gain=0.5`, `association_weight=0.0`, `contrast_threshold` $0.03$ or $0.002$) paired with `SubspaceResidualConditionalReconstruction` (`router="subspace_residual_conditional"`, `subspace_dims=6`, `subspace_nu0=0.10`, `feature_scale=2.1`, `nw_weight=1.0`, `nw_bandwidth=0.65`, `nw_ridge=0.06`, `replay_policy="none"`, `rehearsal_batch=0`). |
| Training and sampling noise | Zero scheduled critic-input or generator-output additive noise (`0.0`); intrinsic task MoG perturbation $\sigma=0.025$ is preserved on every transformed particle row. |
| Parameter averaging and serving | Live unaveraged generator weights (`ema_decay=0.0`, `serve_average=False`). Pre-arrival serving queries only committed subspace/RLS/NW sufficient statistics and preceding detached context $c_{t-1}$. |

<a id="configuration-differences"></a>
## Configuration differences

- `online-subspace-prior` branches from `online-residual-prior-linear-key`, keeping `CausalStateMemory` (`contrast_threshold=0.03`, `normalize_key=False`, `key_gain=0.5`) completely unchanged while introducing `SubspaceResidualConditionalReconstruction` and calmed single-arrival optimization (`d_lr_mult=0.02`, `betas=[0.4, 0.999]`, `lr=0.007`, `reconstruction_weight=100.0`).
- `online-subspace-prior-fine-salience` is a strict single-factor child of `online-subspace-prior` changing only `memory_policy.contrast_threshold` from `0.03` to `0.002` so sparse 4-of-64-pixel precision manipulation actions are not zeroed out near contact.
- When $d_c=0$ (`seq_minimal_dense4`, `seq_minimal_tokens4`, `seq_img_bars4`, `seq_img_dwell_bars4`, `seq_img_variable_dwell_bars4`, `seq_img_rare_mode_bars4`), `SubspaceResidualConditionalReconstruction` delegates sampling to `FeatureConditionalReconstruction` over the precision-whitened innovation basis, preserving exact marginal empirical quantiles.
- Staged innovation rows (`_pending_basis_row`, `_pending_basis_gain`) commit strictly after `GANTrainer.step` succeeds, guaranteeing exact transactional rollback if an optimizer update raises an exception.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [particlestream/subspace_prior.py](../../../particlestream/subspace_prior.py)
- [particlestream/residual_prior.py](../../../particlestream/residual_prior.py)
- [particlestream/feature_prior.py](../../../particlestream/feature_prior.py)
- [particlestream/memory.py](../../../particlestream/memory.py)
- [particlestream/transport.py](../../../particlestream/transport.py)
- [particlestream/tier1.py](../../../particlestream/tier1.py)
- [configs/forge/ideas/online-subspace-prior.json](../../../configs/forge/ideas/online-subspace-prior.json)
- [configs/forge/ideas/online-subspace-prior-fine-salience.json](../../../configs/forge/ideas/online-subspace-prior-fine-salience.json)

</details>

Each section below describes one exact global recipe. Configuration and parent deltas come from the declarations; results come from Forge's current source-bound receipts. Required and stretch evidence stay separate. A final metric can meet its gate while the sustained verdict fails: every one of the last five checks must pass.

The installed engine is pinned to ParticleGAN/Forge 0.9.7. Older version or replay wording in declaration history describes provenance; the effective configuration below controls execution.

<a id="online-subspace-prior"></a>
## `online-subspace-prior`

[Exact declaration](../../../configs/forge/ideas/online-subspace-prior.json)

Parent: [`online-residual-prior-linear-key`](residualprior-bcap.md#online-residual-prior-linear-key).

Declared changes:

- Precision-whitened anchored Gram-Schmidt innovation subspace (subspace_dims=6, abs_tol=0.008, rel_tol=0.12, nu_0=0.10, feature_scale=2.1) plus 2-row deterministic JL tail eliminates blind 8x64 projection nullspace and 1:1 MoG Wiener attenuation without retroactive coordinate drift
- Semiparametric continuous Nadaraya-Watson kernel-ridge residual (nw_weight=1.0, nw_bandwidth=0.65, nw_ridge=0.06) over unquantized per-bucket latent means gated by (1 - g_disc) captures local nonlinear dynamics alongside discrete quantiles
- Single-arrival critic calming and posterior smoothing (d_lr_mult=0.02, reconstruction_weight=100.0, betas=[0.4, 0.999], lr=0.007, lr_anneal_start=0.55, lr_floor=0.06)

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.transport.feature_scale` | `2.0` | `2.1` |
| `memory_policy.transport.nw_bandwidth` | _unset_ | `0.65` |
| `memory_policy.transport.nw_ridge` | _unset_ | `0.06` |
| `memory_policy.transport.nw_weight` | _unset_ | `1.0` |
| `memory_policy.transport.reconstruction_weight` | `10.0` | `100.0` |
| `memory_policy.transport.router` | `"residual_feature_conditional"` | `"subspace_residual_conditional"` |
| `memory_policy.transport.subspace_abs_tol` | _unset_ | `0.008` |
| `memory_policy.transport.subspace_dims` | _unset_ | `6` |
| `memory_policy.transport.subspace_nu0` | _unset_ | `0.1` |
| `memory_policy.transport.subspace_rel_tol` | _unset_ | `0.12` |
| `recipe_overrides.betas` | `[0.0, 0.999]` | `[0.4, 0.999]` |
| `recipe_overrides.d_lr_mult` | _unset_ | `0.02` |
| `recipe_overrides.lr` | `0.005` | `0.007` |
| `recipe_overrides.lr_anneal_start` | `0.5` | `0.55` |
| `recipe_overrides.lr_floor` | `0.05` | `0.06` |
| `recipe_overrides.network_lr_floor` | `0.05` | `0.06` |

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
      "feature_scale": 2.1,
      "feature_tolerance": 0.15,
      "image_center": 0.5,
      "kernel_bandwidth": 0.35,
      "nw_bandwidth": 0.65,
      "nw_ridge": 0.06,
      "nw_weight": 1.0,
      "outcome_decay": 1.0,
      "reconstruction_loss": "binary_cross_entropy",
      "reconstruction_weight": 100.0,
      "rls_ridge": 0.1,
      "router": "subspace_residual_conditional",
      "slab_scale": 1.0,
      "subspace_abs_tol": 0.008,
      "subspace_dims": 6,
      "subspace_nu0": 0.1,
      "subspace_rel_tol": 0.12,
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
      0.4,
      0.999
    ],
    "d_lr_mult": 0.02,
    "loss": "non_saturating",
    "lr": 0.007,
    "lr_anneal_start": 0.55,
    "lr_floor": 0.06,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.06,
    "prior_update": "frozen"
  },
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **PASS**, 10/10 required passes (10 measured), 4/10 stretch passes (10 measured); qualified tier 1.

Receipt identity: candidate_fingerprint `6bca1c10b4cc64672f8c76da12e9a22377a1b3ea31c4e30a3762b977b74649ad`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-subspace-prior-required"></a>
### Required evidence

<a id="online-subspace-prior-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-subspace-prior/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-subspace-prior_seq_img_round_robin_bars4.gif) | 2026-11-14T04:42:30.719+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-subspace-prior/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-subspace-prior_seq_img_dwell_bars4.gif) | 2026-11-14T04:42:43.310+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.008993`; `direction_acc=1` | [receipt](../receipts/cuda/online-subspace-prior/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-subspace-prior_seq_img_translating_bar_next.gif) | 2026-11-14T04:43:02.129+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-subspace-prior/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-subspace-prior_seq_img_fork_rollout.gif) | 2026-11-14T04:43:20.956+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-subspace-prior/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-subspace-prior_seq_img_variable_dwell_bars4.gif) | 2026-11-14T04:43:33.709+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-subspace-prior/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-subspace-prior_seq_img_rare_mode_bars4.gif) | 2026-11-14T04:43:46.392+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.014743` | [receipt](../receipts/cuda/online-subspace-prior/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-subspace-prior_seq_img_delayed_cue.gif) | 2026-11-14T04:44:05.558+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.09375`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-subspace-prior/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-subspace-prior_seq_img_persistent_fork.gif) | 2026-11-14T04:44:50.734+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.014098`; `xy_rmse=0.003055`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-subspace-prior/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-subspace-prior_seq_img_circle_orbit.gif) | 2026-11-14T04:45:30.463+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.016599`; `s2c_rmse=0.015102`; `c2s_rmse=0.018097`; `xy_rmse=0.005131`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-subspace-prior/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-subspace-prior_seq_img_square_circle_cycle.gif) | 2026-11-14T04:47:01.813+00:00 |

<a id="online-subspace-prior-stretch"></a>
### Stretch evidence

<a id="online-subspace-prior-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-subspace-prior/seq_minimal_dense4.json) · [goal.gif](../gifs/online-subspace-prior_seq_minimal_dense4.gif) | 2026-11-14T04:47:05.684+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-subspace-prior/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-subspace-prior_seq_minimal_tokens4.gif) | 2026-11-14T04:47:09.440+00:00 |

<a id="online-subspace-prior-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.039582`; `seen_max_abs_error=0.11184`; `heldout_response_rmse=0.045592`; `heldout_max_abs_error=0.11319`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-subspace-prior/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-subspace-prior_seq_minimal_kernel2x2.gif) | 2026-11-14T04:47:17.342+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **PASS** | 5/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.98169`; `heldout_exact_accuracy=1`; `heldout_min_bit_confidence=0.98208` | [receipt](../receipts/cuda/online-subspace-prior/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-subspace-prior_seq_minimal_addition2bit.gif) | 2026-11-14T04:47:28.656+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **PASS** | 5/5 | `position_delta_rmse_m=0.0033255`; `relative_mse=0.039351`; `moving_direction_accuracy=0.95833`; `finite_fraction=1` | [receipt](../receipts/cuda/online-subspace-prior/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-subspace-prior_seq_robotics_fetch_reach.gif) | 2026-11-14T04:47:54.581+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011294`; `relative_mse=0.84983`; `moving_direction_accuracy=0.5814`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-subspace-prior/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-subspace-prior_seq_robotics_fetch_push.gif) | 2026-11-14T04:48:11.766+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0054987`; `relative_mse=0.18403`; `moving_direction_accuracy=0.80392`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-subspace-prior/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-subspace-prior_seq_robotics_fetch_slide.gif) | 2026-11-14T04:48:38.912+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **PASS** | 5/5 | `position_delta_rmse_m=0.0032385`; `relative_mse=0.066821`; `moving_direction_accuracy=0.84211`; `finite_fraction=1` | [receipt](../receipts/cuda/online-subspace-prior/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-subspace-prior_seq_robotics_fetch_pick_and_place.gif) | 2026-11-14T04:49:00.630+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0076632`; `relative_mse=0.86861`; `moving_direction_accuracy=0.6729`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-subspace-prior/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-subspace-prior_seq_robotics_hand_reach.gif) | 2026-11-14T04:49:23.482+00:00 |

<a id="online-subspace-prior-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.24778`; `interpolation_speed_error=0.17187`; `interpolation_raw_rmse=0.18251`; `interpolation_point_validity=0.22917`; `interpolation_direction_accuracy=0.58333`; `extrapolation_centroid_rmse=0.27326`; `extrapolation_speed_error=0.30099`; `extrapolation_raw_rmse=0.08688`; `extrapolation_point_validity=0.10417`; `extrapolation_direction_accuracy=0.58333`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_raw_rmse <= 0.12; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-subspace-prior/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-subspace-prior_seq_minimal_velocity_extrapolation.gif) | 2026-11-14T04:49:50.695+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-subspace-prior --device cuda`.

<a id="online-subspace-prior-fine-salience"></a>
## `online-subspace-prior-fine-salience`

[Exact declaration](../../../configs/forge/ideas/online-subspace-prior-fine-salience.json)

Parent: [`online-subspace-prior`](subspaceprior-bcap.md#online-subspace-prior).

Declared changes:

- Lower causal salience contrast threshold (contrast_threshold=0.002) so sparse 4-of-64-pixel precision continuous actions remain salient near contact while exact zero-contrast distractors stay gated

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.contrast_threshold` | `0.03` | `0.002` |

### Effective declaration

```json
{
  "memory_policy": {
    "age_horizon": 8.0,
    "association_weight": 0.0,
    "change_threshold": 0.04,
    "context_scale": 3.0,
    "contrast_threshold": 0.002,
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
      "feature_scale": 2.1,
      "feature_tolerance": 0.15,
      "image_center": 0.5,
      "kernel_bandwidth": 0.35,
      "nw_bandwidth": 0.65,
      "nw_ridge": 0.06,
      "nw_weight": 1.0,
      "outcome_decay": 1.0,
      "reconstruction_loss": "binary_cross_entropy",
      "reconstruction_weight": 100.0,
      "rls_ridge": 0.1,
      "router": "subspace_residual_conditional",
      "slab_scale": 1.0,
      "subspace_abs_tol": 0.008,
      "subspace_dims": 6,
      "subspace_nu0": 0.1,
      "subspace_rel_tol": 0.12,
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
      0.4,
      0.999
    ],
    "d_lr_mult": 0.02,
    "loss": "non_saturating",
    "lr": 0.007,
    "lr_anneal_start": 0.55,
    "lr_floor": 0.06,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.06,
    "prior_update": "frozen"
  },
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **PASS**, 10/10 required passes (10 measured), 4/10 stretch passes (10 measured); qualified tier 1.

Receipt identity: candidate_fingerprint `fe23201e150c1e9f7231077fcd11291f369cb84824629456f432477a42beaa41`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-subspace-prior-fine-salience-required"></a>
### Required evidence

<a id="online-subspace-prior-fine-salience-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-subspace-prior-fine-salience/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-subspace-prior-fine-salience_seq_img_round_robin_bars4.gif) | 2026-11-14T04:50:00.499+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-subspace-prior-fine-salience/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-subspace-prior-fine-salience_seq_img_dwell_bars4.gif) | 2026-11-14T04:50:08.489+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.008993`; `direction_acc=1` | [receipt](../receipts/cuda/online-subspace-prior-fine-salience/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-subspace-prior-fine-salience_seq_img_translating_bar_next.gif) | 2026-11-14T04:50:24.755+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-subspace-prior-fine-salience/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-subspace-prior-fine-salience_seq_img_fork_rollout.gif) | 2026-11-14T04:50:43.008+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-subspace-prior-fine-salience/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-subspace-prior-fine-salience_seq_img_variable_dwell_bars4.gif) | 2026-11-14T04:50:51.908+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-subspace-prior-fine-salience/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-subspace-prior-fine-salience_seq_img_rare_mode_bars4.gif) | 2026-11-14T04:51:00.034+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.014743` | [receipt](../receipts/cuda/online-subspace-prior-fine-salience/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-subspace-prior-fine-salience_seq_img_delayed_cue.gif) | 2026-11-14T04:51:15.316+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.09375`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-subspace-prior-fine-salience/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-subspace-prior-fine-salience_seq_img_persistent_fork.gif) | 2026-11-14T04:52:03.878+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.014098`; `xy_rmse=0.003055`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-subspace-prior-fine-salience/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-subspace-prior-fine-salience_seq_img_circle_orbit.gif) | 2026-11-14T04:52:46.009+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.016599`; `s2c_rmse=0.015102`; `c2s_rmse=0.018097`; `xy_rmse=0.005131`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-subspace-prior-fine-salience/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-subspace-prior-fine-salience_seq_img_square_circle_cycle.gif) | 2026-11-14T04:55:19.965+00:00 |

<a id="online-subspace-prior-fine-salience-stretch"></a>
### Stretch evidence

<a id="online-subspace-prior-fine-salience-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-subspace-prior-fine-salience/seq_minimal_dense4.json) · [goal.gif](../gifs/online-subspace-prior-fine-salience_seq_minimal_dense4.gif) | 2026-11-14T04:55:27.001+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-subspace-prior-fine-salience/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-subspace-prior-fine-salience_seq_minimal_tokens4.gif) | 2026-11-14T04:55:34.214+00:00 |

<a id="online-subspace-prior-fine-salience-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.039582`; `seen_max_abs_error=0.11184`; `heldout_response_rmse=0.045592`; `heldout_max_abs_error=0.11319`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-subspace-prior-fine-salience/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-subspace-prior-fine-salience_seq_minimal_kernel2x2.gif) | 2026-11-14T04:55:54.651+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **PASS** | 5/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.98169`; `heldout_exact_accuracy=1`; `heldout_min_bit_confidence=0.98208` | [receipt](../receipts/cuda/online-subspace-prior-fine-salience/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-subspace-prior-fine-salience_seq_minimal_addition2bit.gif) | 2026-11-14T04:56:14.633+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **PASS** | 5/5 | `position_delta_rmse_m=0.0018584`; `relative_mse=0.012289`; `moving_direction_accuracy=0.97917`; `finite_fraction=1` | [receipt](../receipts/cuda/online-subspace-prior-fine-salience/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-subspace-prior-fine-salience_seq_robotics_fetch_reach.gif) | 2026-11-14T04:57:01.035+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012057`; `relative_mse=0.96863`; `moving_direction_accuracy=0.65116`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-subspace-prior-fine-salience/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-subspace-prior-fine-salience_seq_robotics_fetch_push.gif) | 2026-11-14T04:57:48.561+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0060265`; `relative_mse=0.22105`; `moving_direction_accuracy=0.86275`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-subspace-prior-fine-salience/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-subspace-prior-fine-salience_seq_robotics_fetch_slide.gif) | 2026-11-14T04:58:37.893+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **PASS** | 5/5 | `position_delta_rmse_m=0.0032385`; `relative_mse=0.066821`; `moving_direction_accuracy=0.84211`; `finite_fraction=1` | [receipt](../receipts/cuda/online-subspace-prior-fine-salience/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-subspace-prior-fine-salience_seq_robotics_fetch_pick_and_place.gif) | 2026-11-14T04:59:30.664+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0076632`; `relative_mse=0.86861`; `moving_direction_accuracy=0.6729`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-subspace-prior-fine-salience/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-subspace-prior-fine-salience_seq_robotics_hand_reach.gif) | 2026-11-14T05:00:16.999+00:00 |

<a id="online-subspace-prior-fine-salience-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.24778`; `interpolation_speed_error=0.17187`; `interpolation_raw_rmse=0.18251`; `interpolation_point_validity=0.22917`; `interpolation_direction_accuracy=0.58333`; `extrapolation_centroid_rmse=0.33525`; `extrapolation_speed_error=0.5396`; `extrapolation_raw_rmse=0.08688`; `extrapolation_point_validity=0.10417`; `extrapolation_direction_accuracy=0.54167`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_raw_rmse <= 0.12; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-subspace-prior-fine-salience/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-subspace-prior-fine-salience_seq_minimal_velocity_extrapolation.gif) | 2026-11-14T05:00:37.189+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-subspace-prior-fine-salience --device cuda`.

<a id="refresh"></a>
## Refresh

```sh
particlestream compile --device cuda
```

This page is generated alongside the leaderboard from committed source-bound receipts and declarations; editing a generated page cannot change a verdict or earn qualification.

<a id="references"></a>
## References

- [Numerics of Gram-Schmidt Orthogonalization](https://doi.org/10.1016/0024-3795(94)90493-6). Two-pass classical Gram-Schmidt reorthogonalization guaranteeing machine-precision orthonormality of online innovation basis rows.
- [On Estimating Regression](https://doi.org/10.1137/1109020). Nadaraya-Watson kernel regression estimator applied to unquantized per-bucket RLS residuals on continuous manifolds.
- [Smooth Regression Analysis](https://www.jstor.org/stable/25049340). Local kernel smoothing formulation paired with ridge shrinkage back to global linear RLS far from visited states.
- [Adjustment of an Inverse Matrix Corresponding to a Change in One Element of a Given Matrix](https://doi.org/10.1214/aoms/1177729893). Rank-1 Sherman-Morrison inverse-Gram update underlying the global parametric RLS baseline.
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). Causal first-moment smoothing (beta_1=0.4) stabilizing single-arrival convolutional decoder updates across mode switches.
