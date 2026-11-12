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

- GAIL-BCAP forms a five-step single-factor ablation ladder over `online-feature-prior-bce`: `online-gail-bcap-hinge` (10/10 required, 1/10 stretch) isolates Hinge + prioritized BCE survival/sigmoid style bonus + causal return advantage + active policy entropy; `online-gail-bcap-bce` (10/10 required, 1/10 stretch) enables radial Sherman-Morrison associative extrapolation; `online-gail-bcap-momentum` tests $\beta_1=0.5$; `online-gail-bcap-spike-slab` (10/10 required, 3/10 stretch) upgrades associative extrapolation to Mahalanobis spike-and-slab + Markov context bottleneck + linear causal keys; and `online-gail-bcap-snr` (10/10 required, 3/10 stretch) resolves robotics action contrast erasure (`contrast_threshold=0.005`) and posterior latent SNR (`feature_scale=4.0`).
- `online-gail-bcap-snr` and `online-gail-bcap-spike-slab` rank #1 and #2 overall on `single_stream_stability` view 10 (`13/20` total tasks passed: all 10 required Tier 1 tasks plus `seq_minimal_tokens4`, `seq_minimal_addition2bit`, and `seq_robotics_fetch_reach`), with `online-gail-bcap-snr` cutting `seq_robotics_fetch_reach` physical RMSE by 46% (`1.92 mm` at `97.9%` direction accuracy vs `3.53 mm` on `online-residual-prior-linear-key`).
- On `seq_robotics_fetch_pick_and_place`, lowering `contrast_threshold` (`0.03 -> 0.005`) and raising posterior SNR (`feature_scale: 2.0 -> 4.0`) lifts `online-gail-bcap-snr` from `2/5` (`online-gail-bcap-spike-slab`) to `4/5` terminal checks (`2.45 mm` final RMSE, `1.70 mm` live prequential RMSE, `86.0%` peak terminal direction accuracy), missing only pre-cooldown `step=400` (`0.719` vs `0.750`) due to unaveraged `batch_size=1` live-decoder sign jitter around `0.5`.
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

Current `cuda`, seed 0: **PASS**, 10/10 required passes (10 measured), 3/10 stretch passes (10 measured); qualified tier 1.

Receipt identity: candidate_fingerprint `50ab4ca17b2b680a1436cd54833b15defa85c64c55449060db323306ac152727`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-gail-bcap-snr-required"></a>
### Required evidence

<a id="online-gail-bcap-snr-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_img_round_robin_bars4.gif) | 2026-11-12T17:38:01.224+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_img_dwell_bars4.gif) | 2026-11-12T17:38:09.119+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.008297`; `direction_acc=1` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_img_translating_bar_next.gif) | 2026-11-12T17:38:20.105+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_img_fork_rollout.gif) | 2026-11-12T17:38:31.264+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_img_variable_dwell_bars4.gif) | 2026-11-12T17:38:39.233+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_img_rare_mode_bars4.gif) | 2026-11-12T17:38:47.330+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.022543` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_img_delayed_cue.gif) | 2026-11-12T17:38:58.443+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.09375`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_img_persistent_fork.gif) | 2026-11-12T17:39:49.794+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.01578`; `xy_rmse=0.003225`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_img_circle_orbit.gif) | 2026-11-12T17:41:15.622+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.018057`; `s2c_rmse=0.017729`; `c2s_rmse=0.018384`; `xy_rmse=0.006028`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_img_square_circle_cycle.gif) | 2026-11-12T17:44:02.357+00:00 |

<a id="online-gail-bcap-snr-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-snr-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_minimal_dense4.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_minimal_dense4.gif) | 2026-11-12T17:44:08.267+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_minimal_tokens4.gif) | 2026-11-12T17:44:14.616+00:00 |

<a id="online-gail-bcap-snr-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.042176`; `seen_max_abs_error=0.14914`; `heldout_response_rmse=0.040025`; `heldout_max_abs_error=0.093714`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_minimal_kernel2x2.gif) | 2026-11-12T17:44:30.512+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **PASS** | 5/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.96137`; `heldout_exact_accuracy=1`; `heldout_min_bit_confidence=0.96087` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_minimal_addition2bit.gif) | 2026-11-12T17:44:45.574+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **PASS** | 5/5 | `position_delta_rmse_m=0.0019153`; `relative_mse=0.013053`; `moving_direction_accuracy=0.97917`; `finite_fraction=1` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_robotics_fetch_reach.gif) | 2026-11-12T17:45:23.220+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010593`; `relative_mse=0.74765`; `moving_direction_accuracy=0.62791`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_robotics_fetch_push.gif) | 2026-11-12T17:46:06.934+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0069162`; `relative_mse=0.29114`; `moving_direction_accuracy=0.80392`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_robotics_fetch_slide.gif) | 2026-11-12T17:46:42.714+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 4/5 | `position_delta_rmse_m=0.0024542`; `relative_mse=0.038372`; `moving_direction_accuracy=0.78947`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75 | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T17:47:07.997+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0078394`; `relative_mse=0.90903`; `moving_direction_accuracy=0.66822`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_robotics_hand_reach.gif) | 2026-11-12T17:47:33.273+00:00 |

<a id="online-gail-bcap-snr-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.046327`; `interpolation_speed_error=0.050918`; `interpolation_raw_rmse=0.036157`; `interpolation_point_validity=0.16667`; `interpolation_direction_accuracy=0.91667`; `extrapolation_centroid_rmse=0.36643`; `extrapolation_speed_error=0.4341`; `extrapolation_raw_rmse=0.081088`; `extrapolation_point_validity=0.25`; `extrapolation_direction_accuracy=0.66667`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_raw_rmse <= 0.12; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T17:47:52.871+00:00 |

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

Current `cuda`, seed 0: **PASS**, 10/10 required passes (10 measured), 3/10 stretch passes (10 measured); qualified tier 1.

Receipt identity: candidate_fingerprint `bb4ebe57e0f98a9054030c0e09f83523f6a79c2856e10f6472ec0fe6a6308d3a`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-gail-bcap-spike-slab-required"></a>
### Required evidence

<a id="online-gail-bcap-spike-slab-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_img_round_robin_bars4.gif) | 2026-11-12T17:48:01.855+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_img_dwell_bars4.gif) | 2026-11-12T17:48:09.655+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.009513`; `direction_acc=1` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_img_translating_bar_next.gif) | 2026-11-12T17:48:20.459+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_img_fork_rollout.gif) | 2026-11-12T17:48:31.124+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_img_variable_dwell_bars4.gif) | 2026-11-12T17:48:39.024+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_img_rare_mode_bars4.gif) | 2026-11-12T17:48:46.658+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.027101` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_img_delayed_cue.gif) | 2026-11-12T17:48:57.503+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.09375`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_img_persistent_fork.gif) | 2026-11-12T17:49:24.384+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.019609`; `xy_rmse=0.005184`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_img_circle_orbit.gif) | 2026-11-12T17:49:55.399+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.019104`; `s2c_rmse=0.018582`; `c2s_rmse=0.019626`; `xy_rmse=0.005748`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_img_square_circle_cycle.gif) | 2026-11-12T17:50:53.831+00:00 |

<a id="online-gail-bcap-spike-slab-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-spike-slab-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_minimal_dense4.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_minimal_dense4.gif) | 2026-11-12T17:50:57.463+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_minimal_tokens4.gif) | 2026-11-12T17:51:01.338+00:00 |

<a id="online-gail-bcap-spike-slab-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.041768`; `seen_max_abs_error=0.18007`; `heldout_response_rmse=0.035511`; `heldout_max_abs_error=0.083195`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_minimal_kernel2x2.gif) | 2026-11-12T17:51:08.643+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **PASS** | 5/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.95429`; `heldout_exact_accuracy=1`; `heldout_min_bit_confidence=0.9533` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_minimal_addition2bit.gif) | 2026-11-12T17:51:16.091+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **PASS** | 5/5 | `position_delta_rmse_m=0.0034835`; `relative_mse=0.043181`; `moving_direction_accuracy=0.95833`; `finite_fraction=1` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_robotics_fetch_reach.gif) | 2026-11-12T17:51:31.229+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010513`; `relative_mse=0.73647`; `moving_direction_accuracy=0.69767`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_robotics_fetch_push.gif) | 2026-11-12T17:51:48.451+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0074842`; `relative_mse=0.34093`; `moving_direction_accuracy=0.76471`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_robotics_fetch_slide.gif) | 2026-11-12T17:52:06.353+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 2/5 | `position_delta_rmse_m=0.0024022`; `relative_mse=0.036766`; `moving_direction_accuracy=0.73684`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75 | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T17:52:22.844+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0082151`; `relative_mse=0.99823`; `moving_direction_accuracy=0.61682`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_robotics_hand_reach.gif) | 2026-11-12T17:52:38.097+00:00 |

<a id="online-gail-bcap-spike-slab-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.051051`; `interpolation_speed_error=0.056314`; `interpolation_raw_rmse=0.036744`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.83333`; `extrapolation_centroid_rmse=0.26227`; `extrapolation_speed_error=0.24693`; `extrapolation_raw_rmse=0.080967`; `extrapolation_point_validity=0.16667`; `extrapolation_direction_accuracy=0.66667`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T17:53:03.993+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-gail-bcap-spike-slab --device cuda`.

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
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_round_robin_bars4.gif) | 2026-11-12T17:16:20.628+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_dwell_bars4.gif) | 2026-11-12T17:16:34.357+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.006364`; `direction_acc=1` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_translating_bar_next.gif) | 2026-11-12T17:16:52.352+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_fork_rollout.gif) | 2026-11-12T17:17:11.121+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_variable_dwell_bars4.gif) | 2026-11-12T17:17:27.449+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_rare_mode_bars4.gif) | 2026-11-12T17:17:43.648+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.023147` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_delayed_cue.gif) | 2026-11-12T17:18:02.536+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.03125`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_persistent_fork.gif) | 2026-11-12T17:18:39.479+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.011924`; `xy_rmse=0.003893`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_circle_orbit.gif) | 2026-11-12T17:19:15.903+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.012503`; `s2c_rmse=0.011269`; `c2s_rmse=0.013737`; `xy_rmse=0.003717`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_square_circle_cycle.gif) | 2026-11-12T17:20:43.589+00:00 |

<a id="online-gail-bcap-bce-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-bce-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_minimal_dense4.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_minimal_dense4.gif) | 2026-11-12T17:20:50.119+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_minimal_tokens4.gif) | 2026-11-12T17:20:56.845+00:00 |

<a id="online-gail-bcap-bce-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.04189`; `seen_max_abs_error=0.15896`; `heldout_response_rmse=0.086803`; `heldout_max_abs_error=0.18952`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_minimal_kernel2x2.gif) | 2026-11-12T17:21:07.943+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.97076`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=1.1497e-08`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_minimal_addition2bit.gif) | 2026-11-12T17:21:19.492+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0053724`; `relative_mse=0.1027`; `moving_direction_accuracy=0.95833`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_robotics_fetch_reach.gif) | 2026-11-12T17:21:51.047+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011217`; `relative_mse=0.83837`; `moving_direction_accuracy=0.74419`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_robotics_fetch_push.gif) | 2026-11-12T17:22:21.895+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0077145`; `relative_mse=0.36224`; `moving_direction_accuracy=0.80392`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_robotics_fetch_slide.gif) | 2026-11-12T17:22:53.778+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 4/5 | `position_delta_rmse_m=0.0025834`; `relative_mse=0.04252`; `moving_direction_accuracy=0.80702`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T17:23:25.540+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0081327`; `relative_mse=0.97832`; `moving_direction_accuracy=0.58411`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_robotics_hand_reach.gif) | 2026-11-12T17:23:57.347+00:00 |

<a id="online-gail-bcap-bce-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.035795`; `interpolation_speed_error=0.04004`; `interpolation_raw_rmse=0.029099`; `interpolation_point_validity=0.75`; `interpolation_direction_accuracy=0.91667`; `extrapolation_centroid_rmse=0.077977`; `extrapolation_speed_error=0.060972`; `extrapolation_raw_rmse=0.050648`; `extrapolation_point_validity=0.5`; `extrapolation_direction_accuracy=1`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T17:24:30.685+00:00 |

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
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_round_robin_bars4.gif) | 2026-11-12T17:24:49.999+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_dwell_bars4.gif) | 2026-11-12T17:25:08.806+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.007497`; `direction_acc=1` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_translating_bar_next.gif) | 2026-11-12T17:25:28.886+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_fork_rollout.gif) | 2026-11-12T17:25:49.397+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_variable_dwell_bars4.gif) | 2026-11-12T17:26:08.103+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_rare_mode_bars4.gif) | 2026-11-12T17:26:26.420+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.023308` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_delayed_cue.gif) | 2026-11-12T17:26:47.919+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.03125`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_persistent_fork.gif) | 2026-11-12T17:27:34.792+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.011981`; `xy_rmse=0.003659`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_circle_orbit.gif) | 2026-11-12T17:28:29.236+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.011578`; `s2c_rmse=0.01075`; `c2s_rmse=0.012407`; `xy_rmse=0.00359`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_square_circle_cycle.gif) | 2026-11-12T17:30:14.242+00:00 |

<a id="online-gail-bcap-hinge-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-hinge-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_minimal_dense4.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_minimal_dense4.gif) | 2026-11-12T17:30:21.192+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_minimal_tokens4.gif) | 2026-11-12T17:30:27.920+00:00 |

<a id="online-gail-bcap-hinge-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.041775`; `seen_max_abs_error=0.16378`; `heldout_response_rmse=0.26537`; `heldout_max_abs_error=0.43112`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_minimal_kernel2x2.gif) | 2026-11-12T17:30:39.441+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.96474`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=9.6964e-10`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_minimal_addition2bit.gif) | 2026-11-12T17:30:50.712+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.005476`; `relative_mse=0.1067`; `moving_direction_accuracy=0.95833`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_robotics_fetch_reach.gif) | 2026-11-12T17:31:17.422+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.01111`; `relative_mse=0.82236`; `moving_direction_accuracy=0.69767`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_robotics_fetch_push.gif) | 2026-11-12T17:31:32.729+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0075477`; `relative_mse=0.34674`; `moving_direction_accuracy=0.76471`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_robotics_fetch_slide.gif) | 2026-11-12T17:31:57.434+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 1/5 | `position_delta_rmse_m=0.0026558`; `relative_mse=0.044936`; `moving_direction_accuracy=0.73684`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T17:32:22.049+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0081538`; `relative_mse=0.9834`; `moving_direction_accuracy=0.58879`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_robotics_hand_reach.gif) | 2026-11-12T17:32:46.823+00:00 |

<a id="online-gail-bcap-hinge-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.031857`; `interpolation_speed_error=0.039085`; `interpolation_raw_rmse=0.026463`; `interpolation_point_validity=0.83333`; `interpolation_direction_accuracy=0.91667`; `extrapolation_centroid_rmse=0.079113`; `extrapolation_speed_error=0.066138`; `extrapolation_raw_rmse=0.05589`; `extrapolation_point_validity=0.5`; `extrapolation_direction_accuracy=1`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T17:33:12.925+00:00 |

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
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_round_robin_bars4.gif) | 2026-11-12T17:33:29.300+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_dwell_bars4.gif) | 2026-11-12T17:33:44.169+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.00797`; `direction_acc=1` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_translating_bar_next.gif) | 2026-11-12T17:34:00.868+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_fork_rollout.gif) | 2026-11-12T17:34:17.505+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_variable_dwell_bars4.gif) | 2026-11-12T17:34:32.763+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_rare_mode_bars4.gif) | 2026-11-12T17:34:41.905+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.023138` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_delayed_cue.gif) | 2026-11-12T17:34:51.312+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.03125`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_persistent_fork.gif) | 2026-11-12T17:35:09.931+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.013045`; `xy_rmse=0.003758`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_circle_orbit.gif) | 2026-11-12T17:35:31.711+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.012296`; `s2c_rmse=0.011309`; `c2s_rmse=0.013282`; `xy_rmse=0.00323`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_square_circle_cycle.gif) | 2026-11-12T17:36:10.513+00:00 |

<a id="online-gail-bcap-momentum-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-momentum-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_minimal_dense4.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_minimal_dense4.gif) | 2026-11-12T17:36:14.447+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_minimal_tokens4.gif) | 2026-11-12T17:36:18.205+00:00 |

<a id="online-gail-bcap-momentum-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.0353`; `seen_max_abs_error=0.1779`; `heldout_response_rmse=0.0899`; `heldout_max_abs_error=0.23582`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_minimal_kernel2x2.gif) | 2026-11-12T17:36:24.308+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.9756`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=8.2579e-08`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_minimal_addition2bit.gif) | 2026-11-12T17:36:30.131+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0054013`; `relative_mse=0.10381`; `moving_direction_accuracy=0.95833`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_robotics_fetch_reach.gif) | 2026-11-12T17:36:43.408+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010969`; `relative_mse=0.80171`; `moving_direction_accuracy=0.67442`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_robotics_fetch_push.gif) | 2026-11-12T17:36:56.623+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0074193`; `relative_mse=0.33504`; `moving_direction_accuracy=0.7451`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_robotics_fetch_slide.gif) | 2026-11-12T17:37:10.010+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 2/5 | `position_delta_rmse_m=0.0028069`; `relative_mse=0.050196`; `moving_direction_accuracy=0.80702`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T17:37:23.674+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0080597`; `relative_mse=0.96082`; `moving_direction_accuracy=0.6028`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_robotics_hand_reach.gif) | 2026-11-12T17:37:37.178+00:00 |

<a id="online-gail-bcap-momentum-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.031232`; `interpolation_speed_error=0.037349`; `interpolation_raw_rmse=0.025589`; `interpolation_point_validity=0.66667`; `interpolation_direction_accuracy=1`; `extrapolation_centroid_rmse=0.073497`; `extrapolation_speed_error=0.03789`; `extrapolation_raw_rmse=0.053371`; `extrapolation_point_validity=0.58333`; `extrapolation_direction_accuracy=1`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T17:37:51.849+00:00 |

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
- [Bayesian Variable Selection in Linear Regression](https://doi.org/10.1080/01621459.1988.10478693). Spike-and-slab Bayes factor formulation powering Mahalanobis discrete-vs-continuous prototype gating.
