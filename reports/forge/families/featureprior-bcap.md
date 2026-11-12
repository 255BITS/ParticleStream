<!-- Generated Forge family report -->

# FeaturePrior-BCAP

[← Technique leaderboard](../technique-inventory.md)

<a id="technique-overview"></a>
## Technique overview

FeaturePrior-BCAP pairs the fixed BCAP critic input-gradient cap and a detached causal state reader with a bounded observed-feature conditional prior. A fixed deterministic orthogonal 8×64 encoder projects each arriving 1×8×8 frame into latent space, updating at most 32 online feature prototypes and 64 context-outcome histograms only after successful optimizer steps. The generator trains its unchanged decoder via a weight-10 BCE current-observation posterior while freezing the storage prior; critic updates and causal pre-arrival serving draw empirical feature quantiles at the nearest observed past context without generator probes or frame replay.

[Implementation](../../../particlestream/feature_prior.py) · [Mathematical formulation](featureprior-bcap.md#mathematical-formulation) · [Simplified pseudocode](featureprior-bcap.md#simplified-pseudocode) · [Training details](featureprior-bcap.md#training-details) · [Protocol](../../../EXPERIMENTATION.md) · [Terminology](../../../RESEARCH_GLOSSARY.md)

<a id="mathematical-formulation"></a>
## Mathematical formulation

$G$ is the task-owned residual generator, $D$ the raw-score convolutional critic, and $P$ the frozen $K$-row `MoGParticlePrior` ($K=32$, $d_z=8$, intrinsic scale $\sigma=0.025$) with base centers $\mu_i^{\mathrm{base}}\in\mathbb R^{d_z}$. At chronological step $t$, $x_t\in[0,1]^{1\times 8\times 8}$ is the single arriving frame, $\bar x_t=\operatorname{vec}(x_t)\in\mathbb R^{64}$, $\tilde x_t=\bar x_t-\operatorname{mean}(\bar x_t)\mathbf 1_{64}$ is the DC-centered vector, and $c_{t-1}\in\mathbb R^{d_c}$ ($d_c=16$) is the detached causal context from preceding arrivals. $W_k,W_v\in\mathbb R^{d_c\times 64}$ are fixed detached reader projections, $s_t\in\{0,1\}$ is the contrast salience gate ($\tau_{\mathrm{con}}=0.03$), $r_t$ is the retained salient key, $a_t\in\{0,\dots,H_{\mathrm{age}}\}$ ($H_{\mathrm{age}}=8$) is the repetition-age counter, and $\omega_{\mathrm{assoc}}$ is the associative read weight ($0$ in `online-feature-prior-bce`). $W_{\mathrm{enc}}\in\mathbb R^{d_z\times 64}$ is a fixed row-normalized deterministic orthogonal projection ($s_f=2$, $c_0=0.5$), $\hat\mu_m\in\mathbb R^{d_z}$ are online feature prototypes ($\tau_f=0.15$), $k_b^{\mathrm{ctx}}\in\mathbb R^{d_c}$ and $N_{b,m}\in\mathbb N_0$ are context keys and outcome counts ($C_{\max}=64$, $\tau_c=0.25$), $(a)_+=\max(a,0)$, and $\operatorname{softplus}(a)=\log(1+e^a)$.

**Detached Causal State Reader**

$$
\begin{aligned}k_t&=s_t\frac{W_k\tilde x_t}{\lVert W_k\tilde x_t\rVert_2},\qquad r_t=s_t k_t+(1-s_t)r_{t-1},\qquad f_t=k_t+\tfrac12(k_t-k_{t-1})+(1-s_t)r_{t-1}+\omega_{\mathrm{assoc}}\tanh(M_t q_t),\\c_{t,0:d_c-1}&=s_{\mathrm{ctx}}\operatorname{clip}(f_{t,0:d_c-1},-1,1),\qquad c_{t,d_c-1}=s_{\mathrm{ctx}}\frac{a_t}{H_{\mathrm{age}}}.\end{aligned}
$$

Low-contrast arrivals ($s_t=0$) preserve the most recent salient key $r_{t-1}$ instead of overwriting context with normalized sensor noise, while the final coordinate encodes bounded repetition age $a_t/H_{\mathrm{age}}$ ($s_{\mathrm{ctx}}=3$). All writes run detached from optimizer autograd.

**Observed-Feature Encoding and Empirical Quantile Prior**

$$
\begin{aligned}\phi(x_t)&=s_f\,W_{\mathrm{enc}}(\bar x_t-c_0\mathbf 1_{64}),\qquad b^*=\arg\min_{0\le b<C_{\mathrm{occ}}}\lVert k_b^{\mathrm{ctx}}-\tilde c_{t-1}\rVert_2,\\m^*(i)&=\min\!\left(K-1,\;\inf\left\{m\ge 0:\frac{\sum_{j=0}^m N_{b^*,j}}{\sum_{j=0}^{K-1}N_{b^*,j}}>\frac{i+0.5}{K}\right\}\right),\\z_i^{\mathrm{cond}}&=\hat\mu_{m^*(i)}+(z_i^{\mathrm{raw}}-\mu_i^{\mathrm{base}}),\qquad z_i^{\mathrm{post}}=\phi(x_t)+(z_i^{\mathrm{raw}}-\mu_i^{\mathrm{base}}).\end{aligned}
$$

Every draw retains its exact task-owned Gaussian perturbation $\epsilon_i=z_i^{\mathrm{raw}}-\mu_i^{\mathrm{base}}$. Critic sampling and live serving map particle row $i$ through the midpoint empirical CDF at the nearest observed context $b^*$ (falling back to $z_i^{\mathrm{raw}}$ before observations exist), whereas the generator training step substitutes the current-observation code $\phi(x_t)$.

**Single-Arrival Adversarial and Posterior Reconstruction Objectives**

$$
\begin{aligned}\mathcal L_D&=\operatorname{softplus}(-D(x_t,c_{t-1}))+\operatorname{softplus}(D(\hat x_D,c_{t-1}))+\frac{\lambda}{2}\Big[(\lVert g_r\rVert_2-\kappa)_+^2+(\lVert g_f\rVert_2-\kappa)_+^2\Big],\\\mathcal L_G&=\operatorname{softplus}(-D(\hat x_G,c_{t-1}))+\lambda_{\mathrm{rec}}\,\operatorname{BCE}(\hat x_G,x_t).\end{aligned}
$$

Here $\hat x_D=G(z_{i_D}^{\mathrm{cond}},c_{t-1})$ is detached for the critic step and $\hat x_G=G(z_{i_G}^{\mathrm{post}},c_{t-1})$ trains the image decoder at $\lambda_{\mathrm{rec}}=10$, $\lambda=1$, and $\kappa=1$ with uniform single-particle draws $i_D,i_G\sim\mathcal U\{0,\dots,K-1\}$.

<a id="simplified-pseudocode"></a>
## Simplified pseudocode

```text
Initialize fixed orthogonal encoder W_enc, empty prototype table (<=32) and context table (<=64).
For each chronological single-arrival step t:
  Before revealing x_t, serve y_hat = G(z_cond(i_serve, c_{t-1}), c_{t-1}) under an isolated eval RNG.
  Reveal single target frame x_t (batch_size = 1) and encode latent feature phi(x_t) = s_f W_enc (x_bar_t - c_0).
  Draw uniform row i_D and transform z_D = z_cond(i_D, c_{t-1}) via empirical quantiles at nearest context b*.
  Evaluate non-saturating critic loss on x_t and stop_gradient(G(z_D, c_{t-1})) plus fixed BCAP penalty; step D.
  With D fixed, draw uniform row i_G and form current-observation posterior code z_G = phi(x_t) + eps_{i_G}.
  Evaluate non-saturating generator loss + 10 * BCE(G(z_G, c_{t-1}), x_t); step G while keeping base prior P frozen.
  After both updates succeed, merge phi(x_t) into nearest prototype mu_hat_m (tol 0.15) and c_{t-1} into context bucket b (tol 0.25, LRU cap 64); increment N_{b, m}.
  Advance detached causal state c_t = CausalStateMemory.write(x_t) without storing or replaying frames.
```

<a id="training-details"></a>
## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Decoupled non-saturating logistic adversarial loss for $D$ and $G$, paired on $G$ with pixelwise binary cross-entropy reconstruction ($\lambda_{\mathrm{rec}}=10$) from the current-observation feature posterior. |
| Optimizer | Native Adam via `bcap_adam` ($\beta_1=0$, $\beta_2=0.99$, `prior_betas=(0, 0.99)`). The base MoG storage table is frozen (`prior_update="frozen"`). |
| Learning rates and annealing | Base LR $0.005$ with `critic_lr_mult=1.0`, `prior_lr_mult=1.0`, holding constant through $50\%$ of the task horizon (`lr_anneal_start=0.5`) before cosine-decaying to $5\%$ (`lr_floor=0.05`). |
| Parameter-gradient clipping | None (`critic_grad_guard=False`). Critic Lipschitz control acts smoothly through the BCAP input-gradient loss penalty. |
| Critic penalties and anchors | Fixed one-sided squared $L_2$ BCAP cap on real and conditional-prior fake inputs ($\lambda=1.0$, $\kappa=1.0$) every update; no EMA critic anchor. |
| Damping and update guards | No A2 sparse-row damping (`latent_damping_max_rate=0.0`) and no generator-center probes (`probe_law="no_generator_probes_v1"`). |
| Online memory, prior and routing | Detached `CausalStateMemory` ($d_c=16$, `association_weight=0.0`, `contrast_threshold=0.03`, `change_threshold=0.04`, `age_horizon=8`) plus `FeatureConditionalReconstruction` (up to $32$ feature means and $64$ LRU context-outcome histograms; `replay_policy="none"`, `rehearsal_batch=0`). |
| Training and sampling noise | Zero scheduled critic-input or generator-output additive noise (`noise_std_start=0.0`, `gen_output_noise_std=0.0`). Intrinsic task MoG perturbation $\sigma=0.025$ is preserved on every transformed row. |
| Parameter averaging and serving | Live unaveraged generator weights (`ema_decay=0.0`, `serve_average=False`). Serving uses only preceding context $c_{t-1}$ and previously committed feature/outcome summaries. |

<a id="configuration-differences"></a>
## Configuration differences

- FeaturePrior-BCAP established the first Tier 1-qualifying global recipe (`online-feature-prior-bce`: 10/10 required passes, 1/10 stretch passes on `single_stream_stability` view 10), serving as the direct parent of ResidualPrior-BCAP.
- Unlike competitive or context-balanced transport, FeaturePrior-BCAP executes zero extra generator forward probes per update: both $D$ and $G$ draw one uniform particle row at `batch_size=1`.
- Feature prototypes and context-outcome counts commit strictly after the single-arrival critic and generator updates complete, preserving causal pre-arrival serving.
- Because unseen contexts fall back to 1-NN lookup over at most 64 observed context keys while $G$ trains on $\phi(x_t)+\epsilon$, compositional held-out extrapolation (`addition2bit`, `kernel2x2`, `velocity_extrapolation`) and high-DoF continuous state-action manifolds (`robotics`) remain open stretch frontiers.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [particlestream/feature_prior.py](../../../particlestream/feature_prior.py)
- [particlestream/memory.py](../../../particlestream/memory.py)
- [particlestream/transport.py](../../../particlestream/transport.py)
- [particlestream/tier1.py](../../../particlestream/tier1.py)
- [configs/forge/ideas/online-feature-prior-bce.json](../../../configs/forge/ideas/online-feature-prior-bce.json)

</details>

Each section below describes one exact global recipe. Configuration and parent deltas come from the declarations; results come from Forge's current source-bound receipts. Required and stretch evidence stay separate. A final metric can meet its gate while the sustained verdict fails: every one of the last five checks must pass.

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
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-feature-prior-bce/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_img_round_robin_bars4.gif) | 2026-11-12T07:19:34.733+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-feature-prior-bce/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_img_dwell_bars4.gif) | 2026-11-12T07:19:42.169+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.008454`; `direction_acc=1` | [receipt](../receipts/cuda/online-feature-prior-bce/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_img_translating_bar_next.gif) | 2026-11-12T07:19:50.333+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-feature-prior-bce/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_img_fork_rollout.gif) | 2026-11-12T07:19:58.664+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-feature-prior-bce/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_img_variable_dwell_bars4.gif) | 2026-11-12T07:20:05.929+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-feature-prior-bce/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_img_rare_mode_bars4.gif) | 2026-11-12T07:20:13.384+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.024176` | [receipt](../receipts/cuda/online-feature-prior-bce/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_img_delayed_cue.gif) | 2026-11-12T07:20:21.547+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.03125`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-feature-prior-bce/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_img_persistent_fork.gif) | 2026-11-12T07:20:38.610+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.012616`; `xy_rmse=0.003612`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-feature-prior-bce/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_img_circle_orbit.gif) | 2026-11-12T07:20:58.250+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.012665`; `s2c_rmse=0.011734`; `c2s_rmse=0.013596`; `xy_rmse=0.003817`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-feature-prior-bce/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_img_square_circle_cycle.gif) | 2026-11-12T07:21:33.125+00:00 |

<a id="online-feature-prior-bce-stretch"></a>
### Stretch evidence

<a id="online-feature-prior-bce-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-feature-prior-bce/seq_minimal_dense4.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_minimal_dense4.gif) | 2026-11-12T07:21:36.648+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-feature-prior-bce/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_minimal_tokens4.gif) | 2026-11-12T07:21:40.333+00:00 |

<a id="online-feature-prior-bce-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.042742`; `seen_max_abs_error=0.16605`; `heldout_response_rmse=0.26142`; `heldout_max_abs_error=0.43249`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-feature-prior-bce/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_minimal_kernel2x2.gif) | 2026-11-12T07:21:45.714+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.97912`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=6.7075e-16`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-feature-prior-bce/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_minimal_addition2bit.gif) | 2026-11-12T07:21:51.312+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0055261`; `relative_mse=0.10866`; `moving_direction_accuracy=0.95833`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-feature-prior-bce/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_robotics_fetch_reach.gif) | 2026-11-12T07:22:03.269+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010941`; `relative_mse=0.79759`; `moving_direction_accuracy=0.74419`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-feature-prior-bce/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_robotics_fetch_push.gif) | 2026-11-12T07:22:15.792+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0076109`; `relative_mse=0.35257`; `moving_direction_accuracy=0.77451`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-feature-prior-bce/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_robotics_fetch_slide.gif) | 2026-11-12T07:22:28.174+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 4/5 | `position_delta_rmse_m=0.0026655`; `relative_mse=0.045265`; `moving_direction_accuracy=0.84211`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75 | [receipt](../receipts/cuda/online-feature-prior-bce/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T07:22:41.046+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.008179`; `relative_mse=0.98949`; `moving_direction_accuracy=0.58879`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-feature-prior-bce/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_robotics_hand_reach.gif) | 2026-11-12T07:22:53.593+00:00 |

<a id="online-feature-prior-bce-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.029967`; `interpolation_speed_error=0.039319`; `interpolation_raw_rmse=0.026119`; `interpolation_point_validity=0.83333`; `interpolation_direction_accuracy=0.91667`; `extrapolation_centroid_rmse=0.083345`; `extrapolation_speed_error=0.074005`; `extrapolation_raw_rmse=0.053561`; `extrapolation_point_validity=0.58333`; `extrapolation_direction_accuracy=1`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-feature-prior-bce/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-feature-prior-bce_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T07:23:06.933+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-feature-prior-bce --device cuda`.

<a id="refresh"></a>
## Refresh

```sh
particlestream compile --device cuda
```

This page is generated alongside the leaderboard from committed source-bound receipts and declarations; editing a generated page cannot change a verdict or earn qualification.

<a id="references"></a>
## References

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661). Non-saturating logistic adversarial objective paired with the conditional feature posterior.
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). First-order moment optimizer used for generator and critic updates.
- [On the regularization of Wasserstein GANs](https://arxiv.org/abs/1709.08894). Related one-sided squared hinge gradient penalty; BCAP evaluates the cap directly at real and generated endpoints.
- [Using Fast Weights to Attend to the Recent Past](https://arxiv.org/abs/1610.06258). Inspiration for bounded detached fast-weight state maintained alongside slow network weights.
