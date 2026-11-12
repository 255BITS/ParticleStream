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

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-bcap-required"></a>
### Required evidence

<a id="online-bcap-tier-1-required"></a>
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

<a id="online-bcap-stretch"></a>
### Stretch evidence

<a id="online-bcap-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-bcap-tier-2-stretch"></a>
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

<a id="online-bcap-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

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

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-bcap-adam-required"></a>
### Required evidence

<a id="online-bcap-adam-tier-1-required"></a>
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

<a id="online-bcap-adam-stretch"></a>
### Stretch evidence

<a id="online-bcap-adam-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-bcap-adam-tier-2-stretch"></a>
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

<a id="online-bcap-adam-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

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

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-bcap-adam-causal-required"></a>
### Required evidence

<a id="online-bcap-adam-causal-tier-1-required"></a>
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

<a id="online-bcap-adam-causal-stretch"></a>
### Stretch evidence

<a id="online-bcap-adam-causal-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-bcap-adam-causal-tier-2-stretch"></a>
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

<a id="online-bcap-adam-causal-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bcap-adam-causal --device cuda`.

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

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-bcap-balanced-required"></a>
### Required evidence

<a id="online-bcap-balanced-tier-1-required"></a>
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

<a id="online-bcap-balanced-stretch"></a>
### Stretch evidence

<a id="online-bcap-balanced-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-bcap-balanced-tier-2-stretch"></a>
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

<a id="online-bcap-balanced-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bcap-balanced --device cuda`.

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

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-bcap-balanced-bce-required"></a>
### Required evidence

<a id="online-bcap-balanced-bce-tier-1-required"></a>
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

<a id="online-bcap-balanced-bce-stretch"></a>
### Stretch evidence

<a id="online-bcap-balanced-bce-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-bcap-balanced-bce-tier-2-stretch"></a>
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

<a id="online-bcap-balanced-bce-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-bcap-balanced-bce --device cuda`.

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

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-bcap-balanced-frozen-required"></a>
### Required evidence

<a id="online-bcap-balanced-frozen-tier-1-required"></a>
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

<a id="online-bcap-balanced-frozen-stretch"></a>
### Stretch evidence

<a id="online-bcap-balanced-frozen-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-bcap-balanced-frozen-tier-2-stretch"></a>
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

<a id="online-bcap-balanced-frozen-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

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

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-bcap-stateless-required"></a>
### Required evidence

<a id="online-bcap-stateless-tier-1-required"></a>
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

<a id="online-bcap-stateless-stretch"></a>
### Stretch evidence

<a id="online-bcap-stateless-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-bcap-stateless-tier-2-stretch"></a>
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

<a id="online-bcap-stateless-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

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

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-bcap-transport-required"></a>
### Required evidence

<a id="online-bcap-transport-tier-1-required"></a>
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

<a id="online-bcap-transport-stretch"></a>
### Stretch evidence

<a id="online-bcap-transport-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-bcap-transport-tier-2-stretch"></a>
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

<a id="online-bcap-transport-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

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

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-bcap-transport-causal-required"></a>
### Required evidence

<a id="online-bcap-transport-causal-tier-1-required"></a>
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

<a id="online-bcap-transport-causal-stretch"></a>
### Stretch evidence

<a id="online-bcap-transport-causal-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-bcap-transport-causal-tier-2-stretch"></a>
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

<a id="online-bcap-transport-causal-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

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

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-bcap-transport-fit-required"></a>
### Required evidence

<a id="online-bcap-transport-fit-tier-1-required"></a>
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

<a id="online-bcap-transport-fit-stretch"></a>
### Stretch evidence

<a id="online-bcap-transport-fit-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-bcap-transport-fit-tier-2-stretch"></a>
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

<a id="online-bcap-transport-fit-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

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
