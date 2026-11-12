<!-- Generated Forge family report -->

# FastWeight-BCAP

[← Technique leaderboard](../technique-inventory.md)

<a id="technique-overview"></a>
## Technique overview

FastWeight-BCAP extends the preset `bcap_adam` trainer with a detached Hebbian fast-weight memory module (`FastWeightMemory`, $d_c=16$) that updates only on real arrivals and conditions both generator and critic without backpropagating through temporal history. Its strict online declaration (`online-fast-weight-bcap`) preserves the complete `bcap_adam` relativistic + VICReg optimization recipe while disabling frame replay.

[Implementation](../../../particlestream/memory.py) · [Mathematical formulation](fastweight-bcap.md#mathematical-formulation) · [Simplified pseudocode](fastweight-bcap.md#simplified-pseudocode) · [Training details](fastweight-bcap.md#training-details) · [Protocol](../../../EXPERIMENTATION.md) · [Terminology](../../../RESEARCH_GLOSSARY.md)

<a id="mathematical-formulation"></a>
## Mathematical formulation

$G$ is the generator, $D$ the critic, and $P$ the trainable $K$-particle MoG prior ($K=32$, $d_z=8$, $\sigma=0.025$). At step $t$, $x_t\in[0,1]^{1\times 8\times 8}$ is the arriving real frame with flattened vector $\bar x_t=\operatorname{vec}(x_t)\in\mathbb R^{64}$. $W_k,W_v\in\mathbb R^{d_c\times 64}$ ($d_c=16$) are fixed projections inside `FastWeightMemory`, $M_t\in\mathbb R^{d_c\times d_c}$ is the detached Hebbian fast-weight matrix with decay $\gamma=0.90$ and write rate $\eta_w=0.35$, and $c_t\in\mathbb R^{d_c}$ is the detached velocity context passed to step $t+1$. $\Delta(x,y)=D(x,c_{t-1})-D(y,c_{t-1})$ is the paired relativistic score difference and $R_{\mathrm{VICReg}}(P)$ is the preset prior regularizer.

**Detached Hebbian Writer and Thresholded Velocity Reader**

$$
\begin{aligned}k_t&=\frac{W_k\bar x_t}{\lVert W_k\bar x_t\rVert_2},\qquad v_t=\frac{W_v\bar x_t}{\lVert W_v\bar x_t\rVert_2},\qquad M_t=\gamma M_{t-1}+\eta_w\,v_t k_{t-1}^\top,\\c_t&=\begin{cases}3.0\,\dfrac{k_t-k_{t-1}}{\lVert k_t-k_{t-1}\rVert_2},&\lVert k_{t-1}\rVert_2>0\land\lVert k_t-k_{t-1}\rVert_2>0.35,\\[6pt]\mathbf 0_{d_c},&\text{otherwise}.\end{cases}\end{aligned}
$$

Only real observations mutate $(M_t,k_{t-1})$ inside `torch.no_grad()`. Because the original `velocity` reader emits the normalized key difference $k_t-k_{t-1}$ without querying $M_t$, it cleanly disambiguates 1-step motion (`translating_bar_next`) but collapses to $\mathbf 0_{d_c}$ across repeated distractors ($k_t=k_{t-1}$).

**Single-Arrival Relativistic BCAP and VICReg Objectives**

$$
\begin{aligned}\mathcal L_D&=\operatorname{softplus}\big(-\Delta(x_t,\hat x_D)\big)+\frac{\lambda}{2}\Big[(\lVert g_r\rVert_2-\kappa)_+^2+(\lVert g_f\rVert_2-\kappa)_+^2\Big],\\\mathcal L_G&=\operatorname{softplus}\big(\Delta(x_t,\hat x_G)\big)+\lambda_{\mathrm{prior}}R_{\mathrm{VICReg}}(P).\end{aligned}
$$

With empty `recipe_overrides`, `online-fast-weight-bcap` inherits `bcap_adam`'s constant-LR relativistic pairing and repulsive `vicreg` prior regularizer over uniform `batch_size=1` draws $\hat x_D,\hat x_G$.

<a id="simplified-pseudocode"></a>
## Simplified pseudocode

```text
Initialize G, D, trainable MoG prior P and detached FastWeightMemory (d_c = 16).
For each chronological single-arrival step t:
  On predictive streams at arrival 0, warm-start c_0 = FastWeightMemory.write(x_0) without stepping G or D.
  Serve pre-arrival prediction y_hat = G(z_{i_serve}, c_{t-1}) under isolated eval RNG.
  Draw uniform i_D and compute detached fake x_hat_D = stop_gradient(G(z_{i_D}, c_{t-1})).
  Backpropagate relativistic critic loss softplus(D(x_hat_D, c_{t-1}) - D(x_t, c_{t-1})) + BCAP cap; step D.
  With D fixed, draw uniform i_G, compute x_hat_G = G(z_{i_G}, c_{t-1}), and evaluate relativistic G loss + VICReg prior loss.
  Step G and trainable prior P at constant preset learning rates.
  Update detached M_t and emit thresholded velocity context c_t = FastWeightMemory.write(x_t) (zero replay).
```

<a id="training-details"></a>
## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Preset paired relativistic logistic loss (`loss="relativistic"`) on raw critic scores. |
| Optimizer | Native Adam (`bcap_adam` preset: $\beta_1=0$, $\beta_2=0.999$). |
| Learning rates and annealing | Constant unannealed `bcap_adam` rates (`lr=0.00425`, `critic_lr_mult=1.0`, `prior_lr_mult=2.0`, `lr_floor=1.0`). |
| Parameter-gradient clipping | None (`critic_grad_guard=False`). |
| Critic penalties and anchors | Fixed real/fake one-sided squared $L_2$ BCAP cap ($\lambda=1.0$, $\kappa=1.0$). |
| Damping and update guards | None (`latent_damping_max_rate=0.0`); `prior_regularizer="vicreg"` spreads trainable latent centers. |
| Online memory, prior and routing | Detached `FastWeightMemory` (`dim=16`, `decay=0.9`, `write_lr=0.35`, `reader="velocity"`), uniform particle sampling, `replay_policy="none"`, `rehearsal_batch=0`. |
| Training and sampling noise | Zero scheduled critic-input or generator-output additive noise; task-declared MoG $\sigma=0.025$ is preserved. |
| Parameter averaging and serving | Live unaveraged weights (`ema_decay=0.0`, `serve_average=False`). |

<a id="configuration-differences"></a>
## Configuration differences

- `online-fast-weight-bcap` passes `seq_img_translating_bar_next` (`1/10` required passes), proving that detached key-difference conditioning alone resolves 1-step interior motion ambiguity without replay.
- On unconditional multi-mode streams (`bars4`, `dwell_bars4`), pairing `batch_size=1` relativistic score subtraction with unannealed updates and repulsive `vicreg` prior regularization prevents stable 4-mode convergence.
- The historical parent `fast-weight-bcap` omitted explicit `replay_policy="none"` fields and remains preserved as read-only provenance.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [particlestream/models.py](../../../particlestream/models.py)
- [particlestream/memory.py](../../../particlestream/memory.py)
- [particlestream/online.py](../../../particlestream/online.py)
- [particlestream/tier1.py](../../../particlestream/tier1.py)
- [configs/forge/ideas/online-fast-weight-bcap.json](../../../configs/forge/ideas/online-fast-weight-bcap.json)

</details>

Each section below describes one exact global recipe. Configuration and parent deltas come from the declarations; results come from Forge's current source-bound receipts. Required and stretch evidence stay separate. A final metric can meet its gate while the sustained verdict fails: every one of the last five checks must pass.

The installed engine is pinned to ParticleGAN/Forge 0.9.7. Older version or replay wording in declaration history describes provenance; the effective configuration below controls execution.

<a id="online-fast-weight-bcap"></a>
## `online-fast-weight-bcap`

[Exact declaration](../../../configs/forge/ideas/online-fast-weight-bcap.json)

Parent: [`fast-weight-bcap`](fastweight-bcap.md#fast-weight-bcap).

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
    "decay": 0.9,
    "dim": 16,
    "kind": "fast_weight",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "write_lr": 0.35
  },
  "recipe_overrides": {},
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-fast-weight-bcap-required"></a>
### Required evidence

<a id="online-fast-weight-bcap-tier-1-required"></a>
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

<a id="online-fast-weight-bcap-stretch"></a>
### Stretch evidence

<a id="online-fast-weight-bcap-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-fast-weight-bcap-tier-2-stretch"></a>
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

<a id="online-fast-weight-bcap-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-fast-weight-bcap --device cuda`.

<a id="fast-weight-bcap"></a>
## `fast-weight-bcap`

[Exact declaration](../../../configs/forge/ideas/fast-weight-bcap.json)

Parent: [`bcap`](bcap.md#bcap).

Declared changes:

- Detached FastWeightMemory writer/reader context + bounded 64-frame reservoir replay on batch_size=1 stream

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.decay` | _unset_ | `0.9` |
| `memory_policy.dim` | _unset_ | `16` |
| `memory_policy.kind` | `"none"` | `"fast_weight"` |
| `memory_policy.reservoir_size` | _unset_ | `64` |
| `memory_policy.write_lr` | _unset_ | `0.35` |

### Effective declaration

```json
{
  "memory_policy": {
    "decay": 0.9,
    "dim": 16,
    "kind": "fast_weight",
    "reservoir_size": 64,
    "write_lr": 0.35
  },
  "recipe_overrides": {},
  "recipe_preset": "bcap_adam"
}
```

**Historical, read-only.** This declaration does not satisfy explicit `replay_policy=none` with `rehearsal_batch=0`; the executor rejects it before touching its saved evidence. It has no current qualification score.

Strict online successors: [`online-fast-weight-bcap`](fastweight-bcap.md#online-fast-weight-bcap).

[Preserved receipt directory](../receipts/cuda/fast-weight-bcap) · [Archived cohorts](../cohorts) · [Research history](../research-notes.md)

<a id="refresh"></a>
## Refresh

```sh
particlestream compile --device cuda
```

This page is generated alongside the leaderboard from committed source-bound receipts and declarations; editing a generated page cannot change a verdict or earn qualification.

<a id="references"></a>
## References

- [Using Fast Weights to Attend to the Recent Past](https://arxiv.org/abs/1610.06258). Detached fast-weight associative state updated online from arriving activations.
- [The relativistic discriminator: a key element missing from standard GAN](https://arxiv.org/abs/1807.00734). Paired relativistic logistic objective inherited from the `bcap_adam` preset.
- [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906). Variance-covariance prior regularizer inherited from the `bcap_adam` preset.
