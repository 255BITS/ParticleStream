<!-- Generated Forge family report -->

# Global-Stream-BCAP

[← Technique leaderboard](../technique-inventory.md)

<a id="technique-overview"></a>
## Technique overview

Global-Stream-BCAP combines `FastWeightMemory` ($d_c=16$ velocity reader) and the fixed BCAP critic cap with one global `batch_size=1` optimization recipe: decoupled non-saturating logistic loss, an unregularized trainable prior (`prior_regularizer="none"`), faster Adam second-moment tracking ($\beta_2=0.99$), and cosine learning-rate annealing. Its strict online declaration (`online-fast-weight-bcap-ns`) removes the 7-frame reservoir rehearsal batch used by the historical `fast-weight-bcap-ns` recipe.

[Implementation](../../../particlestream/memory.py) · [Mathematical formulation](global-stream-bcap.md#mathematical-formulation) · [Simplified pseudocode](global-stream-bcap.md#simplified-pseudocode) · [Training details](global-stream-bcap.md#training-details) · [Protocol](../../../EXPERIMENTATION.md) · [Terminology](../../../RESEARCH_GLOSSARY.md)

<a id="mathematical-formulation"></a>
## Mathematical formulation

$G$ is the generator, $D$ the critic, and $P$ the trainable $K$-particle MoG prior ($K=32$, $d_z=8$, $\sigma=0.025$) with centers $\{\mu_i\}_{i=0}^{K-1}$. $x_t$ is the single real arrival at step $t\in\{1,\dots,T\}$, $c_{t-1}\in\mathbb R^{16}$ is the detached velocity context from `FastWeightMemory`, $\hat x_D=G(\mu_{i_D}+\sigma\epsilon_D,c_{t-1})$ is the detached critic fake, and $\hat x_G=G(\mu_{i_G}+\sigma\epsilon_G,c_{t-1})$ is the differentiable generator fake. $\eta_0=0.005$ is the base learning rate, $u_t=t/T$ is normalized progress, $u_0=0.5$ is the cosine anneal start, and $f=0.05$ is the LR floor.

**Decoupled Non-Saturating BCAP Objective with Unregularized Prior**

$$
\begin{aligned}\mathcal L_D&=\operatorname{softplus}(-D(x_t,c_{t-1}))+\operatorname{softplus}(D(\hat x_D,c_{t-1}))+\frac{\lambda}{2}\Big[(\lVert g_r\rVert_2-\kappa)_+^2+(\lVert g_f\rVert_2-\kappa)_+^2\Big],\\\mathcal L_G&=\operatorname{softplus}(-D(\hat x_G,c_{t-1})).\end{aligned}
$$

Replacing relativistic pairing with independent real/fake logistic terms avoids single-arrival gradient inversion when $x_t$ and $\hat x_G$ belong to different valid modes, while `prior_regularizer="none"` lets the 32 MoG centers cluster onto discrete templates without repulsive spreading.

**Delayed Cosine Learning-Rate Schedule**

$$
\eta(t)=\begin{cases}\eta_0,&u_t\le u_0,\\[6pt]\eta_0\left(f+\dfrac{1-f}{2}\left[1+\cos\!\left(\pi\dfrac{u_t-u_0}{1-u_0}\right)\right]\right),&u_t>u_0.\end{cases}
$$

Network groups step at $\eta(t)$ and trainable prior centers step at $2\eta(t)$ (`prior_lr_mult=2.0`), holding full plasticity through the first half of the stream before cooling to $5\%$ of $\eta_0$.

<a id="simplified-pseudocode"></a>
## Simplified pseudocode

```text
Initialize G, D, unregularized trainable MoG prior P and detached FastWeightMemory (d_c = 16).
For each chronological single-arrival step t of T:
  Update network LR eta(t) and prior LR 2 * eta(t) from the delayed cosine schedule (u_0 = 0.5, floor = 0.05).
  Serve pre-arrival prediction y_hat = G(z_{i_serve}, c_{t-1}) under isolated eval RNG when predictive.
  Draw uniform i_D and compute detached fake x_hat_D = stop_gradient(G(z_{i_D}, c_{t-1})).
  Backpropagate non-saturating critic loss + fixed BCAP cap (lambda = 1, kappa = 1) on x_t alone; step D.
  With D fixed, draw uniform i_G, compute x_hat_G = G(z_{i_G}, c_{t-1}), and backpropagate softplus(-D(x_hat_G, c_{t-1})).
  Step G and trainable prior row mu_{i_G} without prior repulsion or generator probes.
  Advance detached velocity context c_t = FastWeightMemory.write(x_t) with rehearsal_batch = 0.
```

<a id="training-details"></a>
## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Decoupled non-saturating logistic loss (`loss="non_saturating"`) without auxiliary reconstruction. |
| Optimizer | Native Adam (`bcap_adam`: $\beta_1=0$, $\beta_2=0.99$, `prior_betas=(0, 0.99)`). |
| Learning rates and annealing | Base LR $0.005$, `critic_lr_mult=1.0`, `prior_lr_mult=2.0`, holding until `lr_anneal_start=0.5` then cosine-decaying to `lr_floor=0.05` (`prior_lr_floor=0.05`). |
| Parameter-gradient clipping | None (`critic_grad_guard=False`). |
| Critic penalties and anchors | Fixed real/fake one-sided squared $L_2$ BCAP cap ($\lambda=1.0$, $\kappa=1.0$). |
| Damping and update guards | None (`latent_damping_max_rate=0.0`, `prior_regularizer="none"`). |
| Online memory, prior and routing | Detached `FastWeightMemory` (`dim=16`, `reader="velocity"`), uniform unrouted particle draws. `online-fast-weight-bcap-ns` enforces `replay_policy="none"` and `rehearsal_batch=0` (removing historical `fast-weight-bcap-ns`'s `rehearsal_batch=7`). |
| Training and sampling noise | Zero scheduled critic-input or generator-output additive noise (`0.0`); task MoG $\sigma=0.025$ is preserved. |
| Parameter averaging and serving | Live unaveraged weights (`ema_decay=0.0`, `serve_average=False`). |

<a id="configuration-differences"></a>
## Configuration differences

- Global-Stream-BCAP isolates the exact boundary between replay-assisted streaming and true single-stream online learning: removing the 7-frame FIFO rehearsal batch drops `fast-weight-bcap-ns` to `0/10` required passes (`online-fast-weight-bcap-ns`).
- Without rehearsal, `dwell=16` identical-frame blocks overwrite the critic and pull unrouted trainable particles toward whichever mode currently occupies the stream (`seq_img_dwell_bars4` collapses to 2/4 modes), motivating the subsequent causal-state, context-balanced, and observed-feature prior families.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [particlestream/models.py](../../../particlestream/models.py)
- [particlestream/memory.py](../../../particlestream/memory.py)
- [particlestream/online.py](../../../particlestream/online.py)
- [particlestream/tier1.py](../../../particlestream/tier1.py)
- [configs/forge/ideas/online-fast-weight-bcap-ns.json](../../../configs/forge/ideas/online-fast-weight-bcap-ns.json)
- [configs/forge/ideas/fast-weight-bcap-ns.json](../../../configs/forge/ideas/fast-weight-bcap-ns.json)

</details>

Each section below describes one exact global recipe. Configuration and parent deltas come from the declarations; results come from Forge's current source-bound receipts. Required and stretch evidence stay separate. A final metric can meet its gate while the sustained verdict fails: every one of the last five checks must pass.

The installed engine is pinned to ParticleGAN/Forge 0.9.7. Older version or replay wording in declaration history describes provenance; the effective configuration below controls execution.

<a id="online-fast-weight-bcap-ns"></a>
## `online-fast-weight-bcap-ns`

[Exact declaration](../../../configs/forge/ideas/online-fast-weight-bcap-ns.json)

Parent: [`fast-weight-bcap-ns`](global-stream-bcap.md#fast-weight-bcap-ns).

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
    "d_lr_mult": 1.0,
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "prior_lr_mult": 2.0,
    "prior_regularizer": "none",
    "reg_arm": "b_cap",
    "reg_coeff": 1.0,
    "reg_kappa": 1.0
  },
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-fast-weight-bcap-ns-required"></a>
### Required evidence

<a id="online-fast-weight-bcap-ns-tier-1-required"></a>
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

<a id="online-fast-weight-bcap-ns-stretch"></a>
### Stretch evidence

<a id="online-fast-weight-bcap-ns-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-fast-weight-bcap-ns-tier-2-stretch"></a>
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

<a id="online-fast-weight-bcap-ns-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-fast-weight-bcap-ns --device cuda`.

<a id="fast-weight-bcap-ns"></a>
## `fast-weight-bcap-ns`

[Exact declaration](../../../configs/forge/ideas/fast-weight-bcap-ns.json)

Parent: [`fast-weight-bcap`](fastweight-bcap.md#fast-weight-bcap).

Declared changes:

- Non-saturating GAN loss for batch_size=1 stability
- Cosine learning rate annealing (lr=0.005, betas=(0.0, 0.999), lr_schedule=cosine, lr_anneal_start=0.5, lr_floor=0.05)
- 64-frame rehearsal reservoir spanning full dwell=16 4-mode cycle
- Temporal velocity direction memory with static noise gating (||diff|| > 0.35)
- Joint non-linear conditional critic architecture
- Dwell-aware rehearsal replay and task-isolated context conditioning

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.rehearsal_batch` | _unset_ | `7` |
| `recipe_overrides.betas` | _unset_ | `[0.0, 0.999]` |
| `recipe_overrides.d_lr_mult` | _unset_ | `1.0` |
| `recipe_overrides.loss` | _unset_ | `"non_saturating"` |
| `recipe_overrides.lr` | _unset_ | `0.005` |
| `recipe_overrides.lr_anneal_start` | _unset_ | `0.5` |
| `recipe_overrides.lr_floor` | _unset_ | `0.05` |
| `recipe_overrides.lr_schedule` | _unset_ | `"cosine"` |
| `recipe_overrides.network_lr_floor` | _unset_ | `0.05` |
| `recipe_overrides.prior_lr_mult` | _unset_ | `2.0` |
| `recipe_overrides.prior_regularizer` | _unset_ | `"none"` |
| `recipe_overrides.reg_arm` | _unset_ | `"b_cap"` |
| `recipe_overrides.reg_coeff` | _unset_ | `1.0` |
| `recipe_overrides.reg_kappa` | _unset_ | `1.0` |

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
    "d_lr_mult": 1.0,
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "prior_lr_mult": 2.0,
    "prior_regularizer": "none",
    "reg_arm": "b_cap",
    "reg_coeff": 1.0,
    "reg_kappa": 1.0
  },
  "recipe_preset": "bcap_adam"
}
```

**Historical, read-only.** This declaration does not satisfy explicit `replay_policy=none` with `rehearsal_batch=0`; the executor rejects it before touching its saved evidence. It has no current qualification score.

Strict online successors: [`online-fast-weight-bcap-ns`](global-stream-bcap.md#online-fast-weight-bcap-ns).

[Preserved receipt directory](../receipts/cuda/fast-weight-bcap-ns) · [Archived cohorts](../cohorts) · [Research history](../research-notes.md)

<a id="refresh"></a>
## Refresh

```sh
particlestream compile --device cuda
```

This page is generated alongside the leaderboard from committed source-bound receipts and declarations; editing a generated page cannot change a verdict or earn qualification.

<a id="references"></a>
## References

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661). Non-saturating logistic adversarial objective replacing relativistic score pairing.
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983). Cosine learning-rate annealing schedule applied over the second half of the stream horizon.
- [Using Fast Weights to Attend to the Recent Past](https://arxiv.org/abs/1610.06258). Detached fast-weight memory state maintained without backpropagating through time.
