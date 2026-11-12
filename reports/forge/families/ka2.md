<!-- Generated Forge family report -->

# KA2

[← Technique leaderboard](../technique-inventory.md)

<a id="technique-overview"></a>
## Technique overview

KA2 pairs a critic-local Adam controller (`KA2CriticAdam`) with a readiness-gated two-phase input-gradient penalty and an adaptive EMA critic-gradient anchor. Before critic second-moment readiness crosses its threshold, KA2 applies an RMS-scaled warm-up penalty; once ready, it blends the warm-up term equally with an unnormalized $L_2$ cap and an EMA real-gradient proximity anchor whose release gate and tracking speed respond to Adam moment surprise.

[Implementation](../../../particlestream/tier1.py) · [Mathematical formulation](ka2.md#mathematical-formulation) · [Simplified pseudocode](ka2.md#simplified-pseudocode) · [Training details](ka2.md#training-details) · [Protocol](../../../EXPERIMENTATION.md) · [Terminology](../../../RESEARCH_GLOSSARY.md)

<a id="mathematical-formulation"></a>
## Mathematical formulation

$G$ is the generator, $D$ the live critic with parameters $\theta_t$, $\bar D$ the critic parameter EMA with parameters $\bar\theta_t$, and $C,\bar C$ denote $D,\bar D$ wrapped with any declared fresh critic-input noise. $x_t$ is the single real arrival, $c_{t-1}$ the detached memory context, and $y_t$ the generated sample (including any scheduled output noise). $g_r=\nabla_x C(x_t,c_{t-1})$ and $g_f=\nabla_y C(y_t,c_{t-1})$ are per-sample input gradients across $d=64$ coordinates; $\lambda$ is penalty strength, $\kappa$ is the cap threshold, and $w$ is configured anchor weight (`reg_anchor_weight`). $R(t)=1-\beta_{2,D}^t$ is critic Adam second-moment readiness against threshold $R_{\mathrm{thresh}}=0.55085\approx 1-0.999^{800}$, $W_t\in\{0,1\}$ is the hysteresis anchor gate driven by short-to-baseline moment-surprise ratio $\rho_t$, and $\alpha_t\in[0,1]$ controls critic EMA decay $\delta_t=1-0.1\alpha_t$.

**Early RMS, Late Capped and EMA Anchor Terms**

$$
\begin{aligned}A&=\frac{\lVert g_r\rVert_2^2}{d}+\left(\frac{\lVert g_f\rVert_2}{\sqrt d}-\kappa\right)_+^2,\\B_{\mathrm{cap}}&=(\lVert g_r\rVert_2-\kappa)_+^2+(\lVert g_f\rVert_2-\kappa)_+^2,\qquad P_{\mathrm{anchor}}=\frac{\lVert g_r-\operatorname{stopgrad}(\nabla_x\bar C(x_t,c_{t-1}))\rVert_2^2}{d}.\end{aligned}
$$

$A$ normalizes gradient magnitudes by $d=64$ ($\sqrt d=8$), whereas $B_{\mathrm{cap}}$ caps raw Euclidean norms directly. On the first blended call, $\bar D$ copies $D$ and $P_{\mathrm{anchor}}=0$; subsequent calls penalize real-gradient deviation from $\bar C$.

**Moment-Readiness Handover**

$$
R_{\mathrm{KA2}}=\begin{cases}\dfrac{\lambda}{2}A,&R(t)<R_{\mathrm{thresh}},\\[8pt]\dfrac{\lambda}{2}\left[\dfrac12 A+\dfrac12\left(B_{\mathrm{cap}}+W_t w P_{\mathrm{anchor}}\right)\right],&R(t)\ge R_{\mathrm{thresh}}.\end{cases}
$$

In `particlegan==0.9.7`, handover triggers when $R(t)=1-\beta_{2,D}^t\ge 0.55085$. At preset $\beta_{2,D}=0.999$, a 480-step task reaches $R(480)\approx 0.381<0.55085$ and stays in warm-up $A$; `online-ka2-ready*` sets $\beta_{2,D}=0.99$ so blending engages at step $t=80$ ($R(80)\approx 0.5525$).

**Surprise-Gated Hysteresis and Adaptive Critic EMA**

$$
W_t=\begin{cases}0,&W_{t-1}=1\land\rho_t>3.0,\\1,&W_{t-1}=0\land\rho_t<1.75,\\W_{t-1},&\text{otherwise},\end{cases}\qquad \delta_t=1-0.1\alpha_t,\qquad \bar\theta_t=\delta_t\bar\theta_{t-1}+(1-\delta_t)\theta_t.
$$

Once 25 blended calls establish baseline Adam moment surprise, $\rho_t>3$ releases the anchor ($W_t=0$) and speeds up $\alpha_t$, while $\rho_t<1.75$ restores $W_t=1$. Sixty consecutive high-surprise steps hard-reseed $\bar\theta_t\leftarrow\theta_t$.

<a id="simplified-pseudocode"></a>
## Simplified pseudocode

```text
Configure KA2CriticAdam, K3PGeneratorAdam, noise schedules and critic moment factor beta_{2,D}.
For each chronological single-arrival step t:
  Serve pre-arrival prediction under isolated eval RNG when task is predictive.
  Draw particle row i_D and form detached fake y_D = stop_gradient(G(z_{i_D}, c_{t-1}) + gen_noise).
  Evaluate input gradients g_r and g_f on noise-wrapped critic C; compute early RMS term A.
  If 1 - beta_{2,D}^t < 0.55085, set R_KA2 = 0.5 * lambda * A.
  Otherwise, initialize or query D_bar for P_anchor, update hysteresis gate W_t from surprise ratio rho_t,
    and set R_KA2 = 0.5 * lambda * (0.5 * A + 0.5 * (B_cap + W_t * w * P_anchor)).
  Backpropagate critic loss + R_KA2, apply spike guard after warmup, step D, and update/reseed D_bar from moment surprise.
  With D fixed, draw i_G (uniform or routed), recompute y_G, apply A2 sparse-row damping if enabled, step G/P, and write c_t.
```

<a id="training-details"></a>
## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Paired relativistic logistic in preset `online-ka2`; decoupled non-saturating logistic in `online-ka2-clean`, `online-ka2-light-anchor`, `online-ka2-ready*`, and `online-ka2-transport-causal`. |
| Optimizer | `KA2CriticAdam` and `K3PGeneratorAdam` ($\beta_1=0$; $\beta_{2,D}=0.999$ in `online-ka2*` vs $0.99$ in `online-ka2-ready*` and `online-ka2-transport-causal`). |
| Learning rates and annealing | Preset `online-ka2` uses `lr=0.00425` (`lr_anneal_start=0.6`, `lr_floor=0.01`); clean/ready online variants use `lr=0.005` (`lr_anneal_start=0.5`, `lr_floor=0.05`, `prior_lr_mult=2.0` or `1.0`). |
| Parameter-gradient clipping | Adaptive per-tensor critic spike guard (`critic_grad_guard=True` in preset, disabled in `online-ka2-ready*`) limits tensor RMS to $5\times$ bias-corrected second-moment RMS after 200 steps. |
| Critic penalties and anchors | Readiness-gated blend of early RMS penalty $A$ and late $L_2$ cap $B_{\mathrm{cap}}$ plus surprise-gated EMA gradient anchor ($w=1.0$ default, $w=0.1$ in `light-anchor`, $w=0.25$ in `ready*`). |
| Damping and update guards | A2 sparse latent-row cosine damping (`latent_damping_max_rate=0.5` in preset, disabled in `online-ka2-ready*`). |
| Online memory, prior and routing | `FastWeightMemory` velocity reader (`online-ka2*`), `kind="none"` (`online-ka2-stateless`), or `CausalStateMemory` (`online-ka2-ready-causal`, `online-ka2-transport-causal`); all active recipes enforce `replay_policy="none"` and `rehearsal_batch=0`. |
| Training and sampling noise | Preset `online-ka2` applies decaying critic-input noise ($0.5\to 0$) and generator output noise ($0\to 0.029$); `clean` and `ready*` variants set both to $0.0$ for crisp binary $8\times 8$ gates. |
| Parameter averaging and serving | Critic EMA $\bar D$ adapts in $\delta_t\in[0.9,1.0]$ once blended; generator EMA (`0.995` in preset, `0.0` in `clean`/`ready*`) is not used for serving (`serve_average=False`). |

<a id="configuration-differences"></a>
## Configuration differences

- All measured strict-online KA2 candidates currently score `0/10` required passes (`online-ka2-transport-causal` reaches `1/10` stretch pass on `seq_minimal_tokens4`).
- On 480-step online streams, preset $\beta_{2,D}=0.999$ never reaches blended mode ($R(480)<0.55085$), while $\beta_{2,D}=0.99$ (`online-ka2-ready`) activates the EMA critic anchor across switching regimes where anchoring to stale real-input gradients slows online adaptation.
- Preset additive critic-input noise (`0.5`) and generator output noise (`0.029`) blur sharp binary $8\times 8$ bars below pixel-quality thresholds, motivating zero-noise overrides in `online-ka2-clean` and `online-ka2-ready*`.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [particlestream/tier1.py](../../../particlestream/tier1.py)
- [particlestream/transport.py](../../../particlestream/transport.py)
- [particlestream/memory.py](../../../particlestream/memory.py)
- [configs/forge/ideas/online-ka2.json](../../../configs/forge/ideas/online-ka2.json)
- [configs/forge/ideas/online-ka2-ready.json](../../../configs/forge/ideas/online-ka2-ready.json)
- [configs/forge/ideas/online-ka2-transport-causal.json](../../../configs/forge/ideas/online-ka2-transport-causal.json)

</details>

Each section below describes one exact global recipe. Configuration and parent deltas come from the declarations; results come from Forge's current source-bound receipts. Required and stretch evidence stay separate. A final metric can meet its gate while the sustained verdict fails: every one of the last five checks must pass.

The installed engine is pinned to ParticleGAN/Forge 0.9.7. Older version or replay wording in declaration history describes provenance; the effective configuration below controls execution.

<a id="online-ka2"></a>
## `online-ka2`

[Exact declaration](../../../configs/forge/ideas/online-ka2.json)

Parent: [`stream-ka2`](ka2.md#stream-ka2).

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
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05
  },
  "recipe_preset": "ka2"
}
```

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-ka2-required"></a>
### Required evidence

<a id="online-ka2-tier-1-required"></a>
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

<a id="online-ka2-stretch"></a>
### Stretch evidence

<a id="online-ka2-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-ka2-tier-2-stretch"></a>
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

<a id="online-ka2-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-ka2 --device cuda`.

<a id="online-ka2-clean"></a>
## `online-ka2-clean`

[Exact declaration](../../../configs/forge/ideas/online-ka2-clean.json)

Parent: [`stream-ka2-clean`](ka2.md#stream-ka2-clean).

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
    "d_guard_ratio": 0.0,
    "direct_particle_gain": false,
    "ema_decay": 0.0,
    "input_noise_std": 0.0,
    "latent_damping_max_rate": 0.0,
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "output_noise_std": 0.0,
    "output_noise_warmup": 0.0
  },
  "recipe_preset": "ka2"
}
```

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-ka2-clean-required"></a>
### Required evidence

<a id="online-ka2-clean-tier-1-required"></a>
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

<a id="online-ka2-clean-stretch"></a>
### Stretch evidence

<a id="online-ka2-clean-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-ka2-clean-tier-2-stretch"></a>
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

<a id="online-ka2-clean-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-ka2-clean --device cuda`.

<a id="online-ka2-light-anchor"></a>
## `online-ka2-light-anchor`

[Exact declaration](../../../configs/forge/ideas/online-ka2-light-anchor.json)

Parent: [`stream-ka2-light-anchor`](ka2.md#stream-ka2-light-anchor).

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
    "d_guard_ratio": 0.0,
    "direct_particle_gain": false,
    "ema_decay": 0.0,
    "input_noise_std": 0.0,
    "latent_damping_max_rate": 0.0,
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "output_noise_std": 0.0,
    "output_noise_warmup": 0.0,
    "reg_anchor_weight": 0.1
  },
  "recipe_preset": "ka2"
}
```

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-ka2-light-anchor-required"></a>
### Required evidence

<a id="online-ka2-light-anchor-tier-1-required"></a>
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

<a id="online-ka2-light-anchor-stretch"></a>
### Stretch evidence

<a id="online-ka2-light-anchor-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-ka2-light-anchor-tier-2-stretch"></a>
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

<a id="online-ka2-light-anchor-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-ka2-light-anchor --device cuda`.

<a id="online-ka2-ready"></a>
## `online-ka2-ready`

[Exact declaration](../../../configs/forge/ideas/online-ka2-ready.json)

Parent: [`stream-ka2-ready`](ka2.md#stream-ka2-ready).

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
    "d_betas": [
      0.0,
      0.99
    ],
    "d_guard_ratio": 0.0,
    "direct_particle_gain": false,
    "ema_decay": 0.0,
    "input_noise_std": 0.0,
    "latent_damping_max_rate": 0.0,
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "output_noise_std": 0.0,
    "output_noise_warmup": 0.0
  },
  "recipe_preset": "ka2"
}
```

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-ka2-ready-required"></a>
### Required evidence

<a id="online-ka2-ready-tier-1-required"></a>
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

<a id="online-ka2-ready-stretch"></a>
### Stretch evidence

<a id="online-ka2-ready-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-ka2-ready-tier-2-stretch"></a>
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

<a id="online-ka2-ready-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-ka2-ready --device cuda`.

<a id="online-ka2-ready-causal"></a>
## `online-ka2-ready-causal`

[Exact declaration](../../../configs/forge/ideas/online-ka2-ready-causal.json)

Parent: [`online-ka2-ready`](ka2.md#online-ka2-ready).

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
    "d_betas": [
      0.0,
      0.99
    ],
    "d_guard_ratio": 0.0,
    "direct_particle_gain": false,
    "ema_decay": 0.0,
    "input_noise_std": 0.0,
    "latent_damping_max_rate": 0.0,
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "output_noise_std": 0.0,
    "output_noise_warmup": 0.0
  },
  "recipe_preset": "ka2"
}
```

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-ka2-ready-causal-required"></a>
### Required evidence

<a id="online-ka2-ready-causal-tier-1-required"></a>
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

<a id="online-ka2-ready-causal-stretch"></a>
### Stretch evidence

<a id="online-ka2-ready-causal-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-ka2-ready-causal-tier-2-stretch"></a>
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

<a id="online-ka2-ready-causal-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-ka2-ready-causal --device cuda`.

<a id="online-ka2-stateless"></a>
## `online-ka2-stateless`

[Exact declaration](../../../configs/forge/ideas/online-ka2-stateless.json)

Parent: [`ka2`](ka2.md#ka2).

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
  "recipe_preset": "ka2"
}
```

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-ka2-stateless-required"></a>
### Required evidence

<a id="online-ka2-stateless-tier-1-required"></a>
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

<a id="online-ka2-stateless-stretch"></a>
### Stretch evidence

<a id="online-ka2-stateless-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-ka2-stateless-tier-2-stretch"></a>
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

<a id="online-ka2-stateless-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-ka2-stateless --device cuda`.

<a id="online-ka2-transport-causal"></a>
## `online-ka2-transport-causal`

[Exact declaration](../../../configs/forge/ideas/online-ka2-transport-causal.json)

Parent: [`online-ka2-ready-causal`](ka2.md#online-ka2-ready-causal).

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
    "d_betas": [
      0.0,
      0.99
    ],
    "d_guard_ratio": 0.0,
    "direct_particle_gain": false,
    "ema_decay": 0.0,
    "input_noise_std": 0.0,
    "latent_damping_max_rate": 0.0,
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "output_noise_std": 0.0,
    "output_noise_warmup": 0.0
  },
  "recipe_preset": "ka2"
}
```

Current `cuda`, seed 0: **INCOMPLETE**, 0/10 required passes (0 measured), 0/10 stretch passes (0 measured); qualified tier 0.

<a id="online-ka2-transport-causal-required"></a>
### Required evidence

<a id="online-ka2-transport-causal-tier-1-required"></a>
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

<a id="online-ka2-transport-causal-stretch"></a>
### Stretch evidence

<a id="online-ka2-transport-causal-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

<a id="online-ka2-transport-causal-tier-2-stretch"></a>
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

<a id="online-ka2-transport-causal-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **NOT_RUN** | — | stale receipt: adapter_source_sha256, task_fingerprint differ | — | — |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-ka2-transport-causal --device cuda`.

<a id="ka2"></a>
## `ka2`

[Exact declaration](../../../configs/forge/ideas/ka2.json)

Declared changes:

- KA2 critic EMA anchor on batch_size=1 sequential stream

### Effective declaration

```json
{
  "memory_policy": {
    "kind": "none"
  },
  "recipe_overrides": {},
  "recipe_preset": "ka2"
}
```

**Historical, read-only.** This declaration does not satisfy explicit `replay_policy=none` with `rehearsal_batch=0`; the executor rejects it before touching its saved evidence. It has no current qualification score.

Strict online successors: [`online-ka2-stateless`](ka2.md#online-ka2-stateless).

[Preserved receipt directory](../receipts/cuda/ka2) · [Archived cohorts](../cohorts) · [Research history](../research-notes.md)

<a id="stream-ka2"></a>
## `stream-ka2`

[Exact declaration](../../../configs/forge/ideas/stream-ka2.json)

Declared changes:

- ParticleGAN 0.9.6 public ka2 trainer preset
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
  "recipe_preset": "ka2"
}
```

**Historical, read-only.** This declaration does not satisfy explicit `replay_policy=none` with `rehearsal_batch=0`; the executor rejects it before touching its saved evidence. It has no current qualification score.

Strict online successors: [`online-ka2`](ka2.md#online-ka2).

[Preserved receipt directory](../receipts/cuda/stream-ka2) · [Archived cohorts](../cohorts) · [Research history](../research-notes.md)

<a id="stream-ka2-clean"></a>
## `stream-ka2-clean`

[Exact declaration](../../../configs/forge/ideas/stream-ka2-clean.json)

Parent: [`stream-ka2`](ka2.md#stream-ka2).

Declared changes:

- Disable auxiliary noise, EMA serving, critic guard and latent damping; retain KA2 critic EMA anchor weight=1

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `recipe_overrides.d_guard_ratio` | _unset_ | `0.0` |
| `recipe_overrides.direct_particle_gain` | _unset_ | `false` |
| `recipe_overrides.ema_decay` | _unset_ | `0.0` |
| `recipe_overrides.input_noise_std` | _unset_ | `0.0` |
| `recipe_overrides.latent_damping_max_rate` | _unset_ | `0.0` |
| `recipe_overrides.output_noise_std` | _unset_ | `0.0` |
| `recipe_overrides.output_noise_warmup` | _unset_ | `0.0` |

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
    "d_guard_ratio": 0.0,
    "direct_particle_gain": false,
    "ema_decay": 0.0,
    "input_noise_std": 0.0,
    "latent_damping_max_rate": 0.0,
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "output_noise_std": 0.0,
    "output_noise_warmup": 0.0
  },
  "recipe_preset": "ka2"
}
```

**Historical, read-only.** This declaration does not satisfy explicit `replay_policy=none` with `rehearsal_batch=0`; the executor rejects it before touching its saved evidence. It has no current qualification score.

Strict online successors: [`online-ka2-clean`](ka2.md#online-ka2-clean).

[Preserved receipt directory](../receipts/cuda/stream-ka2-clean) · [Archived cohorts](../cohorts) · [Research history](../research-notes.md)

<a id="stream-ka2-light-anchor"></a>
## `stream-ka2-light-anchor`

[Exact declaration](../../../configs/forge/ideas/stream-ka2-light-anchor.json)

Parent: [`stream-ka2-clean`](ka2.md#stream-ka2-clean).

Declared changes:

- Reduce KA2 EMA anchor weight from 1 to 0.1 globally; retain the KA2 penalty and formulation optimizer

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `recipe_overrides.reg_anchor_weight` | _unset_ | `0.1` |

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
    "d_guard_ratio": 0.0,
    "direct_particle_gain": false,
    "ema_decay": 0.0,
    "input_noise_std": 0.0,
    "latent_damping_max_rate": 0.0,
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "output_noise_std": 0.0,
    "output_noise_warmup": 0.0,
    "reg_anchor_weight": 0.1
  },
  "recipe_preset": "ka2"
}
```

**Historical, read-only.** This declaration does not satisfy explicit `replay_policy=none` with `rehearsal_batch=0`; the executor rejects it before touching its saved evidence. It has no current qualification score.

Strict online successors: [`online-ka2-light-anchor`](ka2.md#online-ka2-light-anchor).

[Preserved receipt directory](../receipts/cuda/stream-ka2-light-anchor) · [Archived cohorts](../cohorts) · [Research history](../research-notes.md)

<a id="stream-ka2-ready"></a>
## `stream-ka2-ready`

[Exact declaration](../../../configs/forge/ideas/stream-ka2-ready.json)

Parent: [`stream-ka2-clean`](ka2.md#stream-ka2-clean).

Declared changes:

- Critic beta2=0.99 (other optimizer betas unchanged) engages KA2 readiness around step 80 within the fixed 480-step budget

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `recipe_overrides.d_betas` | _unset_ | `[0.0, 0.99]` |

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
    "d_betas": [
      0.0,
      0.99
    ],
    "d_guard_ratio": 0.0,
    "direct_particle_gain": false,
    "ema_decay": 0.0,
    "input_noise_std": 0.0,
    "latent_damping_max_rate": 0.0,
    "loss": "non_saturating",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "output_noise_std": 0.0,
    "output_noise_warmup": 0.0
  },
  "recipe_preset": "ka2"
}
```

**Historical, read-only.** This declaration does not satisfy explicit `replay_policy=none` with `rehearsal_batch=0`; the executor rejects it before touching its saved evidence. It has no current qualification score.

Strict online successors: [`online-ka2-ready`](ka2.md#online-ka2-ready).

[Preserved receipt directory](../receipts/cuda/stream-ka2-ready) · [Archived cohorts](../cohorts) · [Research history](../research-notes.md)

<a id="refresh"></a>
## Refresh

```sh
particlestream compile --device cuda
```

This page is generated alongside the leaderboard from committed source-bound receipts and declarations; editing a generated page cannot change a verdict or earn qualification.

<a id="references"></a>
## References

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). Second-moment statistics $v_t$ and bias-correction readiness $1-\beta_2^t$ used by `KA2CriticAdam`.
- [Which Training Methods for GANs do actually Converge?](https://arxiv.org/abs/1801.04406). Zero-centered real-input gradient regularization motivating the early RMS penalty $A$.
- [The relativistic discriminator: a key element missing from standard GAN](https://arxiv.org/abs/1807.00734). Paired relativistic logistic loss used in the default `ka2` preset.
