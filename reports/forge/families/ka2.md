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

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `0d327da8115c91e5d323c0162229bb735c39aba5181ec0d92a6a2508f1799e11`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-ka2-required"></a>
### Required evidence

<a id="online-ka2-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.1875`; `modes=3`; `finite_template_tv=0.8125`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9; modes == 4.0 | [receipt](../receipts/cuda/online-ka2/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-ka2_seq_img_round_robin_bars4.gif) | 2026-11-12T07:23:15.700+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0.03125`; `modes=1`; `finite_template_tv=0.96875`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-ka2_seq_img_dwell_bars4.gif) | 2026-11-12T07:23:23.441+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.5004`; `direction_acc=0.28125`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-ka2_seq_img_translating_bar_next.gif) | 2026-11-12T07:23:32.249+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-ka2_seq_img_fork_rollout.gif) | 2026-11-12T07:23:40.802+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0.0625`; `modes=2`; `finite_template_tv=0.9375`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-ka2_seq_img_variable_dwell_bars4.gif) | 2026-11-12T07:23:48.592+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.84375`; `finite_template_tv=0.1875`; `support_modes=3`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-ka2/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-ka2_seq_img_rare_mode_bars4.gif) | 2026-11-12T07:23:56.163+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.5`; `prediction_rmse=0.28821`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-ka2_seq_img_delayed_cue.gif) | 2026-11-12T07:24:05.110+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-ka2_seq_img_persistent_fork.gif) | 2026-11-12T07:24:19.101+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.20061`; `xy_rmse=1.1256`; `step_accuracy=0.24219`; `orbit_completion=0`; `waypoint_modes=2`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-ka2_seq_img_circle_orbit.gif) | 2026-11-12T07:24:34.706+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.20458`; `s2c_rmse=0.20166`; `c2s_rmse=0.20751`; `xy_rmse=0.9925`; `step_accuracy=0`; `cycle_completion=0`; `waypoint_modes=0`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-ka2_seq_img_square_circle_cycle.gif) | 2026-11-12T07:25:02.164+00:00 |

<a id="online-ka2-stretch"></a>
### Stretch evidence

<a id="online-ka2-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-ka2/seq_minimal_dense4.json) · [goal.gif](../gifs/online-ka2_seq_minimal_dense4.gif) | 2026-11-12T07:25:05.580+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-ka2/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-ka2_seq_minimal_tokens4.gif) | 2026-11-12T07:25:08.825+00:00 |

<a id="online-ka2-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.49227`; `seen_max_abs_error=0.99603`; `heldout_response_rmse=0.41912`; `heldout_max_abs_error=0.69603`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-ka2/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-ka2_seq_minimal_kernel2x2.gif) | 2026-11-12T07:25:14.079+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.10417`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0.125`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-ka2/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-ka2_seq_minimal_addition2bit.gif) | 2026-11-12T07:25:19.107+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0164`; `relative_mse=0.95707`; `moving_direction_accuracy=0.60938`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-ka2_seq_robotics_fetch_reach.gif) | 2026-11-12T07:25:30.134+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.01433`; `relative_mse=1.3682`; `moving_direction_accuracy=0.51744`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-ka2_seq_robotics_fetch_push.gif) | 2026-11-12T07:25:41.238+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012317`; `relative_mse=0.92339`; `moving_direction_accuracy=0.68137`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-ka2_seq_robotics_fetch_slide.gif) | 2026-11-12T07:25:52.576+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012233`; `relative_mse=0.95348`; `moving_direction_accuracy=0.66667`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-ka2_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T07:26:03.512+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0092127`; `relative_mse=1.2554`; `moving_direction_accuracy=0.50117`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-ka2_seq_robotics_hand_reach.gif) | 2026-11-12T07:26:14.727+00:00 |

<a id="online-ka2-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.83971`; `interpolation_speed_error=0.89875`; `interpolation_raw_rmse=0.073697`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.5`; `extrapolation_centroid_rmse=0.97613`; `extrapolation_speed_error=0.90967`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.5625`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-ka2/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-ka2_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T07:26:25.683+00:00 |

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

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `e7d672470cb865075a0e31920abecaea78bd0611c2d25e92fc23ceca937b4c13`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-ka2-clean-required"></a>
### Required evidence

<a id="online-ka2-clean-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.375`; `modes=2`; `finite_template_tv=0.625`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-clean/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-ka2-clean_seq_img_round_robin_bars4.gif) | 2026-11-12T07:26:33.625+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-clean/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-ka2-clean_seq_img_dwell_bars4.gif) | 2026-11-12T07:26:40.330+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.26554`; `direction_acc=0.65625`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2-clean/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-ka2-clean_seq_img_translating_bar_next.gif) | 2026-11-12T07:26:48.244+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2-clean/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-ka2-clean_seq_img_fork_rollout.gif) | 2026-11-12T07:26:55.791+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-clean/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-ka2-clean_seq_img_variable_dwell_bars4.gif) | 2026-11-12T07:27:02.705+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.71875`; `finite_template_tv=0.28125`; `support_modes=3`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-ka2-clean/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-ka2-clean_seq_img_rare_mode_bars4.gif) | 2026-11-12T07:27:09.465+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.51562`; `prediction_rmse=0.29736`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2-clean/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-ka2-clean_seq_img_delayed_cue.gif) | 2026-11-12T07:27:17.447+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2-clean/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-ka2-clean_seq_img_persistent_fork.gif) | 2026-11-12T07:27:30.204+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.22958`; `xy_rmse=1.059`; `step_accuracy=0.015625`; `orbit_completion=0`; `waypoint_modes=3`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2-clean/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-ka2-clean_seq_img_circle_orbit.gif) | 2026-11-12T07:27:45.122+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.21195`; `s2c_rmse=0.19485`; `c2s_rmse=0.22906`; `xy_rmse=1.1484`; `step_accuracy=0`; `cycle_completion=0`; `waypoint_modes=0`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2-clean/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-ka2-clean_seq_img_square_circle_cycle.gif) | 2026-11-12T07:28:10.985+00:00 |

<a id="online-ka2-clean-stretch"></a>
### Stretch evidence

<a id="online-ka2-clean-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-ka2-clean/seq_minimal_dense4.json) · [goal.gif](../gifs/online-ka2-clean_seq_minimal_dense4.gif) | 2026-11-12T07:28:14.150+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-ka2-clean/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-ka2-clean_seq_minimal_tokens4.gif) | 2026-11-12T07:28:17.487+00:00 |

<a id="online-ka2-clean-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.54018`; `seen_max_abs_error=1`; `heldout_response_rmse=0.47447`; `heldout_max_abs_error=0.7`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-ka2-clean/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-ka2-clean_seq_minimal_kernel2x2.gif) | 2026-11-12T07:28:22.207+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.083333`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-ka2-clean/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-ka2-clean_seq_minimal_addition2bit.gif) | 2026-11-12T07:28:27.035+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.11946`; `relative_mse=50.784`; `moving_direction_accuracy=0`; `finite_fraction=0`; failed terminal bounds: finite_fraction == 1.0; moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-clean/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-ka2-clean_seq_robotics_fetch_reach.gif) | 2026-11-12T07:28:36.987+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.013094`; `relative_mse=1.1424`; `moving_direction_accuracy=0.62791`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-clean/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-ka2-clean_seq_robotics_fetch_push.gif) | 2026-11-12T07:28:47.401+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012093`; `relative_mse=0.89015`; `moving_direction_accuracy=0.69608`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-clean/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-ka2-clean_seq_robotics_fetch_slide.gif) | 2026-11-12T07:28:57.545+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.013169`; `relative_mse=1.1049`; `moving_direction_accuracy=0.71491`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-clean/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-ka2-clean_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T07:29:07.887+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0091477`; `relative_mse=1.2378`; `moving_direction_accuracy=0.5514`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-clean/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-ka2-clean_seq_robotics_hand_reach.gif) | 2026-11-12T07:29:18.119+00:00 |

<a id="online-ka2-clean-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.79998`; `interpolation_speed_error=1.021`; `interpolation_raw_rmse=0.073697`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.47917`; `extrapolation_centroid_rmse=0.82188`; `extrapolation_speed_error=0.7749`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.5`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-ka2-clean/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-ka2-clean_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T07:29:28.468+00:00 |

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

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `a4e55cc1635ce80f042db8b4ba21457497dd5d9966ab25098e99f27d55ab502b`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-ka2-light-anchor-required"></a>
### Required evidence

<a id="online-ka2-light-anchor-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.375`; `modes=2`; `finite_template_tv=0.625`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_img_round_robin_bars4.gif) | 2026-11-12T07:29:36.324+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_img_dwell_bars4.gif) | 2026-11-12T07:29:43.228+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.26554`; `direction_acc=0.65625`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_img_translating_bar_next.gif) | 2026-11-12T07:29:50.941+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_img_fork_rollout.gif) | 2026-11-12T07:29:58.708+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_img_variable_dwell_bars4.gif) | 2026-11-12T07:30:05.576+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.71875`; `finite_template_tv=0.28125`; `support_modes=3`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_img_rare_mode_bars4.gif) | 2026-11-12T07:30:12.629+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.51562`; `prediction_rmse=0.29736`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_img_delayed_cue.gif) | 2026-11-12T07:30:20.449+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_img_persistent_fork.gif) | 2026-11-12T07:30:33.362+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.22958`; `xy_rmse=1.059`; `step_accuracy=0.015625`; `orbit_completion=0`; `waypoint_modes=3`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_img_circle_orbit.gif) | 2026-11-12T07:30:48.222+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.21195`; `s2c_rmse=0.19485`; `c2s_rmse=0.22906`; `xy_rmse=1.1484`; `step_accuracy=0`; `cycle_completion=0`; `waypoint_modes=0`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_img_square_circle_cycle.gif) | 2026-11-12T07:31:14.186+00:00 |

<a id="online-ka2-light-anchor-stretch"></a>
### Stretch evidence

<a id="online-ka2-light-anchor-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_minimal_dense4.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_minimal_dense4.gif) | 2026-11-12T07:31:17.315+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_minimal_tokens4.gif) | 2026-11-12T07:31:20.456+00:00 |

<a id="online-ka2-light-anchor-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.54018`; `seen_max_abs_error=1`; `heldout_response_rmse=0.47447`; `heldout_max_abs_error=0.7`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_minimal_kernel2x2.gif) | 2026-11-12T07:31:25.381+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.083333`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_minimal_addition2bit.gif) | 2026-11-12T07:31:30.085+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.11946`; `relative_mse=50.784`; `moving_direction_accuracy=0`; `finite_fraction=0`; failed terminal bounds: finite_fraction == 1.0; moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_robotics_fetch_reach.gif) | 2026-11-12T07:31:40.298+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.013094`; `relative_mse=1.1424`; `moving_direction_accuracy=0.62791`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_robotics_fetch_push.gif) | 2026-11-12T07:31:50.468+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012093`; `relative_mse=0.89015`; `moving_direction_accuracy=0.69608`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_robotics_fetch_slide.gif) | 2026-11-12T07:32:00.856+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.013169`; `relative_mse=1.1049`; `moving_direction_accuracy=0.71491`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T07:32:11.151+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0091477`; `relative_mse=1.2378`; `moving_direction_accuracy=0.5514`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_robotics_hand_reach.gif) | 2026-11-12T07:32:21.588+00:00 |

<a id="online-ka2-light-anchor-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.79998`; `interpolation_speed_error=1.021`; `interpolation_raw_rmse=0.073697`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.47917`; `extrapolation_centroid_rmse=0.82188`; `extrapolation_speed_error=0.7749`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.5`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-ka2-light-anchor/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-ka2-light-anchor_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T07:32:31.749+00:00 |

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

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `231f209dd7dbdd3d067596f4c0d0842ed36a2a1135190e924445869f8fab17fa`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-ka2-ready-required"></a>
### Required evidence

<a id="online-ka2-ready-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.71875`; `modes=3`; `finite_template_tv=0.34375`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-ready/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-ka2-ready_seq_img_round_robin_bars4.gif) | 2026-11-12T07:32:40.162+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0.09375`; `modes=1`; `finite_template_tv=0.90625`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-ready/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-ka2-ready_seq_img_dwell_bars4.gif) | 2026-11-12T07:32:47.397+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.16925`; `direction_acc=0.96875`; failed terminal bounds: next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2-ready/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-ka2-ready_seq_img_translating_bar_next.gif) | 2026-11-12T07:32:55.320+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2-ready/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-ka2-ready_seq_img_fork_rollout.gif) | 2026-11-12T07:33:03.468+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0.09375`; `modes=1`; `finite_template_tv=0.90625`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-ready/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-ka2-ready_seq_img_variable_dwell_bars4.gif) | 2026-11-12T07:33:10.670+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.75`; `finite_template_tv=0.25`; `support_modes=3`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-ka2-ready/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-ka2-ready_seq_img_rare_mode_bars4.gif) | 2026-11-12T07:33:18.000+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.51562`; `prediction_rmse=0.29875`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2-ready/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-ka2-ready_seq_img_delayed_cue.gif) | 2026-11-12T07:33:26.136+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0.0625`; `within_state_modes=1`; `branch_tv=0.9375`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2-ready/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-ka2-ready_seq_img_persistent_fork.gif) | 2026-11-12T07:33:39.562+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.22018`; `xy_rmse=0.92888`; `step_accuracy=0.089844`; `orbit_completion=0`; `waypoint_modes=4`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2-ready/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-ka2-ready_seq_img_circle_orbit.gif) | 2026-11-12T07:33:54.686+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.18624`; `s2c_rmse=0.1815`; `c2s_rmse=0.19097`; `xy_rmse=0.92833`; `step_accuracy=0.042969`; `cycle_completion=0`; `waypoint_modes=1`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2-ready/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-ka2-ready_seq_img_square_circle_cycle.gif) | 2026-11-12T07:34:21.402+00:00 |

<a id="online-ka2-ready-stretch"></a>
### Stretch evidence

<a id="online-ka2-ready-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-ka2-ready/seq_minimal_dense4.json) · [goal.gif](../gifs/online-ka2-ready_seq_minimal_dense4.gif) | 2026-11-12T07:34:24.569+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-ka2-ready/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-ka2-ready_seq_minimal_tokens4.gif) | 2026-11-12T07:34:27.879+00:00 |

<a id="online-ka2-ready-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.51551`; `seen_max_abs_error=1`; `heldout_response_rmse=0.44619`; `heldout_max_abs_error=0.7`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-ka2-ready/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-ka2-ready_seq_minimal_kernel2x2.gif) | 2026-11-12T07:34:32.687+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.083333`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-ka2-ready/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-ka2-ready_seq_minimal_addition2bit.gif) | 2026-11-12T07:34:37.605+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.016269`; `relative_mse=0.94183`; `moving_direction_accuracy=0.57292`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-ready/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-ka2-ready_seq_robotics_fetch_reach.gif) | 2026-11-12T07:34:48.045+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.013558`; `relative_mse=1.2247`; `moving_direction_accuracy=0.40698`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-ready/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-ka2-ready_seq_robotics_fetch_push.gif) | 2026-11-12T07:34:58.869+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.014074`; `relative_mse=1.2055`; `moving_direction_accuracy=0.70588`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-ready/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-ka2-ready_seq_robotics_fetch_slide.gif) | 2026-11-12T07:35:09.601+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011935`; `relative_mse=0.90754`; `moving_direction_accuracy=0.74123`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-ready/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-ka2-ready_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T07:35:20.436+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012402`; `relative_mse=2.2752`; `moving_direction_accuracy=0.55257`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-ready/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-ka2-ready_seq_robotics_hand_reach.gif) | 2026-11-12T07:35:30.958+00:00 |

<a id="online-ka2-ready-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.78891`; `interpolation_speed_error=0.98427`; `interpolation_raw_rmse=0.073698`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.5`; `extrapolation_centroid_rmse=0.90545`; `extrapolation_speed_error=0.77463`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.52083`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-ka2-ready/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-ka2-ready_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T07:35:41.779+00:00 |

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

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `0701ee63be1d12ebd448ee4654f76d102e21cb1c75894f255aa8c85f2115f5d9`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-ka2-ready-causal-required"></a>
### Required evidence

<a id="online-ka2-ready-causal-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.71875`; `modes=3`; `finite_template_tv=0.34375`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_img_round_robin_bars4.gif) | 2026-11-12T07:35:50.155+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0.09375`; `modes=1`; `finite_template_tv=0.90625`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_img_dwell_bars4.gif) | 2026-11-12T07:35:57.524+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.28302`; `direction_acc=0.5`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_img_translating_bar_next.gif) | 2026-11-12T07:36:05.551+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_img_fork_rollout.gif) | 2026-11-12T07:36:13.930+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0.09375`; `modes=1`; `finite_template_tv=0.90625`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_img_variable_dwell_bars4.gif) | 2026-11-12T07:36:21.169+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.75`; `finite_template_tv=0.25`; `support_modes=3`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_img_rare_mode_bars4.gif) | 2026-11-12T07:36:28.662+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.8125`; `prediction_rmse=0.23001`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_img_delayed_cue.gif) | 2026-11-12T07:36:36.885+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=1`; `within_state_modes=1`; `branch_tv=0.5`; `trajectory_consistency=1`; failed terminal bounds: branch_tv <= 0.2; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_img_persistent_fork.gif) | 2026-11-12T07:36:51.440+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.21281`; `xy_rmse=1.0293`; `step_accuracy=0.039062`; `orbit_completion=0`; `waypoint_modes=2`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_img_circle_orbit.gif) | 2026-11-12T07:37:07.937+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.19866`; `s2c_rmse=0.20204`; `c2s_rmse=0.19527`; `xy_rmse=0.99382`; `step_accuracy=0.011719`; `cycle_completion=0`; `waypoint_modes=1`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_img_square_circle_cycle.gif) | 2026-11-12T07:37:37.741+00:00 |

<a id="online-ka2-ready-causal-stretch"></a>
### Stretch evidence

<a id="online-ka2-ready-causal-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_minimal_dense4.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_minimal_dense4.gif) | 2026-11-12T07:37:41.204+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_minimal_tokens4.gif) | 2026-11-12T07:37:44.456+00:00 |

<a id="online-ka2-ready-causal-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.32884`; `seen_max_abs_error=0.98452`; `heldout_response_rmse=0.2448`; `heldout_max_abs_error=0.65409`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_minimal_kernel2x2.gif) | 2026-11-12T07:37:49.907+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.083333`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_minimal_addition2bit.gif) | 2026-11-12T07:37:54.946+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010776`; `relative_mse=0.41319`; `moving_direction_accuracy=0.83333`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_robotics_fetch_reach.gif) | 2026-11-12T07:38:06.417+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.012784`; `relative_mse=1.0889`; `moving_direction_accuracy=0.61628`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_robotics_fetch_push.gif) | 2026-11-12T07:38:18.330+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0095461`; `relative_mse=0.55465`; `moving_direction_accuracy=0.68137`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_robotics_fetch_slide.gif) | 2026-11-12T07:38:30.392+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.006747`; `relative_mse=0.29002`; `moving_direction_accuracy=0.6886`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T07:38:42.119+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0096744`; `relative_mse=1.3844`; `moving_direction_accuracy=0.54673`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_robotics_hand_reach.gif) | 2026-11-12T07:38:54.150+00:00 |

<a id="online-ka2-ready-causal-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.83537`; `interpolation_speed_error=0.65705`; `interpolation_raw_rmse=0.073697`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.45833`; `extrapolation_centroid_rmse=0.95494`; `extrapolation_speed_error=0.77428`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.39583`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-ka2-ready-causal/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-ka2-ready-causal_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T07:39:06.081+00:00 |

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

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `1809eab877b64696bef1d478285799865b8bfdb15f3199b4c751d3895588cfe2`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-ka2-stateless-required"></a>
### Required evidence

<a id="online-ka2-stateless-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.8125`; `modes=4`; `finite_template_tv=0.1875`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9 | [receipt](../receipts/cuda/online-ka2-stateless/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-ka2-stateless_seq_img_round_robin_bars4.gif) | 2026-11-12T07:39:14.692+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-stateless/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-ka2-stateless_seq_img_dwell_bars4.gif) | 2026-11-12T07:39:22.566+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.36696`; `direction_acc=0.5`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2-stateless/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-ka2-stateless_seq_img_translating_bar_next.gif) | 2026-11-12T07:39:31.422+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=0.375`; `within_state_modes=1`; `branch_tv=0.625`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2-stateless/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-ka2-stateless_seq_img_fork_rollout.gif) | 2026-11-12T07:39:39.852+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0`; `modes=0`; `finite_template_tv=1`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-stateless/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-ka2-stateless_seq_img_variable_dwell_bars4.gif) | 2026-11-12T07:39:47.934+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.71875`; `finite_template_tv=0.28125`; `support_modes=3`; `rare_mass=0`; `rare_mass=0`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85; rare_mass >= 0.03125; support_modes == 4.0 | [receipt](../receipts/cuda/online-ka2-stateless/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-ka2-stateless_seq_img_rare_mode_bars4.gif) | 2026-11-12T07:39:55.548+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.5`; `prediction_rmse=0.29904`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2-stateless/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-ka2-stateless_seq_img_delayed_cue.gif) | 2026-11-12T07:40:04.551+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0`; `within_state_modes=0`; `branch_tv=1`; `trajectory_consistency=0`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2-stateless/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-ka2-stateless_seq_img_persistent_fork.gif) | 2026-11-12T07:40:14.997+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.26284`; `xy_rmse=1.0355`; `step_accuracy=0.011719`; `orbit_completion=0`; `waypoint_modes=2`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2-stateless/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-ka2-stateless_seq_img_circle_orbit.gif) | 2026-11-12T07:40:27.285+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.2636`; `s2c_rmse=0.25524`; `c2s_rmse=0.27196`; `xy_rmse=1.2435`; `step_accuracy=0`; `cycle_completion=0`; `waypoint_modes=0`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2-stateless/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-ka2-stateless_seq_img_square_circle_cycle.gif) | 2026-11-12T07:40:48.862+00:00 |

<a id="online-ka2-stateless-stretch"></a>
### Stretch evidence

<a id="online-ka2-stateless-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-ka2-stateless/seq_minimal_dense4.json) · [goal.gif](../gifs/online-ka2-stateless_seq_minimal_dense4.gif) | 2026-11-12T07:40:52.379+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-ka2-stateless/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-ka2-stateless_seq_minimal_tokens4.gif) | 2026-11-12T07:40:55.778+00:00 |

<a id="online-ka2-stateless-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.48256`; `seen_max_abs_error=0.99991`; `heldout_response_rmse=0.40767`; `heldout_max_abs_error=0.69991`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-ka2-stateless/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-ka2-stateless_seq_minimal_kernel2x2.gif) | 2026-11-12T07:41:00.891+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.083333`; `seen_min_bit_confidence=1.452e-29`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=1.452e-29`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-ka2-stateless/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-ka2-stateless_seq_minimal_addition2bit.gif) | 2026-11-12T07:41:05.751+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.016972`; `relative_mse=1.025`; `moving_direction_accuracy=0.5`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-stateless/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-ka2-stateless_seq_robotics_fetch_reach.gif) | 2026-11-12T07:41:16.328+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.014519`; `relative_mse=1.4045`; `moving_direction_accuracy=0.5`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-stateless/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-ka2-stateless_seq_robotics_fetch_push.gif) | 2026-11-12T07:41:26.695+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.013528`; `relative_mse=1.1139`; `moving_direction_accuracy=0.80392`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-stateless/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-ka2-stateless_seq_robotics_fetch_slide.gif) | 2026-11-12T07:41:37.370+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.013106`; `relative_mse=1.0944`; `moving_direction_accuracy=0.59649`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-stateless/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-ka2-stateless_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T07:41:48.311+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0092311`; `relative_mse=1.2604`; `moving_direction_accuracy=0.51519`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-stateless/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-ka2-stateless_seq_robotics_hand_reach.gif) | 2026-11-12T07:41:58.996+00:00 |

<a id="online-ka2-stateless-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.85724`; `interpolation_speed_error=0.66188`; `interpolation_raw_rmse=0.073697`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.16667`; `extrapolation_centroid_rmse=0.98437`; `extrapolation_speed_error=0.77438`; `extrapolation_raw_rmse=0.070665`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.16667`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-ka2-stateless/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-ka2-stateless_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T07:42:08.426+00:00 |

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

Current `cuda`, seed 0: **FAIL**, 0/10 required passes (10 measured), 0/10 stretch passes (10 measured); qualified tier 0.

Receipt identity: candidate_fingerprint `a7991f5d609eecac7ac238f2fe544aa4a28ccd78556ffca825ba4b95bafab8be`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-ka2-transport-causal-required"></a>
### Required evidence

<a id="online-ka2-transport-causal-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **FAIL** | 0/5 | `hq=0.59375`; `modes=4`; `finite_template_tv=0.40625`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.9 | [receipt](../receipts/cuda/online-ka2-transport-causal/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-ka2-transport-causal_seq_img_round_robin_bars4.gif) | 2026-11-12T07:42:21.092+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **FAIL** | 0/5 | `hq=0.65625`; `modes=2`; `finite_template_tv=0.46875`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85; modes == 4.0 | [receipt](../receipts/cuda/online-ka2-transport-causal/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-ka2-transport-causal_seq_img_dwell_bars4.gif) | 2026-11-12T07:42:32.920+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **FAIL** | 0/5 | `next_frame_rmse=0.25993`; `direction_acc=0.60938`; failed terminal bounds: direction_acc >= 0.9; next_frame_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2-transport-causal/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-ka2-transport-causal_seq_img_translating_bar_next.gif) | 2026-11-12T07:42:46.104+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **FAIL** | 0/5 | `hq=0.59375`; `within_state_modes=2`; `branch_tv=0.40625`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85 | [receipt](../receipts/cuda/online-ka2-transport-causal/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-ka2-transport-causal_seq_img_fork_rollout.gif) | 2026-11-12T07:42:59.590+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **FAIL** | 0/5 | `hq=0.40625`; `modes=4`; `finite_template_tv=0.59375`; failed terminal bounds: finite_template_tv <= 0.2; hq >= 0.85 | [receipt](../receipts/cuda/online-ka2-transport-causal/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-ka2-transport-causal_seq_img_variable_dwell_bars4.gif) | 2026-11-12T07:43:11.569+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **FAIL** | 0/5 | `hq=0.5625`; `finite_template_tv=0.4375`; `support_modes=4`; `rare_mass=0.03125`; `rare_mass=0.03125`; failed terminal bounds: finite_template_tv <= 0.15; hq >= 0.85 | [receipt](../receipts/cuda/online-ka2-transport-causal/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-ka2-transport-causal_seq_img_rare_mode_bars4.gif) | 2026-11-12T07:43:23.651+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **FAIL** | 0/5 | `outcome_accuracy=0.70312`; `prediction_rmse=0.27468`; failed terminal bounds: outcome_accuracy >= 0.9; prediction_rmse <= 0.12 | [receipt](../receipts/cuda/online-ka2-transport-causal/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-ka2-transport-causal_seq_img_delayed_cue.gif) | 2026-11-12T07:43:36.736+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **FAIL** | 0/5 | `hq=0.5625`; `within_state_modes=1`; `branch_tv=0.46875`; `trajectory_consistency=0.34375`; failed terminal bounds: branch_tv <= 0.2; hq >= 0.85; trajectory_consistency >= 0.95; within_state_modes == 2.0 | [receipt](../receipts/cuda/online-ka2-transport-causal/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-ka2-transport-causal_seq_img_persistent_fork.gif) | 2026-11-12T07:43:56.546+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **FAIL** | 0/5 | `orbit_rmse=0.15167`; `xy_rmse=0.52253`; `step_accuracy=0.36328`; `orbit_completion=0.03125`; `waypoint_modes=8`; failed terminal bounds: orbit_completion >= 0.9; orbit_rmse <= 0.12; step_accuracy >= 0.9; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2-transport-causal/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-ka2-transport-causal_seq_img_circle_orbit.gif) | 2026-11-12T07:44:18.071+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **FAIL** | 0/5 | `cycle_rmse=0.14255`; `s2c_rmse=0.12912`; `c2s_rmse=0.15599`; `xy_rmse=0.69742`; `step_accuracy=0.070312`; `cycle_completion=0`; `waypoint_modes=6`; failed terminal bounds: c2s_rmse <= 0.12; cycle_completion >= 0.9; cycle_rmse <= 0.12; s2c_rmse <= 0.12; step_accuracy >= 0.9; waypoint_modes == 8.0; xy_rmse <= 0.15 | [receipt](../receipts/cuda/online-ka2-transport-causal/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-ka2-transport-causal_seq_img_square_circle_cycle.gif) | 2026-11-12T07:44:57.945+00:00 |

<a id="online-ka2-transport-causal-stretch"></a>
### Stretch evidence

<a id="online-ka2-transport-causal-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-ka2-transport-causal/seq_minimal_dense4.json) · [goal.gif](../gifs/online-ka2-transport-causal_seq_minimal_dense4.gif) | 2026-11-12T07:45:02.102+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **FAIL** | 0/5 | `valid_fraction=0.09375`; `modes=3`; `mass_tv=0.90625`; failed terminal bounds: mass_tv <= 0.1; modes == 4.0; valid_fraction == 1.0 | [receipt](../receipts/cuda/online-ka2-transport-causal/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-ka2-transport-causal_seq_minimal_tokens4.gif) | 2026-11-12T07:45:06.589+00:00 |

<a id="online-ka2-transport-causal-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.36966`; `seen_max_abs_error=0.96026`; `heldout_response_rmse=0.28065`; `heldout_max_abs_error=0.65361`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-ka2-transport-causal/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-ka2-transport-causal_seq_minimal_kernel2x2.gif) | 2026-11-12T07:45:13.921+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=0.20833`; `seen_min_bit_confidence=0`; `heldout_exact_accuracy=0.125`; `heldout_min_bit_confidence=0`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9; seen_exact_accuracy == 1.0; seen_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-ka2-transport-causal/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-ka2-transport-causal_seq_minimal_addition2bit.gif) | 2026-11-12T07:45:21.159+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0094179`; `relative_mse=0.31562`; `moving_direction_accuracy=0.84896`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-ka2-transport-causal/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-ka2-transport-causal_seq_robotics_fetch_reach.gif) | 2026-11-12T07:45:37.069+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011078`; `relative_mse=0.81773`; `moving_direction_accuracy=0.69767`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-transport-causal/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-ka2-transport-causal_seq_robotics_fetch_push.gif) | 2026-11-12T07:45:53.413+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0089335`; `relative_mse=0.48575`; `moving_direction_accuracy=0.76471`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-transport-causal/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-ka2-transport-causal_seq_robotics_fetch_slide.gif) | 2026-11-12T07:46:09.723+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0044161`; `relative_mse=0.12425`; `moving_direction_accuracy=0.68421`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75 | [receipt](../receipts/cuda/online-ka2-transport-causal/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-ka2-transport-causal_seq_robotics_fetch_pick_and_place.gif) | 2026-11-12T07:46:26.053+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0086015`; `relative_mse=1.0943`; `moving_direction_accuracy=0.58528`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-ka2-transport-causal/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-ka2-transport-causal_seq_robotics_hand_reach.gif) | 2026-11-12T07:46:41.981+00:00 |

<a id="online-ka2-transport-causal-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.82489`; `interpolation_speed_error=0.65216`; `interpolation_raw_rmse=0.07377`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.5`; `extrapolation_centroid_rmse=0.96154`; `extrapolation_speed_error=0.76758`; `extrapolation_raw_rmse=0.071446`; `extrapolation_point_validity=0`; `extrapolation_direction_accuracy=0.45833`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-ka2-transport-causal/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-ka2-transport-causal_seq_minimal_velocity_extrapolation.gif) | 2026-11-12T07:46:57.350+00:00 |

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
