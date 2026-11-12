# ParticleStream research notes

## Stabilized `GAIL-BCAP` Family and 660-Run View 10 Qualification, 2026-11-12

Added the `GAIL-BCAP` technique family (`particlestream/gail_prior.py`,
`GAILConditionalReconstruction`), translating stabilized generative adversarial imitation
control into ParticleGAN's public `PriorManipulator` API under ParticleStream's strict
`batch_size=1` online protocol (`replay_policy="none"`, `rehearsal_batch=0`, `seed=0`,
zero generator probes).

Three global candidate recipes form an explicit single-factor ablation ladder over
`online-feature-prior-bce`:
- **`online-gail-bcap-hinge`** (parent `online-feature-prior-bce`): replaces the non-saturating
  logistic critic with a Hinge margin critic (`loss="hinge"`), increases detached causal-state
  fast-weight `write_lr` (`0.35 -> 0.5`), cancels the default unbounded linear Hinge generator
  slope (`cancel_linear_gan=True`) in favor of prioritized pixelwise BCE survival reconstruction
  minus a bounded sigmoid critic style bonus (`style_bonus_weight=0.1`) modulated by a causal
  post-commit discounted return advantage (`return_discount=0.99`, `advantage_scale=0.25`), and
  draws entropy-regularized categorical quantiles (`policy_entropy_bonus=0.01`) strictly over
  active observed prototypes per context (`assoc_ridge=0.0`, `assoc_max_blend=0.0`).
- **`online-gail-bcap-bce`** (parent `online-gail-bcap-hinge`): enables a detached
  Sherman-Morrison recursive ridge fast-weight operator (`assoc_ridge=0.05`,
  `assoc_forgetting=1.0`, `assoc_blend_scale=0.25`, `assoc_max_blend=1.0`) inside
  `GAILConditionalReconstruction` to smoothly extrapolate radially bounded latent feature
  expectations beyond discrete context tolerance (`d_min > 0.25`) while preserving exact
  discrete multimodal quantiles on observed contexts (`d_min <= 0.25`).
- **`online-gail-bcap-momentum`** (parent `online-gail-bcap-bce`): enables nonzero Adam
  first-moment smoothing (`betas=[0.5, 0.999]`) across sequential `batch_size=1` arrivals.

Fresh source-bound CUDA execution across all **33 executable global recipes × 20 tasks
(660 sequential runs, 20 oracle PASS controls, 45 destructive FAIL controls)** qualifies all
three `GAIL-BCAP` global recipes at **Tier 1 (`PASS`, `10/10` required tasks, `50/50` terminal
checks, `1/10` stretch tasks)** alongside `online-feature-prior-bce`:
- Enabling the Sherman-Morrison associative fast-weight operator in `online-gail-bcap-bce`
  reduces unseen `seq_minimal_kernel2x2` `heldout_response_rmse` by **3.0x** (`0.2654` in
  `online-gail-bcap-hinge` and `0.2614` in `online-feature-prior-bce` down to `0.0868`,
  `heldout_max_abs_error` `0.4311 -> 0.1895`), cuts `seq_img_translating_bar_next`
  `next_frame_rmse` by **25%** (`0.0085 -> 0.0064`), lowers `seq_img_delayed_cue`
  `prediction_rmse` (`0.0242 -> 0.0231`), `seq_img_circle_orbit` `orbit_rmse`
  (`0.0126 -> 0.0119`), `seq_img_square_circle_cycle` `cycle_rmse` (`0.0127 -> 0.0125`), and
  `seq_robotics_fetch_pick_and_place` `position_delta_rmse_m` (`0.0027m -> 0.0026m`, `4/5`
  terminal checks), while preserving `1.0000` quality and exact rare-branch support (`0.0625`)
  on all discrete multimodal tasks.
- Adding `betas=[0.5, 0.999]` in `online-gail-bcap-momentum` further improves
  `seq_minimal_velocity_extrapolation` (`direction_accuracy = 1.0000` on both interpolation
  and extrapolation, `extrapolation_speed_error` `0.0740 -> 0.0379`, **1.95x** lower) and
  `seq_minimal_kernel2x2` `seen_response_rmse` (`0.0419 -> 0.0353`), while slightly increasing
  `seq_robotics_fetch_pick_and_place` terminal variance (`2/5` vs `4/5` terminal checks).

## Bidirectional Square-Circle CycleGAN (`seq_img_square_circle_cycle`) and View 7 Qualification, 2026-11-06

Added `seq_img_square_circle_cycle` as the tenth required task in `single_stream_stability`
revision 7 (`configs/forge/view-history/single_stream_stability-v6.json` is archived).
In `seq_img_square_circle_cycle`, two synchronized orbital domains step clockwise around
`(3.5, 3.5)` at `batch_size=1` with no replay (`seed=0`):
- **Square Orbit ($S$)** along the unit $L_\infty$ perimeter ($\max(|x|, |y|) = 1.0$):
  `SQUARE_WAYPOINTS = ((0, 3), (0, 6), (3, 6), (6, 6), (6, 3), (6, 0), (3, 0), (0, 0))`
- **Circle Orbit ($C$)** along the unit $L_2$ ring ($\sqrt{x^2 + y^2} \approx 1.0$):
  `CIRCLE_WAYPOINTS = ((0, 3), (1, 5), (3, 6), (5, 5), (6, 3), (5, 1), (3, 0), (1, 1))`

A bidirectional CycleGAN pair of public `ParticleGAN` trainers with causal state memories
— forward $(G_{S\to C}, D_C, M_S)$ and reverse $(G_{C\to S}, D_S, M_C)$ — observes one arriving
frame pair per step (`real_batch=1`, `fake_batch=1`, `replayed_examples=0`). Before the arrival
is revealed, `session_s2c.predict()` and `reverse_session.predict()` serve cross-domain
next-frame predictions; then `session_s2c.observe(c_cur, context_frame=s_cur)` and
`reverse_session.observe(s_cur, context_frame=c_cur)` each perform a single `batch_size=1`
update. At each of the 24 checkpoints (`steps 20, 40, ..., 480`), all 32 particles are
warm-started with two preceding waypoints (`W -> NW`) in both domains and rolled out in
**closed-loop cross-fed cycle ($S \to C \to S$ and $C \to S \to C$) for 8 consecutive steps
(`N .. NW`) without teacher forcing**: generated Circle frames feed into cloned $M_C$ to drive
$G_{C\to S}$, and generated Square frames feed into cloned $M_S$ to drive $G_{S\to C}$.

The numerical gates require simultaneous bidirectional pixel accuracy (`cycle_rmse <= 0.12`,
`s2c_rmse <= 0.12`, `c2s_rmse <= 0.12`), normalized 2D coordinate accuracy (`xy_rmse <= 0.15`),
bidirectional per-step waypoint accuracy (`step_accuracy >= 0.90`), full 8-step closed-loop
bidirectional cycle completion (`cycle_completion >= 0.90`), and complete phase support
(`waypoint_modes == 8.0`) across all five terminal checks (`400..480`). Crucially, at all four
diagonal corner phases (`NE, SE, SW, NW`, phases `1, 3, 5, 7`), where Square corners lie at
$\|( \pm 1, \pm 1 )\|_2 = \sqrt{2} \approx 1.414$ (`(0,6), (6,6), (6,0), (0,0)`) and Circle
diagonals lie tucked inward at $\|( \pm \frac{2}{3}, \pm \frac{2}{3} )\|_2 \approx 0.943$
(`(1,5), (5,5), (5,1), (1,1)`), each direction must be strictly closer in pixel RMSE to its
target domain template than to the source domain template. Oracle and destructive controls
verify that identity copy ($S \to S$ / $C \to C$, `step_accuracy = 0.50`, `cycle_completion = 0.0`),
one-way collapse, counter-clockwise orbits, 1-step phase lag, and midpoint corner interpolation
all fail.

Without changing any recipe hyperparameter, architecture, or task-owned resource budget,
the single global `online-feature-prior-bce` recipe qualifies **`10/10` required tasks (`PASS`,
`50/50` terminal checks)** across 4,800 primary training updates (plus 480 reverse-direction
CycleGAN updates) and 240 observations:
- On `seq_img_square_circle_cycle`, `online-feature-prior-bce` reaches `100%` closed-loop
  `cycle_completion` at step 120, holds `cycle_completion = 1.0`, `step_accuracy = 1.0`, and
  `waypoint_modes = 8.0` across all five terminal checks (`400..480`), and finishes at step 480
  with `cycle_rmse = 0.012665`, `s2c_rmse = 0.011734`, `c2s_rmse = 0.013596`, `xy_rmse = 0.003817`,
  and live prequential RMSE `0.016235`.
- Both the canonical Forge checkpoint strip
  (`reports/forge/gifs/online-feature-prior-bce_seq_img_square_circle_cycle.gif`) and a 3-panel
  Square $\leftrightarrow$ Circle CycleGAN training animation
  (`docs/media/readme/square-circle-cycle.gif`) are rendered directly from actual training checkpoints.

## Circular-Orbit Real-Time Point Prediction (`seq_img_circle_orbit`) and View 6 Qualification, 2026-11-06

Added `seq_img_circle_orbit` as the ninth required task in `single_stream_stability`
revision 6 (`configs/forge/view-history/single_stream_stability-v5.json` is archived).
In `seq_img_circle_orbit`, a point particle (`2×2` patch of intensity `0.9` on a `0.1`
background) steps clockwise around eight disjoint circular waypoints
`CIRCLE_WAYPOINTS = ((0, 3), (1, 5), (3, 6), (5, 5), (6, 3), (5, 1), (3, 0), (1, 1))`
(corresponding to 45° increments `N, NE, E, SE, S, SW, W, NW` around `(3.5, 3.5)`),
revealing one frame per chronological arrival (`batch_size=1`, no replay, protocol seed `0`).

Before each arrival is revealed, `OnlineSession.predict()` serves a next-point prediction
conditioned on the preceding detached context, and `OnlineSession.observe()` updates once
(`real_batch=1`, `fake_batch=1`, `replayed_examples=0`). At each of the 24 checkpoints
(`steps 20, 40, ..., 480`), all 32 particles are warm-started with two preceding waypoints
(`W -> NW`) and rolled out in **closed loop for 8 consecutive steps (`N .. NW`) without
teacher forcing**, feeding each particle's generated frame back into a cloned memory state.

The numerical gates require simultaneous pixel accuracy (`orbit_rmse <= 0.12`),
normalized 2D coordinate accuracy (`xy_rmse <= 0.15`), per-step waypoint accuracy
(`step_accuracy >= 0.90`), complete 8-step closed-loop revolution (`orbit_completion >= 0.90`),
and full phase support (`waypoint_modes == 8.0`) across all five terminal checks (`400..480`).
Oracle and destructive controls verify that stuck particles, counter-clockwise orbits,
1-step phase-lagged orbits, incomplete `7/8` arcs, and center-blurred outputs all fail.

Without changing any recipe hyperparameter, architecture, or task-owned resource budget,
the single global `online-feature-prior-bce` recipe qualifies **`9/9` required tasks (`PASS`,
`45/45` terminal checks)** across 4,320 training updates and 216 observations:
- On `seq_img_circle_orbit`, `online-feature-prior-bce` first reaches `100%` closed-loop
  `orbit_completion` at step 120, settles permanently at `100%` from step 200 through step 480,
  and holds `orbit_completion = 1.0`, `step_accuracy = 1.0`, `waypoint_modes = 8.0`,
  `xy_rmse = 0.0036..0.0044` (`<= 0.15`), and `orbit_rmse = 0.0126..0.0134` (`<= 0.12`)
  across all five terminal checks (`400..480`), with final live prequential RMSE `0.015725`.
- Unrouted strict-online baselines (`online-bcap-adam`, `online-bispf-early`,
  `online-ka2-ready`) fail `seq_img_circle_orbit` (`orbit_completion = 0.0312`, `0.1602`,
  and `0.5156` across the terminal window) because open-loop errors compound over the
  8-step self-fed rollout without feature-conditional particle routing.
- In addition to the canonical Forge checkpoint strip
  (`reports/forge/gifs/online-feature-prior-bce_seq_img_circle_orbit.gif`),
  `render_circle_orbit_gif` renders a 3-panel circular-orbit training animation
  (`docs/media/readme/circle-orbit.gif`) directly from actual training checkpoints.

## ParticleGAN and Forge 0.9.7 compatibility, 2026-11-06

Upgraded the published `particlegan` wheel from 0.9.6 to 0.9.7, including its
`particlegan_forge` namespace. The release adds canonical stretch assignments
and a separate per-tier stretch table. All eight ParticleStream tasks remain
required in view 5. Recipe settings, task declarations, adapters, architectures,
480-update budgets, numerical gates and seed 0 are unchanged. All training ran
sequentially with real/fake batch size one and no replay.

Fresh CUDA execution of the single global `online-feature-prior-bce` recipe
passes **8/8 required tasks and 40/40 terminal checks**, covering 3,840 updates
and 192 observations. Every observation and every actual-training GIF is
identical to the archived 0.9.6 cohort. All 192 GIF frames decode successfully;
receipt hashes match the GIFs, training remains finite, and evaluation records
zero RNG deviations. Fresh receipts carry the installed 0.9.7 engine digest;
historical receipts cannot qualify against it.

The full regression suite passes **421 tests** in 40.99 seconds, including
oracle/destructive controls, all three trainer families, serving isolation,
source-bound qualification, archived artifacts and the new stretch-report
integration. Workspace validation accepts all 44 candidates and eight tasks;
`uv pip check` reports all 46 installed packages compatible. The eight full
training runs take 67.412 aggregate seconds including rendering.

The clean environment uses Python 3.13.15, Torch 2.15.0+cu132, NumPy 2.5.4,
Matplotlib 3.11.2 and Pillow 13.0.0. The prior receipts used Torch 2.14.1+cu130;
the measured observation/GIF parity therefore also covers this runtime change.
The previous eight-task receipts, GIFs and audit are preserved byte for byte in
[`cohorts/particlegan-0.9.6-view5/`](cohorts/particlegan-0.9.6-view5/README.md).
The refreshed [audit](feature-prior-audit.json) records package versions,
engine binding, worst terminal gate values and archive integrity checks.

Reproduce with `particlestream validate`,
`particlestream run online-feature-prior-bce --device cuda` and
`python -m pytest -q` in the project environment. Independently audit receipts
and every GIF frame with `python scripts/online_comparison.py
--candidate online-feature-prior-bce --device cuda --audit-only
--summary runs/forge/upgrade-audit.json`.

Recommendation: keep the 0.9.7 pin and the same qualifying global recipe. The
single [leaderboard](technique-inventory.md) remains 8/8 for that recipe;
other candidates retain their historical evidence and require fresh runs for
qualification with the new engine. This upgrade does not extend the validated
480-update horizon or the bounded memory capacity.

## Cohort before the ParticleGAN 0.9.6 upgrade

The original reports are recoverable at git commit
`16f0efc8f872874f915a869636b5c1b063328d04`. Their 16 CUDA receipts and GIFs
remain under `receipts/cuda/` and `gifs/` with their original candidate IDs.
Do not relabel these receipts as results from the upgraded engine.

The recorded winner was `fast-weight-bcap-ns`: all four tasks passed all five
terminal checks. The stateless BCAP baseline passed only the fork task; the
earlier FastWeight-BCAP and KA2 candidates passed zero tasks. The winner combined
non-saturating loss, Adam lr=0.005, cosine annealing, a bounded 64-frame FIFO with
seven replay samples, and temporal difference context.

Historical explanations were hypotheses based on a combined recipe change,
not isolated causal ablations. In particular, the earlier FastWeight candidate's
final motion metrics passed while its sustained terminal window failed. Final
metrics alone do not qualify a task.

Prior repulsion, high learning rates and unconditioned replay have failed in
previous experiments; consult [DEAD_ENDS.md](../../DEAD_ENDS.md) before revisiting
them. Higher-resolution streams and longer memory horizons were proposed for
Tier 2 but are not implemented tasks.

## Upgrade protocol

- Pin the published ParticleGAN 0.9.6 wheel and exercise its actual public
  trainers; remove the pytest import override for the sibling checkout.
- Retain all four tasks, gates, 480-step budgets, seed 0 and five terminal checks.
- Share bounded memory across family candidates and hold each task's network
  architecture fixed. Distinguish a one-frame arriving stream from replay's
  larger optimizer input.
- Keep historical candidate IDs and artifacts intact. New family candidate IDs
  require fresh receipts bound to task, recipe, engine and implementation code.
- Evaluate BCAP/Adam, BCAP/DualNorm, KA2, BISPF and optimized BISPF. Prioritize
  BISPF qualification, recording failures without lowering gates or choosing
  a different recipe for each task.

Current results are generated into [the leaderboard](technique-inventory.md).

## Results: 2026-11-05, ParticleGAN 0.9.6

Ran **40 full-budget CUDA tasks** (10 global candidates × 4 tasks), totaling
19,200 training updates and 413.055 aggregate run seconds, including GIF rendering.
Every run used seed 0, completed 480 updates and 24 observations, reported finite
training state and zero evaluation RNG deviations, and produced a 24-frame
actual-training GIF. The historical 16 receipts/GIFs were retained unchanged.
All four original tasks and every numerical gate remain required.

Three distinct trainer families now qualify: `stream-bispf-early`,
`stream-bcap-adam` and `stream-ka2-ready`. Each passes all five terminal checks
on all four tasks with one complete recipe across the suite. The canonical
leaderboard includes all failed variants and marks pre-upgrade evidence stale.

### BISPF

The initial `stream-bispf` passed both mode-retention tasks, but motion and fork
each passed only four of five terminal checks. Both final checkpoints passed;
the sustained-window rule correctly rejected them. At step 400, motion RMSE was
0.133608 (gate ≤0.12), while fork quality was 0.8125 (gate ≥0.85).

`stream-bispf-early` changes only annealing start from 0.5 to 0.25 of the budget.
It passes all four tasks. Across the five terminal checks, worst motion RMSE is
0.113529, minimum direction accuracy is 0.953125, minimum fork quality is 0.875
and maximum branch TV is 0.125. Its bar-mode quality is at least 0.96875 on both
streams. This is the original BISPF critic with its formulation optimizer and
the explicitly declared non-saturating, no-SBNP, bounded-memory recipe.

The optimized BISPF/DualNorm candidate failed all four tasks. Lowering its global
rates and starting annealing earlier also failed all four. Rate reduction alone
did not recover it; these results do not establish the underlying cause.

### BCAP

`stream-bcap-adam` passes all four tasks. Its terminal minimum quality is 1.0 on
both bar streams and fork, motion accuracy is always 1.0, and worst terminal
motion RMSE is 0.092738. The distinct modern BCAP/DualNorm candidate failed all
four tasks. A public default that works elsewhere is not automatically qualified
on this single-stream image suite.

### KA2

`stream-ka2` and `stream-ka2-clean` each passed three tasks but failed motion.
Removing auxiliary noise, critic guard, latent damping and serving EMA did not
resolve motion by itself. Reducing the clean candidate's anchor weight from 1
to 0.1 produced identical curves on all four tasks.

The public KA2 kernel gates its adaptive phase on second-moment readiness:
`1 - beta2**t >= 0.55085`. At beta2=0.999, readiness at step 480 is only about
0.381, so the lighter anchor could not affect this cohort. `stream-ka2-ready`
changes only critic betas to (0,0.99), retaining generator/prior betas (0,0.999)
and the native KA2 optimizer and penalty. Readiness crosses the threshold around
step 80, and the resulting global recipe passes all four tasks. Worst terminal
motion RMSE is 0.094063 with direction accuracy 1.0; minimum fork quality is
0.875. The change affects both optimizer dynamics and adaptive-phase timing;
this comparison does not isolate the anchor's causal contribution.

### Recommendations

Use **`stream-bispf-early`** as the BISPF representative and
**`stream-bcap-adam`** as the reference with the strongest observed gate margins.
Keep **`stream-ka2-ready`** as the third qualified family. Reproduce these exact
global recipes through the project CLI; do not select a different recipe per
task or substitute the failed optimized BISPF variant for the qualified one.

BISPF and KA2 fork quality reaches 0.875 against a 0.85 gate: losing one more
valid particle among 32 would fail. Qualification is for this fixed Tier 1
protocol. A later Tier 2 should declare harder streams and new gates explicitly;
these results do not qualify larger images or longer rollouts.

### Verification

The final regression suite passes **245 tests**, including oracle/destructive
controls, real public-trainer smoke checks, source-bound family qualification,
independent sustained-gate recomputation and historical artifact preservation.
All **56** receipt/GIF pairs match SHA-256 and decode as **1,344** total frames
(24 frames per 900×300 GIF). Logs remain untracked under `runs/forge/`.


## Strict online Tier 1

The batch-size-1 requirement now applies to real and fake optimizer inputs,
chronological arrival order and serving while learning. Historical replay
receipts and GIFs remain unchanged and no longer qualify against this contract.
The new `online-*` recipes copy their historical parents' complete optimization
settings and explicitly disable replay, including stored-frame substitution.

Both temporal training paths now reveal one frame at a time. Predictions precede
the target arrival; the arriving target trains using the preceding detached
context. Predictive tasks declare one warm-up arrival, so 480 updates consume
481 frames; marginal image tasks use 480 arrivals. Serving is serialized with
training and uses an isolated RNG. Extra serving calls are tested to leave
subsequent parameters, optimizer states and non-evaluation RNG states unchanged.

View revision 4 keeps the four original requirements and adds variable dwell,
rare-mode retention, delayed cue and persistent fork as four held challenges.
They all execute and retain their failures. A held task returns to counted
qualification only when one global recipe passes it with every counted task,
under a separately archived view revision. Holding never cures a counted failure.

The fork law is an explicit task delta: each branch is an independent fair draw
when its first frame arrives. Previous episodes therefore cannot determine the
next outcome. Evaluation separately checks both attainable previous-branch to
stem histories and takes the worst state. The three-step fork keeps both the
particle index and its Gaussian latent perturbation fixed, feeds generated
frames into per-particle cloned memory, and never reads real future frames.
These declarations change task fingerprints and require fresh receipts.

### Strict cohort results — 2026-11-05

Ran **24 full-budget CUDA tasks** (three global recipes × eight tasks), with
protocol seed 0, **11,520 training updates**, **11,532 chronological arrivals**
and **11,520 live predictions**. Aggregate attempt time was 219.3156 seconds,
including evaluation and GIF rendering. Every run completed all 24 observations,
recorded real/fake training batches of one, remained finite, and reported zero
replay, gradient accumulation and evaluation RNG deviations. Each produced a
24-frame, 900×300 actual-training GIF: 576 new animation frames in total.

**None of the three recipes qualifies:** every counted task fails its sustained
window, and every held challenge also fails. Failures remain in the canonical
[leaderboard](technique-inventory.md); no gates or budgets were relaxed. All 56
historical receipt/GIF pairs remain byte-identical, verified by the frozen
112-file manifest in `tests/fixtures/pre_online_artifacts.json`.

- **Acquisition and retention:** BISPF round-robin reaches final quality 0.90625,
  four modes and TV 0.09375, but passes only the last of five terminal checks.
  It is correctly FAIL. BCAP round-robin has four modes but quality 0.78125 and
  TV 0.25; mode count alone is insufficient. Final quality on fixed/variable
  dwell is zero for BCAP and BISPF, and 0.09375 for KA2.
- **Rare modes:** BISPF and KA2 have no valid rare-mode particles at the final
  checkpoint. BCAP retains one rare particle among 32 and all four supports,
  but overall quality 0.53125 and TV 0.46875 still fail. Rare support alone
  cannot establish a correct distribution.
- **Motion:** final direction accuracy is 0.96875 for BISPF and KA2, yet RMSE is
  0.182882 and 0.169249 respectively, above the 0.12 gate. BCAP remains at
  direction accuracy 0.5 and RMSE 0.33884. Correct direction can coexist with
  inaccurate predicted images.
- **Branching:** BCAP's one-step fork has final quality 1.0 but only one branch
  and TV 0.5. The scorer catches sharp mode collapse. All three persistent-fork
  recipes have final trajectory consistency 0 and fail the per-depth gates.
- **Delayed cue:** final accuracy is 0.5–0.515625 and RMSE is about 0.299.
  After four identical distractors the current reader's last-frame difference
  contains no cue information; its accumulated matrix is still not read. This
  does not establish that an actual associative-memory read would fail.
  BCAP's last live prequential error is only 0.11045 because the stream includes
  many easy distractors, illustrating why balanced outcome gates remain separate
  from average live error.

### Interpretation and next work

The bar comparisons jointly remove historical examples and reduce the effective
real/fake batch from eight to one. They do not isolate replay's retention effect
from batch stabilization. Motion and fork did not use replay in the old adapter;
their regressions additionally involve chronological target timing, observation
noise, training every arrival, and, for forks, independent branch choices plus
attainable-state evaluation. Do not attribute those regressions to replay removal.

Keep all three strict recipes as failed reference points. The next useful global
recipe change should stabilize acquisition and retention with one observation
per update; every candidate must still run the whole eight-task suite. A separate
memory ablation should actually query the accumulated state and retain the cue
through distractors, with memory-erasure controls. Neither result is established
by this cohort. Preserve seed 0, the fixed gates, and the one-recipe-across-tasks
rule rather than selecting task-specific fixes.

### Strict cohort verification

The complete regression suite passes **312 tests**, including oracle/destructive
controls, serving/training concurrency, bitwise trajectory invariance under extra
serving calls, public-trainer batch guards, independent receipt grading and
historical artifact preservation. Every new GIF hash matches its receipt and all
576 frames decode. The independent protocol audit found no remaining blocker.

## Strict online recovery

The recovery target is one global recipe passing all eight unchanged task gates.
Experiments retain seed 0, chronological single-image arrivals, one public
ParticleGAN critic/generator update per arrival, and the five-check terminal
window. Failed recipes remain evidence; a final-check improvement is insufficient.

### Initial comparisons

Two previously unrun strict descendants, `online-bcap-stateless` and
`online-fast-weight-bcap-ns`, completed all eight tasks and passed none. The latter
reproduces `online-bcap-adam`'s complete numerical curves on all eight tasks; it
is alias/reproduction evidence rather than an independent improvement. Nine
additional historical descendants were declared for complete comparisons.

Four isolated BISPF ablations completed 32 CUDA runs. Freezing the initial prior
and reducing its learning-rate multiplier from 2 to 0.25 each recover motion
prediction, passing all five terminal checks. Frozen-prior motion has direction
accuracy 1 throughout that window, worst RMSE 0.079487, and final RMSE 0.071489.
Each still fails the other seven tasks. Critic momentum (0.95,0.999) and critic
rate multiplier 0.25 pass no tasks. These results support the individual recipe
deltas under this protocol; they do not establish broad robustness or solve
retention and delayed recall.

The first combined causal-reader/current-observation reconstruction recipe,
`online-bcap-transport-causal`, also fails all eight sustained gates. Final
delayed-cue accuracy rises to 0.765625, with RMSE 0.265376. Persistent-fork
trajectory consistency rises to 0.53125, but worst-depth quality 0.6875 and
branch coverage still fail. Particle-use diagnostics expose incomplete density
coverage and show why visiting all rows globally does not ensure coverage within
each conditioning state. The combined frozen-prior/constant-rate/weight-100
fitting candidate regresses and also passes none; its several simultaneous
changes do not isolate a cause.

### Mechanisms and evidence scope

The causal reader carries current appearance, last-step change, retained salient
content, repetition age and an associative read in the existing context size.
It adds no generator or critic parameters. The competitive reconstruction
recipe uses only the current arriving target, uniform critic draws, one routed
generator draw and 32 sequential inference-only center probes per update. It
clears targets before serving and records probe counts and particle assignments.
The probes add compute; comparison by update count alone does not equalize cost.

Context-balanced routing is a separate declared mechanism: bounded detached
conditioning centroids carry local transport duals. An overused row's dual
increases its future assignment cost within that context. Synthetic controls
test marginal frequencies, rare support and independent conditional coverage;
these controls do not substitute for actual eight-task training receipts.

The first actual context-balanced recipe passes round-robin acquisition and
rare-mode retention across all five terminal checks. Final rare-mode quality
is 1, TV is 0, and rare mass is exactly 2/32. It still fails the other six tasks.
Delayed-cue accuracy reaches 0.953125, but image RMSE 0.218652 exceeds the 0.12
gate. Accuracy alone does not establish correct reconstruction. Fixed dwell has
four modes yet quality 0.78125 and TV 0.21875 still fail; variable dwell remains
unsolved. Motion and persistent rollouts remain important unresolved failures.

Before integrating recovery sources, 72 strict-online receipt/GIF pairs were
copied byte-for-byte into `cohorts/pre-recovery/`, with all eight task declarations.
Its 256-file manifest also protects the original 56 historical pairs. Canonical
paths can receive new source-bound measurements without discarding the archive.
Source changes invalidate qualification of older receipts even when numerical
gates and architectures are unchanged. Current qualification is generated only
in the canonical [leaderboard](technique-inventory.md).

### Qualified conditional-feature prior

`online-feature-prior-bce` is the first single recipe to pass every task and
every terminal check: 8/8 tasks, 40/40 checks. It keeps the task's 32×8 initial
MoG storage table frozen and learns a bounded conditional transform over
observed feature means. A fixed named orthogonal image projection supplies the
current-frame training posterior. The unchanged generator learns image decoding
with BCE weight 10; critic and serving samples use learned empirical quantiles
from previous successful observations. The transform retains the original
Gaussian perturbation and never emits images directly. No previous image is
reused as a training target. This is a new memory/prior mechanism with a BCAP
critic, rather than evidence that optimizer tuning alone repaired all failures.

Across the initial passing terminal window, all density and branch quality
metrics are 1. Motion accuracy is 1 with worst RMSE 0.009068. Delayed-cue accuracy
is 1 with worst RMSE 0.026010. Rare-mode mass is exactly 2/32 and TV is zero.
One-step fork TV is at most 0.03125; persistent-fork TV is at most 0.09375 and
trajectory consistency is 1. Fixed/variable dwell TV is at most 0.03125. These
margins satisfy the original gates without task-specific recipe choices.

The same resolved recipe, memory policy, manipulator settings and named encoder
initialization are used across all tasks. Independent tests confirm that extra
serving leaves subsequent model, memory, optimizer and training RNG state
bit-identical; feature counts commit from the preceding observed context;
exceptions clear transient targets; and persistent evaluation writes generated
frames only into cloned memory. Zeroing the decoder destroys output quality,
confirming that latent feature statistics do not bypass actual training.

View 5 promotes the four formerly held challenges to required tests, archiving
the exact view 4 declaration. Since the public source digest covers the whole
adapter module, the initial eight passing view-4 receipts/GIFs are preserved
under `cohorts/feature-prior-view4/` before fresh full-budget view-5 qualification.
No source binding is patched by hand. Numerical gates, budgets, stream laws,
architectures and protocol seed are unchanged.

Use this recipe as the Tier 1 reference. Its finite memory remains a measured
limit: persistent fork fills all 64 context slots and records 173 evictions
within 480 updates despite passing. A useful next tier should test longer
streams, context churn and richer images with the same global recipe. This
qualification establishes the declared eight-task protocol, not those untested
settings. Earlier failed and alias recipes remain visible in history.

### Complete recovery campaign and verification

Measured all eleven previously unrun historical descendants across 88 full-budget
tasks. None passes the full suite. Original `online-fast-weight-bcap` passes motion
5/5, with final accuracy 1 and RMSE 0.067943, but fails the other seven tasks.
KA2 clean/light-anchor curves are exactly identical on all eight tasks; as with
the earlier BCAP alias, this is reproduction evidence rather than a new gain.
The nine later baseline recipes have an immutable source/declaration/GIF archive
under `cohorts/baselines-d19b9d4/`. Their older source bindings cannot qualify
against the final feature-prior implementation.

The campaign ran 160 full-budget tasks over 19 distinct new recipes, including
the final winner's fresh source-verification repeat: 76,800 training updates,
76,880 chronological arrivals and 3,840 actual-training GIF frames. Aggregate
task-attempt time was 1,708.3484 seconds, including evaluation and rendering.
Every new run stayed finite, used seed 0 and optimizer batches of one, and
recorded zero replay and evaluation RNG deviations. Exploratory declarations
without runs remain NOT_RUN; no outcomes are inferred for them.

Final view-5 qualification is eligible PASS on all eight required tasks, with
zero held tasks. All 192 observations and eight GIF hashes exactly reproduce
the archived first passing cohort under distinct, fresh adapter source bindings.
Independent audit checks every GIF frame, unchanged task gates/resources,
identical global configuration, target cleanup and the real decoder path. The
complete regression suite passes 417 tests. Detailed verification is recorded in
[`feature-prior-audit.json`](feature-prior-audit.json); current qualification
remains in the single [leaderboard](technique-inventory.md).

### Five real robotics dynamics tests (view 10)

The unchanged `online-feature-prior-bce` recipe now runs sequentially on
FetchReach, FetchPush, FetchSlide, FetchPickAndPlace and HandReach using real
Gymnasium-Robotics/MuJoCo transitions. The five final 480-update runs pass 0/5
robotics stretch tests; required qualification remains 10/10. Final physical
errors are 5.53, 10.94, 7.61, 2.67 and 8.18 mm respectively. PickAndPlace passes
four terminal checks but misses the direction gate at update 420. Every run
stays finite, uses optimizer batches of one and records zero replay or evaluation
RNG deviations. All five oracles pass and twenty destructive controls fail.

These are one-step state/action prediction tests under fixed excitation, not
learned robot-control policies. The initial capped affine target encoding failed
on a 10.22 cm Slide transition; its attempts and source are archived before
five fresh runs with one shared invertible encoding. No recipe, physical gate
or update-budget change accompanies that repair. See the
[protocol, measurements, training GIFs and recommendation](robotics-readout.md)
and [machine-readable audit](robotics-audit.json). Retain these tests as stretch
and test the next explicit global conditioning/stability ablation across all
five, without selecting per-task recipes or extending individual budgets.

### Complete recipe refresh (2026-11-07, view 10)

The [current inventory](technique-inventory.md) now measures all 30 executable
recipes on all 20 tasks: 600 fresh, full-budget runs. Every run uses seed 0,
sequential real/fake batches of one and the unchanged task gates. Each candidate
keeps one global recipe across the suite. Numerical failures do not skip the
remaining stretch measurements or grant higher-tier qualification. The cohort
contains 19 numerical PASS and 581 FAIL outcomes, with no INVALID or INCOMPLETE
execution. All 600 runs stay finite and record zero replay and evaluation RNG
deviations. The [refresh audit](refresh-audit.json) records timestamps, execution
sessions and receipt/GIF hashes; interrupted execution resumes only unchanged,
source-current evidence.

The fresh `online-feature-prior-bce` result retains 10/10 required passes and
1/10 stretch passes, with tokens its only stretch success. The best BCAP recipe
passes 3/10 required tasks; BISPF and FastWeight-BCAP reach 1/10 each, while
Global-Stream-BCAP and KA2 reach 0/10. Retain the feature-prior recipe as the
global reference for the required suite. Dense images, held-out kernels and
addition, robotics dynamics and unseen-speed rollout remain numerical failures;
the required result does not establish those capabilities.

Six linked family pages explain the methods and every exact recipe, including
effective settings, parent deltas, gates, sustained checks and actual-training
GIFs. Fourteen historical declarations remain read-only with links to their
strict online successors; their stale evidence is not displayed as a fresh
zero score. Scores always belong to one complete recipe, without per-task
selection.

Validation: 523 regression tests pass, including regrading every fresh receipt,
checking source bindings, shared recipes, batch/RNG contracts, simulator traces
and all 24 GIF frames and hashes. All 20 oracle controls pass and all 45
destructive controls fail. All 4,133 local links, including 2,138 exact section
anchors across the inventory and six family pages, resolve.

### Parametric residual conditional prior and 640-run refresh (2026-11-12, view 10)

The [current inventory](technique-inventory.md) and
[`ResidualPrior-BCAP` family page](families/residualprior-bcap.md) introduce
two strict single-factor online recipes (`online-residual-prior-bce` and
`online-residual-prior-linear-key`) and refresh all 32 executable recipes across
all 20 tasks: 640 fresh, source-bound CUDA runs under protocol seed 0 and
sequential real/fake batch size 1 (`replay_policy: "none"`). The cohort records
43 numerical PASS and 597 FAIL outcomes, with zero INVALID or INCOMPLETE runs,
zero replay or evaluation RNG deviations and finite states throughout. All 20
oracle controls pass and all 45 destructive controls fail; timestamps and
receipt/GIF SHA-256 hashes are recorded in [refresh-audit.json](refresh-audit.json).

`ResidualFeatureConditionalReconstruction` replaces nearest-prototype context
snapping on novel contexts with an online `float64` Sherman-Morrison recursive
least-squares predictor, a precision-weighted spike-and-slab posterior over
context prototypes, empirical-Bayes kernel-gated live residual quantiles and a
Markov bottleneck penalty on direct generator input-context shortcut weights.
With unit-sphere causal-memory keys (`online-residual-prior-bce`,
`normalize_key: true`), the residual prior preserves 10/10 required Tier 1
passes and 1/10 stretch passes (`seq_minimal_tokens4`), isolating that the
parametric conditional prior introduces no regression on discrete or bimodal
forking streams (`seq_img_persistent_fork` 5/5). Removing unit-norm key
projection (`online-residual-prior-linear-key`, `normalize_key: false`,
`key_gain: 0.5`) preserves Euclidean multi-operand superposition in causal state
memory and advances the global reference to 10/10 required passes and 3/10
stretch passes: `seq_minimal_tokens4` (5/5), held-out compositional binary
addition `seq_minimal_addition2bit` (5/5, `heldout_exact_accuracy=1.0`,
`heldout_min_bit_confidence=0.96443` vs `0.0` on `online-feature-prior-bce`) and
continuous kinematics `seq_robotics_fetch_reach` (5/5,
`position_delta_rmse_m=0.0035343 <= 0.005` vs `0.0055261` on
`online-feature-prior-bce`).

Qualification remains Tier 1 because seven stretch tasks remain numerical
failures under the single global recipe: `seq_minimal_dense4` (8-dim fixed
isometric bottleneck), `seq_minimal_kernel2x2` (held-out `heldout_response_rmse`
drops from `0.26142` to `0.038166`, matching seen `0.04305`, but misses the
`0.03` RMSE and `0.05` max-abs pixel uniformity bounds within 192 updates),
`seq_robotics_fetch_pick_and_place` (4/5 terminal checks, `2.80 mm` RMSE),
contact-discontinuous robotics (`fetch_push`, `fetch_slide`, `hand_reach`) and
closed-loop `seq_minimal_velocity_extrapolation`. Validation passes 545
regression tests across all 7 documented technique families, 640 receipts and
all local links and section anchors.

### GAIL-BCAP family, Frontier (2) latent-SNR transport, and 740-run refresh (2026-11-12, view 10)

The [current inventory](technique-inventory.md) and
[`GAIL-BCAP` family page](families/gail-bcap.md) introduce five single-factor
online recipes (`online-gail-bcap-hinge`, `online-gail-bcap-bce`,
`online-gail-bcap-momentum`, `online-gail-bcap-spike-slab` and
`online-gail-bcap-snr`) and refresh all 37 executable recipes across all 20
tasks: 740 fresh, source-bound CUDA runs under protocol seed 0 and sequential
real/fake batch size 1 (`replay_policy: "none"`, `rehearsal_batch: 0`, zero
generator probes). The cohort records 102 numerical PASS and 638 FAIL outcomes,
with zero INVALID or INCOMPLETE runs, zero replay or evaluation RNG deviations
and finite states throughout. All 20 oracle controls pass and all 45 destructive
controls fail; timestamps and receipt/GIF SHA-256 hashes are recorded in
[refresh-audit.json](refresh-audit.json).

`GAILConditionalReconstruction` translates generative adversarial imitation
control into strict online world modeling by pairing a Hinge margin critic and
fixed BCAP input-gradient cap with prioritized pixelwise BCE survival
reconstruction minus a bounded sigmoid critic style bonus
(`cancel_linear_gan=True`), causal return-advantage modulation and active
prototype-support policy entropy (`policy_entropy_bonus=0.01`). All five
`GAIL-BCAP` recipes qualify across 10/10 required Tier 1 tasks (50/50 terminal
checks each). Upgrading radial Sherman-Morrison associative blending
(`online-gail-bcap-bce`, 10/10 required, 1/10 stretch) to `float64`
Sherman-Morrison RLS, precision-weighted Mahalanobis spike-and-slab prototype
transport, empirical-Bayes kernel-gated active-entropy residual quantiles, a
Markov context-bypass penalty (`context_bypass_weight=5.0`) and Euclidean causal
keys (`online-gail-bcap-spike-slab`, `normalize_key: false`, `key_gain: 0.5`)
advances `GAIL-BCAP` to #2 overall (10/10 required, 3/10 stretch:
`seq_minimal_tokens4`, `seq_minimal_addition2bit` held-out exact `1.0` and
`seq_robotics_fetch_reach` `3.48 mm <= 5.0 mm`, while lowering
`seq_minimal_kernel2x2` held-out RMSE to `0.03551`).

Inspecting physical robotics failure modes exposed two orthogonal bottlenecks:
(1) `CausalStateMemory`'s `contrast_threshold=0.03` erased sparse 4-of-64-pixel
action commands on sub-`4 cm` `Fetch*` transitions near grasp contact
(`||a_t - c_0||_2 = 0.025 < 0.03`), leaving multi-episode pick-and-place
partially action-blind (`2/5` terminal checks on `online-gail-bcap-spike-slab`),
and (2) millimetre displacements projected at `feature_scale=2.0` had low
posterior latent SNR relative to fixed task MoG noise `prior_sigma=0.025`.
Lowering `contrast_threshold` to `0.005`, scaling posterior projection by `2x`
(`feature_scale=4.0`, `feature_tolerance=0.30`, `spike_slab_scale=2.0`, yielding
`4x` higher latent SNR) and setting episode-invariant style gain
(`advantage_scale=0.0`) produces `online-gail-bcap-snr`, the #1 global reference
in ParticleStream: 10/10 required passes and 3/10 stretch passes, cutting
`seq_robotics_fetch_reach` physical RMSE by 46% to `1.92 mm` (`0.0019153 m`,
`97.92%` direction accuracy vs `3.53 mm` on `online-residual-prior-linear-key`),
lifting `seq_robotics_fetch_pick_and_place` from `2/5` to `4/5` terminal checks
(`2.45 mm` final RMSE, `1.70 mm` live prequential RMSE, `86.0%` peak terminal
direction accuracy, missing only pre-cooldown `step=400` at `0.719` vs `0.750`
from unaveraged `batch_size=1` live-decoder sign jitter around `0.5`) and
reducing `seq_robotics_fetch_slide` RMSE to `6.92 mm` (`80.39%` direction
accuracy).

Qualification remains Tier 1 because seven stretch tasks remain open under one
unchanged global recipe: `seq_minimal_dense4` (8-dim isometric bottleneck),
`seq_minimal_kernel2x2` (`0.0355`–`0.0400` held-out RMSE vs `0.03` gate under
192 single-sample shared-decoder updates), `seq_robotics_fetch_pick_and_place`
(pre-cooldown live-decoder sign jitter), contact-discontinuous manipulation
(`fetch_push`, `fetch_slide`, 83-DoF `hand_reach`) and closed-loop unseen
boundary columns on `seq_minimal_velocity_extrapolation`. Validation passes 576
regression tests across all 8 documented technique families, 740 receipts and
all 5,147 local links (including 2,651 exact section anchors).
