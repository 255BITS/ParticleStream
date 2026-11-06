# ParticleStream research notes

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
