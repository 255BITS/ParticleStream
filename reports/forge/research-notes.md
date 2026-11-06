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
