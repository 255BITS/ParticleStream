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
Experiment explanations and recommendations will be appended here after runs.
