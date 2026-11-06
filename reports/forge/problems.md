# ParticleStream problem ledger — `single_stream_stability`

Revision 2 retains the four original required tasks, numerical gates and
480-update budgets. Fresh ParticleGAN 0.9.6 receipts qualify three families on
all four tasks: `stream-bispf-early`, `stream-bcap-adam` and `stream-ka2-ready`.
See the [single current leaderboard](technique-inventory.md) for all numerical
results and failures, and [research notes](research-notes.md) for explanations.

## Declared requirements

- **PROB-SS-01 — mode acquisition:** learn all four bar templates from a
  one-observation round-robin stream. Qualified in all three representative
  families, including the quality, mode-count and total-variation gates.
- **PROB-SS-02 — retention:** retain all four templates over dwell=16 blocks.
  Qualified in all three representatives with a bounded 64-frame replay memory.
- **PROB-SS-03 — motion:** distinguish two directions at an ambiguous interior
  position using temporal context. Qualified in all three representatives.
  Initial BISPF missed one terminal check; initial KA2 and clean KA2 failed all
  five motion checks. Their failures remain in the current comparison.
- **PROB-SS-04 — branching:** cover both future branches from one state with
  balanced particle mass. Qualified in all three representatives. BISPF and KA2
  have less quality margin than the BCAP/Adam representative.

## Remaining family-level issues

Modern BCAP/DualNorm and optimized BISPF/DualNorm have not qualified on any task
in this cohort. Conservative optimized-BISPF rates did not fix the failures;
causal diagnosis remains open. Do not relabel these as passes based on another
optimizer's results from the same critic family.

KA2's beta2=0.999 readiness schedule does not reach its adaptive phase within
480 updates. Merely changing its anchor weight cannot fix this cohort. The
qualified KA2 recipe declares critic beta2=0.99 and retains the native optimizer
and penalty. This is a recipe change, not an increase to the task budget.

Higher resolution, longer memory horizons and longer rollouts remain proposed
research directions. No Tier 2 task has been declared or qualified.
