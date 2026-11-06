# Superseded robotics encoding: affine v1

The first two runs finished as numerical FAIL: FetchReach (5.29 mm, terminal
0/5) and FetchPush (11.43 mm, terminal 0/5). FetchSlide was INVALID when a fast
puck transition exceeded the declared ±0.10 m target encoding range, before
the remaining tasks could run. No successful result is claimed for this cohort.

The source, task declarations, view 9 and all available receipts, training GIFs
and simulator traces are copied unchanged here. `manifest.json` records hashes
relative to this directory. Original implementation commit: `5269ae5`.
The original execution events remain in `runs/forge/events.jsonl`.

The replacement applies one invertible arctangent coordinate map to all five
tasks, followed by fresh source-bound runs. It does not clip target motion,
change physical-unit gates or budgets, tune the recipe, or rebind these receipts.
The canonical leaderboard reports only the current task definitions.
