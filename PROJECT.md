# ParticleStream project

ParticleStream compares trainer families on sequential 8×8 image learning and
short-horizon memory tasks. The current declared view is
`single_stream_stability` revision 2; gates and training budgets match revision 1.

## Runtime

- Install the published `particlegan==0.9.6` wheel in the project environment.
- `particlestream.tier1` declares tasks, candidates and the `seq_image` adapter.
- `particlestream.models` supplies the task-owned generator/critic and bounded
  recipe-owned memory. The input stream emits one frame per update; replay
  candidates can train on that frame plus seven historical observations.
- `particlestream.streams` supplies detached data and numerical scorers.
- `particlestream.forge.StreamWorkspace` extends public `ForgeWorkspace` path
  properties, retaining `configs/forge`, `reports/forge` and `runs/forge`.
- The `particlestream` CLI delegates execution, grading, RNG isolation and
  qualification to `particlegan_forge`. Use this CLI instead of upstream's CLI,
  whose default namespace differs.

## Gauntlet

All four tasks remain required: round-robin mode acquisition, retention across
16-frame dwell blocks, direction-sensitive next-frame prediction, and balanced
two-branch generation from a common state. Every task runs 480 updates with
24 observations; all bounds must pass at the final five checks. Seed is always 0.

Compare BCAP/Adam, BCAP/DualNorm, KA2, BISPF and optimized BISPF. Every candidate
uses one global recipe over the suite. Changes are declared in candidate files;
there is no per-task recipe selection. Common conditional network architecture
is task-owned, including zero context for stateless candidates.

## Evidence and checks

- Qualification receipts bind task/candidate fingerprints, seed, device, engine
  and adapter sources. Task fingerprints additionally bind model, stream,
  recipe-construction and initializer-helper implementation code.
- Receipts retain the complete resolved recipe, memory policy, ParticleGAN and
  Torch versions, implementation hashes and actual-training GIF hash.
- Original candidate IDs, 16 pre-upgrade receipts and GIFs remain unchanged.
  Historical passes are not fresh qualification for the new engine.
- `reports/forge/technique-inventory.md` is the only generated leaderboard.
  `EXPERIMENTS_BY_TIER.md` and `EXPERIMENT_MEMORY.md` are generated from the same
  declarations and receipts; `research-notes.md` retains interpretation.
- Tests cover numerical oracle/destructive controls, memory detachment, RNG
  isolation, sustained gates, invalid evidence, family identity, public trainer
  execution, historical artifact integrity and rejection of stale receipts.

See [README.md](README.md) for setup and [EXPERIMENTATION.md](EXPERIMENTATION.md)
for the protocol and declared family deltas.
