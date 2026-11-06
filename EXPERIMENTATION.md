# Experimentation with ParticleStream (`particlegan_forge`)

Read this before proposing or running an experiment. ParticleStream uses [`particlegan`](../ParticleGAN/particlegan) and the namespaced [`particlegan_forge`](../ParticleGAN/particlegan_forge) Tier API to evaluate single-stream (`batch_size=1`) sequential learning and world-model memory under repeatable tiered gates.

Start with:
- [Research glossary](RESEARCH_GLOSSARY.md)
- [Current technique leaderboard](reports/forge/technique-inventory.md)
- [Experiments by tier](reports/forge/EXPERIMENTS_BY_TIER.md)
- [Compiled experiment memory](reports/forge/EXPERIMENT_MEMORY.md)

## Protocol and Ownership

1. **Protocol Seed `0` & Named RNG Streams**:
   - All runs use protocol seed `0`, `particlegan.init.deterministic_orthogonal_`, and `particlegan_forge.NamedStreams` (`init`, `data`, `prior`, `noise`, `eval`).
   - Evaluation callbacks execute inside `streams.isolated_evaluation()`; any mutation of non-`eval` streams sets `unintended_rng_deviations > 0` and grades the run `INVALID`.
2. **Task vs. Recipe Ownership**:
   - **Task owns**: Stream law (`requires_grad=False`), `batch_size=1`, network architectures, initial prior distribution (`MoGParticlePrior` / `ParticlePrior`) and capacity (`num_particles`, `z_dim`), rollout horizon, and numerical gates (`24` observations, last `5` terminal checks passing).
   - **Recipe owns**: Losses, critic penalty, optimizer, prior update/regularization policy, and memory update/decay policy.
3. **One Global Recipe**:
   - Use one global recipe per candidate across all Tier 1 tasks. Compare distinct trainer families without per-task selection. Do not run seed-only experiments or select per-task hyperparameters.

## Tier 1 Tasks (`single_stream_stability`)

| Task | Stream (`batch_size=1`) | What It Tests | Numerical Gates |
| --- | --- | --- | --- |
| `seq_img_round_robin_bars4` | Ordered cycle `0, 1, 2, 3, ...` over 4 `8x8` bar templates (`dwell=1`) | Acquisition and stability at `batch_size=1` without random minibatch shuffling | `hq >= 0.90`, `modes == 4`, `finite_template_tv <= 0.15` |
| `seq_img_dwell_bars4` | Correlated blocks of `dwell=16` identical frames per bar before switching | Retention across correlated blocks (anti-chasing / catastrophic forgetting) | `hq >= 0.85`, `modes == 4`, `finite_template_tv <= 0.20` |
| `seq_img_translating_bar_next` | Bouncing vertical bar (four positions, `0,1,2,3,2,1`) one frame per step | 1-step velocity disambiguation via memory $M_t$ from ambiguous interior frames | `next_frame_rmse <= 0.12`, `direction_acc >= 0.90` |
| `seq_img_fork_rollout` | Top-to-center bar that splits Left or Right | Within-state multimodal particle branching (`MoGParticlePrior`) from identical memory state | `hq >= 0.85`, `within_state_modes == 2`, `branch_tv <= 0.20` |

## Commands

```sh
# With the project virtual environment active, sync current declarations into configs/forge/
python -m particlestream.forge bootstrap

# Validate tasks, views, and candidates (launches no training)
python -m particlestream.forge --root . validate

# Run a candidate across Tier 1 and update the single leaderboard + goal.gif artifacts
python -m particlestream.forge --root . run stream-bispf --view single_stream_stability --through-tier 1 --device cpu

# Tail live execution events
tail -F runs/forge/events.jsonl
tail -F runs/forge/<candidate>/<task>/run.log
```

## Current engine and trainer families

ParticleGAN is pinned to **0.9.6**. Tests import the installed wheel, never the
adjacent source checkout. `particlestream.forge.StreamWorkspace` extends the
public `particlegan_forge.ForgeWorkspace` with this repository's established
paths and canonical `technique-inventory.md` filename. Use the project CLI above;
the upstream CLI uses different default directories and can validate an empty
workspace. Project validation rejects an empty gauntlet.

| Candidate | Trainer family / public preset | Shared memory |
| --- | --- | --- |
| `stream-bcap-adam` | BCAP / `bcap_adam` | 64-frame FIFO, 7 replay samples, velocity context |
| `stream-bcap` | BCAP / `bcap` (DualNorm) | same |
| `stream-ka2` | KA2 / `ka2` | same |
| `stream-bispf` | BISPF / `bispf` | same |
| `stream-bispf-optimized` | BISPF / `bispf_optimized` | same |

All five candidates declare non-saturating loss and cosine annealing from the
halfway point to a 0.05 floor. Adam/formulation variants use lr=0.005 and
betas=(0,0.999); DualNorm variants retain preset rates. BISPF explicitly disables
SBNP (`prior_regularizer=none`, `prior_reg=0`) for discrete-mode clustering and
uses float32. These are declared recipe deltas, not unchanged upstream defaults.
The optimized BISPF candidate retains its BISPF critic and DualNorm optimizer.

The arriving stream emits one observation per step. Replay candidates concatenate
up to seven stored observations into the training tensor; the optimizer's real
batch can therefore contain eight frames. This distinction is part of the recipe.
Network architecture is task-owned and identical across candidates; stateless
candidates use zero context in the same conditional network.

Qualification requires fresh source-bound receipts. Historical receipts and GIFs
remain in place but cannot qualify against the upgraded engine. See
[research notes](reports/forge/research-notes.md) for the previous cohort and
interpretation. Never promote a shortened software check to qualification.
