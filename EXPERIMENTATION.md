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
   - Use one global recipe across all Tier 1 tasks. Do not run seed-only experiments or select per-task hyperparameters.

## Tier 1 Tasks (`single_stream_stability`)

| Task | Stream (`batch_size=1`) | What It Tests | Numerical Gates |
| --- | --- | --- | --- |
| `seq_img_round_robin_bars4` | Ordered cycle `0, 1, 2, 3, ...` over 4 `8x8` bar templates (`dwell=1`) | Acquisition and stability at `batch_size=1` without random minibatch shuffling | `hq >= 0.90`, `modes == 4`, `finite_template_tv <= 0.15` |
| `seq_img_dwell_bars4` | Correlated blocks of `dwell=16` identical frames per bar before switching | Retention across correlated blocks (anti-chasing / catastrophic forgetting) | `hq >= 0.85`, `modes == 4`, `finite_template_tv <= 0.20` |
| `seq_img_translating_bar_next` | Bouncing vertical bar (`0 -> 7 -> 0`) one frame per step | 1-step velocity disambiguation via memory $M_t$ from ambiguous interior frames | `next_frame_rmse <= 0.12`, `direction_acc >= 0.90` |
| `seq_img_fork_rollout` | Top-to-center bar that splits Left or Right | Within-state multimodal particle branching (`MoGParticlePrior`) from identical memory state | `hq >= 0.85`, `within_state_modes == 2`, `branch_tv <= 0.20` |

## Commands

```sh
# Bootstrap / sync declarations into configs/forge/
python -m particlestream.tier1 bootstrap

# Validate tasks, views, and candidates (launches no training)
python -m particlegan_forge --root . --provider particlestream.tier1 validate

# Run a candidate across Tier 1 and update the single leaderboard + goal.gif artifacts
python -m particlegan_forge --root . --provider particlestream.tier1 run bcap --view single_stream_stability --through-tier 1 --device cpu

# Tail live execution events
tail -F runs/forge/events.jsonl
tail -F runs/forge/<candidate>/<task>/run.log
```
