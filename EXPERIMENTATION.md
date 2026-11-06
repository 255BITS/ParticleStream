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
python -m particlestream.forge --root . run online-feature-prior-bce --view single_stream_stability --through-tier 1 --device cpu

# Tail live execution events
tail -F runs/forge/events.jsonl
tail -F runs/forge/<candidate>/<task>/run.log
```

## Strict online execution

ParticleGAN (including `particlegan_forge`) is pinned to **0.9.7**. Tests import
its installed wheel, never the adjacent source checkout. `StreamWorkspace` extends the public
`particlegan_forge.ForgeWorkspace`; use the project CLI so validation checks the
actual `configs/forge` workspace.

Every new run uses chronological, single-frame arrivals and **real and fake
training batches of one**. The online executor has no replay concatenation,
stored-frame substitution, extra replay updates, or gradient accumulation.
Replay is an explicit recipe policy: executable candidates declare
`replay_policy=none` and `rehearsal_batch=0`. Historical replay recipes remain
visible but execution rejects them before touching their saved receipts.

`OnlineSession` supports serving and training throughout the same session.
`predict()` serves from current learned weights and detached context; `observe()`
accepts the next observation and trains. Calls are serialized so inference and
training can share state safely. Serving uses a dedicated evaluation RNG and
must not alter future training, model modes, memory, or training RNG streams.
This permits interleaved inference and learning; it does not promise parallel
GPU kernels or a throughput target.

With an initialized public ParticleGAN trainer, networks and named RNG streams,
the same API used by the experiment adapter is available to an application:

```python
from particlestream import OnlineSession

session = OnlineSession(trainer, generator, critic, memory,
                        predictive=True, serving_generator=serving_rng)
first_frame, _ = stream.next_frame(data_rng)
session.observe(first_frame)  # declared warm-up, no optimizer update
prediction = session.predict()  # serve before fetching the target
next_frame, _ = stream.next_frame(data_rng)
stats = session.observe(next_frame)  # learn using the preceding context
```

For prediction tasks, the first frame is a declared warm-up. A prediction is
made before the next frame is fetched, then that newly arrived frame trains
against the context saved from the preceding arrival. Memory is updated only
after that training update. Thus 480 predictive updates consume 481 arrivals;
unconditional image tasks consume 480 arrivals for 480 updates. Receipts record
arrivals, updates, warm-up, serving calls, batch sizes and prequential errors.

The three new global recipes inherit their parents' optimizer, loss, prior,
annealing and context policies unchanged; their declared delta disables replay:

| New recipe | Historical parent | Trainer preset |
| --- | --- | --- |
| `online-bcap-adam` | `stream-bcap-adam` | `bcap_adam` |
| `online-bispf-early` | `stream-bispf-early` | `bispf` |
| `online-ka2-ready` | `stream-ka2-ready` | `ka2` |

The 32 prior particles remain task-owned capacity and evaluation samples.
They are not a training minibatch. Architectures remain task-owned: all
candidates use the same network for a given task, and stateless recipes use
zero context. Do not tune a separate recipe for each task.

## Additional Tier 1 challenges

All challenges retain 8×8 images, 32 particles, 480 updates, 24 observations and
five required terminal checks. The original four numerical gates stay unchanged.

| Task | Stream | Numerical gates |
| --- | --- | --- |
| `seq_img_variable_dwell_bars4` | Balanced rotating dwell lengths 4, 8, 16 and 32 | `hq>=0.85`, `modes==4`, `finite_template_tv<=0.20` |
| `seq_img_rare_mode_bars4` | Four templates in exact 5:5:5:1 proportions | `hq>=0.85`, distribution TV `<=0.15`, `support_modes==4`, `1/32<=rare_mass<=3/32` |
| `seq_img_delayed_cue` | Cue, four identical distractors, cue-dependent outcome | `outcome_accuracy>=0.90`, `prediction_rmse<=0.12` |
| `seq_img_persistent_fork` | Stem and a three-frame left/right future | At every depth: `hq>=0.85`, `within_state_modes==2`, `branch_tv<=0.20`; `trajectory_consistency>=0.95` |
| `seq_img_circle_orbit` | Clockwise 8-waypoint circular orbit (`N, NE, E, SE, S, SW, W, NW`), one point frame per step | Closed-loop 8-step rollout without teacher forcing: `orbit_rmse<=0.12`, `xy_rmse<=0.15`, `step_accuracy>=0.90`, `orbit_completion>=0.90`, `waypoint_modes==8` |

Rare-mode TV uses the declared nonuniform target and includes invalid mass.
Support means at least one valid particle per mode, not the uniform task's
minimum-frequency rule. Delayed-cue evaluation balances both cue histories;
cue labels, stream indices and future frames are never model inputs. Persistent
fork evaluation keeps each particle's latent identity and Gaussian perturbation,
then feeds its generated frames through a cloned memory state. It receives no
real future frames. Both fork streams draw independent fair branches when the
branch arrives; evaluation checks both attainable previous-branch-to-stem
histories separately and requires the worst state to meet every gate.

The four challenges started **held** in view 4. View 5 restores all four as
**required** after one global recipe passes every task with all five terminal
checks. The exact previous view is archived. No numerical gate, architecture,
stream law or training budget changes during this promotion.

Every task has oracle PASS and destructive FAIL controls. New controls include
latest-mode-only output, rare-mode removal, cue erasure and branch switching
with otherwise perfect per-frame marginals. The historical fast-weight context
only uses the last frame difference; delayed recall is an experiment, not an
assumed capability. The accumulated fast-weight matrix is not currently read.

## Evidence and reproduction

Run the complete new comparison with:

```sh
particlestream bootstrap
particlestream validate
particlestream compare --device cuda
```

Select an individual global recipe with:

```sh
particlestream run online-bcap-adam --device cuda
```

The default comparison selects all declared `online-*` recipes. Historical
`stream-*` receipts and GIFs are preserved and cannot qualify against the
revised execution contract. Changing the adapter or dependent source requires
fresh source-bound receipts. Short software checks never qualify a candidate.

Results, including failures, are generated into the single
[leaderboard](reports/forge/technique-inventory.md). See
[research notes](reports/forge/research-notes.md) for historical interpretation
and current measurements. `goal.gif` frames come from actual training snapshots.

## Strict online recovery recipes

The recovery comparison first gives each previously unrun historical recipe a
new explicit no-replay identity. Four separate BISPF ablations then test a frozen
prior, prior-rate multiplier 0.25, critic betas (0.95,0.999), and critic-rate
multiplier 0.25. Each candidate runs all eight tasks with one complete recipe.

The opt-in `reader=causal_state` policy preserves the same 16-dimensional context
and projection parameter shapes. Its detached, bounded state carries current
appearance, last-step change, retained contrast-bearing content, repetition age,
and an associative-matrix query. It uses arriving pixels only. Cue labels,
templates, stream metadata and future frames are unavailable to the reader.

The opt-in `memory_policy.transport` configuration uses the public ParticleGAN
prior-manipulator interface. The critic still samples the uniform mixture. The
generator routes the current observation to one particle using image MSE plus a
declared usage penalty, and adds reconstruction on that one generated image.
Thirty-two clean particle-center probes execute sequentially without gradients
per update; this is additional inference compute, not additional optimization.
The target clears after the update, including on exceptions. Serving retains
the original prior law and receives no target. Numerical gates, architectures,
480-update budgets and the sustained five-check rule remain unchanged.

The initial transport policy uses reconstruction weight 10, temperature 0,
usage penalty 0.01 and no usage decay. Combined fitting candidates declare every
additional optimizer or memory change in their JSON. These are distinct
experiments; a combination does not isolate the contribution of each change.

Archived task declarations and unchanged receipt/GIF copies are stored under
`reports/forge/cohorts/pre-recovery/`. The manifest also protects the original
historical artifacts. Receipts from earlier implementations cannot qualify
against new source fingerprints; the archive retains their measured outcomes.

The qualified `online-feature-prior-bce` recipe uses a different declared prior
mechanism. A fixed 8×64 orthogonal projection encodes each arriving image; its
initialization comes from the named protocol `init` stream. Up to 32 feature
means and 64 context/outcome summaries learn from successful arrivals. Critic
and serving draws use empirical quantiles at the nearest observed context;
every row preserves its original MoG perturbation. The generator's training
posterior uses only the current arrived frame and BCE weight 10. Statistics
commit after the update succeeds, and serving receives no target or stored
image. The storage prior is frozen while this bounded conditional transform
learns. It uses zero center probes and the same task-owned G/D networks.

Reproduce the qualified recipe with `particlestream run online-feature-prior-bce
--device cuda`. All nine tests are required in view 6 (`single_stream_stability-v5.json`
is archived in `configs/forge/view-history/`). Promoting the view changes
the full adapter-module digest, so the first passing view-4 cohort is archived
in `reports/forge/cohorts/feature-prior-view4/`, the view-5 cohort is archived
in `reports/forge/cohorts/particlegan-0.9.6-view5/`, and fresh runs bind qualification
to the view-6 implementation. Receipts are never manually rebound.
