# Minimal Tier 1 to Tier 3 experiment targets

Targets set on 2026-11-06. Start with five small proofs covering arbitrary
patterns, discrete output, generalization, arithmetic, and pattern extrapolation.
Their combined cap is **960 training updates for one global recipe**. The other
twelve tests are a backlog, not an automatic execution matrix. The five core
proofs are implemented and their first bounded comparison is concluded in the
[experiment readout](minimal-core-readout.md).

The current [leaderboard](technique-inventory.md) records
`online-feature-prior-bce` passing all ten required Tier 1 tasks. Tier 2 and Tier 3
are not yet declared in ParticleStream. Those ten finite image tasks establish
acquisition and seen-state rollout; they do not establish unseen-pattern rules,
integer arithmetic, or a served token interface. Their declarations and results
remain intact.

## Target table

Here, Tier 1 means acquisition, Tier 2 means held-out combinations and conditional
behavior, and Tier 3 means extrapolation or sustained adaptation. This is a
proposed ParticleStream progression. ParticleGAN's own view calls its tiers
smoke, quality, and endurance; its existing assignments are not imported here.

Every number below is a proposed screen target. Controls and budgets need
validation before qualification. **Core** marks the first five proofs; implement
and measure them in table order, skipping backlog rows. The budget column counts
optimizer updates, not examples hidden inside a batch.

| Tier / order | Experiment | Smallest meaningful question and split | Pass gates | Updates | Destructive control that must fail |
| --- | --- | --- | --- | ---: | --- |
| 1 / **Core 1** | Arbitrary dense motifs | Four unrelated, fixed 8×8 binary images; balanced arrivals. Acquisition beyond hand-picked bars, with no generalization claim. | Valid-image fraction ≥0.90 using maximum pixel error ≤0.10; all four modes; mass TV ≤0.15. | 96 | Output their blurred mean, or remove one mode. |
| 1 / **Core 2** | Four discrete tokens | Four balanced symbols, represented during training by four one-hot pixel slots; serving returns an integer ID or INVALID. | 32/32 stratified served probes valid; all four tokens; mass TV ≤0.10; each accepted output has active slot ≥0.90 and every inactive slot ≤0.10. | 96 | Emit four probabilities of 0.25, or collapse to one token. |
| 1 / Backlog | Full-adder acquisition | All eight bit triples `(a,b,carry_in)` produce `(sum,carry_out)`. This is truth-table acquisition. | All 8/8 answers correct, with each correct bit probability ≥0.90. | 96 | Ignore carry-in or produce XOR without carry-out. |
| 1 / Backlog | Continue a known pattern | Learn `ABC`; give two observed periods, then generate twelve steps at each of three starting phases. | All 3/3 complete continuations correct; raw frame RMSE ≤0.12. | 96 | Repeat the last frame or start in the wrong phase. |
| 1 / Backlog | Ordered two-cue recall | `AA`, `AB`, `BA`, `BB`, then blank and an answer query; four distinct answers. | All 4/4 histories answered correctly; raw answer RMSE ≤0.12. | 96 | Erase either cue or reverse cue order. |
| 2 / **Core 3** | Local kernel on unseen patches | Learn the response of `K=[[1,2],[3,4]]/10` on 12 of the sixteen binary 2×2 patches; hold out patch masks `{3,5,10,12}`. | Both seen and held-out response RMSE ≤0.03; maximum absolute response error ≤0.05. | 192 | Zero output, copy an input pixel, or flip the kernel. |
| 2 / **Core 4** | Addition on unseen pairs | Two-bit operands 0..3; train twelve pairs; hold out `(0,1),(1,3),(2,0),(3,2)`. Every operand and possible sum remains represented in training. | Exact sums on 4/4 held-out and 12/12 seen pairs; correct-bit confidence ≥0.90; carry cases reported separately. | 192 | Drop carry, ignore an operand, or shuffle the condition. |
| 2 / Backlog | Motion at unseen locations | A 2×2 patch bounces through columns `1,2,3,4,3,2`; train rows 1,3,5 and test rows 2,4. Give two real frames, then generate four. | All 12/12 held-out row/phase trajectories correct; centroid RMSE ≤0.05 image widths; raw RMSE ≤0.12. | 192 | Snap to a training row or copy the current frame. |
| 2 / Backlog | New pattern from its prefix | Train six motifs `AAB,AAC,BBA,BBC,CCA,CCB`; hold out all six permutations of `ABC`. Give two observed periods, generate two more. | All 6/6 held-out six-step continuations correct; raw frame RMSE ≤0.12. | 192 | Select a memorized training motif or use a fixed cycle. |
| 2 / Backlog | Recall after an unseen delay | Cue A/B, blank distractors, query, answer. Train delays 1,2; evaluate delays 4,8. | All 4/4 cue/delay answers correct; raw answer RMSE ≤0.12. | 192 | Erase the cue or answer from the query alone. |
| 2 / Backlog | Denoising at new corruptions | Four bars × eight training flip positions; evaluate eight disjoint flip positions per bar. | All 32/32 identities correct; worst-case clean-target RMSE ≤0.08. | 96 | Copy the corrupted input: one 0.1↔0.9 pixel gives RMSE 0.10. |
| 2 / Backlog | Bridge a missing observation | Six bounce phases; two visible frames, a visible missing-frame marker, then the next true frame. | All 6/6 next-frame identities correct; raw frame RMSE ≤0.12. | 192 | Erase motion memory or copy the last visible frame. |
| 2 / Backlog | Respond to actions | Four position glyphs × visible L/R action glyph; predict wrapped successor. | All 8/8 position/action answers correct; raw frame RMSE ≤0.12. | 192 | Ignore action: paired L/R cases then disagree. |
| 3 / **Core 5** | Extrapolate an unseen speed | Train signed speeds ±0.025, ±0.050 image widths/step; separately test interpolation ±0.0375 and extrapolation ±0.075. Three real frames, then three generated frames. | Centroid RMSE ≤0.010 image widths; estimated speed error ≤0.010; raw RMSE ≤0.12; valid point at every step; all eight clips retain correct direction. Both split groups must pass. | 384 | Zero velocity or clamp velocity to the nearest training speed. |
| 3 / Backlog | Apply the local rule spatially | Apply Core 3's learned kernel across an image; train four placements and test four disjoint placements with unseen patch combinations. | Worst-position response RMSE ≤0.03; maximum absolute response error ≤0.05. | 384 | Position lookup or an output fixed at a training position. |
| 3 / Backlog | Longer serial addition | Least-significant-bit-first bit-pair episodes of lengths 1,2; test lengths 3,4 with final carry. The same channels are active at every length. | All sixteen frozen test sums exact, eight at each length; correct-bit confidence ≥0.90; carry-chain subset separately exact. | 384 | Reset carry each step or truncate at the training length. |
| 3 / Backlog | Adapt when the law returns | A={bars 0,1} → B={bars 2,3} → A, 128 updates per phase, with no phase labels. Tests reacquisition, not retention of an inactive distribution. | At each phase's last five checks: HQ ≥0.90; active-law TV ≤0.125; forbidden mass ≤1/32. | 384 | Freeze A throughout, or emit a stationary four-mode mixture. |

## Keep the proofs small

- Start with `online-feature-prior-bce` unchanged across the five core tasks.
  The declared recipe delta is **none**. A failure is useful evidence; do not
  expand its budget, tune its recipe, or search seeds after seeing its result.
  A revised recipe would be a separately declared global candidate.
- Retain protocol seed 0, named RNG streams, deterministic orthogonal
  initialization, real and fake batch size 1, one sequential execution lane,
  no replay, no gradient accumulation, and one update per declared training
  observation. Keep the same task-owned architecture and prior across candidates.
  Start from existing width 16, context dimension 16, 32 particles, and z dimension
  8 where compatible; freeze any necessary new task architecture before training.
- The new 96/192/384-update caps are explicit task/protocol deltas from the
  existing 480-update tasks, whose budgets and gates stay unchanged. Use 24
  checks, every 4/8/16 updates respectively; all last five must pass. For the
  A→B→A test, check every sixteen updates and require each phase's final five:
  64..128, 192..256, and 320..384.
- Use four fixed clean prior centers (rows 0,8,16,24), evaluated sequentially, for
  deterministic conditional tests; require the gates for every center. These
  establish behavior at the declared centers. Distribution tests use all 32 rows,
  each with one frozen MoG perturbation drawn from the isolated evaluation stream.
  Their claims concern those 32 stratified served probes, not validity for every
  future random draw. Preserve original prior capacity; record the exact sampling
  contract. Use neither best-of-N selection nor noisy iid categorical counts for
  a tight mass-TV gate. The analytic oracle must honor the same stratification.
- Evaluation uses only the finite cases named in the task manifest. The core
  kernel has four held-out patches, addition four held-out pairs, and speed eight
  clips. Full seen-set retention checks need run only at the final five checks;
  held-out checks run at all 24. Count evaluation forwards, served calls, training
  updates, peak memory, and elapsed time separately. The 960-update cap is not a
  wall-clock estimate or a claim of minimum compute.
- No trained extra baseline is needed to validate a scorer. Use an analytic
  oracle and the cheap corrupted-output controls in the table. First require
  oracle PASS and every named destructive control FAIL, including on the held-out
  subset. If a control passes, repair the task before training a model.
- A whole-image average can hide incorrect sparse pixels. Dense motifs use a
  maximum-pixel gate; kernel tasks score the response region; token/bit tasks
  score every semantic slot; motion also scores centroids. Invalid outputs count
  as errors rather than disappearing from a denominator.
- Use only compact final metrics, provenance receipts, a short explanation, and
  actual-training `goal.gif` snapshots as committed evidence. Tail operational
  events at `runs/forge/events.jsonl`. Keep the existing single generated
  [leaderboard](technique-inventory.md); do not turn this target table into a
  second result leaderboard or combine unrelated task metrics into a scalar.

## Freeze the few details that affect the claim

The kernel mask uses row-major bit order over the 2×2 patch. The response is a
learned weighted local sum, without supplying kernel coefficients to the model.
This first probe establishes one local response law; the Tier 3 placement test
asks whether it transfers across spatial positions. Neither probe establishes
generalization to every possible convolution kernel.

For the speed test, render a bilinear point and use starting x coordinates
0.45 and 0.55 for each of the four signed evaluation speeds: eight clips total.
Choose a fixed interior y coordinate and verify point support remains inside the
image through all six frames. No clipping, wrapping, or boundary bounce supplies
a shortcut. A valid point has total foreground mass in [0.90,1.10], with at least
90% of that mass in the target's bilinear 2×2 support. Use a zero background and
unit-mass target. This prevents a diffuse frame from passing solely by placing
its mean at the right coordinate. The analytic
renderer must satisfy every gate; nearest-training-speed control must fail the
extrapolation gates. This is a bounded extrapolation result, not unrestricted
mathematical induction.

Generate the four dense motif bitmaps once from a named task-definition stream,
record their bytes, and reject accidental duplicates or near-duplicates before
training. Freeze the sixteen serial-addition cases, all new placement masks,
corruption locations, and source/target episode lengths in the task manifest.
Those backlog rows are design targets until their manifests and controls exist.

Known-pattern continuation does not withhold the transition law: it checks
closed-loop horizon extension. New-motif continuation withholds whole motifs;
copying the supplied prefix is the intended rule. The speed test withholds
velocities. These are different claims and should retain separate results.

## Required implementation before execution

The installed ParticleGAN/Forge dependency is pinned to 0.9.7. Inspecting the
adjacent ParticleGAN checkout provides design context; it does not authorize
silently importing that source into a ParticleStream run.

`OnlineSession.observe()` currently requires `(1,1,8,8)` frames, and the current
generator serves sigmoid images. Token and bit tasks therefore need a declared
pixel code and a serving decoder that returns integer values or INVALID. Raw
confidence gates apply before accepting a code. Nearest-template projection
alone cannot establish a discrete-output result.

Kernel and arithmetic tasks also need an explicit causal paired-observation
adapter: source visible before prediction, target unavailable until its training
arrival. Declare whether a pair is one training observation, its context-update
policy, and all source-only warm-up arrivals. Do not silently introduce a second
optimizer update or use `context_frame` to expose a future target. Encode ordered
recall with an explicit query glyph and a reset contract so answer timing is
identifiable. Every arrival and optimizer update belongs in the receipt.

Run every task through `particlegan` and `particlegan_forge`. Held-out evaluation
must restore weights, modes, memory, learned feature/prior summaries and training
RNG streams. Read-only evaluation prefixes may populate cloned memory; rollout
then consumes generated frames only. No held-out target may train weights or
statistics. Rendering must show learned predictions from training checkpoints;
an oracle animation cannot substitute for an actual-training `goal.gif`.

When implementing the expansion, archive the exact old view revision and
declare new task roles explicitly. Frontier proposals can start as `stretch`;
core versus backlog in this plan is execution priority, not a qualification
role. Any change to required qualification needs its own view revision and
fresh compatible evidence. Report ordinary failures and INVALID runs distinctly.

## Reading and execution order

The source review used three subagents for generalization/continuation,
convolution/discrete/math, and tier coverage/other capabilities. Relevant existing
evidence includes [experiment memory](EXPERIMENT_MEMORY.md),
[current tier declarations](EXPERIMENTS_BY_TIER.md),
[ParticleGAN's tier policy](https://github.com/HyperGAN/ParticleGAN/blob/develop/EXPERIMENTATION.md#tiers-and-views),
and [the online session](../../particlestream/online.py).

The core task manifests, source/decoder adapters, scorers and controls are now
implemented. All five bounded tasks ran sequentially after validation. Their
measured outcomes and costs are in the [readout](minimal-core-readout.md); no
backlog test was launched. These remain the original targets for that cohort.
