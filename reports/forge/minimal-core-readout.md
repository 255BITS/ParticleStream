# Five minimal capability proofs

The unchanged `online-feature-prior-bce` recipe passes discrete-token acquisition
in 96 updates. Addition acquires all twelve training pairs but answers none of
the four withheld pairs correctly throughout the final five checks. Arbitrary
dense images, local kernel responses, and unseen-speed rollout fail their
preregistered small-budget gates.

The five sequential runs consumed **960 optimizer updates and 51.35 seconds** on
one NVIDIA RTX A6000. No parameter search, budget extension, seed change, or
training retry followed these results. All runs were finite with zero unintended
RNG deviations. The existing ten required tests still qualify **10/10, Tier 1**;
the five new stretch tests score **1/5**. The canonical
[leaderboard](technique-inventory.md) contains both results.

## Measurements

Values below are from the final checkpoint. Qualification uses all final five
checks, which are retained in the source-bound receipts. Deterministic tasks
require each of four clean prior centers; distribution tasks score 32 fixed,
stratified MoG draws with their perturbations retained.

| Experiment | Outcome | Final measurement | Terminal checks | Updates | Complete run time | Training GIF |
| --- | --- | --- | --- | ---: | ---: | --- |
| Arbitrary dense images | FAIL | Valid fraction 0.00; valid modes 0/4; TV 1.00 | 0/5 | 96 | 8.54 s | [goal.gif](gifs/online-feature-prior-bce_seq_minimal_dense4.gif) |
| Four integer tokens | PASS | Valid fraction 1.00; modes 4/4; TV 0.00 | 5/5 | 96 | 6.26 s | [goal.gif](gifs/online-feature-prior-bce_seq_minimal_tokens4.gif) |
| Held-out local kernel | FAIL | Seen RMSE 0.0427; held-out RMSE 0.2614; held-out maximum error 0.4325 | 0/5 | 192 | 8.15 s | [goal.gif](gifs/online-feature-prior-bce_seq_minimal_kernel2x2.gif) |
| Held-out addition pairs | FAIL | Seen 12/12 exact; held-out 0/4 exact; seen minimum bit confidence 0.9791 | 0/5 | 192 | 9.48 s | [goal.gif](gifs/online-feature-prior-bce_seq_minimal_addition2bit.gif) |
| Unseen-speed rollout | FAIL | Extrapolation centroid RMSE 0.0833 image widths; maximum speed error 0.0740; valid-point fraction 0.5833 | 0/5 | 384 | 18.92 s | [goal.gif](gifs/online-feature-prior-bce_seq_minimal_velocity_extrapolation.gif) |

The local kernel gates require RMSE ≤0.03 and maximum absolute error ≤0.05 on
both seen and held-out inputs. It misses the acquisition gates as well, so this
run does not isolate a generalization-only failure. Dense-image acquisition also
fails at its declared 96-update budget; this is not evidence that the task is
impossible or that the recipe cannot solve it at another preregistered budget.

Addition gives the clearest boundary. Seen accuracy is 1.00 and held-out accuracy
0.00 at every terminal check, with seen minimum correct-bit confidence at least
0.9743. Every operand and possible sum occurs in training. The held-out result
therefore cannot be explained by an unseen output vocabulary. Finite context
lookup is a plausible explanation given the recipe's nearest-context prior, but
this comparison does not ablate that mechanism.

Velocity also fails interpolation: final centroid RMSE is 0.0300 versus the
0.0100 gate. Extrapolation raw image RMSE is only 0.0536 and would pass the 0.12
image gate alone. Position, speed, and foreground-support gates expose errors
that sparse-image RMSE would conceal. Rollouts consume three real prefix frames
and then generated frames only.

Token success concerns the declared integer decoder and its finite 32-draw
probe. Accepted raw outputs must match a whole one-hot code within 0.10,
including its background. An ambiguous soft vector becomes INVALID. This is not
a claim of a native categorical generator or validity for every future random
draw.

## Compute and controls

The 960 updates use real and fake batch size one, with no replay or gradient
accumulation. Independent kernel/addition examples add 384 source-only arrivals;
velocity adds another 384 prefix arrivals. Total arrivals are 1,728: 960 trained
targets plus 768 explicit source-only observations.

Training calls account for 1,920 generator forwards and 5,760 critic forwards.
Serving adds 960 generator forwards. Evaluation adds 5,088 generator forwards
and 13.07 seconds. Complete run time includes initialization, control checking,
evaluation and GIF rendering, so 51.35 seconds is not a pure optimization time.
Peak PyTorch CUDA tensor allocation is 9.13 MiB; process peak RSS is 2.02 GiB.
The unconditional G/D have 10,290 parameters together; conditional G/D have
15,858. All compute details are retained in the receipts.

All five analytic oracles pass and all fifteen destructive controls fail through
the same production scorers. Controls include blurred or missing modes, a soft
one-hot vector, horizontally/vertically flipped kernels, ignored operands and
removed carry, zero velocity, clamped training velocity, blank and diffuse
frames. Held-out kernel and addition controls also fail their held-out gates.
The kernel split was chosen to detect flips before any model was trained.

Software validation passed **497 tests**. Post-run verification regraded all five
receipts with `particlegan_forge`, matched implementation fingerprints and GIF
hashes, checked 24 GIF frames each, verified all budgets and batch sizes, and
confirmed the old ten-task qualification remains compatible. Operational events
are at `runs/forge/events.jsonl`; raw logs are not committed.

## Recommendation and reproduction

Use held-out addition as the next small mechanism comparison: it already
separates successful acquisition from failed generalization with four withheld
cases. Keep this cohort's task definitions, gates and costs intact. Require seen
acquisition before attributing future kernel or dense-image failures to
generalization. Keep the other twelve proposed tests as a backlog until a new
bounded question is selected.

The execution source is commit `93fee25`, using installed ParticleGAN/Forge
0.9.7, protocol seed 0 and one unchanged global recipe. Task definitions are
under `configs/forge/tasks/seq_minimal_*.json`; view 8 preserves the exact old
view 7 in `configs/forge/view-history/`. New tasks are explicitly stretch, so
none of these results claims Tier 2 or Tier 3 qualification.

```sh
particlestream bootstrap
particlestream controls
particlestream core --device cuda
```

The [target plan](minimal-tier-expansion-plan.md) specifies scope and budgets.
The [machine-readable readout](minimal-core-readout.json) links every receipt
and GIF with hashes, cohort keys, final metrics and measured costs.
