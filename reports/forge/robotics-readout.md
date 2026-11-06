# Five Gymnasium-Robotics dynamics experiments

This expansion uses real MuJoCo transitions from `FetchReach-v4`, `FetchPush-v4`,
`FetchSlide-v4`, `FetchPickAndPlace-v4` and `HandReach-v3`. These are bounded,
action-conditioned, one-step world-model tests. Passing does not establish a
learned control policy, goal-reaching success, camera-image prediction or
closed-loop rollout. Rewards and simulator success flags are recorded only as
diagnostics and are never used by the learner or excitation policy.

## Frozen protocol

- One global candidate: `online-feature-prior-bce`; recipe delta `{}` for all five.
  The generator, critic, optimizer, learned feature prior and causal memory
  retain the existing recipe and 8×8 coordinate-raster architecture.
- Protocol seed `0`, named initialization/data/prior/noise/evaluation streams.
  Each simulator reset receives a draw from its split's named stream; these
  episode resets are not alternate experiment seeds.
- Each task trains 480 chronological transitions: twelve 40-step episodes.
  Five tasks run sequentially: 2,400 optimizer updates in total. Real and fake
  batches are one, with no replay or gradient accumulation.
- Before each transition, the full current state (10/25/63 coordinates) arrives
  as `0.5 + 0.5*tanh(state)`, then the chosen action (4/20 coordinates) arrives
  as `0.5 + 0.5*action`. Both fit their separate 64-value frames without
  truncation. Observation memory resets between independent transition pairs;
  learned weights, feature statistics and optimizer state persist.
- Serving completes before `env.step(action)` reveals the target. One later
  target arrival trains once against the saved state/action context. Targets
  never enter observation memory. There are 960 source-only arrivals and 480
  target arrivals per task.
- Targets are gripper displacement for Reach, gripper plus object displacement
  for Push/Slide/PickAndPlace, and all five fingertip displacements for HandReach.
  Coordinates in metres repeat across the 64 target pixels, encoded as
  `0.5 + displacement/0.2`. Out-of-range motion is rejected; it is never clipped.
  Decoding averages only fixed repeat slots, with no target snapping.
- Fixed task-owned excitation uses uniform actions for Reach/HandReach;
  approach/contact/push phases for Push/Slide; approach/close/lift phases for
  PickAndPlace. This collection policy is not trained. No policy search occurs.
- Four separate evaluation-only episodes supply sixteen transitions, at steps
  7, 17, 27 and 37. They are generated once from the named evaluation stream,
  never optimized, and reused at all 24 checkpoints. Four clean latent centres
  (rows 0/8/16/24) are scored independently for every case, without best-of-N
  selection. This is a finite held-out set, not a generalization guarantee.

## Numerical gates and controls

All five terminal checks (updates 400, 420, 440, 460, 480) must meet every gate:

| Metric | Gate | Meaning |
| --- | --- | --- |
| `position_delta_rmse_m` | `<= 0.005` | At most 5 mm RMS position-change error |
| `relative_mse` | `<= 0.80` | At least 20% lower MSE than predicting no motion |
| `moving_direction_accuracy` | `>= 0.75` | Correct sign on coordinates moving at least 1 mm |
| `finite_fraction` | `== 1` | Every raw prediction finite and in its declared range |

The production scorer must pass the exact simulator-target oracle and fail
persistence, reversed motion, shuffled targets and an opposite-action oracle.
The last control restores MuJoCo's complete integration state and simulates
the opposite action in the evaluation-only environment. The observed
no-motion baseline RMS spans 8.22–16.76 mm; object RMS spans 12.55–14.86 mm,
so the manipulation probes include real object motion. These are data/control
measurements before training, not learned-model results.

All five tests are Tier 2 `stretch` in view 9. View 8 is archived byte-for-byte;
the ten required toy tasks and five minimal probe declarations remain intact.
The only generated leaderboard remains [technique-inventory.md](technique-inventory.md).

## Reproduction and evidence

```sh
uv pip install --python .venv/bin/python -e '.[dev,robotics]'
.venv/bin/particlestream bootstrap
.venv/bin/particlestream robotics --controls-only
.venv/bin/particlestream robotics --device cuda
tail -F runs/forge/events.jsonl
```

Dependencies are pinned to Gymnasium 1.4.0, Gymnasium-Robotics 1.4.2 and
MuJoCo 3.3.7, alongside ParticleGAN/Forge 0.9.7. Receipts include resolved recipe,
simulator versions, controls, named reset seeds, compute counts, source hashes,
and a SHA256-bound simulator trace under `reports/forge/traces/`. Training GIFs
show actual checkpoint predictions and targets as coordinate rasters; they are
not rendered camera reconstructions. Full traces also remain in each run folder.

Measured results and recommendations will be added after the fixed-budget runs.

Environment definitions: [Fetch](https://robotics.farama.org/envs/fetch/),
[HandReach](https://robotics.farama.org/envs/shadow_dexterous_hand/reach/).
