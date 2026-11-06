# ParticleStream

An eight-task sequential 8×8 image gauntlet using ParticleGAN and its public
Forge Tier API. Train on one arriving frame at a time while serving predictions
from the same session. Real and fake training batches are one; replay is
explicitly disabled. Each candidate uses one global recipe and protocol seed 0.

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev]'
source .venv/bin/activate
particlestream bootstrap
particlestream validate
particlestream run online-feature-prior-bce --device cuda
particlestream board --device cuda
python -m pytest -q
```

Use `--device cpu` without a CUDA GPU. `python -m particlestream.forge` is
equivalent to `particlestream`. Logs are append-only under
`runs/forge/events.jsonl`; each candidate/task also has `run.log` and `goal.gif`.

Read [EXPERIMENTATION.md](EXPERIMENTATION.md) before experiments. The single
[leaderboard](reports/forge/technique-inventory.md),
[task gates](reports/forge/EXPERIMENTS_BY_TIER.md) and
[research notes](reports/forge/research-notes.md) distinguish current evidence
from preserved historical runs. Passing requires every bound at the final five
of 24 checks over 480 training updates. All eight tasks are required in view 5.

`online-feature-prior-bce` is the first global recipe to pass all eight tasks.
It learns a bounded conditional prior over compressed features from past
observations, while the unchanged image generator learns to decode each arriving
frame with binary cross entropy. Its causal reader retains cues through repeated
distractors. The storage prior stays frozen; the conditional transform learns
without replay, preserves the task's MoG noise, and clears training targets
before serving. This is a new memory/prior recipe, with the BCAP critic.

`particlestream compare --device cuda` runs all declared online recipes, including
exploratory declarations without measurements. Use the individual command above
to reproduce the qualified recipe. Earlier source-bound results and failed
experiments remain available in the research notes and cohort archives.
