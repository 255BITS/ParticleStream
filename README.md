# ParticleStream

A four-task sequential 8×8 image gauntlet using ParticleGAN and its public Forge
Tier API. Compare BCAP, KA2 and BISPF trainer families with one recipe per
candidate across all tasks, protocol seed 0 and unchanged numerical gates.

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev]'
source .venv/bin/activate
particlestream bootstrap
particlestream validate
particlestream run stream-bispf-early --device cuda
particlestream compare --device cuda
particlestream board --device cuda
python -m pytest -q
```

Use `--device cpu` without a CUDA GPU. `python -m particlestream.forge` is
equivalent to `particlestream`. Logs are append-only under
`runs/forge/events.jsonl`; each candidate/task also has `run.log` and `goal.gif`.

Read [EXPERIMENTATION.md](EXPERIMENTATION.md) before experiments. The single
[leaderboard](reports/forge/technique-inventory.md),
[task gates](reports/forge/EXPERIMENTS_BY_TIER.md) and
[research notes](reports/forge/research-notes.md) separate current qualification
from historical evidence. Passing requires all bounds at the final five of 24
checks over 480 training steps per task.
