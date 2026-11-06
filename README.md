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
particlestream compare --device cuda
particlestream run online-bispf-early --device cuda
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
of 24 checks over 480 training updates. Four new challenges are held while their
results remain visible; counted failures continue to block qualification.
