**Experiment work: read [EXPERIMENTATION.md](EXPERIMENTATION.md) and [RESEARCH_GLOSSARY.md](RESEARCH_GLOSSARY.md) first.**
Read [reports/forge/EXPERIMENT_MEMORY.md](reports/forge/EXPERIMENT_MEMORY.md) and [reports/forge/technique-inventory.md](reports/forge/technique-inventory.md) before proposing another idea.

- Do not run seed-only experiments (protocol seed is `0`).
- Be token efficient and keep logs easy to tail (`runs/forge/events.jsonl`).
- Use one global recipe across tasks; declare recipe deltas explicitly.
- Every task must execute through `particlegan` and `particlegan_forge`, declare numerical pass/fail gates, validate with oracle and destructive controls, and produce an actual-training `goal.gif`.
- Only keep one generated leaderboard per goal (`reports/forge/technique-inventory.md`).
- Summarize and give explanations, leaderboard, and recommendations after running experiments.
- commit and push as you go
