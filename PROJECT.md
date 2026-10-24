# Project: ParticleStream Single-Stream Benchmark & Global Recipe Research

## Architecture
- **Upstream Engine (`/nvme/dev/particlegan/ParticleGAN`)**:
  - `particlegan`: Core GAN mechanics, architectures, particles, orthogonal initialization (`particlegan.init`).
  - `particlegan_forge`: Namespaced Tier API (`tiers.py`, `rng.py`, `gif.py`, `contracts.py`). Provides `NamedStreams`, `ForgeWorkspace`, `TaskSpec`, `TierView`, `CandidateSpec`, atomic retiering with `view-history/`, receipt serialization, and report compilation.
- **Downstream Benchmark (`/nvme/dev/particlegan/ParticleStream`)**:
  - `particlestream.streams`: 4 data streams (`SequentialBarsStream`, `TranslatingBarStream`, `ForkStream`), templates, scorers (`score_multimodal_bars`, `score_translating_next`, `score_fork_rollout`).
  - `particlestream.models`: `FastWeightMemory` (Hebbian matrix + reservoir replay), `StreamImageGenerator`, `StreamImageCritic`.
  - `particlestream.tier1`: 4 Tier 1 task declarations, `single_stream_stability` view, candidate definitions, `@REGISTRY.adapter("seq_image")` execution loop.
  - `tests`: `test_tier1.py` and E2E verification suites.
  - `reports/forge`: Durable artifacts (`receipts/`, `gifs/`, `technique-inventory.md`, `EXPERIMENTS_BY_TIER.md`, `EXPERIMENT_MEMORY.md`).

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | `particlegan_forge` CLI `validate` | Validates workspace tasks, views, candidates | M1 | Survey |
| 2 | `particlegan_forge` CLI `board` | Displays candidate qualification matrix for a view | M1 | Survey |
| 3 | `particlegan_forge` CLI `run` | Executes candidates through tier budget with receipts | M1 | Survey |
| 4 | `particlegan_forge` CLI `compile` | Compiles single current leaderboard and tier docs | M1 | Survey |
| 5 | `particlegan_forge` CLI `retier` | Atomic view retiering with `configs/forge/view-history/` | M1 | Survey |
| 6 | Top-level Receipt Fields | Expose `completed_steps`, `all_finite`, `unintended_rng_deviations`, `terminal_passing` | M1 | Survey / Acceptance Criteria |
| 7 | Upstream Test & Packaging Cleanliness | Add `tests/__init__.py` in ParticleGAN, ensure tests pass, commit to `develop` | M1 | Survey |
| 8 | Stream Law `requires_grad=False` | Detached non-differentiable tensors for all streams | M2 | Survey / R2 |
| 9 | Deterministic Orthogonal Init & Seed 0 | `particlegan.init` parameter seeds under protocol seed 0 | M2 | Survey / R2 |
| 10 | Isolated `NamedStreams` & Audit | RNG stream families (`init`, `data`, `prior`, `noise`, `eval`) with zero leakage | M2 | Survey / R2 |
| 11 | Oracle PASS & Destructive FAIL Controls | Explicit unit assertions for all 4 task scorers | M2 | Survey / R2 |
| 12 | 4 Tier 1 Task Scorers & Metrics | `seq_img_round_robin_bars4`, `seq_img_dwell_bars4`, `seq_img_translating_bar_next`, `seq_img_fork_rollout` | M2 | Survey / R2 |
| 13 | Sustained 5-Check Terminal Gate | Last 5 of 24 observation checkpoints simultaneously pass | M2 | Survey / R2 |
| 14 | Baseline Candidates Execution | Execute `bcap` and `ka2` on full budget with receipts | M3 | Survey / R3 |
| 15 | `FastWeightMemory` Temporal Fix | Fix pre-write context bug to enable velocity context for translating bar | M3 | Survey / R3 |
| 16 | Replay Reservoir Capacity Scaling | Increase reservoir size to 64+ frames to prevent catastrophic forgetting on dwell=16 | M3 | Survey / R3 |
| 17 | Batch-Size 1 Discriminator Stabilization | Recipe stabilization for single-stream round-robin bars | M3 | Survey / R3 |
| 18 | Global Single-Stream Recipe Qualification | One single global recipe achieves 4/4 PASS on Tier 1 | M3 | Survey / R3 |
| 19 | Actual-Training GIF Generation | Render 4 valid `goal.gif` / `<candidate>_<task>.gif` artifacts | M4 | Survey / R4 |
| 20 | Full-Budget Qualification Receipts | 4 durable receipts for winning recipe (and baselines) with terminal_passing=5 | M4 | Survey / R4 |
| 21 | Single Current Leaderboard Compilation | Single `reports/forge/technique-inventory.md`, updated tier docs, clean git tree | M4 | Survey / R4 |
| 22 | Research Readout | Comprehensive analysis of baseline failures, winning recipe dynamics, and Tier 2 recommendations | M4 | Survey / R3 / R4 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Upstream Forge Tier API & Git Merge | `particlegan_forge` top-level receipt fields, ParticleGAN test packaging fix, validation, commit/merge to `develop` | none | DONE |
| M2 | Tier 1 Benchmark Suite & Controls Verification | Hardened unit tests in `tests/test_tier1.py` covering all streams nograd, oracle PASS / destructive FAIL for 4 tasks, RNG isolation | none | DONE |
| M3 | Global Single-Stream Recipe Research & 4/4 Qualification | Fix `FastWeightMemory` context, reservoir capacity, stabilize recipe; run baselines (`bcap`, `ka2`) and qualify 1 global recipe to 4/4 PASS on CUDA | M1, M2 | DONE |
| M4 | Artifact Generation, Single Leaderboard & Hygiene | Render GIFs, generate receipts, compile single leaderboard (`technique-inventory.md`), verify `runs/` untracked, final readout | M3 | DONE |

## Interface Contracts
### `particlegan_forge` ↔ `particlestream`
- **Adapter Registration**:
  `@REGISTRY.adapter("seq_image")`
  Signature: `run_seq_image_task(workspace: ForgeWorkspace, task: TaskSpec, candidate: CandidateSpec, seed: int = 0, device: str = "cpu", step_callbacks = None) -> dict`
- **Receipt Schema**:
  ```json
  {
    "task_id": str,
    "candidate_id": str,
    "seed": int,
    "device": str,
    "completed_steps": int,
    "all_finite": bool,
    "unintended_rng_deviations": int,
    "terminal_passing": int,
    "grade": {
      "status": "PASS" | "FAIL" | "INVALID" | "INCOMPLETE",
      "terminal_passing": int,
      "failed_bounds": list
    }
  }
  ```
- **CLI Invocations**:
  - `python -m particlegan_forge --root <dir> --provider particlestream.tier1 validate`
  - `python -m particlegan_forge --root <dir> --provider particlestream.tier1 board --view <view_id> [--device cuda]`
  - `python -m particlegan_forge --root <dir> --provider particlestream.tier1 run <candidate> --view <view_id> [--device cuda]`
  - `python -m particlegan_forge --root <dir> compile [--view <view_id>]`
  - `python -m particlegan_forge --root <dir> retier <view_id> --task <id> --tier <int> --reason <str>`

## Code Layout
- `/nvme/dev/particlegan/ParticleGAN/`:
  - `particlegan/`: core library
  - `particlegan_forge/`: namespaced forge tier engine
  - `tests/`: ParticleGAN tests
- `/nvme/dev/particlegan/ParticleStream/`:
  - `particlestream/`: benchmark package (`tier1.py`, `models.py`, `streams.py`)
  - `configs/forge/`: task and view metadata
  - `tests/`: test suite
  - `reports/forge/`: receipts, GIFs, and markdown leaderboards
  - `runs/`: execution logs (untracked)
