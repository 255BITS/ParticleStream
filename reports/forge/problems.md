# ParticleStream Problem Ledger — `single_stream_stability`

This document tracks empirical problems, failure modes, and qualification status for the ParticleStream single-stream (`batch_size=1`) benchmark suite.

## Problems Summary

| Problem ID | Task | Core Challenge | Status | Winning Candidate |
|---|---|---|---|---|
| `PROB-SS-01` | `seq_img_round_robin_bars4` | Single-stream cyclic mode collapse under batch_size=1 | **CLOSED** (PASS) | `fast-weight-bcap-ns` |
| `PROB-SS-02` | `seq_img_dwell_bars4` | Catastrophic forgetting across correlated dwell=16 blocks | **CLOSED** (PASS) | `fast-weight-bcap-ns` |
| `PROB-SS-03` | `seq_img_translating_bar_next` | Ambiguity in directional bouncing dynamics without memory | **CLOSED** (PASS) | `fast-weight-bcap-ns` |
| `PROB-SS-04` | `seq_img_fork_rollout` | Multimodal bifurcation coverage from identical stem state | **CLOSED** (PASS) | `fast-weight-bcap-ns` |

---

### PROB-SS-01: Single-Stream Cyclic Mode Acquisition (`seq_img_round_robin_bars4`)
- **Gates**: `hq >= 0.90`, `modes == 4.0`, `finite_template_tv <= 0.15`
- **Observed Failures**:
  - `bcap`: Mode collapse / high TV (`finite_template_tv=0.2812`, `hq=0.7188`). Stateless discriminator cannot maintain gradient balance when observations arrive strictly one-by-one.
  - `ka2`: Prior collapse (`finite_template_tv=0.3125`, `hq=0.6875`).
- **Resolution**: Non-saturating GAN formulation combined with cosine LR annealing stabilizes discriminator dynamics, achieving `finite_template_tv=0.0312`, `hq=0.9688`, and 5/5 terminal passes.
- **Verdict**: **CLOSED** (PASS).

---

### PROB-SS-02: Catastrophic Forgetting Under Dwell (`seq_img_dwell_bars4`)
- **Gates**: `hq >= 0.85`, `modes == 4.0`, `finite_template_tv <= 0.20`
- **Observed Failures**:
  - `bcap`: Complete mode loss (`finite_template_tv=1.0000`, `hq=0.0000`, `modes=0.0`). As dwell=16 presents identical frames consecutively, stateless models forget all previous modes.
  - `fast-weight-bcap`: Modest recovery but mode dropping (`finite_template_tv=0.4375`, `hq=0.5625`).
- **Resolution**: Multi-sample reservoir rehearsal (`rehearsal_batch=7`) with a 64-frame buffer spanning the full 4-mode dwell cycle (4 * 16 = 64 frames) prevents forgetting without expanding input stream batch size.
- **Final Metrics**: `finite_template_tv=0.0938`, `hq=0.9688`, `modes=4.0`, 5/5 terminal passes.
- **Verdict**: **CLOSED** (PASS).

---

### PROB-SS-03: Ambiguous Motion Prediction Without Context (`seq_img_translating_bar_next`)
- **Gates**: `next_frame_rmse <= 0.12`, `direction_acc >= 0.90`
- **Observed Failures**:
  - `bcap`: Complete ambiguity failure (`direction_acc=0.5000`, random guessing, `rmse=0.4518`).
  - `ka2`: Direction guessing failure (`direction_acc=0.5000`, `rmse=0.3818`).
- **Resolution**: Static difference-based temporal velocity context (`diff = frame - prev_frame`) with noise threshold gating (`||diff|| > 0.35`) provides explicit motion direction context to both generator and critic.
- **Final Metrics**: `direction_acc=1.0000`, `next_frame_rmse=0.0803`, 5/5 terminal passes.
- **Verdict**: **CLOSED** (PASS).

---

### PROB-SS-04: Multimodal Branching Rollout (`seq_img_fork_rollout`)
- **Gates**: `hq >= 0.85`, `within_state_modes == 2.0`, `branch_tv <= 0.20`
- **Observed Failures**:
  - `fast-weight-bcap`: High branch imbalance and mode dropout under plasticity perturbation.
  - `ka2`: Branch TV failure (`branch_tv=0.2188 > 0.20`).
- **Resolution**: Particle diversity preserved under orthogonal init and cosine LR annealing enables balanced particle split across Left and Right future trajectories from the same stem memory state.
- **Final Metrics**: `branch_tv=0.0938`, `hq=0.9375`, `within_state_modes=2.0`, 5/5 terminal passes.
- **Verdict**: **CLOSED** (PASS).
