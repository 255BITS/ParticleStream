# Dead Ends Log

This file records explored approaches that failed empirically, along with the mathematical/empirical reasons for failure, to prevent oscillation in future iterations. Explorers MUST NOT recommend these failed approaches.

| Iteration | Approach Tried | Why It Failed | Files Touched |
|---|---|---|---|
| M3-Iter1 | Direct GIF writes to `reports/forge/gifs/` during unisolated runs | Overwrote canonical GIFs without writing new receipts, causing Check 5 hash drift | `particlestream/tier1.py` |
| M3-Iter1 | `(step % 2 == 0)` dwell rehearsal | Introduced 5.53:1 update frequency bias (64.8% vs 11.7%), starving past modes below the 12.5% threshold | `particlestream/tier1.py` |
| M3-Iter2 | Learning rate sweep with `lr > 0.006` | High recency gradient oscillation at batch_size=1 causes catastrophic mode collapse to 1 or 2 modes | `particlestream/tier1.py` |
| M3-Iter2 | Prior regularizers (Coulomb repulsion, SBNP, VICReg, SigReg) | Repulsive forces actively oppose the clustering of 32 particles into 4 discrete point-like bar modes, keeping TV high (0.20–0.66) and scoring 0/5 passes | `particlestream/tier1.py` |
| M3-Iter2 | Unconditional/periodic reservoir replay in round-robin stream (`dwell=1`) | Round-robin cycles deterministically through all 4 modes every 4 steps. Random reservoir sampling breaks cyclic sequence, collapsing modes to 2 or 3 | `particlestream/tier1.py` |
| M3-Iter2 | Lowering `reg_coeff` to 0.5 | Collapses B-cap capacity anchor, causing Task 3 direction accuracy to drop to 0.625 (0/5 PASS) and Task 4 to 4/5 PASS | `particlestream/tier1.py` |
| M3-Iter2 | Decoupling prior momentum to `prior_betas = (0, 0.90)` | Causes latent prior to drift out of sync with memory velocity context, dropping Task 3 direction accuracy to 0.656 (0/5 PASS) | `particlestream/tier1.py` |
| M3-Iter2 | Increasing `prior_lr_mult` to 2.5 and `d_lr_mult` to 2.0 with `betas=(0, 0.985)` | High particle plasticity causes particles in Task 4 (`fork_rollout`) to jump between branching trajectories, collapsing Task 4 from 5/5 to 1/5 passes | `particlestream/tier1.py` |
| M3-Iter3 | Checking `len(memory.reservoir) >= memory.reservoir_size` on preallocated tensor buffer | `len(tensor)` returns dimension 0 (64), so `64 >= 64` evaluates to `True` unconditionally at step 0, prematurely triggering reservoir replay before all modes are populated | `particlestream/tier1.py` |
| M3-Iter3 | Optimizing Total Variation (`TV <= 0.15`) without improving particle validity (`hq >= 0.90`) | Reverse triangle inequality proves $\text{finite\_template\_tv} \ge 1.0 - \text{hq}$. At $hq=0.8125$, the theoretical minimum TV is $0.1875$. Particles are already uniformly balanced; convergence failure is strictly due to boundary particles with $RMSE \in (0.13, 0.18)$ | `particlestream/streams.py` |
