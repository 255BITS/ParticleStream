# Technique leaderboard — `single_stream_stability` (view 10)

Generated from `particlegan_forge` qualification of receipts under `reports/forge/receipts/cuda`; protocol seed 0, sequential real/fake batch size 1.

Click a technique for its explanation, a recipe for its exact configuration and parent delta, or a score for gates, measurements, source-bound receipts and actual-training GIFs. One global recipe supplies every score in a row. Stretch results do not grant qualification credit.

**Latest refresh: completed, 600/600 runs recorded.** Started 2026-11-07T16:02:10.829072+00:00; completed 2026-11-07T18:20:55.653601+00:00. Controls: 20 oracle PASS, 45 destructive FAIL. [Refresh audit and artifact hashes](refresh-audit.json).

`*` means at least one current task result is missing. A displayed zero counts passes; the measured column distinguishes an observed failure from missing or stale evidence. Historical recipes are listed separately and receive no current score.

## Representative recipes

One complete recipe per technique, ordered by required qualification and passes, then stretch passes. Incomplete recipes are considered only when a technique has no complete required measurement. Ties use measured coverage and recipe name.

| Technique | Exact global recipe | Tier 1 required | Tier 2 required | Tier 3 required | Total required | Measured tasks | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [BCAP](families/bcap.md) | [`online-bcap`](families/bcap.md#online-bcap) | [0/10*](families/bcap.md#online-bcap-tier-1-required) | — | — | [0/10*](families/bcap.md#online-bcap-required) | 0/20 | `INCOMPLETE` |
| [BISPF](families/bispf.md) | [`online-bispf`](families/bispf.md#online-bispf) | [0/10*](families/bispf.md#online-bispf-tier-1-required) | — | — | [0/10*](families/bispf.md#online-bispf-required) | 0/20 | `INCOMPLETE` |
| [FastWeight-BCAP](families/fastweight-bcap.md) | [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | [0/10*](families/fastweight-bcap.md#online-fast-weight-bcap-tier-1-required) | — | — | [0/10*](families/fastweight-bcap.md#online-fast-weight-bcap-required) | 0/20 | `INCOMPLETE` |
| [Global-Stream-BCAP](families/global-stream-bcap.md) | [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | [0/10*](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-1-required) | — | — | [0/10*](families/global-stream-bcap.md#online-fast-weight-bcap-ns-required) | 0/20 | `INCOMPLETE` |
| [FeaturePrior-BCAP](families/featureprior-bcap.md) | [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | [0/10*](families/featureprior-bcap.md#online-feature-prior-bce-tier-1-required) | — | — | [0/10*](families/featureprior-bcap.md#online-feature-prior-bce-required) | 0/20 | `INCOMPLETE` |
| [GAIL-BCAP](families/gail-bcap.md) | [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | [0/10*](families/gail-bcap.md#online-gail-bcap-bce-tier-1-required) | — | — | [0/10*](families/gail-bcap.md#online-gail-bcap-bce-required) | 0/20 | `INCOMPLETE` |
| [KA2](families/ka2.md) | [`online-ka2`](families/ka2.md#online-ka2) | [0/10*](families/ka2.md#online-ka2-tier-1-required) | — | — | [0/10*](families/ka2.md#online-ka2-required) | 0/20 | `INCOMPLETE` |

## All executable recipes

| Technique | Exact global recipe | Tier 1 required | Tier 2 required | Tier 3 required | Total required | Measured tasks | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [BCAP](families/bcap.md) | [`online-bcap`](families/bcap.md#online-bcap) | [0/10*](families/bcap.md#online-bcap-tier-1-required) | — | — | [0/10*](families/bcap.md#online-bcap-required) | 0/20 | `INCOMPLETE` |
| [BCAP](families/bcap.md) | [`online-bcap-adam`](families/bcap.md#online-bcap-adam) | [0/10*](families/bcap.md#online-bcap-adam-tier-1-required) | — | — | [0/10*](families/bcap.md#online-bcap-adam-required) | 0/20 | `INCOMPLETE` |
| [BCAP](families/bcap.md) | [`online-bcap-adam-causal`](families/bcap.md#online-bcap-adam-causal) | [0/10*](families/bcap.md#online-bcap-adam-causal-tier-1-required) | — | — | [0/10*](families/bcap.md#online-bcap-adam-causal-required) | 0/20 | `INCOMPLETE` |
| [BCAP](families/bcap.md) | [`online-bcap-balanced`](families/bcap.md#online-bcap-balanced) | [0/10*](families/bcap.md#online-bcap-balanced-tier-1-required) | — | — | [0/10*](families/bcap.md#online-bcap-balanced-required) | 0/20 | `INCOMPLETE` |
| [BCAP](families/bcap.md) | [`online-bcap-balanced-bce`](families/bcap.md#online-bcap-balanced-bce) | [0/10*](families/bcap.md#online-bcap-balanced-bce-tier-1-required) | — | — | [0/10*](families/bcap.md#online-bcap-balanced-bce-required) | 0/20 | `INCOMPLETE` |
| [BCAP](families/bcap.md) | [`online-bcap-balanced-frozen`](families/bcap.md#online-bcap-balanced-frozen) | [0/10*](families/bcap.md#online-bcap-balanced-frozen-tier-1-required) | — | — | [0/10*](families/bcap.md#online-bcap-balanced-frozen-required) | 0/20 | `INCOMPLETE` |
| [BCAP](families/bcap.md) | [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) | [0/10*](families/bcap.md#online-bcap-stateless-tier-1-required) | — | — | [0/10*](families/bcap.md#online-bcap-stateless-required) | 0/20 | `INCOMPLETE` |
| [BCAP](families/bcap.md) | [`online-bcap-transport`](families/bcap.md#online-bcap-transport) | [0/10*](families/bcap.md#online-bcap-transport-tier-1-required) | — | — | [0/10*](families/bcap.md#online-bcap-transport-required) | 0/20 | `INCOMPLETE` |
| [BCAP](families/bcap.md) | [`online-bcap-transport-causal`](families/bcap.md#online-bcap-transport-causal) | [0/10*](families/bcap.md#online-bcap-transport-causal-tier-1-required) | — | — | [0/10*](families/bcap.md#online-bcap-transport-causal-required) | 0/20 | `INCOMPLETE` |
| [BCAP](families/bcap.md) | [`online-bcap-transport-fit`](families/bcap.md#online-bcap-transport-fit) | [0/10*](families/bcap.md#online-bcap-transport-fit-tier-1-required) | — | — | [0/10*](families/bcap.md#online-bcap-transport-fit-required) | 0/20 | `INCOMPLETE` |
| [BISPF](families/bispf.md) | [`online-bispf`](families/bispf.md#online-bispf) | [0/10*](families/bispf.md#online-bispf-tier-1-required) | — | — | [0/10*](families/bispf.md#online-bispf-required) | 0/20 | `INCOMPLETE` |
| [BISPF](families/bispf.md) | [`online-bispf-early`](families/bispf.md#online-bispf-early) | [0/10*](families/bispf.md#online-bispf-early-tier-1-required) | — | — | [0/10*](families/bispf.md#online-bispf-early-required) | 0/20 | `INCOMPLETE` |
| [BISPF](families/bispf.md) | [`online-bispf-early-causal`](families/bispf.md#online-bispf-early-causal) | [0/10*](families/bispf.md#online-bispf-early-causal-tier-1-required) | — | — | [0/10*](families/bispf.md#online-bispf-early-causal-required) | 0/20 | `INCOMPLETE` |
| [BISPF](families/bispf.md) | [`online-bispf-frozen`](families/bispf.md#online-bispf-frozen) | [0/10*](families/bispf.md#online-bispf-frozen-tier-1-required) | — | — | [0/10*](families/bispf.md#online-bispf-frozen-required) | 0/20 | `INCOMPLETE` |
| [BISPF](families/bispf.md) | [`online-bispf-momentum`](families/bispf.md#online-bispf-momentum) | [0/10*](families/bispf.md#online-bispf-momentum-tier-1-required) | — | — | [0/10*](families/bispf.md#online-bispf-momentum-required) | 0/20 | `INCOMPLETE` |
| [BISPF](families/bispf.md) | [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) | [0/10*](families/bispf.md#online-bispf-optimized-tier-1-required) | — | — | [0/10*](families/bispf.md#online-bispf-optimized-required) | 0/20 | `INCOMPLETE` |
| [BISPF](families/bispf.md) | [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) | [0/10*](families/bispf.md#online-bispf-optimized-cool-tier-1-required) | — | — | [0/10*](families/bispf.md#online-bispf-optimized-cool-required) | 0/20 | `INCOMPLETE` |
| [BISPF](families/bispf.md) | [`online-bispf-slow-critic`](families/bispf.md#online-bispf-slow-critic) | [0/10*](families/bispf.md#online-bispf-slow-critic-tier-1-required) | — | — | [0/10*](families/bispf.md#online-bispf-slow-critic-required) | 0/20 | `INCOMPLETE` |
| [BISPF](families/bispf.md) | [`online-bispf-slow-prior`](families/bispf.md#online-bispf-slow-prior) | [0/10*](families/bispf.md#online-bispf-slow-prior-tier-1-required) | — | — | [0/10*](families/bispf.md#online-bispf-slow-prior-required) | 0/20 | `INCOMPLETE` |
| [BISPF](families/bispf.md) | [`online-bispf-transport-causal`](families/bispf.md#online-bispf-transport-causal) | [0/10*](families/bispf.md#online-bispf-transport-causal-tier-1-required) | — | — | [0/10*](families/bispf.md#online-bispf-transport-causal-required) | 0/20 | `INCOMPLETE` |
| [FastWeight-BCAP](families/fastweight-bcap.md) | [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | [0/10*](families/fastweight-bcap.md#online-fast-weight-bcap-tier-1-required) | — | — | [0/10*](families/fastweight-bcap.md#online-fast-weight-bcap-required) | 0/20 | `INCOMPLETE` |
| [Global-Stream-BCAP](families/global-stream-bcap.md) | [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | [0/10*](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-1-required) | — | — | [0/10*](families/global-stream-bcap.md#online-fast-weight-bcap-ns-required) | 0/20 | `INCOMPLETE` |
| [FeaturePrior-BCAP](families/featureprior-bcap.md) | [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | [0/10*](families/featureprior-bcap.md#online-feature-prior-bce-tier-1-required) | — | — | [0/10*](families/featureprior-bcap.md#online-feature-prior-bce-required) | 0/20 | `INCOMPLETE` |
| [GAIL-BCAP](families/gail-bcap.md) | [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | [0/10*](families/gail-bcap.md#online-gail-bcap-bce-tier-1-required) | — | — | [0/10*](families/gail-bcap.md#online-gail-bcap-bce-required) | 0/20 | `INCOMPLETE` |
| [GAIL-BCAP](families/gail-bcap.md) | [`online-gail-bcap-hinge`](families/gail-bcap.md#online-gail-bcap-hinge) | [0/10*](families/gail-bcap.md#online-gail-bcap-hinge-tier-1-required) | — | — | [0/10*](families/gail-bcap.md#online-gail-bcap-hinge-required) | 0/20 | `INCOMPLETE` |
| [GAIL-BCAP](families/gail-bcap.md) | [`online-gail-bcap-momentum`](families/gail-bcap.md#online-gail-bcap-momentum) | [0/10*](families/gail-bcap.md#online-gail-bcap-momentum-tier-1-required) | — | — | [0/10*](families/gail-bcap.md#online-gail-bcap-momentum-required) | 0/20 | `INCOMPLETE` |
| [KA2](families/ka2.md) | [`online-ka2`](families/ka2.md#online-ka2) | [0/10*](families/ka2.md#online-ka2-tier-1-required) | — | — | [0/10*](families/ka2.md#online-ka2-required) | 0/20 | `INCOMPLETE` |
| [KA2](families/ka2.md) | [`online-ka2-clean`](families/ka2.md#online-ka2-clean) | [0/10*](families/ka2.md#online-ka2-clean-tier-1-required) | — | — | [0/10*](families/ka2.md#online-ka2-clean-required) | 0/20 | `INCOMPLETE` |
| [KA2](families/ka2.md) | [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) | [0/10*](families/ka2.md#online-ka2-light-anchor-tier-1-required) | — | — | [0/10*](families/ka2.md#online-ka2-light-anchor-required) | 0/20 | `INCOMPLETE` |
| [KA2](families/ka2.md) | [`online-ka2-ready`](families/ka2.md#online-ka2-ready) | [0/10*](families/ka2.md#online-ka2-ready-tier-1-required) | — | — | [0/10*](families/ka2.md#online-ka2-ready-required) | 0/20 | `INCOMPLETE` |
| [KA2](families/ka2.md) | [`online-ka2-ready-causal`](families/ka2.md#online-ka2-ready-causal) | [0/10*](families/ka2.md#online-ka2-ready-causal-tier-1-required) | — | — | [0/10*](families/ka2.md#online-ka2-ready-causal-required) | 0/20 | `INCOMPLETE` |
| [KA2](families/ka2.md) | [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) | [0/10*](families/ka2.md#online-ka2-stateless-tier-1-required) | — | — | [0/10*](families/ka2.md#online-ka2-stateless-required) | 0/20 | `INCOMPLETE` |
| [KA2](families/ka2.md) | [`online-ka2-transport-causal`](families/ka2.md#online-ka2-transport-causal) | [0/10*](families/ka2.md#online-ka2-transport-causal-tier-1-required) | — | — | [0/10*](families/ka2.md#online-ka2-transport-causal-required) | 0/20 | `INCOMPLETE` |

## Stretch experiments

These measurements retain their declared gates and budgets even when a required task fails. They do not promote a recipe past an unmet lower-tier requirement.

| Technique | Recipe | Tier 1 stretch | Tier 2 stretch | Tier 3 stretch | Total stretch | Measured stretch |
| --- | --- | --- | --- | --- | --- | --- |
| [BCAP](families/bcap.md) | [`online-bcap`](families/bcap.md#online-bcap) | [0/2*](families/bcap.md#online-bcap-tier-1-stretch) | [0/7*](families/bcap.md#online-bcap-tier-2-stretch) | [0/1*](families/bcap.md#online-bcap-tier-3-stretch) | [0/10*](families/bcap.md#online-bcap-stretch) | 0/10 |
| [BCAP](families/bcap.md) | [`online-bcap-adam`](families/bcap.md#online-bcap-adam) | [0/2*](families/bcap.md#online-bcap-adam-tier-1-stretch) | [0/7*](families/bcap.md#online-bcap-adam-tier-2-stretch) | [0/1*](families/bcap.md#online-bcap-adam-tier-3-stretch) | [0/10*](families/bcap.md#online-bcap-adam-stretch) | 0/10 |
| [BCAP](families/bcap.md) | [`online-bcap-adam-causal`](families/bcap.md#online-bcap-adam-causal) | [0/2*](families/bcap.md#online-bcap-adam-causal-tier-1-stretch) | [0/7*](families/bcap.md#online-bcap-adam-causal-tier-2-stretch) | [0/1*](families/bcap.md#online-bcap-adam-causal-tier-3-stretch) | [0/10*](families/bcap.md#online-bcap-adam-causal-stretch) | 0/10 |
| [BCAP](families/bcap.md) | [`online-bcap-balanced`](families/bcap.md#online-bcap-balanced) | [0/2*](families/bcap.md#online-bcap-balanced-tier-1-stretch) | [0/7*](families/bcap.md#online-bcap-balanced-tier-2-stretch) | [0/1*](families/bcap.md#online-bcap-balanced-tier-3-stretch) | [0/10*](families/bcap.md#online-bcap-balanced-stretch) | 0/10 |
| [BCAP](families/bcap.md) | [`online-bcap-balanced-bce`](families/bcap.md#online-bcap-balanced-bce) | [0/2*](families/bcap.md#online-bcap-balanced-bce-tier-1-stretch) | [0/7*](families/bcap.md#online-bcap-balanced-bce-tier-2-stretch) | [0/1*](families/bcap.md#online-bcap-balanced-bce-tier-3-stretch) | [0/10*](families/bcap.md#online-bcap-balanced-bce-stretch) | 0/10 |
| [BCAP](families/bcap.md) | [`online-bcap-balanced-frozen`](families/bcap.md#online-bcap-balanced-frozen) | [0/2*](families/bcap.md#online-bcap-balanced-frozen-tier-1-stretch) | [0/7*](families/bcap.md#online-bcap-balanced-frozen-tier-2-stretch) | [0/1*](families/bcap.md#online-bcap-balanced-frozen-tier-3-stretch) | [0/10*](families/bcap.md#online-bcap-balanced-frozen-stretch) | 0/10 |
| [BCAP](families/bcap.md) | [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) | [0/2*](families/bcap.md#online-bcap-stateless-tier-1-stretch) | [0/7*](families/bcap.md#online-bcap-stateless-tier-2-stretch) | [0/1*](families/bcap.md#online-bcap-stateless-tier-3-stretch) | [0/10*](families/bcap.md#online-bcap-stateless-stretch) | 0/10 |
| [BCAP](families/bcap.md) | [`online-bcap-transport`](families/bcap.md#online-bcap-transport) | [0/2*](families/bcap.md#online-bcap-transport-tier-1-stretch) | [0/7*](families/bcap.md#online-bcap-transport-tier-2-stretch) | [0/1*](families/bcap.md#online-bcap-transport-tier-3-stretch) | [0/10*](families/bcap.md#online-bcap-transport-stretch) | 0/10 |
| [BCAP](families/bcap.md) | [`online-bcap-transport-causal`](families/bcap.md#online-bcap-transport-causal) | [0/2*](families/bcap.md#online-bcap-transport-causal-tier-1-stretch) | [0/7*](families/bcap.md#online-bcap-transport-causal-tier-2-stretch) | [0/1*](families/bcap.md#online-bcap-transport-causal-tier-3-stretch) | [0/10*](families/bcap.md#online-bcap-transport-causal-stretch) | 0/10 |
| [BCAP](families/bcap.md) | [`online-bcap-transport-fit`](families/bcap.md#online-bcap-transport-fit) | [0/2*](families/bcap.md#online-bcap-transport-fit-tier-1-stretch) | [0/7*](families/bcap.md#online-bcap-transport-fit-tier-2-stretch) | [0/1*](families/bcap.md#online-bcap-transport-fit-tier-3-stretch) | [0/10*](families/bcap.md#online-bcap-transport-fit-stretch) | 0/10 |
| [BISPF](families/bispf.md) | [`online-bispf`](families/bispf.md#online-bispf) | [0/2*](families/bispf.md#online-bispf-tier-1-stretch) | [0/7*](families/bispf.md#online-bispf-tier-2-stretch) | [0/1*](families/bispf.md#online-bispf-tier-3-stretch) | [0/10*](families/bispf.md#online-bispf-stretch) | 0/10 |
| [BISPF](families/bispf.md) | [`online-bispf-early`](families/bispf.md#online-bispf-early) | [0/2*](families/bispf.md#online-bispf-early-tier-1-stretch) | [0/7*](families/bispf.md#online-bispf-early-tier-2-stretch) | [0/1*](families/bispf.md#online-bispf-early-tier-3-stretch) | [0/10*](families/bispf.md#online-bispf-early-stretch) | 0/10 |
| [BISPF](families/bispf.md) | [`online-bispf-early-causal`](families/bispf.md#online-bispf-early-causal) | [0/2*](families/bispf.md#online-bispf-early-causal-tier-1-stretch) | [0/7*](families/bispf.md#online-bispf-early-causal-tier-2-stretch) | [0/1*](families/bispf.md#online-bispf-early-causal-tier-3-stretch) | [0/10*](families/bispf.md#online-bispf-early-causal-stretch) | 0/10 |
| [BISPF](families/bispf.md) | [`online-bispf-frozen`](families/bispf.md#online-bispf-frozen) | [0/2*](families/bispf.md#online-bispf-frozen-tier-1-stretch) | [0/7*](families/bispf.md#online-bispf-frozen-tier-2-stretch) | [0/1*](families/bispf.md#online-bispf-frozen-tier-3-stretch) | [0/10*](families/bispf.md#online-bispf-frozen-stretch) | 0/10 |
| [BISPF](families/bispf.md) | [`online-bispf-momentum`](families/bispf.md#online-bispf-momentum) | [0/2*](families/bispf.md#online-bispf-momentum-tier-1-stretch) | [0/7*](families/bispf.md#online-bispf-momentum-tier-2-stretch) | [0/1*](families/bispf.md#online-bispf-momentum-tier-3-stretch) | [0/10*](families/bispf.md#online-bispf-momentum-stretch) | 0/10 |
| [BISPF](families/bispf.md) | [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) | [0/2*](families/bispf.md#online-bispf-optimized-tier-1-stretch) | [0/7*](families/bispf.md#online-bispf-optimized-tier-2-stretch) | [0/1*](families/bispf.md#online-bispf-optimized-tier-3-stretch) | [0/10*](families/bispf.md#online-bispf-optimized-stretch) | 0/10 |
| [BISPF](families/bispf.md) | [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) | [0/2*](families/bispf.md#online-bispf-optimized-cool-tier-1-stretch) | [0/7*](families/bispf.md#online-bispf-optimized-cool-tier-2-stretch) | [0/1*](families/bispf.md#online-bispf-optimized-cool-tier-3-stretch) | [0/10*](families/bispf.md#online-bispf-optimized-cool-stretch) | 0/10 |
| [BISPF](families/bispf.md) | [`online-bispf-slow-critic`](families/bispf.md#online-bispf-slow-critic) | [0/2*](families/bispf.md#online-bispf-slow-critic-tier-1-stretch) | [0/7*](families/bispf.md#online-bispf-slow-critic-tier-2-stretch) | [0/1*](families/bispf.md#online-bispf-slow-critic-tier-3-stretch) | [0/10*](families/bispf.md#online-bispf-slow-critic-stretch) | 0/10 |
| [BISPF](families/bispf.md) | [`online-bispf-slow-prior`](families/bispf.md#online-bispf-slow-prior) | [0/2*](families/bispf.md#online-bispf-slow-prior-tier-1-stretch) | [0/7*](families/bispf.md#online-bispf-slow-prior-tier-2-stretch) | [0/1*](families/bispf.md#online-bispf-slow-prior-tier-3-stretch) | [0/10*](families/bispf.md#online-bispf-slow-prior-stretch) | 0/10 |
| [BISPF](families/bispf.md) | [`online-bispf-transport-causal`](families/bispf.md#online-bispf-transport-causal) | [0/2*](families/bispf.md#online-bispf-transport-causal-tier-1-stretch) | [0/7*](families/bispf.md#online-bispf-transport-causal-tier-2-stretch) | [0/1*](families/bispf.md#online-bispf-transport-causal-tier-3-stretch) | [0/10*](families/bispf.md#online-bispf-transport-causal-stretch) | 0/10 |
| [FastWeight-BCAP](families/fastweight-bcap.md) | [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | [0/2*](families/fastweight-bcap.md#online-fast-weight-bcap-tier-1-stretch) | [0/7*](families/fastweight-bcap.md#online-fast-weight-bcap-tier-2-stretch) | [0/1*](families/fastweight-bcap.md#online-fast-weight-bcap-tier-3-stretch) | [0/10*](families/fastweight-bcap.md#online-fast-weight-bcap-stretch) | 0/10 |
| [Global-Stream-BCAP](families/global-stream-bcap.md) | [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | [0/2*](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-1-stretch) | [0/7*](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-2-stretch) | [0/1*](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-3-stretch) | [0/10*](families/global-stream-bcap.md#online-fast-weight-bcap-ns-stretch) | 0/10 |
| [FeaturePrior-BCAP](families/featureprior-bcap.md) | [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | [0/2*](families/featureprior-bcap.md#online-feature-prior-bce-tier-1-stretch) | [0/7*](families/featureprior-bcap.md#online-feature-prior-bce-tier-2-stretch) | [0/1*](families/featureprior-bcap.md#online-feature-prior-bce-tier-3-stretch) | [0/10*](families/featureprior-bcap.md#online-feature-prior-bce-stretch) | 0/10 |
| [GAIL-BCAP](families/gail-bcap.md) | [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | [0/2*](families/gail-bcap.md#online-gail-bcap-bce-tier-1-stretch) | [0/7*](families/gail-bcap.md#online-gail-bcap-bce-tier-2-stretch) | [0/1*](families/gail-bcap.md#online-gail-bcap-bce-tier-3-stretch) | [0/10*](families/gail-bcap.md#online-gail-bcap-bce-stretch) | 0/10 |
| [GAIL-BCAP](families/gail-bcap.md) | [`online-gail-bcap-hinge`](families/gail-bcap.md#online-gail-bcap-hinge) | [0/2*](families/gail-bcap.md#online-gail-bcap-hinge-tier-1-stretch) | [0/7*](families/gail-bcap.md#online-gail-bcap-hinge-tier-2-stretch) | [0/1*](families/gail-bcap.md#online-gail-bcap-hinge-tier-3-stretch) | [0/10*](families/gail-bcap.md#online-gail-bcap-hinge-stretch) | 0/10 |
| [GAIL-BCAP](families/gail-bcap.md) | [`online-gail-bcap-momentum`](families/gail-bcap.md#online-gail-bcap-momentum) | [0/2*](families/gail-bcap.md#online-gail-bcap-momentum-tier-1-stretch) | [0/7*](families/gail-bcap.md#online-gail-bcap-momentum-tier-2-stretch) | [0/1*](families/gail-bcap.md#online-gail-bcap-momentum-tier-3-stretch) | [0/10*](families/gail-bcap.md#online-gail-bcap-momentum-stretch) | 0/10 |
| [KA2](families/ka2.md) | [`online-ka2`](families/ka2.md#online-ka2) | [0/2*](families/ka2.md#online-ka2-tier-1-stretch) | [0/7*](families/ka2.md#online-ka2-tier-2-stretch) | [0/1*](families/ka2.md#online-ka2-tier-3-stretch) | [0/10*](families/ka2.md#online-ka2-stretch) | 0/10 |
| [KA2](families/ka2.md) | [`online-ka2-clean`](families/ka2.md#online-ka2-clean) | [0/2*](families/ka2.md#online-ka2-clean-tier-1-stretch) | [0/7*](families/ka2.md#online-ka2-clean-tier-2-stretch) | [0/1*](families/ka2.md#online-ka2-clean-tier-3-stretch) | [0/10*](families/ka2.md#online-ka2-clean-stretch) | 0/10 |
| [KA2](families/ka2.md) | [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) | [0/2*](families/ka2.md#online-ka2-light-anchor-tier-1-stretch) | [0/7*](families/ka2.md#online-ka2-light-anchor-tier-2-stretch) | [0/1*](families/ka2.md#online-ka2-light-anchor-tier-3-stretch) | [0/10*](families/ka2.md#online-ka2-light-anchor-stretch) | 0/10 |
| [KA2](families/ka2.md) | [`online-ka2-ready`](families/ka2.md#online-ka2-ready) | [0/2*](families/ka2.md#online-ka2-ready-tier-1-stretch) | [0/7*](families/ka2.md#online-ka2-ready-tier-2-stretch) | [0/1*](families/ka2.md#online-ka2-ready-tier-3-stretch) | [0/10*](families/ka2.md#online-ka2-ready-stretch) | 0/10 |
| [KA2](families/ka2.md) | [`online-ka2-ready-causal`](families/ka2.md#online-ka2-ready-causal) | [0/2*](families/ka2.md#online-ka2-ready-causal-tier-1-stretch) | [0/7*](families/ka2.md#online-ka2-ready-causal-tier-2-stretch) | [0/1*](families/ka2.md#online-ka2-ready-causal-tier-3-stretch) | [0/10*](families/ka2.md#online-ka2-ready-causal-stretch) | 0/10 |
| [KA2](families/ka2.md) | [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) | [0/2*](families/ka2.md#online-ka2-stateless-tier-1-stretch) | [0/7*](families/ka2.md#online-ka2-stateless-tier-2-stretch) | [0/1*](families/ka2.md#online-ka2-stateless-tier-3-stretch) | [0/10*](families/ka2.md#online-ka2-stateless-stretch) | 0/10 |
| [KA2](families/ka2.md) | [`online-ka2-transport-causal`](families/ka2.md#online-ka2-transport-causal) | [0/2*](families/ka2.md#online-ka2-transport-causal-tier-1-stretch) | [0/7*](families/ka2.md#online-ka2-transport-causal-tier-2-stretch) | [0/1*](families/ka2.md#online-ka2-transport-causal-tier-3-stretch) | [0/10*](families/ka2.md#online-ka2-transport-causal-stretch) | 0/10 |

## Readout and recommendation

No measured recipe currently qualifies for this view. The leading measured recipe is [`online-bcap`](families/bcap.md#online-bcap), with 0/10 required passes (0 measured). Complete missing evidence and address the recorded failures without changing gates or selecting recipes per task.

## Historical declarations

These declarations are read-only because they lack the explicit no-replay policy required by the current executor. Their saved artifacts remain available; stale historical receipts are not fresh failures. Strict online successors declare the replay change under new identities.

| Technique | Historical recipe | Execution | Strict online successor |
| --- | --- | --- | --- |
| [BCAP](families/bcap.md) | [`bcap`](families/bcap.md#bcap) | READ_ONLY | [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) |
| [BCAP](families/bcap.md) | [`stream-bcap`](families/bcap.md#stream-bcap) | READ_ONLY | [`online-bcap`](families/bcap.md#online-bcap) |
| [BCAP](families/bcap.md) | [`stream-bcap-adam`](families/bcap.md#stream-bcap-adam) | READ_ONLY | [`online-bcap-adam`](families/bcap.md#online-bcap-adam) |
| [BISPF](families/bispf.md) | [`stream-bispf`](families/bispf.md#stream-bispf) | READ_ONLY | [`online-bispf`](families/bispf.md#online-bispf) |
| [BISPF](families/bispf.md) | [`stream-bispf-early`](families/bispf.md#stream-bispf-early) | READ_ONLY | [`online-bispf-early`](families/bispf.md#online-bispf-early) |
| [BISPF](families/bispf.md) | [`stream-bispf-optimized`](families/bispf.md#stream-bispf-optimized) | READ_ONLY | [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) |
| [BISPF](families/bispf.md) | [`stream-bispf-optimized-cool`](families/bispf.md#stream-bispf-optimized-cool) | READ_ONLY | [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) |
| [FastWeight-BCAP](families/fastweight-bcap.md) | [`fast-weight-bcap`](families/fastweight-bcap.md#fast-weight-bcap) | READ_ONLY | [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) |
| [Global-Stream-BCAP](families/global-stream-bcap.md) | [`fast-weight-bcap-ns`](families/global-stream-bcap.md#fast-weight-bcap-ns) | READ_ONLY | [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) |
| [KA2](families/ka2.md) | [`ka2`](families/ka2.md#ka2) | READ_ONLY | [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) |
| [KA2](families/ka2.md) | [`stream-ka2`](families/ka2.md#stream-ka2) | READ_ONLY | [`online-ka2`](families/ka2.md#online-ka2) |
| [KA2](families/ka2.md) | [`stream-ka2-clean`](families/ka2.md#stream-ka2-clean) | READ_ONLY | [`online-ka2-clean`](families/ka2.md#online-ka2-clean) |
| [KA2](families/ka2.md) | [`stream-ka2-light-anchor`](families/ka2.md#stream-ka2-light-anchor) | READ_ONLY | [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) |
| [KA2](families/ka2.md) | [`stream-ka2-ready`](families/ka2.md#stream-ka2-ready) | READ_ONLY | [`online-ka2-ready`](families/ka2.md#online-ka2-ready) |

## Per-Task Breakdown

The declared gates below apply at every terminal check. Detailed final readings, terminal failures, timestamps and receipts are linked from each recipe's evidence.

<a id="seq_img_round_robin_bars4"></a>
### Tier 1: `seq_img_round_robin_bars4` (required)

Acquire and hold all 4 8x8 bar modes from a deterministic round-robin stream (batch_size=1, dwell=1).

Gates: `hq >= 0.9`, `modes == 4.0`, `finite_template_tv <= 0.15`. Budget: 480 updates; 24 checks; last 5 must pass.

| Recipe | Verdict | Terminal checks | Evidence |
| --- | --- | --- | --- |
| [`online-bcap`](families/bcap.md#online-bcap) | NOT_RUN | — | [details](families/bcap.md#online-bcap-tier-1-required) |
| [`online-bcap-adam`](families/bcap.md#online-bcap-adam) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-tier-1-required) |
| [`online-bcap-adam-causal`](families/bcap.md#online-bcap-adam-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-causal-tier-1-required) |
| [`online-bcap-balanced`](families/bcap.md#online-bcap-balanced) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-tier-1-required) |
| [`online-bcap-balanced-bce`](families/bcap.md#online-bcap-balanced-bce) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-bce-tier-1-required) |
| [`online-bcap-balanced-frozen`](families/bcap.md#online-bcap-balanced-frozen) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-frozen-tier-1-required) |
| [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) | NOT_RUN | — | [details](families/bcap.md#online-bcap-stateless-tier-1-required) |
| [`online-bcap-transport`](families/bcap.md#online-bcap-transport) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-tier-1-required) |
| [`online-bcap-transport-causal`](families/bcap.md#online-bcap-transport-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-causal-tier-1-required) |
| [`online-bcap-transport-fit`](families/bcap.md#online-bcap-transport-fit) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-fit-tier-1-required) |
| [`online-bispf`](families/bispf.md#online-bispf) | NOT_RUN | — | [details](families/bispf.md#online-bispf-tier-1-required) |
| [`online-bispf-early`](families/bispf.md#online-bispf-early) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-tier-1-required) |
| [`online-bispf-early-causal`](families/bispf.md#online-bispf-early-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-causal-tier-1-required) |
| [`online-bispf-frozen`](families/bispf.md#online-bispf-frozen) | NOT_RUN | — | [details](families/bispf.md#online-bispf-frozen-tier-1-required) |
| [`online-bispf-momentum`](families/bispf.md#online-bispf-momentum) | NOT_RUN | — | [details](families/bispf.md#online-bispf-momentum-tier-1-required) |
| [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-tier-1-required) |
| [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-cool-tier-1-required) |
| [`online-bispf-slow-critic`](families/bispf.md#online-bispf-slow-critic) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-critic-tier-1-required) |
| [`online-bispf-slow-prior`](families/bispf.md#online-bispf-slow-prior) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-prior-tier-1-required) |
| [`online-bispf-transport-causal`](families/bispf.md#online-bispf-transport-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-transport-causal-tier-1-required) |
| [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | NOT_RUN | — | [details](families/fastweight-bcap.md#online-fast-weight-bcap-tier-1-required) |
| [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | NOT_RUN | — | [details](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-1-required) |
| [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | NOT_RUN | — | [details](families/featureprior-bcap.md#online-feature-prior-bce-tier-1-required) |
| [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-bce-tier-1-required) |
| [`online-gail-bcap-hinge`](families/gail-bcap.md#online-gail-bcap-hinge) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-hinge-tier-1-required) |
| [`online-gail-bcap-momentum`](families/gail-bcap.md#online-gail-bcap-momentum) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-momentum-tier-1-required) |
| [`online-ka2`](families/ka2.md#online-ka2) | NOT_RUN | — | [details](families/ka2.md#online-ka2-tier-1-required) |
| [`online-ka2-clean`](families/ka2.md#online-ka2-clean) | NOT_RUN | — | [details](families/ka2.md#online-ka2-clean-tier-1-required) |
| [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) | NOT_RUN | — | [details](families/ka2.md#online-ka2-light-anchor-tier-1-required) |
| [`online-ka2-ready`](families/ka2.md#online-ka2-ready) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-tier-1-required) |
| [`online-ka2-ready-causal`](families/ka2.md#online-ka2-ready-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-causal-tier-1-required) |
| [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) | NOT_RUN | — | [details](families/ka2.md#online-ka2-stateless-tier-1-required) |
| [`online-ka2-transport-causal`](families/ka2.md#online-ka2-transport-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-transport-causal-tier-1-required) |

<a id="seq_img_dwell_bars4"></a>
### Tier 1: `seq_img_dwell_bars4` (required)

Retain all 4 8x8 bar modes across correlated blocks of dwell=16 identical frames (batch_size=1).

Gates: `hq >= 0.85`, `modes == 4.0`, `finite_template_tv <= 0.2`. Budget: 480 updates; 24 checks; last 5 must pass.

| Recipe | Verdict | Terminal checks | Evidence |
| --- | --- | --- | --- |
| [`online-bcap`](families/bcap.md#online-bcap) | NOT_RUN | — | [details](families/bcap.md#online-bcap-tier-1-required) |
| [`online-bcap-adam`](families/bcap.md#online-bcap-adam) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-tier-1-required) |
| [`online-bcap-adam-causal`](families/bcap.md#online-bcap-adam-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-causal-tier-1-required) |
| [`online-bcap-balanced`](families/bcap.md#online-bcap-balanced) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-tier-1-required) |
| [`online-bcap-balanced-bce`](families/bcap.md#online-bcap-balanced-bce) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-bce-tier-1-required) |
| [`online-bcap-balanced-frozen`](families/bcap.md#online-bcap-balanced-frozen) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-frozen-tier-1-required) |
| [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) | NOT_RUN | — | [details](families/bcap.md#online-bcap-stateless-tier-1-required) |
| [`online-bcap-transport`](families/bcap.md#online-bcap-transport) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-tier-1-required) |
| [`online-bcap-transport-causal`](families/bcap.md#online-bcap-transport-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-causal-tier-1-required) |
| [`online-bcap-transport-fit`](families/bcap.md#online-bcap-transport-fit) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-fit-tier-1-required) |
| [`online-bispf`](families/bispf.md#online-bispf) | NOT_RUN | — | [details](families/bispf.md#online-bispf-tier-1-required) |
| [`online-bispf-early`](families/bispf.md#online-bispf-early) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-tier-1-required) |
| [`online-bispf-early-causal`](families/bispf.md#online-bispf-early-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-causal-tier-1-required) |
| [`online-bispf-frozen`](families/bispf.md#online-bispf-frozen) | NOT_RUN | — | [details](families/bispf.md#online-bispf-frozen-tier-1-required) |
| [`online-bispf-momentum`](families/bispf.md#online-bispf-momentum) | NOT_RUN | — | [details](families/bispf.md#online-bispf-momentum-tier-1-required) |
| [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-tier-1-required) |
| [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-cool-tier-1-required) |
| [`online-bispf-slow-critic`](families/bispf.md#online-bispf-slow-critic) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-critic-tier-1-required) |
| [`online-bispf-slow-prior`](families/bispf.md#online-bispf-slow-prior) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-prior-tier-1-required) |
| [`online-bispf-transport-causal`](families/bispf.md#online-bispf-transport-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-transport-causal-tier-1-required) |
| [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | NOT_RUN | — | [details](families/fastweight-bcap.md#online-fast-weight-bcap-tier-1-required) |
| [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | NOT_RUN | — | [details](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-1-required) |
| [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | NOT_RUN | — | [details](families/featureprior-bcap.md#online-feature-prior-bce-tier-1-required) |
| [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-bce-tier-1-required) |
| [`online-gail-bcap-hinge`](families/gail-bcap.md#online-gail-bcap-hinge) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-hinge-tier-1-required) |
| [`online-gail-bcap-momentum`](families/gail-bcap.md#online-gail-bcap-momentum) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-momentum-tier-1-required) |
| [`online-ka2`](families/ka2.md#online-ka2) | NOT_RUN | — | [details](families/ka2.md#online-ka2-tier-1-required) |
| [`online-ka2-clean`](families/ka2.md#online-ka2-clean) | NOT_RUN | — | [details](families/ka2.md#online-ka2-clean-tier-1-required) |
| [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) | NOT_RUN | — | [details](families/ka2.md#online-ka2-light-anchor-tier-1-required) |
| [`online-ka2-ready`](families/ka2.md#online-ka2-ready) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-tier-1-required) |
| [`online-ka2-ready-causal`](families/ka2.md#online-ka2-ready-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-causal-tier-1-required) |
| [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) | NOT_RUN | — | [details](families/ka2.md#online-ka2-stateless-tier-1-required) |
| [`online-ka2-transport-causal`](families/ka2.md#online-ka2-transport-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-transport-causal-tier-1-required) |

<a id="seq_img_translating_bar_next"></a>
### Tier 1: `seq_img_translating_bar_next` (required)

Predict the next 8x8 frame of a bouncing bar from ambiguous interior frames using memory (batch_size=1).

Gates: `next_frame_rmse <= 0.12`, `direction_acc >= 0.9`. Budget: 480 updates; 24 checks; last 5 must pass.

| Recipe | Verdict | Terminal checks | Evidence |
| --- | --- | --- | --- |
| [`online-bcap`](families/bcap.md#online-bcap) | NOT_RUN | — | [details](families/bcap.md#online-bcap-tier-1-required) |
| [`online-bcap-adam`](families/bcap.md#online-bcap-adam) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-tier-1-required) |
| [`online-bcap-adam-causal`](families/bcap.md#online-bcap-adam-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-causal-tier-1-required) |
| [`online-bcap-balanced`](families/bcap.md#online-bcap-balanced) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-tier-1-required) |
| [`online-bcap-balanced-bce`](families/bcap.md#online-bcap-balanced-bce) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-bce-tier-1-required) |
| [`online-bcap-balanced-frozen`](families/bcap.md#online-bcap-balanced-frozen) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-frozen-tier-1-required) |
| [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) | NOT_RUN | — | [details](families/bcap.md#online-bcap-stateless-tier-1-required) |
| [`online-bcap-transport`](families/bcap.md#online-bcap-transport) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-tier-1-required) |
| [`online-bcap-transport-causal`](families/bcap.md#online-bcap-transport-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-causal-tier-1-required) |
| [`online-bcap-transport-fit`](families/bcap.md#online-bcap-transport-fit) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-fit-tier-1-required) |
| [`online-bispf`](families/bispf.md#online-bispf) | NOT_RUN | — | [details](families/bispf.md#online-bispf-tier-1-required) |
| [`online-bispf-early`](families/bispf.md#online-bispf-early) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-tier-1-required) |
| [`online-bispf-early-causal`](families/bispf.md#online-bispf-early-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-causal-tier-1-required) |
| [`online-bispf-frozen`](families/bispf.md#online-bispf-frozen) | NOT_RUN | — | [details](families/bispf.md#online-bispf-frozen-tier-1-required) |
| [`online-bispf-momentum`](families/bispf.md#online-bispf-momentum) | NOT_RUN | — | [details](families/bispf.md#online-bispf-momentum-tier-1-required) |
| [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-tier-1-required) |
| [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-cool-tier-1-required) |
| [`online-bispf-slow-critic`](families/bispf.md#online-bispf-slow-critic) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-critic-tier-1-required) |
| [`online-bispf-slow-prior`](families/bispf.md#online-bispf-slow-prior) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-prior-tier-1-required) |
| [`online-bispf-transport-causal`](families/bispf.md#online-bispf-transport-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-transport-causal-tier-1-required) |
| [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | NOT_RUN | — | [details](families/fastweight-bcap.md#online-fast-weight-bcap-tier-1-required) |
| [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | NOT_RUN | — | [details](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-1-required) |
| [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | NOT_RUN | — | [details](families/featureprior-bcap.md#online-feature-prior-bce-tier-1-required) |
| [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-bce-tier-1-required) |
| [`online-gail-bcap-hinge`](families/gail-bcap.md#online-gail-bcap-hinge) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-hinge-tier-1-required) |
| [`online-gail-bcap-momentum`](families/gail-bcap.md#online-gail-bcap-momentum) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-momentum-tier-1-required) |
| [`online-ka2`](families/ka2.md#online-ka2) | NOT_RUN | — | [details](families/ka2.md#online-ka2-tier-1-required) |
| [`online-ka2-clean`](families/ka2.md#online-ka2-clean) | NOT_RUN | — | [details](families/ka2.md#online-ka2-clean-tier-1-required) |
| [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) | NOT_RUN | — | [details](families/ka2.md#online-ka2-light-anchor-tier-1-required) |
| [`online-ka2-ready`](families/ka2.md#online-ka2-ready) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-tier-1-required) |
| [`online-ka2-ready-causal`](families/ka2.md#online-ka2-ready-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-causal-tier-1-required) |
| [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) | NOT_RUN | — | [details](families/ka2.md#online-ka2-stateless-tier-1-required) |
| [`online-ka2-transport-causal`](families/ka2.md#online-ka2-transport-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-transport-causal-tier-1-required) |

<a id="seq_img_fork_rollout"></a>
### Tier 1: `seq_img_fork_rollout` (required)

Cover both Left and Right future branches from the same stem memory state across particles (batch_size=1).

Gates: `hq >= 0.85`, `within_state_modes == 2.0`, `branch_tv <= 0.2`. Budget: 480 updates; 24 checks; last 5 must pass.

| Recipe | Verdict | Terminal checks | Evidence |
| --- | --- | --- | --- |
| [`online-bcap`](families/bcap.md#online-bcap) | NOT_RUN | — | [details](families/bcap.md#online-bcap-tier-1-required) |
| [`online-bcap-adam`](families/bcap.md#online-bcap-adam) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-tier-1-required) |
| [`online-bcap-adam-causal`](families/bcap.md#online-bcap-adam-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-causal-tier-1-required) |
| [`online-bcap-balanced`](families/bcap.md#online-bcap-balanced) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-tier-1-required) |
| [`online-bcap-balanced-bce`](families/bcap.md#online-bcap-balanced-bce) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-bce-tier-1-required) |
| [`online-bcap-balanced-frozen`](families/bcap.md#online-bcap-balanced-frozen) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-frozen-tier-1-required) |
| [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) | NOT_RUN | — | [details](families/bcap.md#online-bcap-stateless-tier-1-required) |
| [`online-bcap-transport`](families/bcap.md#online-bcap-transport) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-tier-1-required) |
| [`online-bcap-transport-causal`](families/bcap.md#online-bcap-transport-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-causal-tier-1-required) |
| [`online-bcap-transport-fit`](families/bcap.md#online-bcap-transport-fit) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-fit-tier-1-required) |
| [`online-bispf`](families/bispf.md#online-bispf) | NOT_RUN | — | [details](families/bispf.md#online-bispf-tier-1-required) |
| [`online-bispf-early`](families/bispf.md#online-bispf-early) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-tier-1-required) |
| [`online-bispf-early-causal`](families/bispf.md#online-bispf-early-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-causal-tier-1-required) |
| [`online-bispf-frozen`](families/bispf.md#online-bispf-frozen) | NOT_RUN | — | [details](families/bispf.md#online-bispf-frozen-tier-1-required) |
| [`online-bispf-momentum`](families/bispf.md#online-bispf-momentum) | NOT_RUN | — | [details](families/bispf.md#online-bispf-momentum-tier-1-required) |
| [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-tier-1-required) |
| [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-cool-tier-1-required) |
| [`online-bispf-slow-critic`](families/bispf.md#online-bispf-slow-critic) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-critic-tier-1-required) |
| [`online-bispf-slow-prior`](families/bispf.md#online-bispf-slow-prior) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-prior-tier-1-required) |
| [`online-bispf-transport-causal`](families/bispf.md#online-bispf-transport-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-transport-causal-tier-1-required) |
| [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | NOT_RUN | — | [details](families/fastweight-bcap.md#online-fast-weight-bcap-tier-1-required) |
| [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | NOT_RUN | — | [details](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-1-required) |
| [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | NOT_RUN | — | [details](families/featureprior-bcap.md#online-feature-prior-bce-tier-1-required) |
| [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-bce-tier-1-required) |
| [`online-gail-bcap-hinge`](families/gail-bcap.md#online-gail-bcap-hinge) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-hinge-tier-1-required) |
| [`online-gail-bcap-momentum`](families/gail-bcap.md#online-gail-bcap-momentum) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-momentum-tier-1-required) |
| [`online-ka2`](families/ka2.md#online-ka2) | NOT_RUN | — | [details](families/ka2.md#online-ka2-tier-1-required) |
| [`online-ka2-clean`](families/ka2.md#online-ka2-clean) | NOT_RUN | — | [details](families/ka2.md#online-ka2-clean-tier-1-required) |
| [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) | NOT_RUN | — | [details](families/ka2.md#online-ka2-light-anchor-tier-1-required) |
| [`online-ka2-ready`](families/ka2.md#online-ka2-ready) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-tier-1-required) |
| [`online-ka2-ready-causal`](families/ka2.md#online-ka2-ready-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-causal-tier-1-required) |
| [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) | NOT_RUN | — | [details](families/ka2.md#online-ka2-stateless-tier-1-required) |
| [`online-ka2-transport-causal`](families/ka2.md#online-ka2-transport-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-transport-causal-tier-1-required) |

<a id="seq_img_variable_dwell_bars4"></a>
### Tier 1: `seq_img_variable_dwell_bars4` (required)

Retain four modes across balanced variable dwell blocks of 4–32 arriving frames.

Gates: `hq >= 0.85`, `modes == 4.0`, `finite_template_tv <= 0.2`. Budget: 480 updates; 24 checks; last 5 must pass.

| Recipe | Verdict | Terminal checks | Evidence |
| --- | --- | --- | --- |
| [`online-bcap`](families/bcap.md#online-bcap) | NOT_RUN | — | [details](families/bcap.md#online-bcap-tier-1-required) |
| [`online-bcap-adam`](families/bcap.md#online-bcap-adam) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-tier-1-required) |
| [`online-bcap-adam-causal`](families/bcap.md#online-bcap-adam-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-causal-tier-1-required) |
| [`online-bcap-balanced`](families/bcap.md#online-bcap-balanced) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-tier-1-required) |
| [`online-bcap-balanced-bce`](families/bcap.md#online-bcap-balanced-bce) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-bce-tier-1-required) |
| [`online-bcap-balanced-frozen`](families/bcap.md#online-bcap-balanced-frozen) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-frozen-tier-1-required) |
| [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) | NOT_RUN | — | [details](families/bcap.md#online-bcap-stateless-tier-1-required) |
| [`online-bcap-transport`](families/bcap.md#online-bcap-transport) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-tier-1-required) |
| [`online-bcap-transport-causal`](families/bcap.md#online-bcap-transport-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-causal-tier-1-required) |
| [`online-bcap-transport-fit`](families/bcap.md#online-bcap-transport-fit) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-fit-tier-1-required) |
| [`online-bispf`](families/bispf.md#online-bispf) | NOT_RUN | — | [details](families/bispf.md#online-bispf-tier-1-required) |
| [`online-bispf-early`](families/bispf.md#online-bispf-early) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-tier-1-required) |
| [`online-bispf-early-causal`](families/bispf.md#online-bispf-early-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-causal-tier-1-required) |
| [`online-bispf-frozen`](families/bispf.md#online-bispf-frozen) | NOT_RUN | — | [details](families/bispf.md#online-bispf-frozen-tier-1-required) |
| [`online-bispf-momentum`](families/bispf.md#online-bispf-momentum) | NOT_RUN | — | [details](families/bispf.md#online-bispf-momentum-tier-1-required) |
| [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-tier-1-required) |
| [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-cool-tier-1-required) |
| [`online-bispf-slow-critic`](families/bispf.md#online-bispf-slow-critic) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-critic-tier-1-required) |
| [`online-bispf-slow-prior`](families/bispf.md#online-bispf-slow-prior) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-prior-tier-1-required) |
| [`online-bispf-transport-causal`](families/bispf.md#online-bispf-transport-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-transport-causal-tier-1-required) |
| [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | NOT_RUN | — | [details](families/fastweight-bcap.md#online-fast-weight-bcap-tier-1-required) |
| [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | NOT_RUN | — | [details](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-1-required) |
| [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | NOT_RUN | — | [details](families/featureprior-bcap.md#online-feature-prior-bce-tier-1-required) |
| [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-bce-tier-1-required) |
| [`online-gail-bcap-hinge`](families/gail-bcap.md#online-gail-bcap-hinge) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-hinge-tier-1-required) |
| [`online-gail-bcap-momentum`](families/gail-bcap.md#online-gail-bcap-momentum) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-momentum-tier-1-required) |
| [`online-ka2`](families/ka2.md#online-ka2) | NOT_RUN | — | [details](families/ka2.md#online-ka2-tier-1-required) |
| [`online-ka2-clean`](families/ka2.md#online-ka2-clean) | NOT_RUN | — | [details](families/ka2.md#online-ka2-clean-tier-1-required) |
| [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) | NOT_RUN | — | [details](families/ka2.md#online-ka2-light-anchor-tier-1-required) |
| [`online-ka2-ready`](families/ka2.md#online-ka2-ready) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-tier-1-required) |
| [`online-ka2-ready-causal`](families/ka2.md#online-ka2-ready-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-causal-tier-1-required) |
| [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) | NOT_RUN | — | [details](families/ka2.md#online-ka2-stateless-tier-1-required) |
| [`online-ka2-transport-causal`](families/ka2.md#online-ka2-transport-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-transport-causal-tier-1-required) |

<a id="seq_img_rare_mode_bars4"></a>
### Tier 1: `seq_img_rare_mode_bars4` (required)

Retain a rare fourth mode in a sequential stream with exact 5:5:5:1 frequencies.

Gates: `hq >= 0.85`, `finite_template_tv <= 0.15`, `support_modes == 4.0`, `rare_mass >= 0.03125`, `rare_mass <= 0.09375`. Budget: 480 updates; 24 checks; last 5 must pass.

| Recipe | Verdict | Terminal checks | Evidence |
| --- | --- | --- | --- |
| [`online-bcap`](families/bcap.md#online-bcap) | NOT_RUN | — | [details](families/bcap.md#online-bcap-tier-1-required) |
| [`online-bcap-adam`](families/bcap.md#online-bcap-adam) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-tier-1-required) |
| [`online-bcap-adam-causal`](families/bcap.md#online-bcap-adam-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-causal-tier-1-required) |
| [`online-bcap-balanced`](families/bcap.md#online-bcap-balanced) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-tier-1-required) |
| [`online-bcap-balanced-bce`](families/bcap.md#online-bcap-balanced-bce) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-bce-tier-1-required) |
| [`online-bcap-balanced-frozen`](families/bcap.md#online-bcap-balanced-frozen) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-frozen-tier-1-required) |
| [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) | NOT_RUN | — | [details](families/bcap.md#online-bcap-stateless-tier-1-required) |
| [`online-bcap-transport`](families/bcap.md#online-bcap-transport) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-tier-1-required) |
| [`online-bcap-transport-causal`](families/bcap.md#online-bcap-transport-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-causal-tier-1-required) |
| [`online-bcap-transport-fit`](families/bcap.md#online-bcap-transport-fit) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-fit-tier-1-required) |
| [`online-bispf`](families/bispf.md#online-bispf) | NOT_RUN | — | [details](families/bispf.md#online-bispf-tier-1-required) |
| [`online-bispf-early`](families/bispf.md#online-bispf-early) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-tier-1-required) |
| [`online-bispf-early-causal`](families/bispf.md#online-bispf-early-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-causal-tier-1-required) |
| [`online-bispf-frozen`](families/bispf.md#online-bispf-frozen) | NOT_RUN | — | [details](families/bispf.md#online-bispf-frozen-tier-1-required) |
| [`online-bispf-momentum`](families/bispf.md#online-bispf-momentum) | NOT_RUN | — | [details](families/bispf.md#online-bispf-momentum-tier-1-required) |
| [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-tier-1-required) |
| [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-cool-tier-1-required) |
| [`online-bispf-slow-critic`](families/bispf.md#online-bispf-slow-critic) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-critic-tier-1-required) |
| [`online-bispf-slow-prior`](families/bispf.md#online-bispf-slow-prior) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-prior-tier-1-required) |
| [`online-bispf-transport-causal`](families/bispf.md#online-bispf-transport-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-transport-causal-tier-1-required) |
| [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | NOT_RUN | — | [details](families/fastweight-bcap.md#online-fast-weight-bcap-tier-1-required) |
| [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | NOT_RUN | — | [details](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-1-required) |
| [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | NOT_RUN | — | [details](families/featureprior-bcap.md#online-feature-prior-bce-tier-1-required) |
| [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-bce-tier-1-required) |
| [`online-gail-bcap-hinge`](families/gail-bcap.md#online-gail-bcap-hinge) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-hinge-tier-1-required) |
| [`online-gail-bcap-momentum`](families/gail-bcap.md#online-gail-bcap-momentum) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-momentum-tier-1-required) |
| [`online-ka2`](families/ka2.md#online-ka2) | NOT_RUN | — | [details](families/ka2.md#online-ka2-tier-1-required) |
| [`online-ka2-clean`](families/ka2.md#online-ka2-clean) | NOT_RUN | — | [details](families/ka2.md#online-ka2-clean-tier-1-required) |
| [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) | NOT_RUN | — | [details](families/ka2.md#online-ka2-light-anchor-tier-1-required) |
| [`online-ka2-ready`](families/ka2.md#online-ka2-ready) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-tier-1-required) |
| [`online-ka2-ready-causal`](families/ka2.md#online-ka2-ready-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-causal-tier-1-required) |
| [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) | NOT_RUN | — | [details](families/ka2.md#online-ka2-stateless-tier-1-required) |
| [`online-ka2-transport-causal`](families/ka2.md#online-ka2-transport-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-transport-causal-tier-1-required) |

<a id="seq_img_delayed_cue"></a>
### Tier 1: `seq_img_delayed_cue` (required)

Predict a cue-dependent outcome after four pixel-identical intervening observations.

Gates: `outcome_accuracy >= 0.9`, `prediction_rmse <= 0.12`. Budget: 480 updates; 24 checks; last 5 must pass.

| Recipe | Verdict | Terminal checks | Evidence |
| --- | --- | --- | --- |
| [`online-bcap`](families/bcap.md#online-bcap) | NOT_RUN | — | [details](families/bcap.md#online-bcap-tier-1-required) |
| [`online-bcap-adam`](families/bcap.md#online-bcap-adam) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-tier-1-required) |
| [`online-bcap-adam-causal`](families/bcap.md#online-bcap-adam-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-causal-tier-1-required) |
| [`online-bcap-balanced`](families/bcap.md#online-bcap-balanced) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-tier-1-required) |
| [`online-bcap-balanced-bce`](families/bcap.md#online-bcap-balanced-bce) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-bce-tier-1-required) |
| [`online-bcap-balanced-frozen`](families/bcap.md#online-bcap-balanced-frozen) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-frozen-tier-1-required) |
| [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) | NOT_RUN | — | [details](families/bcap.md#online-bcap-stateless-tier-1-required) |
| [`online-bcap-transport`](families/bcap.md#online-bcap-transport) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-tier-1-required) |
| [`online-bcap-transport-causal`](families/bcap.md#online-bcap-transport-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-causal-tier-1-required) |
| [`online-bcap-transport-fit`](families/bcap.md#online-bcap-transport-fit) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-fit-tier-1-required) |
| [`online-bispf`](families/bispf.md#online-bispf) | NOT_RUN | — | [details](families/bispf.md#online-bispf-tier-1-required) |
| [`online-bispf-early`](families/bispf.md#online-bispf-early) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-tier-1-required) |
| [`online-bispf-early-causal`](families/bispf.md#online-bispf-early-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-causal-tier-1-required) |
| [`online-bispf-frozen`](families/bispf.md#online-bispf-frozen) | NOT_RUN | — | [details](families/bispf.md#online-bispf-frozen-tier-1-required) |
| [`online-bispf-momentum`](families/bispf.md#online-bispf-momentum) | NOT_RUN | — | [details](families/bispf.md#online-bispf-momentum-tier-1-required) |
| [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-tier-1-required) |
| [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-cool-tier-1-required) |
| [`online-bispf-slow-critic`](families/bispf.md#online-bispf-slow-critic) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-critic-tier-1-required) |
| [`online-bispf-slow-prior`](families/bispf.md#online-bispf-slow-prior) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-prior-tier-1-required) |
| [`online-bispf-transport-causal`](families/bispf.md#online-bispf-transport-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-transport-causal-tier-1-required) |
| [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | NOT_RUN | — | [details](families/fastweight-bcap.md#online-fast-weight-bcap-tier-1-required) |
| [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | NOT_RUN | — | [details](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-1-required) |
| [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | NOT_RUN | — | [details](families/featureprior-bcap.md#online-feature-prior-bce-tier-1-required) |
| [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-bce-tier-1-required) |
| [`online-gail-bcap-hinge`](families/gail-bcap.md#online-gail-bcap-hinge) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-hinge-tier-1-required) |
| [`online-gail-bcap-momentum`](families/gail-bcap.md#online-gail-bcap-momentum) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-momentum-tier-1-required) |
| [`online-ka2`](families/ka2.md#online-ka2) | NOT_RUN | — | [details](families/ka2.md#online-ka2-tier-1-required) |
| [`online-ka2-clean`](families/ka2.md#online-ka2-clean) | NOT_RUN | — | [details](families/ka2.md#online-ka2-clean-tier-1-required) |
| [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) | NOT_RUN | — | [details](families/ka2.md#online-ka2-light-anchor-tier-1-required) |
| [`online-ka2-ready`](families/ka2.md#online-ka2-ready) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-tier-1-required) |
| [`online-ka2-ready-causal`](families/ka2.md#online-ka2-ready-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-causal-tier-1-required) |
| [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) | NOT_RUN | — | [details](families/ka2.md#online-ka2-stateless-tier-1-required) |
| [`online-ka2-transport-causal`](families/ka2.md#online-ka2-transport-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-transport-causal-tier-1-required) |

<a id="seq_img_persistent_fork"></a>
### Tier 1: `seq_img_persistent_fork` (required)

Generate both three-step fork futures while preserving each particle's branch without teacher forcing.

Gates: `hq >= 0.85`, `within_state_modes == 2.0`, `branch_tv <= 0.2`, `trajectory_consistency >= 0.95`. Budget: 480 updates; 24 checks; last 5 must pass.

| Recipe | Verdict | Terminal checks | Evidence |
| --- | --- | --- | --- |
| [`online-bcap`](families/bcap.md#online-bcap) | NOT_RUN | — | [details](families/bcap.md#online-bcap-tier-1-required) |
| [`online-bcap-adam`](families/bcap.md#online-bcap-adam) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-tier-1-required) |
| [`online-bcap-adam-causal`](families/bcap.md#online-bcap-adam-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-causal-tier-1-required) |
| [`online-bcap-balanced`](families/bcap.md#online-bcap-balanced) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-tier-1-required) |
| [`online-bcap-balanced-bce`](families/bcap.md#online-bcap-balanced-bce) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-bce-tier-1-required) |
| [`online-bcap-balanced-frozen`](families/bcap.md#online-bcap-balanced-frozen) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-frozen-tier-1-required) |
| [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) | NOT_RUN | — | [details](families/bcap.md#online-bcap-stateless-tier-1-required) |
| [`online-bcap-transport`](families/bcap.md#online-bcap-transport) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-tier-1-required) |
| [`online-bcap-transport-causal`](families/bcap.md#online-bcap-transport-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-causal-tier-1-required) |
| [`online-bcap-transport-fit`](families/bcap.md#online-bcap-transport-fit) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-fit-tier-1-required) |
| [`online-bispf`](families/bispf.md#online-bispf) | NOT_RUN | — | [details](families/bispf.md#online-bispf-tier-1-required) |
| [`online-bispf-early`](families/bispf.md#online-bispf-early) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-tier-1-required) |
| [`online-bispf-early-causal`](families/bispf.md#online-bispf-early-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-causal-tier-1-required) |
| [`online-bispf-frozen`](families/bispf.md#online-bispf-frozen) | NOT_RUN | — | [details](families/bispf.md#online-bispf-frozen-tier-1-required) |
| [`online-bispf-momentum`](families/bispf.md#online-bispf-momentum) | NOT_RUN | — | [details](families/bispf.md#online-bispf-momentum-tier-1-required) |
| [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-tier-1-required) |
| [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-cool-tier-1-required) |
| [`online-bispf-slow-critic`](families/bispf.md#online-bispf-slow-critic) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-critic-tier-1-required) |
| [`online-bispf-slow-prior`](families/bispf.md#online-bispf-slow-prior) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-prior-tier-1-required) |
| [`online-bispf-transport-causal`](families/bispf.md#online-bispf-transport-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-transport-causal-tier-1-required) |
| [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | NOT_RUN | — | [details](families/fastweight-bcap.md#online-fast-weight-bcap-tier-1-required) |
| [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | NOT_RUN | — | [details](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-1-required) |
| [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | NOT_RUN | — | [details](families/featureprior-bcap.md#online-feature-prior-bce-tier-1-required) |
| [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-bce-tier-1-required) |
| [`online-gail-bcap-hinge`](families/gail-bcap.md#online-gail-bcap-hinge) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-hinge-tier-1-required) |
| [`online-gail-bcap-momentum`](families/gail-bcap.md#online-gail-bcap-momentum) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-momentum-tier-1-required) |
| [`online-ka2`](families/ka2.md#online-ka2) | NOT_RUN | — | [details](families/ka2.md#online-ka2-tier-1-required) |
| [`online-ka2-clean`](families/ka2.md#online-ka2-clean) | NOT_RUN | — | [details](families/ka2.md#online-ka2-clean-tier-1-required) |
| [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) | NOT_RUN | — | [details](families/ka2.md#online-ka2-light-anchor-tier-1-required) |
| [`online-ka2-ready`](families/ka2.md#online-ka2-ready) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-tier-1-required) |
| [`online-ka2-ready-causal`](families/ka2.md#online-ka2-ready-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-causal-tier-1-required) |
| [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) | NOT_RUN | — | [details](families/ka2.md#online-ka2-stateless-tier-1-required) |
| [`online-ka2-transport-causal`](families/ka2.md#online-ka2-transport-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-transport-causal-tier-1-required) |

<a id="seq_img_circle_orbit"></a>
### Tier 1: `seq_img_circle_orbit` (required)

Trace an 8-waypoint circular orbit in real time by predicting subsequent (x, y) point frames across a closed-loop 8-step rollout without teacher forcing.

Gates: `orbit_rmse <= 0.12`, `xy_rmse <= 0.15`, `step_accuracy >= 0.9`, `orbit_completion >= 0.9`, `waypoint_modes == 8.0`. Budget: 480 updates; 24 checks; last 5 must pass.

| Recipe | Verdict | Terminal checks | Evidence |
| --- | --- | --- | --- |
| [`online-bcap`](families/bcap.md#online-bcap) | NOT_RUN | — | [details](families/bcap.md#online-bcap-tier-1-required) |
| [`online-bcap-adam`](families/bcap.md#online-bcap-adam) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-tier-1-required) |
| [`online-bcap-adam-causal`](families/bcap.md#online-bcap-adam-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-causal-tier-1-required) |
| [`online-bcap-balanced`](families/bcap.md#online-bcap-balanced) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-tier-1-required) |
| [`online-bcap-balanced-bce`](families/bcap.md#online-bcap-balanced-bce) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-bce-tier-1-required) |
| [`online-bcap-balanced-frozen`](families/bcap.md#online-bcap-balanced-frozen) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-frozen-tier-1-required) |
| [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) | NOT_RUN | — | [details](families/bcap.md#online-bcap-stateless-tier-1-required) |
| [`online-bcap-transport`](families/bcap.md#online-bcap-transport) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-tier-1-required) |
| [`online-bcap-transport-causal`](families/bcap.md#online-bcap-transport-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-causal-tier-1-required) |
| [`online-bcap-transport-fit`](families/bcap.md#online-bcap-transport-fit) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-fit-tier-1-required) |
| [`online-bispf`](families/bispf.md#online-bispf) | NOT_RUN | — | [details](families/bispf.md#online-bispf-tier-1-required) |
| [`online-bispf-early`](families/bispf.md#online-bispf-early) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-tier-1-required) |
| [`online-bispf-early-causal`](families/bispf.md#online-bispf-early-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-causal-tier-1-required) |
| [`online-bispf-frozen`](families/bispf.md#online-bispf-frozen) | NOT_RUN | — | [details](families/bispf.md#online-bispf-frozen-tier-1-required) |
| [`online-bispf-momentum`](families/bispf.md#online-bispf-momentum) | NOT_RUN | — | [details](families/bispf.md#online-bispf-momentum-tier-1-required) |
| [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-tier-1-required) |
| [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-cool-tier-1-required) |
| [`online-bispf-slow-critic`](families/bispf.md#online-bispf-slow-critic) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-critic-tier-1-required) |
| [`online-bispf-slow-prior`](families/bispf.md#online-bispf-slow-prior) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-prior-tier-1-required) |
| [`online-bispf-transport-causal`](families/bispf.md#online-bispf-transport-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-transport-causal-tier-1-required) |
| [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | NOT_RUN | — | [details](families/fastweight-bcap.md#online-fast-weight-bcap-tier-1-required) |
| [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | NOT_RUN | — | [details](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-1-required) |
| [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | NOT_RUN | — | [details](families/featureprior-bcap.md#online-feature-prior-bce-tier-1-required) |
| [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-bce-tier-1-required) |
| [`online-gail-bcap-hinge`](families/gail-bcap.md#online-gail-bcap-hinge) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-hinge-tier-1-required) |
| [`online-gail-bcap-momentum`](families/gail-bcap.md#online-gail-bcap-momentum) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-momentum-tier-1-required) |
| [`online-ka2`](families/ka2.md#online-ka2) | NOT_RUN | — | [details](families/ka2.md#online-ka2-tier-1-required) |
| [`online-ka2-clean`](families/ka2.md#online-ka2-clean) | NOT_RUN | — | [details](families/ka2.md#online-ka2-clean-tier-1-required) |
| [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) | NOT_RUN | — | [details](families/ka2.md#online-ka2-light-anchor-tier-1-required) |
| [`online-ka2-ready`](families/ka2.md#online-ka2-ready) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-tier-1-required) |
| [`online-ka2-ready-causal`](families/ka2.md#online-ka2-ready-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-causal-tier-1-required) |
| [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) | NOT_RUN | — | [details](families/ka2.md#online-ka2-stateless-tier-1-required) |
| [`online-ka2-transport-causal`](families/ka2.md#online-ka2-transport-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-transport-causal-tier-1-required) |

<a id="seq_img_square_circle_cycle"></a>
### Tier 1: `seq_img_square_circle_cycle` (required)

Translate bidirectionally between an 8-waypoint square perimeter and an 8-waypoint circular orbit in closed loop without teacher forcing (CycleGAN S<->C).

Gates: `cycle_rmse <= 0.12`, `s2c_rmse <= 0.12`, `c2s_rmse <= 0.12`, `xy_rmse <= 0.15`, `step_accuracy >= 0.9`, `cycle_completion >= 0.9`, `waypoint_modes == 8.0`. Budget: 480 updates; 24 checks; last 5 must pass.

| Recipe | Verdict | Terminal checks | Evidence |
| --- | --- | --- | --- |
| [`online-bcap`](families/bcap.md#online-bcap) | NOT_RUN | — | [details](families/bcap.md#online-bcap-tier-1-required) |
| [`online-bcap-adam`](families/bcap.md#online-bcap-adam) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-tier-1-required) |
| [`online-bcap-adam-causal`](families/bcap.md#online-bcap-adam-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-causal-tier-1-required) |
| [`online-bcap-balanced`](families/bcap.md#online-bcap-balanced) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-tier-1-required) |
| [`online-bcap-balanced-bce`](families/bcap.md#online-bcap-balanced-bce) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-bce-tier-1-required) |
| [`online-bcap-balanced-frozen`](families/bcap.md#online-bcap-balanced-frozen) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-frozen-tier-1-required) |
| [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) | NOT_RUN | — | [details](families/bcap.md#online-bcap-stateless-tier-1-required) |
| [`online-bcap-transport`](families/bcap.md#online-bcap-transport) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-tier-1-required) |
| [`online-bcap-transport-causal`](families/bcap.md#online-bcap-transport-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-causal-tier-1-required) |
| [`online-bcap-transport-fit`](families/bcap.md#online-bcap-transport-fit) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-fit-tier-1-required) |
| [`online-bispf`](families/bispf.md#online-bispf) | NOT_RUN | — | [details](families/bispf.md#online-bispf-tier-1-required) |
| [`online-bispf-early`](families/bispf.md#online-bispf-early) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-tier-1-required) |
| [`online-bispf-early-causal`](families/bispf.md#online-bispf-early-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-causal-tier-1-required) |
| [`online-bispf-frozen`](families/bispf.md#online-bispf-frozen) | NOT_RUN | — | [details](families/bispf.md#online-bispf-frozen-tier-1-required) |
| [`online-bispf-momentum`](families/bispf.md#online-bispf-momentum) | NOT_RUN | — | [details](families/bispf.md#online-bispf-momentum-tier-1-required) |
| [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-tier-1-required) |
| [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-cool-tier-1-required) |
| [`online-bispf-slow-critic`](families/bispf.md#online-bispf-slow-critic) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-critic-tier-1-required) |
| [`online-bispf-slow-prior`](families/bispf.md#online-bispf-slow-prior) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-prior-tier-1-required) |
| [`online-bispf-transport-causal`](families/bispf.md#online-bispf-transport-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-transport-causal-tier-1-required) |
| [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | NOT_RUN | — | [details](families/fastweight-bcap.md#online-fast-weight-bcap-tier-1-required) |
| [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | NOT_RUN | — | [details](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-1-required) |
| [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | NOT_RUN | — | [details](families/featureprior-bcap.md#online-feature-prior-bce-tier-1-required) |
| [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-bce-tier-1-required) |
| [`online-gail-bcap-hinge`](families/gail-bcap.md#online-gail-bcap-hinge) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-hinge-tier-1-required) |
| [`online-gail-bcap-momentum`](families/gail-bcap.md#online-gail-bcap-momentum) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-momentum-tier-1-required) |
| [`online-ka2`](families/ka2.md#online-ka2) | NOT_RUN | — | [details](families/ka2.md#online-ka2-tier-1-required) |
| [`online-ka2-clean`](families/ka2.md#online-ka2-clean) | NOT_RUN | — | [details](families/ka2.md#online-ka2-clean-tier-1-required) |
| [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) | NOT_RUN | — | [details](families/ka2.md#online-ka2-light-anchor-tier-1-required) |
| [`online-ka2-ready`](families/ka2.md#online-ka2-ready) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-tier-1-required) |
| [`online-ka2-ready-causal`](families/ka2.md#online-ka2-ready-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-causal-tier-1-required) |
| [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) | NOT_RUN | — | [details](families/ka2.md#online-ka2-stateless-tier-1-required) |
| [`online-ka2-transport-causal`](families/ka2.md#online-ka2-transport-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-transport-causal-tier-1-required) |

<a id="seq_minimal_dense4"></a>
### Tier 1: `seq_minimal_dense4` (stretch)

Acquire four arbitrary dense 8x8 patterns without replay.

Gates: `valid_fraction >= 0.9`, `modes == 4.0`, `mass_tv <= 0.15`. Budget: 96 updates; 24 checks; last 5 must pass.

| Recipe | Verdict | Terminal checks | Evidence |
| --- | --- | --- | --- |
| [`online-bcap`](families/bcap.md#online-bcap) | NOT_RUN | — | [details](families/bcap.md#online-bcap-tier-1-stretch) |
| [`online-bcap-adam`](families/bcap.md#online-bcap-adam) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-tier-1-stretch) |
| [`online-bcap-adam-causal`](families/bcap.md#online-bcap-adam-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-causal-tier-1-stretch) |
| [`online-bcap-balanced`](families/bcap.md#online-bcap-balanced) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-tier-1-stretch) |
| [`online-bcap-balanced-bce`](families/bcap.md#online-bcap-balanced-bce) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-bce-tier-1-stretch) |
| [`online-bcap-balanced-frozen`](families/bcap.md#online-bcap-balanced-frozen) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-frozen-tier-1-stretch) |
| [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) | NOT_RUN | — | [details](families/bcap.md#online-bcap-stateless-tier-1-stretch) |
| [`online-bcap-transport`](families/bcap.md#online-bcap-transport) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-tier-1-stretch) |
| [`online-bcap-transport-causal`](families/bcap.md#online-bcap-transport-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-causal-tier-1-stretch) |
| [`online-bcap-transport-fit`](families/bcap.md#online-bcap-transport-fit) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-fit-tier-1-stretch) |
| [`online-bispf`](families/bispf.md#online-bispf) | NOT_RUN | — | [details](families/bispf.md#online-bispf-tier-1-stretch) |
| [`online-bispf-early`](families/bispf.md#online-bispf-early) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-tier-1-stretch) |
| [`online-bispf-early-causal`](families/bispf.md#online-bispf-early-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-causal-tier-1-stretch) |
| [`online-bispf-frozen`](families/bispf.md#online-bispf-frozen) | NOT_RUN | — | [details](families/bispf.md#online-bispf-frozen-tier-1-stretch) |
| [`online-bispf-momentum`](families/bispf.md#online-bispf-momentum) | NOT_RUN | — | [details](families/bispf.md#online-bispf-momentum-tier-1-stretch) |
| [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-tier-1-stretch) |
| [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-cool-tier-1-stretch) |
| [`online-bispf-slow-critic`](families/bispf.md#online-bispf-slow-critic) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-critic-tier-1-stretch) |
| [`online-bispf-slow-prior`](families/bispf.md#online-bispf-slow-prior) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-prior-tier-1-stretch) |
| [`online-bispf-transport-causal`](families/bispf.md#online-bispf-transport-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-transport-causal-tier-1-stretch) |
| [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | NOT_RUN | — | [details](families/fastweight-bcap.md#online-fast-weight-bcap-tier-1-stretch) |
| [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | NOT_RUN | — | [details](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-1-stretch) |
| [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | NOT_RUN | — | [details](families/featureprior-bcap.md#online-feature-prior-bce-tier-1-stretch) |
| [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-bce-tier-1-stretch) |
| [`online-gail-bcap-hinge`](families/gail-bcap.md#online-gail-bcap-hinge) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-hinge-tier-1-stretch) |
| [`online-gail-bcap-momentum`](families/gail-bcap.md#online-gail-bcap-momentum) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-momentum-tier-1-stretch) |
| [`online-ka2`](families/ka2.md#online-ka2) | NOT_RUN | — | [details](families/ka2.md#online-ka2-tier-1-stretch) |
| [`online-ka2-clean`](families/ka2.md#online-ka2-clean) | NOT_RUN | — | [details](families/ka2.md#online-ka2-clean-tier-1-stretch) |
| [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) | NOT_RUN | — | [details](families/ka2.md#online-ka2-light-anchor-tier-1-stretch) |
| [`online-ka2-ready`](families/ka2.md#online-ka2-ready) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-tier-1-stretch) |
| [`online-ka2-ready-causal`](families/ka2.md#online-ka2-ready-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-causal-tier-1-stretch) |
| [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) | NOT_RUN | — | [details](families/ka2.md#online-ka2-stateless-tier-1-stretch) |
| [`online-ka2-transport-causal`](families/ka2.md#online-ka2-transport-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-transport-causal-tier-1-stretch) |

<a id="seq_minimal_tokens4"></a>
### Tier 1: `seq_minimal_tokens4` (stretch)

Serve four integer token IDs with strict raw one-hot confidence.

Gates: `valid_fraction == 1.0`, `modes == 4.0`, `mass_tv <= 0.1`. Budget: 96 updates; 24 checks; last 5 must pass.

| Recipe | Verdict | Terminal checks | Evidence |
| --- | --- | --- | --- |
| [`online-bcap`](families/bcap.md#online-bcap) | NOT_RUN | — | [details](families/bcap.md#online-bcap-tier-1-stretch) |
| [`online-bcap-adam`](families/bcap.md#online-bcap-adam) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-tier-1-stretch) |
| [`online-bcap-adam-causal`](families/bcap.md#online-bcap-adam-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-causal-tier-1-stretch) |
| [`online-bcap-balanced`](families/bcap.md#online-bcap-balanced) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-tier-1-stretch) |
| [`online-bcap-balanced-bce`](families/bcap.md#online-bcap-balanced-bce) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-bce-tier-1-stretch) |
| [`online-bcap-balanced-frozen`](families/bcap.md#online-bcap-balanced-frozen) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-frozen-tier-1-stretch) |
| [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) | NOT_RUN | — | [details](families/bcap.md#online-bcap-stateless-tier-1-stretch) |
| [`online-bcap-transport`](families/bcap.md#online-bcap-transport) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-tier-1-stretch) |
| [`online-bcap-transport-causal`](families/bcap.md#online-bcap-transport-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-causal-tier-1-stretch) |
| [`online-bcap-transport-fit`](families/bcap.md#online-bcap-transport-fit) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-fit-tier-1-stretch) |
| [`online-bispf`](families/bispf.md#online-bispf) | NOT_RUN | — | [details](families/bispf.md#online-bispf-tier-1-stretch) |
| [`online-bispf-early`](families/bispf.md#online-bispf-early) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-tier-1-stretch) |
| [`online-bispf-early-causal`](families/bispf.md#online-bispf-early-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-causal-tier-1-stretch) |
| [`online-bispf-frozen`](families/bispf.md#online-bispf-frozen) | NOT_RUN | — | [details](families/bispf.md#online-bispf-frozen-tier-1-stretch) |
| [`online-bispf-momentum`](families/bispf.md#online-bispf-momentum) | NOT_RUN | — | [details](families/bispf.md#online-bispf-momentum-tier-1-stretch) |
| [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-tier-1-stretch) |
| [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-cool-tier-1-stretch) |
| [`online-bispf-slow-critic`](families/bispf.md#online-bispf-slow-critic) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-critic-tier-1-stretch) |
| [`online-bispf-slow-prior`](families/bispf.md#online-bispf-slow-prior) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-prior-tier-1-stretch) |
| [`online-bispf-transport-causal`](families/bispf.md#online-bispf-transport-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-transport-causal-tier-1-stretch) |
| [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | NOT_RUN | — | [details](families/fastweight-bcap.md#online-fast-weight-bcap-tier-1-stretch) |
| [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | NOT_RUN | — | [details](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-1-stretch) |
| [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | NOT_RUN | — | [details](families/featureprior-bcap.md#online-feature-prior-bce-tier-1-stretch) |
| [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-bce-tier-1-stretch) |
| [`online-gail-bcap-hinge`](families/gail-bcap.md#online-gail-bcap-hinge) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-hinge-tier-1-stretch) |
| [`online-gail-bcap-momentum`](families/gail-bcap.md#online-gail-bcap-momentum) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-momentum-tier-1-stretch) |
| [`online-ka2`](families/ka2.md#online-ka2) | NOT_RUN | — | [details](families/ka2.md#online-ka2-tier-1-stretch) |
| [`online-ka2-clean`](families/ka2.md#online-ka2-clean) | NOT_RUN | — | [details](families/ka2.md#online-ka2-clean-tier-1-stretch) |
| [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) | NOT_RUN | — | [details](families/ka2.md#online-ka2-light-anchor-tier-1-stretch) |
| [`online-ka2-ready`](families/ka2.md#online-ka2-ready) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-tier-1-stretch) |
| [`online-ka2-ready-causal`](families/ka2.md#online-ka2-ready-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-causal-tier-1-stretch) |
| [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) | NOT_RUN | — | [details](families/ka2.md#online-ka2-stateless-tier-1-stretch) |
| [`online-ka2-transport-causal`](families/ka2.md#online-ka2-transport-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-transport-causal-tier-1-stretch) |

<a id="seq_minimal_kernel2x2"></a>
### Tier 2: `seq_minimal_kernel2x2` (stretch)

Learn one local kernel response on twelve patches and evaluate four unseen patches.

Gates: `seen_response_rmse <= 0.03`, `seen_max_abs_error <= 0.05`, `heldout_response_rmse <= 0.03`, `heldout_max_abs_error <= 0.05`. Budget: 192 updates; 24 checks; last 5 must pass.

| Recipe | Verdict | Terminal checks | Evidence |
| --- | --- | --- | --- |
| [`online-bcap`](families/bcap.md#online-bcap) | NOT_RUN | — | [details](families/bcap.md#online-bcap-tier-2-stretch) |
| [`online-bcap-adam`](families/bcap.md#online-bcap-adam) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-tier-2-stretch) |
| [`online-bcap-adam-causal`](families/bcap.md#online-bcap-adam-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-causal-tier-2-stretch) |
| [`online-bcap-balanced`](families/bcap.md#online-bcap-balanced) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-tier-2-stretch) |
| [`online-bcap-balanced-bce`](families/bcap.md#online-bcap-balanced-bce) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-bce-tier-2-stretch) |
| [`online-bcap-balanced-frozen`](families/bcap.md#online-bcap-balanced-frozen) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-frozen-tier-2-stretch) |
| [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) | NOT_RUN | — | [details](families/bcap.md#online-bcap-stateless-tier-2-stretch) |
| [`online-bcap-transport`](families/bcap.md#online-bcap-transport) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-tier-2-stretch) |
| [`online-bcap-transport-causal`](families/bcap.md#online-bcap-transport-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-causal-tier-2-stretch) |
| [`online-bcap-transport-fit`](families/bcap.md#online-bcap-transport-fit) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-fit-tier-2-stretch) |
| [`online-bispf`](families/bispf.md#online-bispf) | NOT_RUN | — | [details](families/bispf.md#online-bispf-tier-2-stretch) |
| [`online-bispf-early`](families/bispf.md#online-bispf-early) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-tier-2-stretch) |
| [`online-bispf-early-causal`](families/bispf.md#online-bispf-early-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-causal-tier-2-stretch) |
| [`online-bispf-frozen`](families/bispf.md#online-bispf-frozen) | NOT_RUN | — | [details](families/bispf.md#online-bispf-frozen-tier-2-stretch) |
| [`online-bispf-momentum`](families/bispf.md#online-bispf-momentum) | NOT_RUN | — | [details](families/bispf.md#online-bispf-momentum-tier-2-stretch) |
| [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-tier-2-stretch) |
| [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-cool-tier-2-stretch) |
| [`online-bispf-slow-critic`](families/bispf.md#online-bispf-slow-critic) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-critic-tier-2-stretch) |
| [`online-bispf-slow-prior`](families/bispf.md#online-bispf-slow-prior) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-prior-tier-2-stretch) |
| [`online-bispf-transport-causal`](families/bispf.md#online-bispf-transport-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-transport-causal-tier-2-stretch) |
| [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | NOT_RUN | — | [details](families/fastweight-bcap.md#online-fast-weight-bcap-tier-2-stretch) |
| [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | NOT_RUN | — | [details](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-2-stretch) |
| [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | NOT_RUN | — | [details](families/featureprior-bcap.md#online-feature-prior-bce-tier-2-stretch) |
| [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-bce-tier-2-stretch) |
| [`online-gail-bcap-hinge`](families/gail-bcap.md#online-gail-bcap-hinge) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-hinge-tier-2-stretch) |
| [`online-gail-bcap-momentum`](families/gail-bcap.md#online-gail-bcap-momentum) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-momentum-tier-2-stretch) |
| [`online-ka2`](families/ka2.md#online-ka2) | NOT_RUN | — | [details](families/ka2.md#online-ka2-tier-2-stretch) |
| [`online-ka2-clean`](families/ka2.md#online-ka2-clean) | NOT_RUN | — | [details](families/ka2.md#online-ka2-clean-tier-2-stretch) |
| [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) | NOT_RUN | — | [details](families/ka2.md#online-ka2-light-anchor-tier-2-stretch) |
| [`online-ka2-ready`](families/ka2.md#online-ka2-ready) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-tier-2-stretch) |
| [`online-ka2-ready-causal`](families/ka2.md#online-ka2-ready-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-causal-tier-2-stretch) |
| [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) | NOT_RUN | — | [details](families/ka2.md#online-ka2-stateless-tier-2-stretch) |
| [`online-ka2-transport-causal`](families/ka2.md#online-ka2-transport-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-transport-causal-tier-2-stretch) |

<a id="seq_minimal_addition2bit"></a>
### Tier 2: `seq_minimal_addition2bit` (stretch)

Add two-bit operands on held-out operand combinations.

Gates: `seen_exact_accuracy == 1.0`, `seen_min_bit_confidence >= 0.9`, `heldout_exact_accuracy == 1.0`, `heldout_min_bit_confidence >= 0.9`. Budget: 192 updates; 24 checks; last 5 must pass.

| Recipe | Verdict | Terminal checks | Evidence |
| --- | --- | --- | --- |
| [`online-bcap`](families/bcap.md#online-bcap) | NOT_RUN | — | [details](families/bcap.md#online-bcap-tier-2-stretch) |
| [`online-bcap-adam`](families/bcap.md#online-bcap-adam) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-tier-2-stretch) |
| [`online-bcap-adam-causal`](families/bcap.md#online-bcap-adam-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-causal-tier-2-stretch) |
| [`online-bcap-balanced`](families/bcap.md#online-bcap-balanced) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-tier-2-stretch) |
| [`online-bcap-balanced-bce`](families/bcap.md#online-bcap-balanced-bce) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-bce-tier-2-stretch) |
| [`online-bcap-balanced-frozen`](families/bcap.md#online-bcap-balanced-frozen) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-frozen-tier-2-stretch) |
| [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) | NOT_RUN | — | [details](families/bcap.md#online-bcap-stateless-tier-2-stretch) |
| [`online-bcap-transport`](families/bcap.md#online-bcap-transport) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-tier-2-stretch) |
| [`online-bcap-transport-causal`](families/bcap.md#online-bcap-transport-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-causal-tier-2-stretch) |
| [`online-bcap-transport-fit`](families/bcap.md#online-bcap-transport-fit) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-fit-tier-2-stretch) |
| [`online-bispf`](families/bispf.md#online-bispf) | NOT_RUN | — | [details](families/bispf.md#online-bispf-tier-2-stretch) |
| [`online-bispf-early`](families/bispf.md#online-bispf-early) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-tier-2-stretch) |
| [`online-bispf-early-causal`](families/bispf.md#online-bispf-early-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-causal-tier-2-stretch) |
| [`online-bispf-frozen`](families/bispf.md#online-bispf-frozen) | NOT_RUN | — | [details](families/bispf.md#online-bispf-frozen-tier-2-stretch) |
| [`online-bispf-momentum`](families/bispf.md#online-bispf-momentum) | NOT_RUN | — | [details](families/bispf.md#online-bispf-momentum-tier-2-stretch) |
| [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-tier-2-stretch) |
| [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-cool-tier-2-stretch) |
| [`online-bispf-slow-critic`](families/bispf.md#online-bispf-slow-critic) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-critic-tier-2-stretch) |
| [`online-bispf-slow-prior`](families/bispf.md#online-bispf-slow-prior) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-prior-tier-2-stretch) |
| [`online-bispf-transport-causal`](families/bispf.md#online-bispf-transport-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-transport-causal-tier-2-stretch) |
| [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | NOT_RUN | — | [details](families/fastweight-bcap.md#online-fast-weight-bcap-tier-2-stretch) |
| [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | NOT_RUN | — | [details](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-2-stretch) |
| [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | NOT_RUN | — | [details](families/featureprior-bcap.md#online-feature-prior-bce-tier-2-stretch) |
| [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-bce-tier-2-stretch) |
| [`online-gail-bcap-hinge`](families/gail-bcap.md#online-gail-bcap-hinge) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-hinge-tier-2-stretch) |
| [`online-gail-bcap-momentum`](families/gail-bcap.md#online-gail-bcap-momentum) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-momentum-tier-2-stretch) |
| [`online-ka2`](families/ka2.md#online-ka2) | NOT_RUN | — | [details](families/ka2.md#online-ka2-tier-2-stretch) |
| [`online-ka2-clean`](families/ka2.md#online-ka2-clean) | NOT_RUN | — | [details](families/ka2.md#online-ka2-clean-tier-2-stretch) |
| [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) | NOT_RUN | — | [details](families/ka2.md#online-ka2-light-anchor-tier-2-stretch) |
| [`online-ka2-ready`](families/ka2.md#online-ka2-ready) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-tier-2-stretch) |
| [`online-ka2-ready-causal`](families/ka2.md#online-ka2-ready-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-causal-tier-2-stretch) |
| [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) | NOT_RUN | — | [details](families/ka2.md#online-ka2-stateless-tier-2-stretch) |
| [`online-ka2-transport-causal`](families/ka2.md#online-ka2-transport-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-transport-causal-tier-2-stretch) |

<a id="seq_robotics_fetch_reach"></a>
### Tier 2: `seq_robotics_fetch_reach` (stretch)

Predict action-conditioned physical motion in FetchReach-v4 on held-out simulator episodes.

Gates: `position_delta_rmse_m <= 0.005`, `relative_mse <= 0.8`, `moving_direction_accuracy >= 0.75`, `finite_fraction == 1.0`. Budget: 480 updates; 24 checks; last 5 must pass.

| Recipe | Verdict | Terminal checks | Evidence |
| --- | --- | --- | --- |
| [`online-bcap`](families/bcap.md#online-bcap) | NOT_RUN | — | [details](families/bcap.md#online-bcap-tier-2-stretch) |
| [`online-bcap-adam`](families/bcap.md#online-bcap-adam) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-tier-2-stretch) |
| [`online-bcap-adam-causal`](families/bcap.md#online-bcap-adam-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-causal-tier-2-stretch) |
| [`online-bcap-balanced`](families/bcap.md#online-bcap-balanced) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-tier-2-stretch) |
| [`online-bcap-balanced-bce`](families/bcap.md#online-bcap-balanced-bce) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-bce-tier-2-stretch) |
| [`online-bcap-balanced-frozen`](families/bcap.md#online-bcap-balanced-frozen) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-frozen-tier-2-stretch) |
| [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) | NOT_RUN | — | [details](families/bcap.md#online-bcap-stateless-tier-2-stretch) |
| [`online-bcap-transport`](families/bcap.md#online-bcap-transport) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-tier-2-stretch) |
| [`online-bcap-transport-causal`](families/bcap.md#online-bcap-transport-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-causal-tier-2-stretch) |
| [`online-bcap-transport-fit`](families/bcap.md#online-bcap-transport-fit) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-fit-tier-2-stretch) |
| [`online-bispf`](families/bispf.md#online-bispf) | NOT_RUN | — | [details](families/bispf.md#online-bispf-tier-2-stretch) |
| [`online-bispf-early`](families/bispf.md#online-bispf-early) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-tier-2-stretch) |
| [`online-bispf-early-causal`](families/bispf.md#online-bispf-early-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-causal-tier-2-stretch) |
| [`online-bispf-frozen`](families/bispf.md#online-bispf-frozen) | NOT_RUN | — | [details](families/bispf.md#online-bispf-frozen-tier-2-stretch) |
| [`online-bispf-momentum`](families/bispf.md#online-bispf-momentum) | NOT_RUN | — | [details](families/bispf.md#online-bispf-momentum-tier-2-stretch) |
| [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-tier-2-stretch) |
| [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-cool-tier-2-stretch) |
| [`online-bispf-slow-critic`](families/bispf.md#online-bispf-slow-critic) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-critic-tier-2-stretch) |
| [`online-bispf-slow-prior`](families/bispf.md#online-bispf-slow-prior) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-prior-tier-2-stretch) |
| [`online-bispf-transport-causal`](families/bispf.md#online-bispf-transport-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-transport-causal-tier-2-stretch) |
| [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | NOT_RUN | — | [details](families/fastweight-bcap.md#online-fast-weight-bcap-tier-2-stretch) |
| [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | NOT_RUN | — | [details](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-2-stretch) |
| [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | NOT_RUN | — | [details](families/featureprior-bcap.md#online-feature-prior-bce-tier-2-stretch) |
| [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-bce-tier-2-stretch) |
| [`online-gail-bcap-hinge`](families/gail-bcap.md#online-gail-bcap-hinge) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-hinge-tier-2-stretch) |
| [`online-gail-bcap-momentum`](families/gail-bcap.md#online-gail-bcap-momentum) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-momentum-tier-2-stretch) |
| [`online-ka2`](families/ka2.md#online-ka2) | NOT_RUN | — | [details](families/ka2.md#online-ka2-tier-2-stretch) |
| [`online-ka2-clean`](families/ka2.md#online-ka2-clean) | NOT_RUN | — | [details](families/ka2.md#online-ka2-clean-tier-2-stretch) |
| [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) | NOT_RUN | — | [details](families/ka2.md#online-ka2-light-anchor-tier-2-stretch) |
| [`online-ka2-ready`](families/ka2.md#online-ka2-ready) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-tier-2-stretch) |
| [`online-ka2-ready-causal`](families/ka2.md#online-ka2-ready-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-causal-tier-2-stretch) |
| [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) | NOT_RUN | — | [details](families/ka2.md#online-ka2-stateless-tier-2-stretch) |
| [`online-ka2-transport-causal`](families/ka2.md#online-ka2-transport-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-transport-causal-tier-2-stretch) |

<a id="seq_robotics_fetch_push"></a>
### Tier 2: `seq_robotics_fetch_push` (stretch)

Predict action-conditioned physical motion in FetchPush-v4 on held-out simulator episodes.

Gates: `position_delta_rmse_m <= 0.005`, `relative_mse <= 0.8`, `moving_direction_accuracy >= 0.75`, `finite_fraction == 1.0`. Budget: 480 updates; 24 checks; last 5 must pass.

| Recipe | Verdict | Terminal checks | Evidence |
| --- | --- | --- | --- |
| [`online-bcap`](families/bcap.md#online-bcap) | NOT_RUN | — | [details](families/bcap.md#online-bcap-tier-2-stretch) |
| [`online-bcap-adam`](families/bcap.md#online-bcap-adam) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-tier-2-stretch) |
| [`online-bcap-adam-causal`](families/bcap.md#online-bcap-adam-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-causal-tier-2-stretch) |
| [`online-bcap-balanced`](families/bcap.md#online-bcap-balanced) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-tier-2-stretch) |
| [`online-bcap-balanced-bce`](families/bcap.md#online-bcap-balanced-bce) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-bce-tier-2-stretch) |
| [`online-bcap-balanced-frozen`](families/bcap.md#online-bcap-balanced-frozen) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-frozen-tier-2-stretch) |
| [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) | NOT_RUN | — | [details](families/bcap.md#online-bcap-stateless-tier-2-stretch) |
| [`online-bcap-transport`](families/bcap.md#online-bcap-transport) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-tier-2-stretch) |
| [`online-bcap-transport-causal`](families/bcap.md#online-bcap-transport-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-causal-tier-2-stretch) |
| [`online-bcap-transport-fit`](families/bcap.md#online-bcap-transport-fit) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-fit-tier-2-stretch) |
| [`online-bispf`](families/bispf.md#online-bispf) | NOT_RUN | — | [details](families/bispf.md#online-bispf-tier-2-stretch) |
| [`online-bispf-early`](families/bispf.md#online-bispf-early) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-tier-2-stretch) |
| [`online-bispf-early-causal`](families/bispf.md#online-bispf-early-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-causal-tier-2-stretch) |
| [`online-bispf-frozen`](families/bispf.md#online-bispf-frozen) | NOT_RUN | — | [details](families/bispf.md#online-bispf-frozen-tier-2-stretch) |
| [`online-bispf-momentum`](families/bispf.md#online-bispf-momentum) | NOT_RUN | — | [details](families/bispf.md#online-bispf-momentum-tier-2-stretch) |
| [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-tier-2-stretch) |
| [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-cool-tier-2-stretch) |
| [`online-bispf-slow-critic`](families/bispf.md#online-bispf-slow-critic) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-critic-tier-2-stretch) |
| [`online-bispf-slow-prior`](families/bispf.md#online-bispf-slow-prior) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-prior-tier-2-stretch) |
| [`online-bispf-transport-causal`](families/bispf.md#online-bispf-transport-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-transport-causal-tier-2-stretch) |
| [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | NOT_RUN | — | [details](families/fastweight-bcap.md#online-fast-weight-bcap-tier-2-stretch) |
| [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | NOT_RUN | — | [details](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-2-stretch) |
| [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | NOT_RUN | — | [details](families/featureprior-bcap.md#online-feature-prior-bce-tier-2-stretch) |
| [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-bce-tier-2-stretch) |
| [`online-gail-bcap-hinge`](families/gail-bcap.md#online-gail-bcap-hinge) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-hinge-tier-2-stretch) |
| [`online-gail-bcap-momentum`](families/gail-bcap.md#online-gail-bcap-momentum) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-momentum-tier-2-stretch) |
| [`online-ka2`](families/ka2.md#online-ka2) | NOT_RUN | — | [details](families/ka2.md#online-ka2-tier-2-stretch) |
| [`online-ka2-clean`](families/ka2.md#online-ka2-clean) | NOT_RUN | — | [details](families/ka2.md#online-ka2-clean-tier-2-stretch) |
| [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) | NOT_RUN | — | [details](families/ka2.md#online-ka2-light-anchor-tier-2-stretch) |
| [`online-ka2-ready`](families/ka2.md#online-ka2-ready) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-tier-2-stretch) |
| [`online-ka2-ready-causal`](families/ka2.md#online-ka2-ready-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-causal-tier-2-stretch) |
| [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) | NOT_RUN | — | [details](families/ka2.md#online-ka2-stateless-tier-2-stretch) |
| [`online-ka2-transport-causal`](families/ka2.md#online-ka2-transport-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-transport-causal-tier-2-stretch) |

<a id="seq_robotics_fetch_slide"></a>
### Tier 2: `seq_robotics_fetch_slide` (stretch)

Predict action-conditioned physical motion in FetchSlide-v4 on held-out simulator episodes.

Gates: `position_delta_rmse_m <= 0.005`, `relative_mse <= 0.8`, `moving_direction_accuracy >= 0.75`, `finite_fraction == 1.0`. Budget: 480 updates; 24 checks; last 5 must pass.

| Recipe | Verdict | Terminal checks | Evidence |
| --- | --- | --- | --- |
| [`online-bcap`](families/bcap.md#online-bcap) | NOT_RUN | — | [details](families/bcap.md#online-bcap-tier-2-stretch) |
| [`online-bcap-adam`](families/bcap.md#online-bcap-adam) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-tier-2-stretch) |
| [`online-bcap-adam-causal`](families/bcap.md#online-bcap-adam-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-causal-tier-2-stretch) |
| [`online-bcap-balanced`](families/bcap.md#online-bcap-balanced) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-tier-2-stretch) |
| [`online-bcap-balanced-bce`](families/bcap.md#online-bcap-balanced-bce) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-bce-tier-2-stretch) |
| [`online-bcap-balanced-frozen`](families/bcap.md#online-bcap-balanced-frozen) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-frozen-tier-2-stretch) |
| [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) | NOT_RUN | — | [details](families/bcap.md#online-bcap-stateless-tier-2-stretch) |
| [`online-bcap-transport`](families/bcap.md#online-bcap-transport) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-tier-2-stretch) |
| [`online-bcap-transport-causal`](families/bcap.md#online-bcap-transport-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-causal-tier-2-stretch) |
| [`online-bcap-transport-fit`](families/bcap.md#online-bcap-transport-fit) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-fit-tier-2-stretch) |
| [`online-bispf`](families/bispf.md#online-bispf) | NOT_RUN | — | [details](families/bispf.md#online-bispf-tier-2-stretch) |
| [`online-bispf-early`](families/bispf.md#online-bispf-early) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-tier-2-stretch) |
| [`online-bispf-early-causal`](families/bispf.md#online-bispf-early-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-causal-tier-2-stretch) |
| [`online-bispf-frozen`](families/bispf.md#online-bispf-frozen) | NOT_RUN | — | [details](families/bispf.md#online-bispf-frozen-tier-2-stretch) |
| [`online-bispf-momentum`](families/bispf.md#online-bispf-momentum) | NOT_RUN | — | [details](families/bispf.md#online-bispf-momentum-tier-2-stretch) |
| [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-tier-2-stretch) |
| [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-cool-tier-2-stretch) |
| [`online-bispf-slow-critic`](families/bispf.md#online-bispf-slow-critic) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-critic-tier-2-stretch) |
| [`online-bispf-slow-prior`](families/bispf.md#online-bispf-slow-prior) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-prior-tier-2-stretch) |
| [`online-bispf-transport-causal`](families/bispf.md#online-bispf-transport-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-transport-causal-tier-2-stretch) |
| [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | NOT_RUN | — | [details](families/fastweight-bcap.md#online-fast-weight-bcap-tier-2-stretch) |
| [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | NOT_RUN | — | [details](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-2-stretch) |
| [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | NOT_RUN | — | [details](families/featureprior-bcap.md#online-feature-prior-bce-tier-2-stretch) |
| [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-bce-tier-2-stretch) |
| [`online-gail-bcap-hinge`](families/gail-bcap.md#online-gail-bcap-hinge) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-hinge-tier-2-stretch) |
| [`online-gail-bcap-momentum`](families/gail-bcap.md#online-gail-bcap-momentum) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-momentum-tier-2-stretch) |
| [`online-ka2`](families/ka2.md#online-ka2) | NOT_RUN | — | [details](families/ka2.md#online-ka2-tier-2-stretch) |
| [`online-ka2-clean`](families/ka2.md#online-ka2-clean) | NOT_RUN | — | [details](families/ka2.md#online-ka2-clean-tier-2-stretch) |
| [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) | NOT_RUN | — | [details](families/ka2.md#online-ka2-light-anchor-tier-2-stretch) |
| [`online-ka2-ready`](families/ka2.md#online-ka2-ready) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-tier-2-stretch) |
| [`online-ka2-ready-causal`](families/ka2.md#online-ka2-ready-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-causal-tier-2-stretch) |
| [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) | NOT_RUN | — | [details](families/ka2.md#online-ka2-stateless-tier-2-stretch) |
| [`online-ka2-transport-causal`](families/ka2.md#online-ka2-transport-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-transport-causal-tier-2-stretch) |

<a id="seq_robotics_fetch_pick_and_place"></a>
### Tier 2: `seq_robotics_fetch_pick_and_place` (stretch)

Predict action-conditioned physical motion in FetchPickAndPlace-v4 on held-out simulator episodes.

Gates: `position_delta_rmse_m <= 0.005`, `relative_mse <= 0.8`, `moving_direction_accuracy >= 0.75`, `finite_fraction == 1.0`. Budget: 480 updates; 24 checks; last 5 must pass.

| Recipe | Verdict | Terminal checks | Evidence |
| --- | --- | --- | --- |
| [`online-bcap`](families/bcap.md#online-bcap) | NOT_RUN | — | [details](families/bcap.md#online-bcap-tier-2-stretch) |
| [`online-bcap-adam`](families/bcap.md#online-bcap-adam) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-tier-2-stretch) |
| [`online-bcap-adam-causal`](families/bcap.md#online-bcap-adam-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-causal-tier-2-stretch) |
| [`online-bcap-balanced`](families/bcap.md#online-bcap-balanced) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-tier-2-stretch) |
| [`online-bcap-balanced-bce`](families/bcap.md#online-bcap-balanced-bce) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-bce-tier-2-stretch) |
| [`online-bcap-balanced-frozen`](families/bcap.md#online-bcap-balanced-frozen) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-frozen-tier-2-stretch) |
| [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) | NOT_RUN | — | [details](families/bcap.md#online-bcap-stateless-tier-2-stretch) |
| [`online-bcap-transport`](families/bcap.md#online-bcap-transport) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-tier-2-stretch) |
| [`online-bcap-transport-causal`](families/bcap.md#online-bcap-transport-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-causal-tier-2-stretch) |
| [`online-bcap-transport-fit`](families/bcap.md#online-bcap-transport-fit) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-fit-tier-2-stretch) |
| [`online-bispf`](families/bispf.md#online-bispf) | NOT_RUN | — | [details](families/bispf.md#online-bispf-tier-2-stretch) |
| [`online-bispf-early`](families/bispf.md#online-bispf-early) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-tier-2-stretch) |
| [`online-bispf-early-causal`](families/bispf.md#online-bispf-early-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-causal-tier-2-stretch) |
| [`online-bispf-frozen`](families/bispf.md#online-bispf-frozen) | NOT_RUN | — | [details](families/bispf.md#online-bispf-frozen-tier-2-stretch) |
| [`online-bispf-momentum`](families/bispf.md#online-bispf-momentum) | NOT_RUN | — | [details](families/bispf.md#online-bispf-momentum-tier-2-stretch) |
| [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-tier-2-stretch) |
| [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-cool-tier-2-stretch) |
| [`online-bispf-slow-critic`](families/bispf.md#online-bispf-slow-critic) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-critic-tier-2-stretch) |
| [`online-bispf-slow-prior`](families/bispf.md#online-bispf-slow-prior) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-prior-tier-2-stretch) |
| [`online-bispf-transport-causal`](families/bispf.md#online-bispf-transport-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-transport-causal-tier-2-stretch) |
| [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | NOT_RUN | — | [details](families/fastweight-bcap.md#online-fast-weight-bcap-tier-2-stretch) |
| [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | NOT_RUN | — | [details](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-2-stretch) |
| [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | NOT_RUN | — | [details](families/featureprior-bcap.md#online-feature-prior-bce-tier-2-stretch) |
| [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-bce-tier-2-stretch) |
| [`online-gail-bcap-hinge`](families/gail-bcap.md#online-gail-bcap-hinge) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-hinge-tier-2-stretch) |
| [`online-gail-bcap-momentum`](families/gail-bcap.md#online-gail-bcap-momentum) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-momentum-tier-2-stretch) |
| [`online-ka2`](families/ka2.md#online-ka2) | NOT_RUN | — | [details](families/ka2.md#online-ka2-tier-2-stretch) |
| [`online-ka2-clean`](families/ka2.md#online-ka2-clean) | NOT_RUN | — | [details](families/ka2.md#online-ka2-clean-tier-2-stretch) |
| [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) | NOT_RUN | — | [details](families/ka2.md#online-ka2-light-anchor-tier-2-stretch) |
| [`online-ka2-ready`](families/ka2.md#online-ka2-ready) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-tier-2-stretch) |
| [`online-ka2-ready-causal`](families/ka2.md#online-ka2-ready-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-causal-tier-2-stretch) |
| [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) | NOT_RUN | — | [details](families/ka2.md#online-ka2-stateless-tier-2-stretch) |
| [`online-ka2-transport-causal`](families/ka2.md#online-ka2-transport-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-transport-causal-tier-2-stretch) |

<a id="seq_robotics_hand_reach"></a>
### Tier 2: `seq_robotics_hand_reach` (stretch)

Predict action-conditioned physical motion in HandReach-v3 on held-out simulator episodes.

Gates: `position_delta_rmse_m <= 0.005`, `relative_mse <= 0.8`, `moving_direction_accuracy >= 0.75`, `finite_fraction == 1.0`. Budget: 480 updates; 24 checks; last 5 must pass.

| Recipe | Verdict | Terminal checks | Evidence |
| --- | --- | --- | --- |
| [`online-bcap`](families/bcap.md#online-bcap) | NOT_RUN | — | [details](families/bcap.md#online-bcap-tier-2-stretch) |
| [`online-bcap-adam`](families/bcap.md#online-bcap-adam) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-tier-2-stretch) |
| [`online-bcap-adam-causal`](families/bcap.md#online-bcap-adam-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-causal-tier-2-stretch) |
| [`online-bcap-balanced`](families/bcap.md#online-bcap-balanced) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-tier-2-stretch) |
| [`online-bcap-balanced-bce`](families/bcap.md#online-bcap-balanced-bce) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-bce-tier-2-stretch) |
| [`online-bcap-balanced-frozen`](families/bcap.md#online-bcap-balanced-frozen) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-frozen-tier-2-stretch) |
| [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) | NOT_RUN | — | [details](families/bcap.md#online-bcap-stateless-tier-2-stretch) |
| [`online-bcap-transport`](families/bcap.md#online-bcap-transport) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-tier-2-stretch) |
| [`online-bcap-transport-causal`](families/bcap.md#online-bcap-transport-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-causal-tier-2-stretch) |
| [`online-bcap-transport-fit`](families/bcap.md#online-bcap-transport-fit) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-fit-tier-2-stretch) |
| [`online-bispf`](families/bispf.md#online-bispf) | NOT_RUN | — | [details](families/bispf.md#online-bispf-tier-2-stretch) |
| [`online-bispf-early`](families/bispf.md#online-bispf-early) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-tier-2-stretch) |
| [`online-bispf-early-causal`](families/bispf.md#online-bispf-early-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-causal-tier-2-stretch) |
| [`online-bispf-frozen`](families/bispf.md#online-bispf-frozen) | NOT_RUN | — | [details](families/bispf.md#online-bispf-frozen-tier-2-stretch) |
| [`online-bispf-momentum`](families/bispf.md#online-bispf-momentum) | NOT_RUN | — | [details](families/bispf.md#online-bispf-momentum-tier-2-stretch) |
| [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-tier-2-stretch) |
| [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-cool-tier-2-stretch) |
| [`online-bispf-slow-critic`](families/bispf.md#online-bispf-slow-critic) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-critic-tier-2-stretch) |
| [`online-bispf-slow-prior`](families/bispf.md#online-bispf-slow-prior) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-prior-tier-2-stretch) |
| [`online-bispf-transport-causal`](families/bispf.md#online-bispf-transport-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-transport-causal-tier-2-stretch) |
| [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | NOT_RUN | — | [details](families/fastweight-bcap.md#online-fast-weight-bcap-tier-2-stretch) |
| [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | NOT_RUN | — | [details](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-2-stretch) |
| [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | NOT_RUN | — | [details](families/featureprior-bcap.md#online-feature-prior-bce-tier-2-stretch) |
| [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-bce-tier-2-stretch) |
| [`online-gail-bcap-hinge`](families/gail-bcap.md#online-gail-bcap-hinge) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-hinge-tier-2-stretch) |
| [`online-gail-bcap-momentum`](families/gail-bcap.md#online-gail-bcap-momentum) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-momentum-tier-2-stretch) |
| [`online-ka2`](families/ka2.md#online-ka2) | NOT_RUN | — | [details](families/ka2.md#online-ka2-tier-2-stretch) |
| [`online-ka2-clean`](families/ka2.md#online-ka2-clean) | NOT_RUN | — | [details](families/ka2.md#online-ka2-clean-tier-2-stretch) |
| [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) | NOT_RUN | — | [details](families/ka2.md#online-ka2-light-anchor-tier-2-stretch) |
| [`online-ka2-ready`](families/ka2.md#online-ka2-ready) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-tier-2-stretch) |
| [`online-ka2-ready-causal`](families/ka2.md#online-ka2-ready-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-causal-tier-2-stretch) |
| [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) | NOT_RUN | — | [details](families/ka2.md#online-ka2-stateless-tier-2-stretch) |
| [`online-ka2-transport-causal`](families/ka2.md#online-ka2-transport-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-transport-causal-tier-2-stretch) |

<a id="seq_minimal_velocity_extrapolation"></a>
### Tier 3: `seq_minimal_velocity_extrapolation` (stretch)

Predict three unseen-speed point frames from three observed frames, separating interpolation from extrapolation.

Gates: `interpolation_centroid_rmse <= 0.01`, `interpolation_speed_error <= 0.01`, `interpolation_raw_rmse <= 0.12`, `interpolation_point_validity == 1.0`, `interpolation_direction_accuracy == 1.0`, `extrapolation_centroid_rmse <= 0.01`, `extrapolation_speed_error <= 0.01`, `extrapolation_raw_rmse <= 0.12`, `extrapolation_point_validity == 1.0`, `extrapolation_direction_accuracy == 1.0`. Budget: 384 updates; 24 checks; last 5 must pass.

| Recipe | Verdict | Terminal checks | Evidence |
| --- | --- | --- | --- |
| [`online-bcap`](families/bcap.md#online-bcap) | NOT_RUN | — | [details](families/bcap.md#online-bcap-tier-3-stretch) |
| [`online-bcap-adam`](families/bcap.md#online-bcap-adam) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-tier-3-stretch) |
| [`online-bcap-adam-causal`](families/bcap.md#online-bcap-adam-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-adam-causal-tier-3-stretch) |
| [`online-bcap-balanced`](families/bcap.md#online-bcap-balanced) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-tier-3-stretch) |
| [`online-bcap-balanced-bce`](families/bcap.md#online-bcap-balanced-bce) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-bce-tier-3-stretch) |
| [`online-bcap-balanced-frozen`](families/bcap.md#online-bcap-balanced-frozen) | NOT_RUN | — | [details](families/bcap.md#online-bcap-balanced-frozen-tier-3-stretch) |
| [`online-bcap-stateless`](families/bcap.md#online-bcap-stateless) | NOT_RUN | — | [details](families/bcap.md#online-bcap-stateless-tier-3-stretch) |
| [`online-bcap-transport`](families/bcap.md#online-bcap-transport) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-tier-3-stretch) |
| [`online-bcap-transport-causal`](families/bcap.md#online-bcap-transport-causal) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-causal-tier-3-stretch) |
| [`online-bcap-transport-fit`](families/bcap.md#online-bcap-transport-fit) | NOT_RUN | — | [details](families/bcap.md#online-bcap-transport-fit-tier-3-stretch) |
| [`online-bispf`](families/bispf.md#online-bispf) | NOT_RUN | — | [details](families/bispf.md#online-bispf-tier-3-stretch) |
| [`online-bispf-early`](families/bispf.md#online-bispf-early) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-tier-3-stretch) |
| [`online-bispf-early-causal`](families/bispf.md#online-bispf-early-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-early-causal-tier-3-stretch) |
| [`online-bispf-frozen`](families/bispf.md#online-bispf-frozen) | NOT_RUN | — | [details](families/bispf.md#online-bispf-frozen-tier-3-stretch) |
| [`online-bispf-momentum`](families/bispf.md#online-bispf-momentum) | NOT_RUN | — | [details](families/bispf.md#online-bispf-momentum-tier-3-stretch) |
| [`online-bispf-optimized`](families/bispf.md#online-bispf-optimized) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-tier-3-stretch) |
| [`online-bispf-optimized-cool`](families/bispf.md#online-bispf-optimized-cool) | NOT_RUN | — | [details](families/bispf.md#online-bispf-optimized-cool-tier-3-stretch) |
| [`online-bispf-slow-critic`](families/bispf.md#online-bispf-slow-critic) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-critic-tier-3-stretch) |
| [`online-bispf-slow-prior`](families/bispf.md#online-bispf-slow-prior) | NOT_RUN | — | [details](families/bispf.md#online-bispf-slow-prior-tier-3-stretch) |
| [`online-bispf-transport-causal`](families/bispf.md#online-bispf-transport-causal) | NOT_RUN | — | [details](families/bispf.md#online-bispf-transport-causal-tier-3-stretch) |
| [`online-fast-weight-bcap`](families/fastweight-bcap.md#online-fast-weight-bcap) | NOT_RUN | — | [details](families/fastweight-bcap.md#online-fast-weight-bcap-tier-3-stretch) |
| [`online-fast-weight-bcap-ns`](families/global-stream-bcap.md#online-fast-weight-bcap-ns) | NOT_RUN | — | [details](families/global-stream-bcap.md#online-fast-weight-bcap-ns-tier-3-stretch) |
| [`online-feature-prior-bce`](families/featureprior-bcap.md#online-feature-prior-bce) | NOT_RUN | — | [details](families/featureprior-bcap.md#online-feature-prior-bce-tier-3-stretch) |
| [`online-gail-bcap-bce`](families/gail-bcap.md#online-gail-bcap-bce) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-bce-tier-3-stretch) |
| [`online-gail-bcap-hinge`](families/gail-bcap.md#online-gail-bcap-hinge) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-hinge-tier-3-stretch) |
| [`online-gail-bcap-momentum`](families/gail-bcap.md#online-gail-bcap-momentum) | NOT_RUN | — | [details](families/gail-bcap.md#online-gail-bcap-momentum-tier-3-stretch) |
| [`online-ka2`](families/ka2.md#online-ka2) | NOT_RUN | — | [details](families/ka2.md#online-ka2-tier-3-stretch) |
| [`online-ka2-clean`](families/ka2.md#online-ka2-clean) | NOT_RUN | — | [details](families/ka2.md#online-ka2-clean-tier-3-stretch) |
| [`online-ka2-light-anchor`](families/ka2.md#online-ka2-light-anchor) | NOT_RUN | — | [details](families/ka2.md#online-ka2-light-anchor-tier-3-stretch) |
| [`online-ka2-ready`](families/ka2.md#online-ka2-ready) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-tier-3-stretch) |
| [`online-ka2-ready-causal`](families/ka2.md#online-ka2-ready-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-ready-causal-tier-3-stretch) |
| [`online-ka2-stateless`](families/ka2.md#online-ka2-stateless) | NOT_RUN | — | [details](families/ka2.md#online-ka2-stateless-tier-3-stretch) |
| [`online-ka2-transport-causal`](families/ka2.md#online-ka2-transport-causal) | NOT_RUN | — | [details](families/ka2.md#online-ka2-transport-causal-tier-3-stretch) |

## Reproduction

```sh
particlestream validate
particlestream compare --all-tasks --device cuda
# Resume only unchanged runs from this refresh, if interrupted:
particlestream compare --all-tasks --resume --device cuda
# Regenerate this inventory and every linked explanation without training:
particlestream compile --device cuda
```

[Protocol](../../EXPERIMENTATION.md) · [Task contracts](EXPERIMENTS_BY_TIER.md) · [Experiment memory](EXPERIMENT_MEMORY.md) · [Historical interpretation](research-notes.md)
