# Results and interpretation

## Main comparisons (requested budget 0.20)

| Analysis | Result | Interpretation |
|---|---|---|
| DLPFC matched-mask refit | TSB-cov 0.868, GraphMask 0.858, random 0.833 (macro F1) | Edges useful to a fresh GCN; parity with GraphMask, not superiority |
| Mouse matched-mask refit | TSB-cov 0.885 vs random 0.786 | Ten algorithmic repeats on one section |
| DLPFC end-to-end | TSB-cov 0.784, TSB-fixed 0.828, full-graph GCN 0.947 | Coverage costs gated accuracy; no predictive advantage |
| Mouse end-to-end | TSB-cov 0.926 vs graph-free MLP 0.922 | Spatial context is not shown to be necessary |
| Target separation (well-predicted targets) | permuted minus real correlation 0.461 (TSB-fixed), 0.449 (TSB-dual), 0.271 (TSB-cov) | Coverage attenuates specificity |
| Two regression programs (excitatory vs oligodendrocyte) | TSB-dual 0.30 vs 0.82 permuted; TSB-cov 0.81 vs 0.83 | Separation without coverage is not a loss-type artifact |

Mean exported fractions at a requested 0.20 are 0.205 (DLPFC) and 0.218 (mouse). Expected L0, the fraction of deterministic gates above a threshold, and the top-k exported fraction are different quantities.

## Caveats that affect interpretation

- Under TSB-cov, the six gene-program maps are about as similar to each other as permuted-target maps (0.819 vs 0.830); use TSB-dual to contrast programs.
- Under TSB-dual, poorly predicted programs also differ from one another (0.58 vs 0.82 permuted), so distinct maps alone are not evidence of target information.
- Reliability does not imply informativeness: microglia (R2 -0.18) has consensus reliability 0.80.
- The held-out-marker and co-expression audits and the 18-program panel use TSB-fixed. In the marker audit, 2.39 is the expression-matched control's Moran's I ratio, not an adjusted TSB effect (TSB: 2.77).
- Paired section tests average seeds but not donor dependence (three donors); they are consistency checks, not population tests.
- Holm correction over 21 tests supports TSB-cov over random at all three budgets, and over GraphMask only at 0.10.
- Spearman-Brown extrapolations are optimistic heuristics, not measured reliability.
- Poor target predictability can reflect features, noise, resolution, model, or optimization, not necessarily absent spatial structure.
- Tiling reduces memory substantially, but memory still grows with graph size; compare runtimes at matched node counts.
- Regression budgets remain an order of magnitude less accurate than classification budgets.
