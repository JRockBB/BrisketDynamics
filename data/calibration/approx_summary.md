# Approximation scoreboard (vs 2-chain NUTS reference)

W1 = per-site 1-Wasserstein to NUTS in unconstrained z (mean over sites / worst site / ignition pair); sd ratio = approx/NUTS ignition widths.

| method | W1 mean | W1 max (site) | W1 a0 | W1 k_spread | sd ratio a0 | sd ratio k_spread | khat | IS ESS | wall |
|---|---|---|---|---|---|---|---|---|---|
| TMB-ign | 0.095 | 0.258 (u_leak) | 0.158 | 0.148 | 0.89 | 0.86 | — | — | 51 min |
| TMB-PSIS | 0.108 | 0.407 (k_spread) | 0.122 | 0.407 | 0.80 | 0.80 | 0.47 | 77 | — |
| Laplace-GN | 0.148 | 0.747 (k_spread) | 0.200 | 0.747 | 0.64 | 0.52 | — | — | — |
