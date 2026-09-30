"""
    sous_vide(cfg=MeatConfig(); T_bath, h_bath=95.0)

The water bath as a cooking environment: the rest environment from t = 0
(`η_rest = 1` kills evaporation exactly — vacuum bag — `ε_rest = 0`,
`h_rest` the bath film coefficient). Returns `(cfg, t_pull)`; pass the
`t_pull` (negative: bathed from the start) to `meat_solve`. Validated
against Baldwin's heating tables and the exact conduction series with
zero fitted parameters (scripts/run_sous_vide.py, ≤ 8% on 21 cases).
"""
sous_vide(cfg::MeatConfig=MeatConfig(); T_bath::Real, h_bath::Real=95.0) = (
    cfg=with_config(
        cfg; T_rest=float(T_bath), h_rest=float(h_bath), ε_rest=0.0, η_rest=1.0
    ),
    t_pull=-3000.0,
)
