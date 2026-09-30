"""
    chicken()

Poultry-class preset: the MECHANISMS are the deliverable, the constants
are placeholders (no poultry data has touched this repo). Skin barrier =
finite `k_skin` in series with the boundary layer (heat passes,
evaporation throttled — why poultry does not stall like a brisket);
drumstick-like geometry = cylinder curvature; lower marbling. Doneness
for poultry is NOT the collagen state C — use the pasteurization
lethality integral (app control.py safety machinery).
"""
chicken() = (
    θ=[3.8e-9, 0.0018, 10.0, 0.025, 1.0, 5.6e-4, 0.20, 0.6],  # placeholder
    cfg=with_config(MeatConfig(); k_skin=2.0e-4, geom_m=1.0, F0=0.08),
)
