"""
    sausage()

Sausage preset: a small cylinder (R ≈ 15 mm) whose CASING is the
existing skin-barrier mechanism (`k_skin` throttles evaporation while
heat passes — why sausages burst instead of stalling), high fat, myosin
doneness note: score with the lethality integral, not collagen.
Constants are placeholders.
"""
sausage() = (
    θ=[3.8e-9, 0.0018, 10.0, 0.015, 1.0, 5.6e-4, 0.20, 0.6],  # placeholder
    cfg=with_config(MeatConfig(); geom_m=1.0, k_skin=1.5e-4, F0=0.35),
)
