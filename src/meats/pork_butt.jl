"""
    pork_butt()

Pulled-pork preset: the "free transfer" case — a bone-in butt is
near-spherical (geom_m = 2, R ≈ 5 cm), fattier than a brisket flat, and
uses the SAME collagen doneness kinetics. Constants are placeholders
except where the pooling record backs them (k_e held within ±8 % across
five cooks including two pork butts — docs/pooling.md).
"""
pork_butt() = (
    θ=[3.8e-9, 0.0018, 10.0, 0.05, 1.0, 5.6e-4, 0.20, 0.6],  # placeholder
    cfg=with_config(MeatConfig(); geom_m=2.0, F0=0.30),
)
