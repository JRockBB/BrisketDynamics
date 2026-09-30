"""
    brisket()

The calibrated cut: a packer-brisket flat as the symmetric 1-D slab the
2013 cook log calibrated (docs/writeup.md; posterior under
data/calibration/). Returns `(θ, cfg)` — θ is the default parameter vector
the calibration priors are centered on, cfg the slab MeatConfig.
"""
brisket() = (
    θ=[3.8e-9, 0.0018, 10.0, 0.03, 1.0, 5.6e-4, 0.20, 0.6],
    cfg=MeatConfig(),
)
