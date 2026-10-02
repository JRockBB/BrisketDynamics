"""
    chuck()

Chuck-roast preset: a boneless beef shoulder roast, thicker than a brisket flat (half-thickness ~4 cm) and more
marbled (fat placed between the brisket's and a pork butt's, F0 = 0.28 kg per kg of dry matter). Constants are
placeholders pending a chuck cook log; the collagen kinetics are the brisket's.
"""
chuck() = (
    θ=[3.8e-9, 0.0018, 10.0, 0.04, 1.0, 5.6e-4, 0.20, 0.6],  # placeholder
    cfg=with_config(MeatConfig(); F0=0.28),
)
