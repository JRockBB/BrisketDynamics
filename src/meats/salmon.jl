"""
    hot_smoked_salmon()

Hot-smoked salmon preset: the doneness-kinetics-as-configuration claim
made concrete — the Arrhenius triple (Ea, k_ref, T_ref) swaps from beef
collagen to a fish-myosin scale (denatures decades lower: T_ref 50 °C),
on a thin slab, with no collagenase route (`k_enz0 = 0`: myosin doneness
is not tenderized by a 55 °C hold). Fish holds far less water-binding
collagen; constants are placeholders for the mechanism demo.
"""
hot_smoked_salmon() = (
    θ=[4.5e-9, 0.0018, 10.0, 0.015, 1.0, 4.0e-4, 0.15, 0.6],  # placeholder
    cfg=with_config(
        MeatConfig();
        F0=0.28, W0=2.0, Ea=2.5e5, k_ref=2.0e-6, T_ref=323.15, k_enz0=0.0,
    ),
)
