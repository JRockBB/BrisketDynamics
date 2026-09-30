"""
    ribs()

Spare-rib preset: a thin slab (half-thickness ~12 mm) where the wrap
physics is mandatory rather than optional (docs/roadmap.md item on
generalization) — the crutch constants are literature-fixed pending a
wrapped-cook log, so treat quality outputs as mechanism demos, not
predictions. Constants are placeholders.
"""
ribs() = (
    θ=[3.8e-9, 0.0018, 10.0, 0.012, 1.0, 5.6e-4, 0.20, 0.6],  # placeholder
    cfg=MeatConfig(),
)
