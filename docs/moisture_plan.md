# Moisture Physics: Fat Rendering + Denaturation Expulsion

Gap audit Tier 1 #1 and #2 (docs/model_gaps.md), built as one unit because
they share the moisture story: perceived juiciness = retained water AND
retained rendered fat; total cook loss = evaporated water + dripped water
(purge) + drained fat. Branch `moisture-physics`.

## Physics

**Fat state F(x,t)** — un-rendered fat fraction per node (dry basis),
initial F0 (marbling; per-brisket constant, flat ≈ 0.20–0.25):

    dF/dt = − k_f0 · σ((T − T_render)/δ_render) · F

Beef fat melts over ~32–49 °C, but *rendering* (drainage from ruptured
cells) becomes meaningful from ~54 °C — the smooth switch sits at
T_render ≈ 55 °C. Rendered fat splits: fraction η_drain drains (drip
loss); the rest is retained in-tissue as liquid fat (internal basting),
computed algebraically R = (F0 − F)·(1 − η_drain) — no extra state.

**Water expulsion source** — the dominant cook-loss mechanism: protein
shrinkage squeezes water out (myosin 50–65 °C, actin 70–75 °C). The W
equation gains a relaxation sink toward a temperature-dependent
equilibrium water-holding capacity:

    dW/dt += − k_e · softplus_δ(W − W_eq(T))
    W_eq(T) = W0 · (1 − a1·σ((T−T_m1)/δ_m1) − a2·σ((T−T_m2)/δ_m2))

with T_m1 ≈ 58 °C (myosin), T_m2 ≈ 72 °C (actin) per the denaturation
literature; a1, a2 sized so endpoint cook-loss vs temperature matches the
published beef curves (~15% loss by 60 °C, 30–35% by 80 °C+). Expelled
water drips (purge). Mass closure is algebraic — no new integral states:

    dripped_water = (W0·nx − ΣW_final) − ∫ j_evap dt · (area scaling)

**States**: [T | W | C | F] × nx + [B, S]  →  4·nx + 2 = 162. F is a pure
sink (no feedback into T/W in v1 — fat's thermal-property and
evaporation-barrier effects are documented exclusions; see gaps #3, #6).
Wait — F HAS one feedback: the expulsion sink enters dW. F itself feeds
nothing in v1.

**New outputs** (all additive, Differentiable):
- `juiciness` — perceptual: (ΣW_f + β_fat·ΣR_f) / (nx·(W0 + β_fat·F0·(1−η_drain)))
  with β_fat ≈ 1.8 (fat's per-mass contribution to perceived moistness;
  sensory placeholder, named as such). `w_retained` (water-only) is kept
  for continuity and the scale calibration.
- `fat_rendered` — mean rendered fraction (F0 − F̄)/F0 at pull.
- `mass_evap`, `mass_drip_w`, `mass_drip_f` — the cook-loss decomposition
  the scale experiment observes (tallow catch = drip_w + drip_f;
  before/after = all three).
- New plain input `f0_fat` (per-brisket marbling override; default cfg).

**Objective**: the taste dryness term switches from water-only
`w_retained` to `juiciness`. Nothing else changes.

## Calibration path (why this is Tier-1 buildable NOW)
- k_f0/T_render: tallow melt/render literature (32–49 °C melt, ~54 °C+
  drainage).
- W_eq / k_e: endpoint cook-loss-vs-temperature curves (well published).
- The scale experiment (roadmap) now identifies channels instead of being
  confounded: drippings = drip_w + drip_f (tallow separates from purge on
  cooling — measurable separately!), total loss closes the budget.
- All constants literature-anchored, uncalibrated pending that data;
  cited in MeatConfig comments.

## Steps
1. meat.jl: constants; state relayout (F block after C; B,S move to
   4nx+1/2 — every A[...] slice audited); RHS (dF, expulsion sink in dW);
   outputs (juiciness, fat_rendered, mass split); solve/vjp/cotangents.
2. Julia tests FIRST: mass-closure test (retained + evap + drip_w +
   drip_f = initial, to solver tolerance); endpoint cook-loss curve in
   literature range (hotter endpoint → more loss); fat renders more when
   hotter/longer; juiciness(hot-fast) < juiciness(low-slow)?? — let the
   model answer; FD directions incl. one through the expulsion path.
3. api: outputs + f0_fat input; sentinels; coupled.py dict.
4. Composed check: one FD direction through juiciness.
5. Rerun fullplan + wraptime ONCE under (new objective + new physics) —
   the deferred single rerun; regenerate figures; docs sync (writeup
   taste taxonomy: moisture is now water + rendered fat; gaps #1/#2
   status DONE).

## Risks
- State relayout is the landmine (B/S index shift breaks silently if any
  slice is missed — the mass-closure test catches budget errors, the
  existing bark tests catch B/S mis-indexing).
- Expulsion + evaporation double-counting: the sink and surface flux both
  remove water — physically distinct channels (purge vs vapor); the
  closure test enforces the budget.
- Stiffness: expulsion relaxation adds rate k_e ~ 1/(20 min) — mild vs
  existing stiffness.

Sources: tallow melt/render ranges ([ScienceDirect tallow overview](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/tallow),
render guides); denaturation/WHC ([IJFST 2025 beef WHC](https://academic.oup.com/ijfst/article/60/1/vvaf076/8102298),
[Meat Sci. shrinkage myosin/actin](https://www.sciencedirect.com/science/article/abs/pii/S0309174021000978),
[Food Res. Int. cooking-loss structure](https://www.sciencedirect.com/science/article/abs/pii/S0963996916304033)).
