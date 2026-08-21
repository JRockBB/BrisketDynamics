# Model Gap Audit — the quality layer's requirements list

Written 2026-08-20 after the fat-rendering gap surfaced. Root cause, stated
plainly: this project has two layers — **calibration** (fit the instrumented
cook) and **quality** (optimize the perfect brisket) — and the
identifiability-first scoping rule ("model only what a temperature log can
see") was correct for the first and silently, wrongly applied to the second,
where the right test is *does this physics move the quality answer?* Fat,
expulsion, carryover, and safety are invisible to a thermometer and
load-bearing for quality; they were excluded by rule mismatch, not by
decision. This document is the standing requirements list that the quality
layer should have had from day one. The goal was always the perfect brisket.

Every entry: what's missing, what it corrupts, build cost, and how it would
be calibrated/observed.

## Tier 1 — moves quality answers; in-window or explicitly confessed

1. **Fat rendering.** No fat phase exists (states are lean meat + water).
   Corrupts: juiciness (a large share of perceived moistness is rendered
   fat, not water — `w_retained` is water-only), the planned mass-loss
   calibration (cook loss = water + drained fat; attributing all to
   evaporation biases D_eff/k_m high), bark texture. Cost: ~bark-sized
   (per-node fat fraction, smooth melt/drainage kinetics). Calibration:
   excellent — beef fat melting curves are literature; drippings (tallow
   catch) + trim weights are direct observables. STATUS: DONE (joint
   build with #2, branch `moisture-physics`): per-node fat state F with
   smooth melt/drainage kinetics (T_render ≈ 55 °C, η_drain split),
   `juiciness` = retained water + β_fat-weighted retained rendered fat,
   and the full cook-loss decomposition evap_w/drip_w/drip_f with exact
   algebraic mass closure (tested to 1e-8). β_fat and the split remain
   literature/sensory placeholders pending the scale experiment.
2. **Denaturation-driven moisture expulsion.** The dominant juice-loss
   mechanism in cooked meat is protein shrinkage squeezing water out
   (myosin ~40–60 °C; actin ~66–73 °C drives the major cook-loss step) —
   our W field only diffuses and evaporates; there is no T-driven
   expulsion source. Corrupts: the juiciness mechanism (totals can fit
   while the physics mis-attributes), schedule sensitivity of juiciness.
   Cost: source term with Arrhenius-like kinetics per node; ~bark-sized.
   Calibration: cook-loss-vs-temperature literature is strong; the scale
   experiment separates channels once fat is also modeled. STATUS: DONE
   (joint build with #1): relaxation sink toward a temperature-dependent
   equilibrium water-holding capacity W_eq(T) with myosin (~58 °C) and
   actin (~72 °C) denaturation steps; endpoint wet-basis cook loss lands
   in the published 30–60% band and the wrapped-cook stall now
   self-attenuates late (the crutch's advantage is mid-cook, ~14 K near
   9 h). Consequence measured: the real-cook replay loss at the frozen
   pre-physics MAP moved +922 — the calibration was refit under the new
   physics (see writeup).
3. **Shrinkage.** Meat loses 20–30% volume; L is fixed. Couples to #2
   (shrinkage IS the expulsion mechanism) and shortens conduction paths
   late. Cost: moderate (effective L(t) or moving mesh). STATUS: confess
   in writeup; post-hackathon.
4. **Food-safety constraint.** The planner recommends schedules with
   nothing bounding danger-zone dwell or verifying pasteurization. The one
   gap with an ethics dimension for a tool that tells people how to cook
   meat. Cost: HOURS — integrate a standard lethality curve from the
   existing T field; constrain in every planner. STATUS: DONE,
   curve-faithful: integrated surface lethality F = ∫10^((Ts−60°C)/z)dt
   (z = 6 °C) required to reach the FSIS ~7-log equivalent (720
   60°C-seconds), penalized on log-shortfall, PLUS a 4 h danger-zone
   dwell cap for the growth phase — both unconditional in the single
   objective. Verified: normal cooks pay ~0; a cold smolder start pays
   ~1160 with 5x gradient pressure.
5. **Rest / carryover.** The deadline formulation leans on "the hold" while
   modeling nothing after the pull: carryover (core rises several °C),
   juice redistribution. The pull decision is biased without it. Cost:
   small (lumped post-pull model). STATUS: next after #1/#2 if window
   holds.
6. **Humidity feedback.** Fixed p_v_pit is documented as a confound, but
   it is worse: the meat's own evaporation humidifies the pit and
   suppresses further evaporation — a two-way MASS coupling ignored while
   the two-way heat coupling is modeled. The water pan — the most debated
   pitmaster lever — is invisible. Cost: moderate (pit humidity state +
   evap feedback). STATUS: confess; post-hackathon with wet-bulb sensor
   (instrument roadmap).

## Tier 2 — affects specific shipped claims

7. **h_c independent of damper.** Real airflow changes convection, not
   just pit temperature; control authority is currently thermally
   mediated only. Touches all control realism. Cost: small (h_c(u) map);
   identifiability needs the characterization burn.
8. **Point vs flat.** Two overlapping muscles, different fat/thickness; we
   model a homogeneous flat slab. Burnt ends and slice variance live
   here. Post-hackathon (pairs with quasi-3D).
9. **Early-phase condensation.** Cold meat below pit dew point GAINS
   surface water initially; evaporation is clamped at zero instead. Also
   the era when smoke adheres. Small build; pairs with #6.
10. **Smoke ring.** NO/myoglobin chemistry; cosmetic but judged under
    appearance. Modelable (ring depth = NO exposure vs denaturation
    front). Post-hackathon novelty.
11. **3D/edges** (roadmapped). Edges cook faster; burnt ends exist.

## Tier 3 — real, but data-blocked or second-order

Grain anisotropy and slice direction; pH/aging/WHC animal variation;
wind/weather on UA(t); mid-cook fuel additions; spritz/baste practice;
multi-meat loading; rub chemistry beyond salt (roadmapped); smoke quality
(assessed separately — thin-blue-smoke entry on the roadmap).

## Standing rule going forward

Any objective term that consumes a model output must cite this document:
either the physics behind it is modeled at Tier-adequate fidelity, or the
term's caveat names the gap. New quality dimensions get audited against
meat science FIRST, not against what the current state vector happens to
contain.
