# Model gaps — what the quality layer still does not model

Unmodeled physics and placeholders only. For what *is* implemented, see
`docs/model.md`. Do not treat optimizer output as a USDA cook, a judge
score, or a 95% on-time guarantee.

## Tier 1 — moves quality answers

1. **Shrinkage.** Meat loses 20–30% volume; `L` is fixed. Late-cook
   conduction paths stay too long. The expulsion sink is a water-holding
   relaxation, not a moving mesh. Cost: moderate (`L(t)` or moving mesh).

2. **Rest / carryover.** Nothing is modeled after the pull. Core still
   rises several °C and juice redistributes on the board; the deadline
   formulation leans on "the hold" anyway. Cost: small (lumped post-pull
   model).

3. **Humidity feedback.** Pit vapor pressure is a fixed 2 kPa. Meat
   evaporation does not humidify the pit; a water pan is invisible.
   Calibrated `k_m` is conditional on that constant. Cost: moderate (pit
   humidity state + evap mass coupling). Needs a wet-bulb sensor to
   identify.

## Tier 2 — corrupts specific claims

4. **Food safety is not a constraint.** The objective has weighted
   softplus terms on *surface* lethality and danger-zone dwell
   (`safety_report` prints the residuals). Intact-muscle assumption; the
   cold-spot field is not checked; unsafe plans are not rejected. Do not
   read a low safety term as pasteurization.

5. **\(h_c\) independent of damper.** Airflow changes convection, not just
   pit temperature. Control authority is thermally mediated only.
   Needs the open-loop characterization burn to identify an `h_c(u)` map.

6. **Point vs flat.** Homogeneous slab. Two muscles, different
   fat/thickness, burnt ends, and slice variance are unmodeled.

7. **No condensation.** Cold meat below pit dew point cannot gain surface
   water; evaporation is clamped at zero. That is also when smoke adheres.
   Pairs with humidity (#3).

8. **No smoke ring.** NO/myoglobin chemistry is unmodeled.

9. **No 3D / edges.** Edges cook faster; burnt ends exist.

## Tier 3 — data-blocked or second-order

Grain anisotropy and slice direction; pH/aging/WHC animal variation;
wind/weather on `UA(t)`; mid-cook fuel additions; spritz/baste; multi-meat
loading; rub chemistry beyond salt; smoke *quality* (thin blue vs
smolder).

## Limitations of what is already implemented

These are not missing features. They are how the current physics is wrong
or uncalibrated:

- **Purge sink.** Softplus still removes \(\sim k_e \delta_e \ln 2\) water
  at \(W = W_{\mathrm{eq}}\). Calibrated \(k_e\) absorbs the offset.
- **Evaporation.** Moisture-availability sigmoid × soft-clamped vapor
  deficit. Water activity is used for bark, not for \(p_v(T, a_w)\).
- **Quality scales.** Collagen \(C=0.90\), bark \(k_0\times 5\), \(L^*\)
  affine map, charcoal-burn smoke proxy, and \(\beta_{\mathrm{fat}}\) are
  literature or placeholders — not texture or color calibrated. KCBS
  weights do not make those proxies into judge scores.
- **Likelihood.** Real-cook residuals are autocorrelated. The iid Gaussian
  overcounts information.
- **K=2 Picard.** AD is exact for the unrolled map. The last meat heat
  load is not fed back; `check_picard` warns above a 2 K residual.

## Rule

Any new objective term either has Tier-adequate physics behind it or names
the gap here. Audit new quality dimensions against meat science first, not
against the current state vector.
