# Model gaps — what the quality layer still does not model

Unmodeled physics and placeholders only. For what *is* implemented, see
`docs/model.md`. Do not treat optimizer output as a USDA cook, a judge
score, or a 95% on-time guarantee.

## Tier 1 — moves quality answers

Shrinkage, rest / carryover and pit humidity are modelled
(`docs/tier1_plan.md`); their residues are under *Limitations* below.

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
   With the pit humidity now dynamic, the two-sided flux is a one-line
   change in `evap_flux_fb`.

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
- **Shrinkage.** Isotropic (`n_shrink = 1/3`); the anisotropy is a config
  constant, uncalibrated (plausible band 1/3–2/3 — meat shrinks across the
  fibre first). The lateral share scales the exposed area by the slab
  *mean* volume (a prismatic slab); \(\rho c_p\) stays fixed per physical
  volume (no composition dependence); the dry mass lost as drained fat (a
  few %) is ignored in the metric. Identified by a ruler and a scale, not
  by the probe. The calibrated artifacts (MAP, VI, NUTS, control) were fit
  without it.
- **Humidity.** Quasi-steady, small-\(\omega\) vapor balance with
  literature constants (AFR 20, \(y_w\) 0.28 kg/kg, ambient
  \(\omega\) 0.007, no pan) that only a wet-bulb probe can validate; the
  air flow is tied to the draft \(g(u)\) rather than measured; the
  standalone meat solve (no pit) uses a constant 2 kPa. The calibrated
  \(k_m\) was fit with that constant. Humidity is also an unmodelled
  confound for the MPC ambient filter.
- **Rest.** A pull blend swaps the surface environment (`docs/tier1_plan.md`
  §1); the rest constants are literature presets (cambro / board), the
  rest-phase \(k_m\) is the pit value scaled by \(h_r/h_c\), and the blend
  smears the pull over \(\pm 4\delta_p \approx 20\) min. Nothing here is
  calibrated — the cook log ends at the pull. Modelled carryover is under
  0.5 K; the hold's continued collagen conversion is the large effect,
  which makes the uncalibrated collagen scale (\(C = 0.90\), `k_ref`) the
  binding constant for quality at serve.
- **Evaporation.** Moisture-availability sigmoid × soft-clamped vapor
  deficit. Water activity is used for bark, not for \(p_v(T, a_w)\).
- **Quality scales.** Collagen \(C=0.90\), bark \(k_0\times 5\), \(L^*\)
  affine map, charcoal-burn smoke proxy, and \(\beta_{\mathrm{fat}}\) are
  literature or placeholders — not texture or color calibrated. KCBS
  weights do not make those proxies into judge scores.
- **Likelihood.** Real-cook calibration is AR(1) on contiguous non-lid
  runs; that stops counting a smooth discrepancy as 111 independent
  Kelvins. Fan duty is still treated as exact, ambient is still 300 K,
  and AR(1) is still a discrepancy process, not probe noise.
- **K=2 Picard.** AD is exact for the unrolled map. The last meat heat
  load is not fed back; `check_picard` warns above a 2 K residual.

## Rule

Any new objective term either has Tier-adequate physics behind it or names
the gap here. Audit new quality dimensions against meat science first, not
against the current state vector.
