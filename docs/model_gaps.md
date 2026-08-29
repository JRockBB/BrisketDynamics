# Model gaps — what the quality layer still does not model

Unmodeled physics and placeholders only. For what *is* implemented, see
`docs/model.md`. Do not treat optimizer output as a USDA cook, a judge
score, or a 95% on-time guarantee.

## Tier 1 — moves quality answers

Shrinkage, rest / carryover and pit humidity are modelled
(`docs/tier1_plan.md`); their residues are under *Limitations* below.

## Tier 2 — corrupts specific claims

Food safety on the cold spot, condensation, draft convection and the
smoke ring are modelled; their residues are under *Limitations* below.

6. **Point vs flat.** Homogeneous slab. Two muscles, different
   fat/thickness, burnt ends, and slice variance are unmodeled.

9. **No 3D / edges.** Edges cook faster; burnt ends exist.

## Tier 3 — data-blocked or second-order

Grain anisotropy and slice direction; pH/aging/WHC animal variation;
wind/weather on `UA(t)` (unobserved — the MPC's ambient filter absorbs an
effective-loss disturbance); rub chemistry beyond salt; smoke *quality*
(thin blue vs smolder). Fuel additions, the spritz policy and multi-meat
loading are modelled; their residues are under *Limitations*.

## Limitations of what is already implemented

These are not missing features. They are how the current physics is wrong
or uncalibrated:

- **Food safety.** Lethality and danger-zone dwell are evaluated on the
  surface and on the core node of a symmetric 1D slab — the cold spot of
  *this* geometry, not of a real point-and-flat; intact-muscle assumption
  for the surface. The fullplan scan rejects cook lengths whose raw
  residuals fail; the MPC's gradient loop only penalizes and reports.
  Not a certification.
- **Condensation.** Ungated two-sided flux with the same mass-transfer
  coefficient both ways and the condensate absorbed into the surface
  node's water (no film, no run-off); it acts through the dew-point
  crossing in the first half hour, which no probe in the cook log sees.
- **Draft convection.** One exponent for heat and mass transfer, the
  same at every damper position, quoted at a fixed reference draft; the
  draft velocity itself is never measured, so \(m_{\mathrm{conv}}\) is
  identified only through the fan-duty variation of a closed-loop log
  and sits on a ridge with \(h_c\). The open-loop characterization burn
  would pin it.
- **Smoke ring.** A penetration-depth model, not the chemistry: one
  effective diffusivity, a denaturation gate at 62 °C, "lit" as the whole
  dependence on the fire, and constants at literature scale with no ring
  measurement behind them. An output, never a term in the objective.
- **Fuel addition.** One lit chimney of fixed mass dumped over five
  minutes, any time the planner likes, at most once per cook; unlit
  additions are not modelled.
- **Lid lifts.** The loss per lift is a lumped-mass energy loss
  (\(UA_{\mathrm{open}}\) from convective exchange + bed radiation, an
  estimate, not a measurement); a dome probe's 25 K one-minute transient
  is not something a one-state pit can show.
- **Spritz.** One dose per cook (chosen by the planner within a light-to-
  heavy band), one liquid temperature, one interval; a fixed absorbed
  fraction (20 %, assumed — with full absorption the planner sprayed its
  way to juiciness) soaks into the surface node and the rest flashes off
  within the one-minute pulse — no film state, so the wet window is the
  pulse, not a film's life; no run-off; stops at the wrap and the pull.
- **Multi-meat.** Identical briskets sharing the pit's heat and vapor;
  no shielding, stacking or grate-position differences.
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

## Rule

Any new objective term either has Tier-adequate physics behind it or names
the gap here. Audit new quality dimensions against meat science first, not
against the current state vector.
