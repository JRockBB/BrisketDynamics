# Roadmap

What is left, what is not modelled, and how the implemented physics is
still wrong. Equations of what exists: `docs/model.md`. Do not treat
optimizer output as a USDA cook, a judge score, or a 95% on-time guarantee.

1. **Close the loop for real: MPC as the controller.** Productionize the
   receding-horizon re-planner; integrate with HeaterMeter (open-source fan
   controller — our planner can sit above its PID as the planning layer).
   The library stops advising and starts piloting. The wrap, pull, chimney
   and spritz events already commit irreversibly once executed; remaining:
   pointing the re-planner at a real pit, and a weather input — wind and
   weather on `UA(t)` are unobserved today and land in the MPC's ambient
   filter as an effective-loss disturbance.
2. **Reverse mode only when the parameter count demands it.** Forward mode
   is the primary derivative now (one Dual solve per JVP, one chunked pass
   per Jacobian; docs/monolith.md). A discrete adjoint (Enzyme through a
   fixed-step stiff stepper) becomes worth its cost only for hundreds of
   parameters — dense controls, spatial fields, bark chemistry. The 2-D
   axisymmetric slab (item 9) is the concrete trigger: ~3000 states kill
   forward mode, and the investment there is this item plus stencil
   Jacobian sparsity/coloring — not a modeling-framework change (item 3).
3. **Monolith: done in shape (a)** (`src/monolith.jl`, one Tesseract around
   the whole physics). **MTK/acausal: NOT pursued (decided 2026-09-03).**
   Phase A of the generalization tested the premise empirically: cuts and
   cookers turned out to be *configurations of shared physics*
   (`src/meats/`, `src/cookers/` — four cookers cost two config fields and
   two config recipes), so the composition layer already exists as thin
   presets over `common.jl` cores, and the system-level composition
   boundary is the Tesseract itself. MTK's strengths map badly here:
   nothing is a DAE (the humidity loop is deliberately quasi-steady
   closed-form), its first-class events are discrete — the smooth blends
   are the differentiability story and would have to be re-derived inside
   components — and large MOL stencils are its weak spot. Revisit ONLY on
   one of these triggers, and then for NEW components with the monolith
   as oracle, never as a rewrite: (a) heterogeneous multi-meat wiring
   becomes painful (N different cuts sharing one pit — the hand-rolled
   form is a vector of meat state-blocks generalizing `n_meat`); (b)
   outside contributors need to extend the model without touching the
   core (item 10); (c) an actual DAE enters (flow networks). Small fixed
   vessel graphs (water pan = +2 pit states; two-zone offset = +2 states
   and a duct coupling) are component enrichment, hand-written per
   archetype. The multi-dimensional slab (item 9) activates item 2
   (adjoint + Jacobian sparsity at thousands of states), not this item.
4. **The ~$150 instrument package + data flywheel.** In value order: a
   two-hour open-loop *characterization burn* (step the fan 10→30→15%, no
   meat, no controller — free, and the only way to identify the pit's fast
   fan-response dynamics and the draft exponent \(m_{\mathrm{conv}}\),
   which closed-loop cook logs provably hide); a *scale* (brisket
   before/after + fuel before/ash after — mass loss is the single most
   informative missing observable; directly identifies D_eff/k_m, the
   shrinkage anisotropy and the burn ridge); a second meat probe at known
   shallow depth (probe depth as a nuisance latent); DIY wet-bulb probe
   (thermocouple + wick — measures the exact stall-relevant quantity;
   industrial smokehouse practice) to make humidity an input and retire
   the p_v,pit confound; IR spot readings of the surface (h_c/k_m
   separation); a dome probe through a lid lift and a spritz (the 25 K
   one-minute transient a one-state pit cannot show).
5. **Hierarchical multi-cook pooling.** Pit parameters are properties of the
   smoker (pool tightly across cooks); meat parameters are per-brisket
   (partial pooling → a personal prior over "briskets I buy" — this is
   where pH, aging and water-holding-capacity variation between animals
   lives). Every logged cook tightens the twin. Designed experiment worth
   running early: water pan in vs. out, two otherwise-identical cooks, to
   identify the humidity pathway.

   *Which posterior to use per log, measured against the 22 h NUTS
   reference (scored 2026-09-02, per-site 1-Wasserstein in z;
   `data/calibration/approx_summary.md`).* Gauss-Newton Laplace: 2.5 s a
   log and within 10-20 % of the NUTS widths on 13 of 15 sites; it misses
   only k_spread and a0 (ignition — cook-specific, not pooled). The TMB /
   INLA nested-Laplace marginal on those two coordinates
   (`scripts/run_tmb_ignition.py`: 11x11 grid, converged inner Newton,
   ½ log det correction, ~51 min) closes most of that: W1-to-NUTS 0.095
   vs the Laplace's 0.148, tied with an oracle fit to the NUTS draws.
   Guard the family with `scripts/run_psis_laplace.py --proposal tmb
   --moment-match`: Zhang-Stephens Pareto-khat against the true ODE
   (0.76 raw; importance-weighted moment matching repairs it to 0.47),
   with the sample-size-dependent threshold. Measured and retired
   (2026-09-02 attic tarball): profile nested grids, structured t/mixture
   ignition families, 2-D/5-D surrogate-transition samplers (4+ h, no W1
   gain over the grid), and ADVI/IWAE (5-10x too narrow or divergent).
   The residual honest gap — every method still ~0.86-0.89x NUTS width on
   ignition and weakest on u_leak — is the 19-D sloppy-ribbon coupling a
   2-D-marginal family cannot carry; more data (item 4), not more
   machinery.

   Two-stage hierarchical pooling (no joint NUTS): per-log GN Laplace, then
   a Gaussian pool — pit shared, meat partial, ignition iid. Synthetic
   two-cook check: `scripts/run_hier_synth.py`.

   **Not RxInfer / CBFE for this.** Constrained Bethe Free Energy is the
   online-during-cook estimator (a reactive factor graph over pit state and
   brisket parameters once the model is pure Julia). Batch calibration is
   one stiff ODE likelihood; message passing on that graph is just
   Laplace/VI on a custom node. Do not rebuild batch calibration as a
   factor graph to match NUTS.
6. **Wrap and rest constants.** Calibrate η_evap / ε_wrap against a wrapped
   cook log — posts document the wrap moment, making the wrap event a
   known-input step-response experiment — via the two-cook joint fit
   (pooling, above); the rest constants (cambro / board) against a log that
   runs past the pull. Log acquisition is user-owned, like the 2013 log. A
   state-triggered wrap ("wrap at 165 °F internal") needs event
   sensitivities — natural in the Dyad/monolith exploration where events
   are first-class.
7. **Salt & seasoning physics + salting-time optimization.** Dry-brining is
   diffusion (D_salt ~1e-10 m²/s, well-pinned in the ham-curing literature —
   take it as a constant, don't calibrate): an isothermal fridge-temperature
   pre-cook solve from t_salt to cook start yields the initial salt profile
   S0(x); during the cook a fourth PDE state S(x,t) couples through (a) a
   water-holding-capacity modification of the moisture sink — landing
   exactly in the `w_retained` juiciness output — (b) a water-activity
   factor a_w(S) on p_sat in the evaporation BC (smooth, stays
   differentiable), and optionally (c) a small denaturation-temperature
   shift. "When to salt" is structurally the same decision variable as
   "when to light the fire" and joins the same chance-constrained planning
   problem; d(juiciness)/d(t_salt) flows through the pre-cook horizon for
   free. Identifiability discipline: temperature logs are nearly blind to
   salt, so this ships only with the *scale* (item 4) and a designed
   experiment — identical cuts salted at −24 h / −1 h / unsalted, weighed
   before and after; mass loss vs salting time is the curve the WHC
   coupling fits. Non-salt rub effects (sugar browning, emissivity, surface
   mass transfer) belong to the bark item below, not here.
8. **Bark model calibration.** The Maillard and smoke states, the pit
   `mdot` forcing and the wrap-aware \(L^*\) objective are in; the
   constants are literature scale. Calibrate against standardized-photo
   colorimetry: hourly lid-open photos with a color card, \(L^*\) read off
   the calibrated photos, a wrapped/unwrapped pair to separate the smoke
   term from the Maillard term. Optionally penetration-force texture. Add
   **smoke quality** (the "thin blue smoke" variable): clean vs smoldering
   combustion changes deposition chemistry at the same mass — plausibly a
   differentiable function of airflow/burn state, making bitterness a
   controllable output. To our knowledge unpublished territory. (The
   two-sided mahogany band is already in the objective.)
9. **Fidelity where new sensors justify it.** Newman product-rule quasi-3D
   center-line correction (~3× cost, differentiable, cheap taste of 3D);
   full 2D axisymmetric surface model *only* once thermal-camera surface
   fields exist to calibrate it (also feeds bark distribution / burnt ends,
   the point-vs-flat split — two muscles of different fat and thickness —
   and grain direction, which none of a 1D slab can carry).
10. **Community/product shape.** Publish as a usable open-source stack
    (calibrate-from-your-log CLI, schedule/MPC frontend); write up the bark
    and pooling work properly if the data supports it.
11. **Generalize across meats and cookers.** The architecture already is the
    generalization — heat source ⇄ meat through (T_source, q_load), i.e. swap
    one Tesseract, keep the other. The model family is small: geometry mode
    (slab/cylinder/sphere — one curvature term), surface-BC mode (evaporative
    / skin-barrier for poultry / saturated-steam / wrapped), source model
    (charcoal / pellet-auger / two-zone offset / rock reservoir / electric),
    and doneness kinetics (collagen, myosin, pasteurization — all Arrhenius).
    Progress (2026-09-03): the GEOMETRY axis is in and validated —
    `MeatConfig.geom_m` puts the slab/cylinder/sphere curvature into the
    material-coordinate operator (exact (1+m) center limit; bit-identical
    slab at m = 0), and sous vide needed no new BC code at all (the bath
    is the rest environment from t = 0 with η_rest = 1). Validation with
    zero fitted parameters: `scripts/run_sous_vide.py` reproduces the
    exact conduction series at Baldwin's α/h to ≤ 8% on all 21 table
    cases (his published cylinder/sphere columns are safety-padded ~+15%
    and ~+35% above that exact solution; the slab column, us, and theory
    agree to ~1%; `figures/sous_vide_baldwin.png`). The doneness-kinetics
    axis is configuration, not code: collagen constants (Ea, k_ref,
    T_ref) are `MeatConfig` fields (a myosin-doneness preset is just
    different literature constants), and pasteurization is the existing
    lethality integral. The remaining modes followed on
    2026-09-03: skin barrier (`MeatConfig.k_skin`, a series conductance —
    heat passes, evaporation throttled; the poultry mechanism, constants
    uncalibrated), saturated steam (pure configuration: `pv_base_knots =
    p_sat(T_env)` — the two-sided flux condenses and the stall never
    forms; the pibil mechanism), and an electric source
    (`PitConfig.p_heater`: heat = p_heater·u, nothing burns — no fuel
    loss, no combustion water, no smoke, bark goes Maillard-only). Each
    has a physics-consistency test; none besides sous vide has an
    external validation yet, and exposing geom_m/k_skin/p_heater through
    the Tesseract schema is one optional field each (like t_f) when a
    consumer needs them. Remaining: transfer claims beyond sous vide need
    data.
    Notable cases: pulled pork ≈ free transfer (geometry switch); ribs make
    wrap physics mandatory; a combi oven identifies meat parameters under
    controlled T+RH for transfer to any smoker; sous vide validates against
    Baldwin's tables — DONE, above; and the sealed earth pit (pibil/imu) is the jewel —
    saturated steam kills the stall entirely, the source is an uncontrolled
    thermal discharge, and "control" becomes design (rock mass, seal time) in
    an uninstrumented cooking tradition.

## Not modelled

A homogeneous 1D slab: no point-vs-flat split, no edges or burnt ends, no
grain direction or slice variance (item 9); no animal-to-animal pH, aging
or water-holding capacity (item 5); wind and weather on `UA(t)` (item 1);
rub chemistry beyond salt (items 7, 8); smoke *quality* as opposed to
smoke mass (item 8).

## Residues of the implemented physics

Not missing features — how the current physics is wrong or uncalibrated,
and which item above closes it.

- **Food safety.** Lethality and danger-zone dwell on the surface and the
  core node of a symmetric slab — the cold spot of *this* geometry, not of
  a point-and-flat; intact-muscle assumption for the surface. The fullplan
  scan rejects cook lengths whose raw residuals fail; the MPC only
  penalizes and reports. Not a certification. (item 9)
- **Condensation.** Ungated two-sided flux with one mass-transfer
  coefficient both ways, the condensate absorbed into the surface node's
  water (no film, no run-off); it acts through the dew-point crossing in
  the first half hour, which no probe in the cook log sees. (item 4:
  wet-bulb, IR)
- **Draft convection.** One exponent for heat and mass transfer, the same
  at every damper position, quoted at a fixed reference draft; the draft
  velocity is never measured, so \(m_{\mathrm{conv}}\) sits at its prior
  (identified only through the fan-duty variation of a closed-loop log,
  on a ridge with \(h_c\)). (item 4: characterization burn)
- **Smoke ring.** A penetration-depth model, not the chemistry: one
  effective diffusivity, a denaturation gate at 62 °C, "lit" as the whole
  dependence on the fire, constants at literature scale with no ring
  measurement behind them. An output, never a term in the objective.
  (item 8)
- **Fuel addition.** One lit chimney of fixed mass over five minutes, at
  most once per cook; unlit additions are not modelled. (item 1)
- **Lid lifts.** A lumped-mass energy loss per lift (\(UA_{\mathrm{open}}\)
  from convective exchange + bed radiation — an estimate, not a
  measurement); a dome probe's one-minute transient is not something a
  one-state pit can show. (item 4: dome probe)
- **Spritz.** One dose, one liquid temperature, one interval per cook; a
  fixed absorbed fraction (20 %, assumed — with full absorption the planner
  sprayed its way to juiciness) soaks into the surface node and the rest
  flashes off within the one-minute pulse: no film state, no run-off; stops
  at the wrap and the pull. (item 4: scale)
- **Multi-meat.** Identical briskets sharing the pit's heat and vapor; no
  shielding, stacking or grate-position differences. (item 9)
- **Water budget.** The calibrated real cook ends with 4 % of its water
  retained: holding the 10 h stall against the fitted heat input
  (\(h_c \approx 12\) W/m²K plus black-enclosure radiation, ~750 W/m²)
  takes ~0.3 g/m²/s of evaporation, ~11 kg/m² per face, against the
  14.5 kg/m² a 3.9 cm slab holds; a real flat loses ~40 %. Either the
  heat input to the meat is ~2× too high or the slab is thermally thin
  for the water it carries. The scale settles it (item 4); the old
  moisture switch hid this by capping surface loss.
- **Purge sink.** Softplus still removes \(\sim k_e \delta_e \ln 2\) water
  at \(W = W_{\mathrm{eq}}\); the calibrated \(k_e\) absorbs the offset.
- **Shrinkage.** Isotropic (`n_shrink = 1/3`); the anisotropy is a config
  constant, uncalibrated (plausible band 1/3–2/3 — meat shrinks across the
  fibre first). The lateral share scales the exposed area by the slab
  *mean* volume; \(\rho c_p\) stays fixed per physical volume; the dry mass
  lost as drained fat (a few %) is ignored in the metric. Identified by a
  ruler and a scale, not by the probe. (item 4)
- **Humidity.** Quasi-steady, small-\(\omega\) vapor balance with
  literature constants (AFR 20, \(y_w\) 0.28 kg/kg, ambient \(\omega\)
  0.007, no pan) that only a wet-bulb probe can validate; the air flow is
  tied to the draft \(g(u)\) rather than measured; the standalone meat
  solve (no pit) uses a constant 2 kPa. Also an unmodelled confound for
  the MPC ambient filter. (items 4, 5)
- **Rest.** A pull blend swaps the surface environment; the rest constants
  are literature presets (cambro / board), the rest-phase \(k_m\) is the
  pit value scaled by \(h_r/h_c\), and the blend smears the pull over
  \(\pm 4\delta_p \approx 20\) min. Nothing here is calibrated — the cook
  log ends at the pull. Modelled carryover is under 0.5 K; the hold's
  continued collagen conversion is the large effect, which makes the
  uncalibrated collagen scale (\(C = 0.90\), `k_ref`) the binding constant
  for quality at serve. (item 6)
- **Thermophysical properties.** Choi–Okos with ash lumped into protein
  and the component densities fixed (their 5–95 °C drift is under 3 %); no
  bound-water correction to \(c_p\); the Oswin isotherm's two constants are
  literature values shared with the bark model.
- **Quality scales.** Collagen \(C=0.90\), bark \(k_0\times 5\) and the
  dry-side browning floor \(g_{\mathrm{dry}} = 0.5\), \(L^*\)
  affine map, charcoal-burn smoke proxy, and \(\beta_{\mathrm{fat}}\) are
  literature or placeholders — not texture or color calibrated. KCBS
  weights do not make those proxies into judge scores. (item 8)
- **Likelihood.** Real-cook calibration is AR(1) on contiguous non-lid
  runs; fan duty is treated as exact, ambient is 300 K, and AR(1) is a
  discrepancy process, not probe noise. (item 5)

Any new objective term either has physics behind it or names its gap here.
Audit new quality dimensions against meat science first, not against the
current state vector.
