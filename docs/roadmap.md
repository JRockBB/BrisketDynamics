# Roadmap

What is left, split by what it takes: **modeling goals** need code and
design work only; **calibration & data goals** need logs, instruments,
or a cook. Completed work and the results behind it: `docs/results.md`.
Equations of what exists: `docs/model.md`. Independent review of
correctness, theory gaps, and next steps: `docs/review.md` (its
laptop-only items 1–7 were acted on 2026-09-07 — `docs/results.md`,
review fixes; item 8, the output-feedback estimator, is open; no cook or
instruments until travel ends). Do not treat
optimizer output as a USDA cook, a judge score, or a 95% on-time
guarantee.

## Calibration & data goals

- **C1 — Fly the pilot on a real pit.** The pilot is built and validated
  against the mock HeaterMeter, with a live forecast feedforward
  (`docs/results.md`); what remains needs an actual cook: shadow mode
  first (`brisket pilot --wall-pace 1 --advisory-only` at real
  hardware), then executed-quality scoring on hardware (today it is
  model-estimated — a real pit has no ground-truth juiciness).
- **C2 — The ~$150 instrument package + data flywheel.** The sensor-value
  table (`docs/results.md`, 2026-09-08) ranks the additions on the
  calibrated cook: a humidity sensor first (it measures the burn rate
  through the combustion water and breaks the pit ridge in closed loop),
  then a scale (the only observer of the water-holding latents), then a
  second probe at known depth, then IR, then the ruler. In value order as
  planned before that table:
  a two-hour open-loop *characterization burn* (step the fan 10→30→15%,
  no meat, no controller — free, and the only way to identify the pit's
  fast fan-response dynamics and the draft exponent
  \(m_{\mathrm{conv}}\), which closed-loop cook logs provably hide); a
  *scale* (brisket before/after + fuel before/ash after — mass loss is
  the single most informative missing observable; directly identifies
  D_eff/k_m, the shrinkage anisotropy and the burn ridge); a second meat
  probe at known shallow depth (probe depth as a nuisance latent); DIY
  wet-bulb probe (thermocouple + wick — measures the exact
  stall-relevant quantity; industrial smokehouse practice) to make
  humidity an input and retire the p_v,pit confound; IR spot readings of
  the surface (h_c/k_m separation); a dome probe through a lid lift and
  a spritz (the 25 K one-minute transient a one-state pit cannot show).
  A thermal camera (or IR grid scan) joins the list for the 2-D line:
  bark-DISTRIBUTION constants are the only uncalibrated piece of the
  slab2d Tesseract (`docs/results.md`).
- **C3 — More pooling cooks.** The two-stage pool is built and has run
  on three tiers (`docs/results.md`, `docs/pooling.md`); every further
  logged cook tightens the twin. Still wanted: any SMOKER log with
  BEFORE/AFTER weights (none found; pre-cook weights are common, yield
  pairs are not), and the water-pan designed experiment — pan in vs.
  out, two otherwise-identical cooks — which now has its model term
  (the pan states) and a concrete motivation (the same-model WSM pair's
  UA spread; its second unit ran a dry terra-cotta dish).
- **C4 — Wrap and rest constants.** The butcher-paper step response
  pinned the total post-wrap input reduction but not the η_evap/ε_wrap
  split (`docs/results.md`) — a FOIL-wrapped log (maximum contrast) or a
  known-thickness cook breaks the tie; the rest constants (cambro /
  board) need a log that runs past the pull with the probe left in.
  Hunt state (2026-09-04, both graphs threads walked): Antti's
  Texas-crutch-at-24 h BGE chart is a dead Dropbox link (no archive);
  Tim Ka's paper-wrapped Akorn brisket (dual point/flat probes),
  ChrisB's BGE wrap pair and Marc Strickland's pulled-early butt are
  TVWBB attachments login-gated at full size. **Unlock: a TVWBB
  account** makes those four immediately usable.
- **C5 — Salt calibration.** Temperature logs are nearly blind to salt,
  so this ships only with the scale (C2) and a designed experiment:
  identical cuts salted at −24 h / −1 h / unsalted, weighed before and
  after; mass loss vs salting time is the curve the WHC coupling
  (S_surf/β_whc/k_aw, all placeholder scale) fits. Non-salt rub effects
  (sugar browning, emissivity, surface mass transfer) belong to C6.
- **C6 — Bark + smoke-quality calibration.** The Maillard and smoke
  states, the `mdot` forcing, the wrap-aware \(L^*\) objective and the
  `bitterness` output are in; the constants are literature scale.
  Calibrate against standardized-photo colorimetry: hourly lid-open
  photos with a color card, \(L^*\) read off the calibrated photos, a
  wrapped/unwrapped pair to separate the smoke term from the Maillard
  term; optionally penetration-force texture. Smoke-quality constants
  calibrate via a smolder-forced designed cook (choke the vents for an
  hour; taste / photograph the difference). To our knowledge unpublished
  territory. (The two-sided mahogany band is already in the objective.)
- **C7 — Earn the transfer claims.** The architecture is validated twice
  with zero fitted parameters and the notable cases are one-call
  compositions (`docs/results.md`); what remains is data for the
  placeholder constants: the poultry skin barrier, saturated steam, the
  pellet feed / earth-pit reservoir / firebox-duct / kamado-actuator and
  bone constants, and any instrumented log from the uncommon cookers. A
  combi-oven cook (controlled T+RH) remains the best designed experiment
  for meat parameters that transfer to any smoker.

## Online estimator (review item 8)

The estimator lives in Julia next to the physics as an RxInfer factor
graph (`estimator/`, its own project). The acceptance gate is passed:
on the real 19 h cook the graph reproduces the Gauss–Newton MAP within a
third of a standard deviation and the Fisher widths within 3 %
(`docs/results.md`, RxInfer estimator). Its structure is fixed by three
measured RxInfer behaviours (per-residual nodes, one latent block,
affine relinearization; `estimator/test_rules.jl` guards them). What
remains, in order:

- **More channels.** The registry (`estimator/channels.jl`) carries
  pit and meat probes at any depth, surface IR, humidity, a scale and
  static readings; still to add: fuel mass, bark color, the wet-bulb as
  a psychrometric map of the vapor pressure, ambient and rest-environment
  readings. Exposed area as a latent (so a footprint ruler and a scale
  constrain area and thickness separately) needs the parameter-layout
  change — a ninth meat parameter through the Tesseract columns and the
  calibration artifacts.
- **Noise as graph structure.** The AR(1) discrepancy and its scale as
  nodes (RxInfer's AR node) instead of a fixed whitening; the pooled
  posterior as the prior in data form (one source for priors).
- **Streaming.** Hourly batches through `autoupdates`, the previous
  posterior as the next linearization point (the iterated EKF per
  batch), the pilot consuming mean and draws through juliacall.
- **Mock-pit runs.** Matched and mismatched plants, two channel sets,
  the posterior-width-versus-hour figure and the decision drift.
- **Upstream.** File the two RxInfer findings: the silent prior on a
  singular forward output covariance; iterations not moving the
  linearization point.

## Product

The `brisket` CLI shipped, the pooling work is written up, and the
install story is settled — the clone (or a container) IS the
distribution, no registry publishing (`docs/results.md`,
`docs/pooling.md`, `docs/install.md`). Remaining: a schedule/MPC
frontend beyond the CLI; the bark write-up rides on C6.

## Not modelled

A 2-D axisymmetric surrogate at highest fidelity (a point-over-flat
stack is available, inert by default), not a true packer geometry: the
slab2d Tesseract shadow-scores plans to serve (true cold spot, come-up
time, bark distribution, a doubled-grid check — `docs/results.md`) while
the planners optimize on the 1-D monolith; no animal-to-animal pH, aging or
water-holding capacity (C3); wind and weather on `UA(t)` beyond the
pilot's forecast feedforward + ambient filter (C1); rub chemistry beyond
salt (C5, C6); smoke *quality* constants as opposed to smoke mass (C6).

## Residues of the implemented physics

Not missing features — how the current physics is wrong or uncalibrated,
and which goal above closes it.

- **Food safety.** Lethality on the surface and the core node of a
  symmetric slab — the cold spot of *this* geometry, not of a
  point-and-flat; intact-muscle assumption for the surface; the Appendix A
  surface-humidity condition is not modelled. The come-up time
  (10–54.4 °C, 6 h) is reported and, for brisket, not binding (FSIS
  Scientific Gap 6); `cut_critical` binds it for products where Appendix
  A's CUT applies. The fullplan scan rejects cook lengths whose raw
  residuals fail; the MPC only penalizes and reports. Not a certification.
  (The slab2d shadow report scores the true grid cold spot to serve after
  planning — `run_shadow2d.py`.)
- **Condensation.** Ungated two-sided flux with one mass-transfer
  coefficient both ways, the condensate absorbed into the surface node's
  water (no film, no run-off); it acts through the dew-point crossing in
  the first half hour, which no probe in the cook log sees. (C2:
  wet-bulb, IR)
- **Draft convection.** One exponent for heat and mass transfer, the same
  at every damper position, quoted at a fixed reference draft; the draft
  velocity is never measured, so \(m_{\mathrm{conv}}\) sits at its prior
  (identified only through the fan-duty variation of a closed-loop log,
  on a ridge with \(h_c\)). (C2: characterization burn)
- **Smoke ring.** A penetration-depth model, not the chemistry: one
  effective diffusivity, a denaturation gate at 62 °C, "lit" as the whole
  dependence on the fire, constants at literature scale with no ring
  measurement behind them. An output, never a term in the objective. (C6)
- **Fuel addition.** Lit chimneys or wood-split pulse trains of fixed
  mass; unlit additions are not modelled. The addition time (opt_add)
  and split interval (opt_splits) are planner decisions; the per-split
  mass is config. (C1)
- **Lid lifts.** A lumped-mass energy loss per lift (\(UA_{\mathrm{open}}\)
  from convective exchange + bed radiation — an estimate, not a
  measurement); a dome probe's one-minute transient is not something a
  one-state pit can show. (C2: dome probe)
- **Spritz.** One dose, one liquid temperature, one interval per cook; a
  fixed absorbed fraction (20 %, assumed — with full absorption the planner
  sprayed its way to juiciness) soaks into the surface node and the rest
  flashes off within the one-minute pulse: no film state, no run-off; stops
  at the wrap and the pull. (C2: scale)
- **Multi-meat.** Identical briskets sharing the pit's heat and vapor; no
  shielding, stacking or grate-position differences — n_meat scales one
  cut's exchange in both fidelities. (C3)
- **Water pan.** Two states with placeholder conductances (UA_pan, g_pan);
  the pan's humidity balance is closed (it sees its own vapor) but the
  pan and the meat are closed sequentially, not jointly, and the pan holds
  plain water (no juice/fat film). (C3: the pan experiment)
- **Sources.** Firebox mass/loss/duct conductance, the pellet feed, the
  gas water yield, the earth-pit reservoir and the kamado actuator hint
  are placeholder scale (a firebox/chamber probe pair calibrates the
  offset; the keg pool backs only the kamado vessel block). (C7)
- **Bone.** Thermal-only: moisture/collagen in the bone region still
  evolve as meat, and bone is not yet excluded from the water
  bookkeeping. (C7)
- **Salt.** D_salt is literature by design; S_surf/β_whc/k_aw are
  placeholder scale; salt–bark interaction is not modelled. (C5)
- **Water budget.** The calibrated real cook ends with 4 % of its water
  retained: holding the 10 h stall against the fitted heat input
  (\(h_c \approx 12\) W/m²K plus black-enclosure radiation, ~750 W/m²)
  takes ~0.3 g/m²/s of evaporation, ~11 kg/m² per face, against the
  14.5 kg/m² a 3.9 cm slab holds; a real flat loses ~40 %. Either the
  heat input to the meat is ~2× too high or the slab is thermally thin
  for the water it carries. The scale settles it (C2); the old moisture
  switch hid this by capping surface loss. The \((h_c, D_{\mathrm{eff}},
  L)\) sensitivity map on the calibrated cook (`docs/results.md`,
  water-budget scan) says which of those the scale should expect to find.
- **Purge sink.** Softplus still removes \(\sim k_e \delta_e \ln 2\) water
  at \(W = W_{\mathrm{eq}}\); the calibrated \(k_e\) absorbs the offset.
- **Shrinkage.** Isotropic (`n_shrink = 1/3`); the anisotropy is a config
  constant, uncalibrated (plausible band 1/3–2/3 — meat shrinks across the
  fibre first). The lateral share scales the exposed area by the slab
  *mean* volume; \(\rho c_p\) stays fixed per physical volume; the dry mass
  lost as drained fat (a few %) is ignored in the metric. Identified by a
  ruler and a scale, not by the probe. (C2)
- **Humidity.** Quasi-steady, small-\(\omega\) vapor balance with
  literature constants (AFR 20, \(y_w\) 0.28 kg/kg, ambient \(\omega\)
  0.007) that only a wet-bulb probe can validate; the air flow is tied to
  the draft \(g(u)\) rather than measured; the standalone meat solve (no
  pit) uses a constant 2 kPa. Also an unmodelled confound for the MPC
  ambient filter. (C2, C3)
- **Rest.** A pull blend swaps the surface environment; the rest constants
  are literature presets (cambro / board), the rest-phase \(k_m\) is the
  pit value scaled by \(h_r/h_c\), and the blend smears the pull over
  \(\pm 4\delta_p \approx 20\) min. Nothing here is calibrated — the cook
  log ends at the pull. Modelled carryover is under 0.5 K; the hold's
  continued collagen conversion is the large effect, which makes the
  uncalibrated collagen scale (\(C = 0.90\), `k_ref`) the binding constant
  for quality at serve. (C4)
- **Thermophysical properties.** Choi–Okos with ash lumped into protein
  and the component densities fixed (their 5–95 °C drift is under 3 %); no
  bound-water correction to \(c_p\); the Oswin isotherm's two constants are
  literature values shared with the bark model.
- **Quality scales.** Collagen \(C=0.90\) and the enzymatic route's
  scale (\(k_{\mathrm{enz}}\): 24 h at 55 °C from Baldwin / Tornberg, one
  state for two mechanisms), bark \(k_0\times 5\) and the dry-side
  browning floor \(g_{\mathrm{dry}} = 0.5\), \(L^*\) affine map,
  charcoal-burn smoke proxy, and \(\beta_{\mathrm{fat}}\) are literature
  or placeholders — not texture or color calibrated. KCBS weights do not
  make those proxies into judge scores. (C6; texture data for the two
  routes: a Warner–Bratzler or probe-force series on a designed hold)
- **Likelihood.** Real-cook calibration is AR(1) on contiguous non-lid
  runs; fan duty is treated as exact, per-log ambients are scalars, and
  AR(1) is a discrepancy process, not probe noise. (C3)

Any new objective term either has physics behind it or names its gap here.
Audit new quality dimensions against meat science first, not against the
current state vector.
