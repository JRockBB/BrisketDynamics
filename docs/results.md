# Results record

Completed roadmap work and the decisions behind it, moved out of
`docs/roadmap.md` (which holds only what remains). Equations:
`docs/model.md`. Derivative verification: `docs/verification.md`.

## One monolith Tesseract; MTK not pursued (2026-09-03)

The monolith landed in shape (a): `src/monolith.jl`, one Tesseract around
the whole coupled physics. The MTK/acausal question was then tested
empirically by Phase A of the generalization: cuts and cookers turned out
to be *configurations of shared physics* (`src/meats/`, `src/cookers/` —
four cookers cost two config fields and two config recipes), so the
composition layer already exists as thin presets over `common.jl` cores,
and the system-level composition boundary is the Tesseract itself. MTK's
strengths map badly here: nothing is a DAE (the humidity loop is
deliberately quasi-steady closed-form), its first-class events are
discrete — the smooth blends are the differentiability story and would
have to be re-derived inside components — and large MOL stencils are its
weak spot. Revisit ONLY on one of these triggers, and then for NEW
components with the monolith as oracle, never as a rewrite: (a)
heterogeneous multi-meat wiring becomes painful (N different cuts sharing
one pit — the hand-rolled form is a vector of meat state-blocks
generalizing `n_meat`); (b) outside contributors need to extend the model
without touching the core; (c) an actual DAE enters (flow networks).
Small fixed vessel graphs (water pan = +2 pit states; two-zone offset =
+2 states and a duct coupling) are component enrichment, hand-written per
archetype. The multi-dimensional slab activates the adjoint/sparsity
item, not a framework change.

## Which posterior per log (scored 2026-09-02)

Against the 22 h NUTS reference (per-site 1-Wasserstein in z;
`data/calibration/approx_summary.md`): Gauss-Newton Laplace is 2.5 s a
log and within 10-20 % of the NUTS widths on 13 of 15 sites; it misses
only k_spread and a0 (ignition — cook-specific, not pooled). The TMB /
INLA nested-Laplace marginal on those two coordinates
(`scripts/run_tmb_ignition.py`: 11x11 grid, converged inner Newton,
half log det correction, ~51 min) closes most of that: W1-to-NUTS 0.095
vs the Laplace's 0.148, tied with an oracle fit to the NUTS draws. Guard
the family with `scripts/run_psis_laplace.py --proposal tmb
--moment-match`: Zhang-Stephens Pareto-khat against the true ODE (0.76
raw; importance-weighted moment matching repairs it to 0.47), with the
sample-size-dependent threshold. Measured and retired (2026-09-02 attic
tarball): profile nested grids, structured t/mixture ignition families,
2-D/5-D surrogate-transition samplers (4+ h, no W1 gain over the grid),
and ADVI/IWAE (5-10x too narrow or divergent). The residual honest gap —
every method still ~0.86-0.89x NUTS width on ignition and weakest on
u_leak — is the 19-D sloppy-ribbon coupling a 2-D-marginal family cannot
carry; more data, not more machinery.

**Not RxInfer / CBFE for batch calibration.** Constrained Bethe Free
Energy is the online-during-cook estimator (a reactive factor graph over
pit state and brisket parameters once the model is pure Julia). Batch
calibration is one stiff ODE likelihood; message passing on that graph is
just Laplace/VI on a custom node. Do not rebuild batch calibration as a
factor graph to match NUTS.

## First real two-cook pool: cross-smoker (2026-09-03)

`scripts/run_pool_real.py`: mache_2013 + phillip_2018 (a 15 lb packer at
~1 °F, found, digitized and provenance'd by `tools/loghunt/`; partial
log, different smoker → meat partial, pit per-cook). Result: D_eff agrees
within 7 % across the two cooks, k_e within 7 %, a_myosin/L/m_conv close
— the "briskets I buy" block behaves — while the spread sits on the known
confounds (h_c/k_m on the draft ridge between different rigs; k_mult
absorbing the uniform-interior slice start). Fits: rms pit 1.33/1.87 K,
core 3.02/1.43 K (`figures/pool_real.png`,
`data/calibration/pool_real.json`). Synthetic two-cook mechanics check:
`scripts/run_hier_synth.py`.

## Same-smoker pool: one keg, three cooks (2026-09-03)

`run_pool_real.py --set keg`: Dale Ward's Big Steel Keg — overnight
225 °F brisket '13 (11.7 h), hot-and-fast 325 °F pork butt '14 (4.8 h),
overnight 225 °F butt '15 (11.7 h, RotoDamper era) — found, digitized
(`hm_dark` palette; lid events read from the HM UI's own lid-detect bands
where present) and provenance'd by `tools/loghunt/`. Roles flipped vs the
cross-smoker set: vessel identity (C_th, UA, dH_eff) fully shared,
actuator map (r_max, u_leak) per-cook (the airflow hardware changed
across the years), meat + ignition per-cook. Fits: rms pit 2.6/4.2/2.6 K,
core 1.1/1.4/1.0 K. Headlines: the two overnight-225 cooks TWO YEARS
apart, on different actuator hardware, agree on the vessel (UA within
11 %, dH_eff within 5 %, C_th within 27 %) — isolating the '14 outlier as
operating-point/short-horizon, not vessel drift; k_e holds within 3.5 %
and a_myosin within 8 % across all three cooks (brisket AND butts — meat
chemistry transfers); u_leak spans 0.29-0.88 per-cook (hardware + vent
settings, incl. a documented mid-cook vent change in '15). Caveat: the
pooled shared block is ridge-coupled (C_th/UA trade off; the joint
precision-weighted mean with K−1 prior-dedup can land outside the
per-coordinate MAP hull) — the cross-cook agreement above is the
meaningful validation, and it was checked against a falsy-default pooling
bug fixed en route (explicit-empty role lists no longer fall back to
defaults; regression-tested). (`figures/pool_keg.png`,
`data/calibration/pool_keg.json`.)

## Same-model pool: two WSM 18.5 units (2026-09-04)

`run_pool_real.py --set wsm`: mache_2013 x wbegg_2014 (a stock WSM 18.5
8 lb pork butt from a 32 F Australian winter start, fully documented
timeline, found and digitized by `tools/loghunt/` — the 225 F setpoint
line sits ON the 225 gridline, a 0.5 px anchor cross-check). A NEW
pooling tier: two different physical units of one smoker model, vessel
identity PARTIALLY pooled (tau2), everything else per-cook. Fits: mache
reproduces its historical MAP exactly (F=133.2); wbegg rms pit 5.25 K
(the pit sits ~5 C low through the cold morning — the scalar 278 K
day-mean ambient and the terra-cotta-dish-for-water-pan swap are the
stated confounds), core 1.42 K over 20.8 h. Results: **dH_eff agrees
within 8 % across units** (fuel energy density is a design+charcoal
property — it pools hard; hypermean 1.53e7); UA spreads 52 %
(16.7 vs 11.0 — unit + configuration + ambient-estimate, which is what
partial pooling is for; shrunk 18.0/10.2, hypermean 13.5); C_th stays on
its known sloppy ridge (14x apart — closed-loop logs cannot pin it; the
characterization burn can). Meat side: **k_e now holds within +/-8 %
across FIVE cooks** (mache, three kegs, wbegg: 0.51-0.57e-3) spanning
brisket and pork butt, two smoker models, and a 60 F ambient range.
(`figures/pool_wsm.png`, `data/calibration/pool_wsm.json`.)

## Wrap constants from a documented butcher-paper wrap (2026-09-04)

`scripts/run_wrap_fit.py` on `erickson_2013`, found by `tools/loghunt/`
(12 lb brisket, Char-Griller Kamado, ~15 h; BUTCHER PAPER wrap documented
in prose AND in the chart's own off/on lid-band pair at 8.75/9.15 h; new
watermark-tolerant `hm_flot_jpeg` palette). The observed core DROPS
~7 °C across the wrap — the brisket spent ~24 min off the pit and the
probe was re-seated — so the forward model is three chained segments (pit
→ room interval → pit-wrapped; full state restarts through T/W/C/F fields
+ bark/smoke/ring) with the probe's material depth as a parameter on each
side of the wrap. Measured pit trace as the boundary forcing; 8-parameter
FD-LM with z-space priors. Primary fit (L = 0.045, point-end probe; every
nuisance lands physical — h_c 11.8, k_m/k_e at priors): **η_evap = 0.17
[0.13, 0.23]** — butcher paper passes ~83 % of the evaporation (paper
breathes; the mechanism behind "paper preserves bark") — and **ε_wrap =
0.31 [0.26, 0.37]** (prior-bound); rms 3.6 K; counterfactuals at the fit:
foil would have finished hottest, unwrapped middle, paper slowest under
this ramped pit. Identifiability caveat (kept in the JSON pair): an
L = 0.033 variant fits identically (rms 3.6) but trades the
radiative/convective split (h_c → 1.3, η_evap 0.80 / ε_wrap 0.16) — ONE
log pins the total post-wrap input reduction, not the full η/ε split; a
FOIL-wrapped log (contrast) or a known-thickness cook breaks the tie.
(`data/calibration/wrap_fit_erickson{,_L33}.json`,
`figures/wrap_fit_erickson{,_L33}.png`.)

## Salt physics: the structure (2026-09-04)

The salting item's model half is in (`src/meats/common.jl`): a fifth PDE
field s(x,t) — dry-basis salt — with the dry brine as the ANALYTIC
semi-infinite profile `s0 = S_surf·erfc(depth/(2√(D_salt·t_salt)))` (no
pre-cook solve; a smooth dependency-free erfc), zero-flux surface during
the cook (salt does not evaporate), and two couplings: water-holding
capacity (`W_eq → W_eq·(1 + β_whc·s)` in the purge sink — salted muscle
purges less) and Raoult-scale water-activity depression of surface
evaporation (`a_w` factor `1/(1 + k_aw·s)`). `t_salt = 0` is EXACTLY
unsalted (both couplings are the identity — tested against absurd
coupling constants), salt is conserved to solver tolerance, a 96 h brine
raises retained water and lowers evaporation directionally, and the
state-restart contract covers the salt field (`Ssalt0f`). D_salt is a
ham-curing literature constant by design; S_surf/β_whc/k_aw are
placeholder scales awaiting the scale + designed experiment. The
monolith carries the field (206 states; FD gradient gates unchanged);
`salt_field` is internal to the Tesseract like geom_m/k_skin until a
consumer needs it, and t_salt lives in MeatConfig until the planner
promotes it to a differentiable decision variable.

## Generalization: geometry, modes, and two external validations

Phase A (2026-09-03): the GEOMETRY axis is in and validated —
`MeatConfig.geom_m` puts the slab/cylinder/sphere curvature into the
material-coordinate operator (exact (1+m) center limit; bit-identical
slab at m = 0), and sous vide needed no new BC code at all (the bath is
the rest environment from t = 0 with η_rest = 1). The doneness-kinetics
axis is configuration, not code: collagen constants (Ea, k_ref, T_ref)
are `MeatConfig` fields, and pasteurization is the existing lethality
integral. The remaining modes followed: skin barrier
(`MeatConfig.k_skin`, a series conductance — heat passes, evaporation
throttled; the poultry mechanism, constants uncalibrated), saturated
steam (pure configuration: `pv_base_knots = p_sat(T_env)` — the
two-sided flux condenses and the stall never forms; the pibil mechanism),
and an electric source (`PitConfig.p_heater`: heat = p_heater·u, nothing
burns — no fuel loss, no combustion water, no smoke, bark goes
Maillard-only). Each has a physics-consistency test, and exposing
geom_m/k_skin/p_heater through the Tesseract schema is one optional field
each (like t_f) when a consumer needs them.

**Sous vide vs Baldwin (zero fitted parameters):**
`scripts/run_sous_vide.py` reproduces the exact conduction series at
Baldwin's α/h to ≤ 8 % on all 21 table cases (his published
cylinder/sphere columns are safety-padded ~+15 % and ~+35 % above that
exact solution; the slab column, us, and theory agree to ~1 %;
`figures/sous_vide_baldwin.png`).

**Oven roast vs Papasidero et al. 2015, CET 43:151 (zero fitted
parameters; the first validation with measured WEIGHT LOSS):**
`scripts/run_oven_roast.py` — a 1.4 kg beef cylinder (18 x 10 cm),
domestic oven 22 → 180 °C cold start, stopped at core 65 °C, with core /
5-mm-depth / intermediate thermocouples and dry-basis water content
2.38 → 1.75 kg/kg (digitized by `scripts/digitize_cet2015.py`,
overlay-verified, into `data/validation/cet2015_roast.json`). With
repo-default θ, the paper's own h = 10 W/m²K and ε = 0.91, and
geom_m = 1.2 as the eigenvalue-matched finite-cylinder equivalent:
time-to-65 °C within x1.04-1.16 of measurement across the
geometry/preheat sensitivity band, core rms 2.6-6.9 °C, and predicted
water loss x1.06-1.27 of measured — where the paper's own fitted COMSOL
model overshot the loss ~2x. The purge + surface-evaporation moisture
physics transfers from smoker calibration to an oven regime it never saw
(`figures/oven_roast_cet2015.png`,
`data/validation/oven_roast_cet2015.json`).

## Smoke quality: the structure (2026-09-04)

The "thin blue smoke" variable is in (`burn_cleanliness`,
`PitConfig.r_smolder/δ_smolder`; monolith outputs `smoke_clean` and
`bitterness`): combustion cleanliness is a logistic in the
draft-to-lit-bed ratio — a choked damper over a large lit bed smolders,
an open draft burns clean — and the DIRTY share of the exact deposition
integrand that drives the bark smoke state integrates to a bitterness
output (a quadrature over saved trajectories; no new ODE states, so the
solve cost is unchanged). Tested: a choked burn is dirtier per unit
deposited than an open one at the same fire, and the electric source is
exactly bitterness-free (nothing burns). Constants are placeholder
scale — bitterness calibration needs a smolder-forced designed cook or
a tasting panel (roadmap item 7 keeps that).

## Composability presets + the quasi-3D fidelity level (2026-09-04)

The notable cases are now one-call compositions: `pork_butt()` (the
near-spherical cut — geom_m = 2, R ≈ 5 cm, fattier; same collagen
kinetics; k_e backed by the five-cook pooling record), `pellet()` (a
controlled-feed source: `PitConfig.m_auger` replaces the Minion bed law
with `m_auger·u·taper` — the damper input becomes the auger duty,
combustion water and smoke stay on, the bed-spread state is inert), and
`earth_pit()` (the pibil/imu as PURE COMPOSITION: the rock reservoir is
the pit's precharged lumped mass — C_th = m_rock·c_rock, Tpit0 = 700 K,
no fuel, r_max = 0 → no smoke — discharging through UA into the meat
under the existing `pibil_pv` saturated-steam environment; "control"
becomes rock mass and seal time; a sealed source exposed a
divide-by-zero in the humidity air-mass flow, now floored). The
fidelity ladder gains its first rung above 1-D: `newman_center` — the
Newman product rule combining per-axis 1-D solves into a quasi-3D
center trajectory (exact for linear conduction under constant ambient;
the documented approximation here; a finite brick costs ~3 solves).
Tested: feed law ignores the bed, the earth pit cooks with zero burn,
Newman bounds and its infinite-axis reduction. Julia 252/252, app
67/67. All constants placeholder scale; the combi-oven case was already
reachable (electric source + humidity forcing).

## The cook pilot, validated without hardware (2026-09-04)

`brisket_app/pit_io.py` (one pit contract: `HeaterMeterClient` speaks the
LinkMeter v14 REST API — status GET, setpoint POST + apikey — and
`SimPit` is the calibrated plant behind the same contract with an
HM-style PI turning setpoints into damper duty), `brisket_app/pilot.py`
(the `run_cook` control loop lifted out of simulation: re-plans schedule
+ wrap + pull each interval, pushes the PLANNED PIT TRAJECTORY as
setpoints, advances the belief state under MEASURED fan duty, snaps the
pit state to the sensor, runs the ambient KF on the innovation with an
optional hourly FORECAST as feedforward, and journals everything as
JSONL), `scripts/run_pilot.py` (the daemon; wrap/pull are advisories on
hardware, auto-executed on sim backends), `scripts/mock_heatermeter.py`
(the sim plant behind the real v14 endpoints) and
`scripts/plot_pilot.py`. Validation: the same 11-h fullplan cook flown
twice against an unannounced −15 K cold front at 2 h — in-process, and
over HTTP through the REAL client path against the mock — lands nearly
identically (wrap re-timed 7.06 → 6.83 h, pull 8.39 → 10.07/10.10 h; the
KF recovers ~60 % of the ambient drop and the setpoint plan compensates
with a ~300 °F push; `figures/pilot_http.png`,
`data/control/pilot_{sim,http}*.jsonl/json`).

## The `brisket` CLI (2026-09-04)

`brisket_app/cli.py`: one `brisket` entry point (`uv run brisket ...`
from app/) — `logs`, `digitize`, `calibrate` (any registry log OR a user
CSV via the cook-log column convention; GN-MAP + Laplace; writes a
`--params`-compatible `fit_<name>.json`), and verbatim forwards to the
canonical scripts for `plan`, `pilot`, `pool`, `mock-pit`. The your-log →
your-plan → your-pilot flow is three commands (README "Using it on your
own cook").

## The weights hunt and its lab resolution

No smoker log with BEFORE/AFTER weights was found in the graphs thread or
the keg threads (pre-cook weights are common, yield pairs are not). The
lab route resolved the moisture-physics validation instead (the CET oven
roast above). Blonder's stall experiments (genuineideas.com; core-temp
curves at known smoker temps, coarse loss figures) remain a soft
cross-check option.
