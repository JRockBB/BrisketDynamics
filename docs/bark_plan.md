# Differentiable Bark Model — Research Summary & Feature Plan

Status: steps 1-5 IMPLEMENTED on branch `bark-model` (2026-08-19):
Julia bark states + pit mdot + smoke forcing + coupled wiring + forward
validation (`figures/bark_model.png`; Julia 68/68, app 21/21, composed
damper->mdot->smoke->bark gradient FD-verified). Step 6 done (bark-aware wrap
objective, L* target replacing the evaporation-integral proxy; overdone
band added after the fixed-horizon run parked at C=0.9999). Open: step 7
(photo calibration, blocked on data) and composing DURATION x wrap x
schedule (timeplan outer scan around the joint optimization) so surplus
hours shorten the cook instead of overshooting doneness.
Research pass 2026-08-19; sources at the end.

## 1. What the literature actually supports

**Browning kinetics (transferable from bakery science).** The
best-established surface-color models are the bread-crust browning kinetics
of Purlis & Salvadori: **first-order kinetics in lightness L***, with an
Arrhenius rate in *surface temperature* (reported E_a ≈ 64 kJ/mol,
k₀ ≈ 4.2×10⁴ s⁻¹ for bread crust) and a water-activity-dependent
correction; they state onset thresholds of T ≳ 120 °C and a_w ≲ 0.6 for
bread. Independently, the classic Maillard literature establishes a
**bell-shaped rate dependence on water activity peaking at a_w ≈ 0.5–0.8**
(mobility-limited below, dilution-limited above), and E_a values of
~100–150 kJ/mol for sugar–amine model systems.

**Two facts force adaptations for BBQ:**
1. A smoker surface never reaches 120 °C at a 107–135 °C pit — yet bark
   demonstrably forms. Resolution: the bread "onset" is not a physical
   switch but where the Arrhenius tail becomes visible in a ~30-min bake;
   a 12–18 h cook integrates the slow tail. We keep pure smooth Arrhenius
   (no threshold) — which is also exactly what differentiability wants.
2. Bark is **not only Maillard**. The mechanistic picture (consistent
   across amazingribs/serious sources, filtered from folklore): a
   protein **pellicle** forms on the moist surface; **smoke phenolics
   (guaiacol, syringol) deposit onto the surface while it is wet** —
   deposition effectively stops once the surface dries or is wrapped;
   Maillard + polymerization then darken and harden the layer; rendered
   fat modulates texture. Practical corollaries the model should
   reproduce: hot-and-fast dries the surface early → less smoke color;
   wrapping freezes bark development; the "bark window" is the first
   4–6 h while the surface is moist.

**Color measurement (calibration path).** Computer-vision color
measurement is standard in food science: photos with a **reference color
card** in-frame, RGB→CIELAB conversion, quantified as L*a*b* or a
browning index (BI). Colorimeter-vs-camera comparisons validate the
approach. No prior work was found calibrating a browning **model**
against photos by gradient-based inversion — the "to our knowledge
unpublished" claim survives this research pass.

## 2. Model design

Two additional surface ODE states **inside the meat solve (Julia)** —
physics lives in the physics component, the states ride the existing
continuous adjoint natively, and the v2 feedback path (crust resistance
altering moisture/heat transport, plausibly part of stall break) becomes a
one-line RHS change instead of a rewrite. The cross-component coupling
uses the established forcing-knots pattern: the pit exports its exact burn
rate (`mdot`, new output), the orchestrator passes it to the meat as
`smoke_knots` (new differentiable forcing input, same plumbing as
`tpit_knots`). v1 has NO feedback into T/W/C — bark states are pure
integrators, so existing physics is unchanged (solver step sizes shift
microscopically; test baselines recompute).

**Water activity** from surface moisture via a smooth sorption isotherm
(Oswin form, meat parameters from literature):
    a_w(W_s) = 1 / (1 + (k_ox / W_s)^n_ox)            # smooth in W_s

**Maillard/browning state** B ∈ [0, 1) (normalized color development):
    dB/dt = k_B(T_s, a_w) · (1 − B)
    k_B = k₀ · exp(−E_a/R · (1/T_s − 1/T_ref)) · g(a_w)
    g(a_w) = exp(−((a_w − a_pk)² / (2 σ_aw²)))        # bell, a_pk ≈ 0.65
Integrated as ODE state 3nx+1 of the meat system (4nx+1 since the moisture build added the fat block F) (linear in B given the
surface trajectory; FBDF handles it incidentally).

**Smoke deposition state** S (accumulated phenolics):
    dS/dt = k_dep · wet(a_w) · φ_smoke(t) · (1 − s_wrap(t))
    wet(a_w) = logistic((a_w − a_dry)/δ_dry)           # needs a moist surface
    φ_smoke  = lininterp(t, smoke_knots)               # pit burn rate forcing
State 3nx+2. The smoke forcing crosses the component boundary exactly like
the pit-temperature forcing does — a genuinely two-Tesseract quantity,
wired in the orchestrator, differentiated through both VJPs.

**Observables:**
    L*(t) = L₀ − ΔL_B·B − ΔL_S·S        (a*, b* analogous if wanted)
    BI    = standard browning index from L*a*b*
    bark texture proxy = surface-layer dryness (∫ or final W in the top
    cells — already available from W_field)

**Wrap interaction for free:** the existing `t_wrap` blend s(t) enters
dS/dt directly (foil blocks smoke contact), and wrapping raises surface
a_w off the Maillard bell — the model reproduces "wrapping freezes bark"
mechanistically.

**Parameters** (all literature-anchored, none calibrated until photos
exist): k₀, E_a (init from bread: 64 kJ/mol), a_pk, σ_aw, k_ox, n_ox,
k_dep, a_dry, δ_dry, ΔL_B, ΔL_S, L₀. Sensitivity study replaces
calibration in phase 1.

## 3. Implementation plan (branch `bark-model`)

1. **Julia core (`src/meat.jl`, `src/pit.jl`)**: bark constants + isotherm
   + g(a_w) helpers in meat; states 3nx+1 (B), 3nx+2 (S) in the meat RHS
   with `smoke_knots` as a second forcing block in `p`
   (`p = vcat(θ, Tpit_knots, smoke_knots, t_wrap)` — t_wrap stays the
   tail); pit gains an exact `mdot` output (it already computes ṁ).
   Outputs: `bark_b`, `bark_s` trajectories + `l_star` at tsave.
2. **Tesseracts**: meat — `smoke_knots` Differentiable input (default
   zeros = no smoke), `bark_b`/`bark_s`/`l_star` Differentiable outputs,
   VJP returns `dsmoke_knots`; pit — `mdot` output with its ForwardDiff
   VJP row.
3. **Orchestrator (`coupled.py`)**: wire `pit["mdot"]` → meat
   `smoke_knots` through the Picard loop (barrier includes it); expose
   bark outputs in the result dict.
4. **Forward validation (no data needed):** Julia physics testset +
   demo script reproducing the four qualitative laws — (a) low-slow
   beats hot-fast on smoke color; (b) wrap freezes bark; (c) bark window
   = first 4–6 h (moist surface); (d) monotone darkening, saturating
   late. Figure `figures/bark_model.png`, 3 cook styles.
5. **Verification (small)**: Julia FD directions for one `smoke_knots`
   knot and bark-output cotangents; existing suites green (baselines
   recompute where solver stepping shifted).
6. **Control integration:** replace the wrap objective's crude bark
   proxy (∫ j_evap dt) with model L* shortfall; re-run `run_wraptime.py`.
7. **Calibration (blocked on data, design now):** photo protocol —
   phone photo + color card at each hourly lid-open plus final;
   `scripts/bark_colorimetry.py` extracting card-corrected L*a*b* from
   photos; hierarchical fit once ≥2 instrumented cooks exist.

## 4. Identifiability (designed before any fitting)

- **A single end-of-cook photo** pins ONE number (final L*, maybe BI) —
  against ~5 free scale parameters it identifies only the lumped product;
  everything else rides priors. Honest but weak.
- **A time-lapse (hourly lid-open photos)** pins the *shape*: onset
  timing → k₀·g path; the early-vs-late split of darkening → ΔL_S vs
  ΔL_B (smoke deposits early on the wet surface; Maillard accumulates
  late and hot). This is the minimum viable calibration target.
- **Two cooks at different pit temps** (or one wrapped/one not) separate
  E_a from k₀ and kill the biggest degeneracy; the wrap comparison also
  isolates ΔL_S (foil blocks deposition).
- Salt/rub note: rub salt depresses surface a_w (ties to the
  salt-physics roadmap item); sugar in rub adds a caramelization channel
  deliberately out of scope for v1.

## 5. Risks

- **a_w model for a rubbed meat surface** is the weakest physics link
  (isotherms are for plain meat; salt/sugar rubs shift them). Mitigation:
  treat a_pk/σ_aw as effective parameters; sensitivity study.
- **Smoke flux proxy** (∝ burn rate) ignores combustion quality
  (smoldering vs clean burn — the folklore "thin blue smoke" variable).
  Named limitation; a draft/damper dependence could be added later.
- **Color from photos under smoker lighting** — the color card handles
  illumination; grease on lens and steam are the practical enemies
  (protocol: wipe, wait 10 s after lid open).

## Sources

- Purlis & Salvadori, "Modelling the browning of bread during baking"
  (Food Research Int., 2009); Purlis, "Browning development in bakery
  products — a review" (J. Food Eng., 2010): first-order L* kinetics,
  Arrhenius + a_w dependence, E_a ≈ 64 kJ/mol, onset T>120 °C / a_w<0.6.
  https://www.sciencedirect.com/science/article/abs/pii/S0963996909000817
- Maillard a_w bell (peak ~0.5–0.8): Eichner & Karel (J. Agric. Food
  Chem., 1972); Labuza lineage.
  https://pubs.acs.org/doi/10.1021/jf60180a025
- Buera et al. (J. Food Sci., 1987): browning kinetics of sugar–glycine
  systems, activation energies 25.7–36.6 kcal/mol.
  https://ift.onlinelibrary.wiley.com/doi/abs/10.1111/j.1365-2621.1987.tb14276.x
- Bark mechanism (pellicle, smoke phenolic deposition on wet surface,
  fat rendering; folklore filtered): Meathead, "What is Bark",
  amazingribs.com; Tasting Table bark-science explainer.
  https://amazingribs.com/more-technique-and-science/more-cooking-science/what-is-bark/
- Camera colorimetry with reference cards; L*a*b* from RGB; CVS vs
  colorimeter equivalence: León et al. (Food Res. Int., 2006);
  J. Food Meas. Charact. (2016) comparison.
  https://www.sciencedirect.com/science/article/abs/pii/S0963996906000470
