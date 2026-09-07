# Independent review (2026-09-07)

Physics, calibration, control, 2-D line, tests/CI, and docs as of the
working tree on 2026-09-07. Theory-gap claims are cited. This is a
snapshot review, not a living plan: remaining work lives in
`docs/roadmap.md`; equations in `docs/model.md`; completed results in
`docs/results.md`. The 30 Aug equation audit
(`docs/equation_correctness_review.md`) is historical — several of its
physics findings were later fixed in code.

**Bottom line.** The engineering is unusually strong for a research
codebase: the stall is a real evaporative boundary condition, derivatives
are actually used, residues are named rather than hidden, and the CET
2015 / Baldwin checks are the right kind of external test. The binding
problems are not missing features. They are (1) a water budget that is
physically wrong by about a factor of two, (2) quality and safety proxies
that are not the quantities they are named after, (3) a 1-D planner /
2-D shadow mismatch that currently overstates 2-D “verification,” and
(4) documentation and CI that have not kept up with the state vector.
When a pit is available, the highest-leverage work is a scale plus an
open-loop burn, not more uncalibrated physics. Until then (travel: no
cook, no instruments), the next work is code and docs that do not need
new data — an honest 2-D shadow, green CI, a safety term that matches
FSIS brisket practice, and an equation sheet that matches the state
vector.

---

## What is already in good shape

The stall mechanism is the right one. Evaporative cooling, not collagen
or fat rendering, is the established explanation, including Blonder’s
sponge-versus-fat tests and the Modernist Cuisine wet-bulb / vacuum-seal
contrast ([AmazingRibs / Blonder](https://amazingribs.com/more-technique-and-science/more-cooking-science/understanding-and-beating-barbecue-stall/);
[Ars Technica summary](https://arstechnica.com/science/2018/09/let-science-be-your-guide-for-the-perfect-labor-day-bbq/);
Myhrvold et al., *Modernist Cuisine*). The code’s two-sided,
isotherm-driven flux \(j \propto a_w p_{\mathrm{sat}} - p_v\) is the
food-engineering form of that story, and wrapping suppressing evaporation
is the Texas-crutch mechanism, not a hack.

Differentiation is doing real work. Gauss–Newton MAP in ~100 s, Fisher
ridges that a finite-difference Hessian cannot resolve, NUTS with a
Fisher-seeded metric, and Optax plans through the same Tesseract are a
coherent stack. The measured last-bit FBDF jitter (~4×10⁻⁵ relative,
amplified near the MAP) is treated as a fact, not papered over.

The project’s own `docs/roadmap.md` “Residues” list is mostly accurate.
The earlier `docs/equation_correctness_review.md` (30 Aug) caught real
issues (isotherm not in \(j\), frozen Choi–Okos, cook-loss identity
test). Several of those were later fixed in code. Treat that review as
historical, not current.

External checks that actually constrain physics: Baldwin tables at zero
fitted parameters; Papasidero et al. 2015 oven roast with measured
weight loss (the paper’s own COMSOL overshot loss ~2×; this model is
closer). Those are the strongest transfer claims in the repo.

---

## Correctness and implementation

### Critical

**1. Water budget: the calibrated cook retains ~4% of its water.**
`docs/roadmap.md` already states this: holding a 10 h stall against
\(h_c \approx 12\,\mathrm{W\,m^{-2}K^{-1}}\) plus enclosure radiation
implies ~11 kg/m² evaporated per face against ~14.5 kg/m² stored in a
3.9 cm slab, while a real flat loses ~40%. Either heat input is ~2× too
high or the slab is thermally thin for the water it carries. This is not
a documentation footnote. It is a failed conservation/identifiability
check that contaminates juiciness, wrap value, spritz value, and the
11 h “don’t spritz / don’t add fuel” plan. The planner is optimizing a
meat that dries to jerky; a model that retained 50–60% would likely wrap
earlier, cook longer, and maybe spritz. Until a before/after mass is in
the likelihood, **do not treat planned juiciness or wrap time as
physical**.

**2. 2-D shadow does not pull or rest.**
`shadow_report` and `tesseracts/slab2d` take wrap, salt, and spritz, not
\(t_{\mathrm{pull}}\). The published 11 h plan pulls at ~8.4 h and rests
~2.6 h. The shadow cooks the full horizon in the pit. That overcooks
collagen, over-dries, and inflates lethality. The reported
\(C_{\min} \approx 0.99997\) and retention 0.189 are consistent with
“2.6 extra hours on the fire,” not with a faithful 2-D replay of the
1-D serve state. Default geometry is also a different object: radius
0.10 m by hand, height \(= 2L\) (~8 cm), 16×14 nodes — a squat cylinder,
not a packer point-and-flat, and coarse for a “true cold spot.”

**3. CI does not run the Python suite.**
`.github/workflows/CI.yml` runs `uv run pytest tests/test_local_apply.py`.
That file is not in the tree. README claims `uv run pytest tests/ -q`.
Julia tests run; composed-gradient, MPC, planner-decision, and slab2d
Tesseract tests do not run on CI. A merge can break the JAX/Tesseract
boundary with a green badge.

### High

**4. State-count and equation-sheet drift.**
The monolith is no longer 166 ODEs. Code builds meat as \(5 n_x + 4\)
(T, W, C, F, salt, B, S, R, wrap-latch) plus six pit states (including
pan and firebox): **210** with defaults `nx=40`. README,
`docs/writeup.md`, `docs/model.md`, `docs/monolith.md`, and
`docs/submission.md` still say 166 / 163. `src/meats/slab2d.jl` header
still describes a v1 with no shrink/wrap/salt while the file implements
all of those. `app/src/brisket_app/timeplan.py` still calls
differentiable duration “the next step” after `run_duration_opt.py`
shipped. Anyone reproducing from the equation sheet will implement the
wrong system.

**5. Food-safety terms are the wrong constraints for brisket.**
The lethality integral at 60 °C / \(z = 6\,\mathrm{K}\) /
\(F_{\mathrm{req}} = 720\,\mathrm{s}\) is the FSIS Appendix A 6.5-log
*Salmonella* line at 140 °F / 12 min
([FSIS time–temperature table](https://www.michiganfoodsafety.com/wp-content/uploads/2024/03/USDA-Appendix-A-Time-and-Temperature-Combinations.pdf)).
That part is labelled correctly now. The **4 h danger-zone cap** is not.
USDA’s 40–140 °F / 2–4 h rule is a **holding** rule for cooked food left
out
([FSIS Danger Zone](https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/danger-zone-40f-140f)),
not a roasting come-up limit. The relevant commercial analogue is
heating come-up time (CUT) 50–130 °F \(\le\) 6 h for *S. aureus*
enterotoxin, and FSIS explicitly lists **brisket as Scientific Gap 6**:
long CUT is expected and the 6 h option need not be a critical operating
parameter until more research exists
([Federal Register 2021-26993](https://www.federalregister.gov/documents/2021/12/14/2021-26993/fsis-guidelines-for-small-and-very-small-meat-and-poultry-establishments-regarding-cooking-and-stabilization);
[FSIS webinar](https://www.fsis.usda.gov/sites/default/files/media_file/2022-05/2021_Appendix_A_and_B_Webinar_May%2023_and_24.pdf)).
A 4 h / 4.4–60 °C penalty with weight 500 will fight every real
low-and-slow cook. Fullplan “rejects raw residuals”; MPC only penalizes
— so published plans can look “safe” because they are hot-and-fast
enough to clear a mis-specified dwell, not because they match FSIS
brisket practice.

**6. MPC and the pilot are not the same problem.**
`mpc.py` is perfect-state-feedback MPC: plant and controller share the
model (plus a known-class ambient disturbance), and the controller reads
the full T/W/C/F/B/S field. A HeaterMeter gives pit temperature, one or
two probes, and fan duty. The pilot snaps pit state to the sensor and
runs an ambient KF; meat fields stay open-loop. The 15 K cold-front demo
is a model-in-the-loop robustness check, not evidence that the same
replan would work on hardware. The repo says this; the writeup still
reads like a deployed controller.

**7. Water-pan humidity is not a closed balance.**
`_pan_algebra` evaporates against the **pre-pan** \(p_v\). The pan does
not see its own contribution. Roadmap names this; it is still a
sign/feedback error for the pan-in/pan-out experiment that is supposed
to explain the WSM UA spread.

### Medium

**8. Softplus purge leak.** At \(W = W_{\mathrm{eq}}\) the sink is still
\(\sim k_e \delta_e \ln 2\). Calibrated \(k_e\) absorbs it; long holds
dry below the stated water-holding equilibrium. Fine for 8–12 h cooks;
bad for overnight holds and rest.

**9. No interior enthalpies.** Purge leaves as liquid (no \(\lambda\)),
fat melt and collagen conversion carry no latent heat, radiation uses
gas temperature as a black enclosure. Relative to the stall sink these
are small, but they bias the post-stall climb and the water–energy
coupling that the budget already fails.

**10. Bone is thermal-only.** Moisture and collagen in the bone region
still evolve as meat. “Cold near the bone” is then impossible by
construction except via geometry; the folklore is about joints and
thermal mass, which the 1-D blend does not have.

**11. 2-D humidity is one step lagged; 2-D rest is missing.** Coupled 2-D
is first-order operator-split. Combined with (2), shadow scores are not
an oracle for the 1-D plan’s serve state.

**12. Robust control sample size.** \(E[J]\) over 8 posterior draws,
evaluated on 30 held-out. With 15-D ridges, 8-sample SAA can overfit
those 8. The headline (nominal \(J = 5.3\pm 9.6\) vs hedge
\(1.81\pm 0.60\)) is directionally right — wrap earlier, pull later —
but the “zero deadline margin” claim is not a 95% guarantee, which the
roadmap already warns.

**13. NUTS ESS on the leak/burn ridge is ~170.** Split-\(\hat r \le 1.010\)
and 0 divergences are good; widths on \(r_{\max}\), \(u_{\mathrm{leak}}\),
\(\sigma_{\mathrm{pit}}\) are still noisy. Laplace/Fisher matching on
13/15 sites is the more honest UQ statement than “the posterior.”

**14. `opt_salt` has no counterweight.** Enabling it pins brine at the
48 h bound (the spritz lesson). Do not expose it in the CLI until C5
exists.

### Low / maintainability

- Choi–Okos parallel mixing for \(k\) over-conducts relative to
  Maxwell–Eucken / series-parallel meat models (van der Sman 2008-type
  arguments); \(k_{\mathrm{mult}}\) can hide it.
- Magnus \(p_{\mathrm{sat}}\) is ~2.6% high at 100 °C; stall band is fine.
- Oswin \(k_{ox}=0.10\), \(n_{ox}=1.5\) is a bread/drying-style isotherm,
  not a fitted meat desorption isotherm (see theory).
- `round()` in spritz timing is piecewise smooth, not a bug, but dense
  spritz policies will have derivative kinks.
- Dual sentinel lesson (`Dual(0,1) <= 0` is false) is documented and
  easy to regress; keep those tests.
- Untracked figures/GIFs and `scratchpad/` will keep the tree noisy.

---

## Theory gaps (cited)

These are not “add more states.” They are places where the current
closures cannot represent the literature mechanism even after
calibration.

### T1. Moisture transport is Fickian; cooking meat is poroelastic

Interior water moves as \(\partial_t W = \nabla\cdot(D_{\mathrm{eff}}\nabla W)\)
minus a local purge sink. The cooking-meat literature treats muscle as a
swelling polymer gel: moisture flux is pressure-driven (Darcy), with
swelling pressure from Flory–Rehner / water-holding capacity, not a
moisture gradient
([van der Sman, *Meat Sci.* 2007](https://doi.org/10.1016/j.meatsci.2007.02.014);
[van der Sman, *Meat Sci.* 2013](https://doi.org/10.1016/j.meatsci.2013.03.027);
[Feyissa et al. 2013](https://doi.org/10.1111/jfpe.70370)
and follow-ons). Consequences:

- \(D_{\mathrm{eff}}\) is an effective lump of permeability, viscosity,
  and WHC slope. It will not transfer across temperature, salt, or pH.
- Drip is a local ODE, not capillary channel flow. van der Sman (2013)
  needed extra drip-channel dynamics above boiling or at long times —
  exactly BBQ timescales.
- The 4% retained-water failure is what this gap looks like in a fitted
  1-D slab: heat input and \(D_{\mathrm{eff}}\) conspire to empty the
  domain.

The CET 2015 oven-roast agreement on *total* loss does not close this.
That cook is short, hotter, and cylindrical; a 19 h stall is a different
regime.

### T2. \(W_{\mathrm{eq}}(T)\) is not water-holding capacity

Two logistic steps at myosin ~58 °C and actin ~72 °C match DSC
denaturation peaks, but WHC also depends on pH, salt, and sarcomere
length. Flory–Rehner is the structure that lets salt and pH enter the
same potential
([van der Sman 2013](https://doi.org/10.1016/j.meatsci.2013.03.027)).
The salt coupling \(W_{\mathrm{eq}}\to W_{\mathrm{eq}}(1+\beta_{\mathrm{whc}} s)\)
is a local linearization of that idea; \(\beta_{\mathrm{whc}}\) is a
placeholder, so C5 cannot be identified from temperature logs (the
roadmap is right).

### T3. Tenderness is not one Arrhenius collagen state

Defaults: \(E_a = 173\,\mathrm{kJ\,mol^{-1}}\),
\(k_{\mathrm{ref}} = 10^{-6}\,\mathrm{s^{-1}}\) at 60 °C \(\Rightarrow\)
\(\tau \approx 8\,\mathrm{h}\) at 80 °C, \(\tau \approx 0.7\,\mathrm{h}\)
at 95 °C, and \(\tau \approx 700\,\mathrm{h}\) at 55 °C. DSC helix–coil
denaturation of intramuscular collagen is typically **300–340 kJ/mol**
([Ishiwatari et al., JSFE](https://doi.org/10.11301/jsfe.12.19),
sarcoplasmic+collagen \(3.26\times 10^2\,\mathrm{kJ\,mol^{-1}}\);
[Vaskoska et al., *Int. J. Biol. Macromol.* 2018](https://doi.org/10.1016/j.ijbiomac.2018.02.132),
~322–338 kJ/mol). Cooking tenderness is **two mechanisms**:

1. **Below ~60 °C:** collagenase remains active; 24 h at 55–60 °C
   tenderizes tough beef (Tornberg 2005; Bouton & Harris 1981;
   [Baldwin sous-vide guide](https://douglasbaldwin.com/sous-vide.html)).
2. **Near 80 °C:** solubilization/gelatinization on a 12–24 h scale
   (Davey et al. 1976; Tornberg 2005).

A single low-\(E_a\) Arrhenius referenced at 60 °C interpolates BBQ
pitmaster pulls (the 19 h / 89.5 °C / \(C=0.874\) anchor) and **misses
sous-vide collagenase entirely**. \(C=0.90\) as “probe tender” is one
judgment on one cook, not a texture calibration. KCBS weights
(0.14 / 0.57 / 0.29 from 0.5600 / 2.2972 / 1.1428) are correctly
normalized
([KCBS 2025/2026 procedures](https://meatandtaterbbq.com/wp-content/uploads/2025/05/KCBS_JUDGING_PROCEDURES_2025.pdf));
they do not turn \(C\) and \(L^*\) into judge scores.

### T4. Stall break is sorption + interior supply; isotherm is uncalibrated and isothermal

Oswin is used for meat drying and is sometimes competitive with GAB
([Mewa et al. 2019](https://doi.org/10.37017/jeae.v5i1.45);
Motarjemi 1988 via
[Comaposada et al. 2001](https://www.sciencedirect.com/science/article/abs/pii/S026087740100053X)),
but GAB / Peleg / DLP usually win, and **isotherms shift with
temperature** (Clausius–Clapeyron). Constants \(k_{ox}=0.10\),
\(n_{ox}=1.5\) are bread-like; at raw \(W_0=2.33\), \(a_w\approx 1\), so
the stall is wet-bulb limited as it should be. Stall *break* and late
drying are set by this isotherm plus \(D_{\mathrm{eff}}\). Without
gravimetric \(a_w(W,T)\) for cooked beef, \(k_m\) and \(D_{\mathrm{eff}}\)
remain a ridge — the same ridge the water budget is sitting on.

### T5. Smoke ring is not NO diffusion with a 62 °C gate

The implemented model is a penetration-depth ODE gated at 62 °C. The
chemistry is: combustion \(\mathrm{NO}/\mathrm{NO}_2/\mathrm{CO}\)
dissolve in **surface moisture**, form nitrite, then nitrosomyoglobin;
the ring stops when the surface dries or myoglobin denatures, not at a
core-temperature trip
([Blonder, Genuine Ideas](https://www.genuineideas.com/ArticlesIndex/srasmokeringmoist.html);
[AmazingRibs smoke-ring mythbusting](https://amazingribs.com/more-technique-and-science/more-cooking-science/smoke-ring-mythbusting/)).
Myoglobin denaturation is species- and ligand-dependent and can sit
nearer 70–77 °C than 62 °C. Leaving ring out of \(J\) is correct;
treating \(D_{\mathrm{ring}}=1.5\times 10^{-9}\) as “NO-assisted
diffusivity” is a curve fit without a ring measurement.

### T6. Bark Maillard is a bread-crust clone

Purlis bread-crust \(E_a \approx 64\,\mathrm{kJ\,mol^{-1}}\) with a
water-activity bell is a reasonable *shape*, and the dry-side floor
\(g_{\mathrm{dry}}=0.5\) matches “crust browns because it dries.” Meat
bark is also smoke phenols, sugar in the rub, and melting fat.
\(k_0\times 5\) and the \(L^*\) affine map are uncalibrated.
Hot-and-fast vs low-and-slow vs wrap lore is a qualitative sanity check,
not a color model.

### T7. Quasi-steady pit humidity and one-state pit

Residence time ~30 s justifies algebraic \(p_v\) during a hold. It does
not justify it during a lid lift (the 25 K one-minute dome transient) or
for wet-bulb as an input. Blonder’s stall temperature moves with pit
humidity; Modernist Cuisine measured RH peaking then falling. AFR,
\(y_w\), \(\omega_{\mathrm{amb}}\) are literature constants. Closed-loop
fan logs cannot separate \(UA\) from \(\Delta H_{\mathrm{eff}}\) or
\(h_c\) from \(k_m\) — that is identifiability theory, not a bigger ODE
(Gutenkunst et al., *PLoS Comput. Biol.* 2007,
[doi:10.1371/journal.pcbi.0030189](https://doi.org/10.1371/journal.pcbi.0030189)).
The characterization burn in C2 is the correct experiment.

### T8. Newman product is exact only for linear conduction, constant ambient

Newman (1936) factors 1-D solutions for the linear heat equation.
Evaporative (nonlinear) BCs, shrinkage, and a moving pit temperature
violate the hypotheses. Using it as a “cheap 3-D taste” is fine if
labelled; it is not a bound on the 2-D stall.

### T9. Two-stage Laplace pooling is not a hierarchical posterior

The role assignment (vessel shared, meat partial, ignition iid) is the
right scientific claim. The procedure — per-log Laplace Gaussians, then
a precision-weighted product with \(K-1\) prior dedup — is Gelman’s
“secret weapon” / linear empirical Bayes, not joint hierarchical NUTS.
Known failure modes: (i) Laplace misses ignition tails (they measured
this); (ii) pooling a subset of correlated coordinates can put the
shared point **outside** the per-cook hull (they measured this too);
(iii) \(\tau^2\) with \(K=2\) is barely identified; (iv) digitized forum
logs are not a random sample of cooks. Joint hierarchical ODE Bayes is
expensive (Gelman et al. 1996 lineage;
[Dass et al. two-step Laplace](https://ar5iv.labs.arxiv.org/html/1607.07203));
more logs beat a fancier pooler, which `docs/pooling.md` already says.

### T10. PSIS-\(\hat k = 0.76\) means the cheap posterior is not a drop-in NUTS replacement

Vehtari, Gelman & Gabry (2017) treat \(\hat k > 0.7\) as unreliable for
importance sampling; moment matching to 0.47 (Paananen et al. 2021) is
the right repair and should stay in the pipeline. Do not quote TMB/INLA
widths as “the posterior” on ignition without the PSIS gate.

### T11. Salt diffusivity is in the right decade; the couplings are not identified

Ham-curing \(D_{\mathrm{NaCl}}\) is typically
\(1\)–\(7\times 10^{-10}\,\mathrm{m^2 s^{-1}}\)
([Vestergaard et al., *Meat Sci.* 2008](https://doi.org/10.1016/j.meatsci.2008.03.002);
[Graiver et al., *J. Food Eng.* 2006](https://doi.org/10.1016/j.jfoodeng.2005.08.018)).
\(10^{-10}\) is a fair cold-cure default; it should rise with cook
temperature (not implemented). \(\beta_{\mathrm{whc}}\) and \(k_{aw}\)
need mass and \(a_w\), not a probe.

### T12. Discrete adjoint of FBDF is still open

Enzyme through IMEX-BE is the right *structure* for 2-D (crossover
~10–26 parameters on the proxies). Production derivatives remain chunked
ForwardDiff through adaptive FBDF. A discrete adjoint of the *adaptive*
stepper is a different object (step-size sensitivities; see Sanz-Serna /
Hairer & Wanner; SciMLSensitivity caveats). Do not assume the 5× reverse
IMEX number applies to the monolith the planners actually call.

---

## Documentation, product, and process

| Issue | Why it matters |
|---|---|
| README / writeup still framed as the Aug 3–31 hackathon, while `docs/results.md` runs through 2026-09-04 | Attribution and “all code in window” claims are stale |
| `equation_correctness_review.md` not marked superseded | Readers will think H2/H5 are still open |
| Install story correctly refuses PyPI; CI app job is broken | New contributors get a false green |
| Pilot has never seen a pit | C1 is the only test that can falsify the twin |
| No smoker yield pair in the log hunt | Moisture physics is identified from temperature, which is the wrong observable |

The honesty elsewhere is a strength: “not a USDA cook,” “not a judge
score,” “not a 95% on-time guarantee” should stay in the CLI output, not
only in `docs/roadmap.md`.

---

## Best next steps

Constraint as of 2026-09-07: **no cook, no pit, no instruments**
(travel). C1–C7 and any recalibration wait. Do **not** add more cooker
presets, bitterness states, or 3-D geometry in the meantime — extra
uncalibrated physics makes the planner more confident and less true.
The water-budget failure is still the binding physics problem; it cannot
be *closed* without a scale, but it can be *isolated* in software.

### Now (laptop only)

1. **Fix the 2-D shadow before quoting it.** Pull/rest on the 2-D
   Tesseract; match 1-D serve time; document geometry (do not default
   \(R=0.10\) if \(L\) is a flat half-thickness); refine the grid or
   report discretization error on \(T_{\mathrm{cold}}\). Until then,
   treat `slab2d_plan.png` as a visualization, not a safety audit.

2. **Fix CI.** Point the app job at tests that exist (`app/tests/`, or
   an explicit subset). Gate the composed gradient test. State-count
   assertions in Julia tests so 166 cannot creep back into docs.

3. **Doc pass.** One current state dimension (210 at `nx=40`, not 166);
   `model.md` humidity/pan/salt/latch in sync with `monolith.jl`;
   `slab2d.jl` header matching the file; `timeplan.py` no longer calling
   duration-opt “the next step.” The Aug 30 equation review is already
   marked historical.

4. **Rewrite safety, do not tighten it.** Replace the 4 h USDA holding
   dwell with FSIS CUT (50–130 °F, 6 h) as a *report*, and encode
   Scientific Gap 6 for brisket so the optimizer is not forced
   hot-and-fast. Keep the \(F\)-value; add humidity as a lethality caveat
   (Appendix A RH requirements), which this model does not certify. No
   new data required — this is a wrong constraint, not a missing
   measurement.

5. **Close the pan humidity algebra.** Drive pan evaporation against the
   post-pan \(p_v\) (or a joint algebraic root), not the pre-pan value.
   Needed before the pan-in/pan-out cook can mean anything.

6. **Water-budget sensitivity, not a new moisture PDE.** On the
   calibrated real-cook grid, map which \((h_c, D_{\mathrm{eff}}, L)\)
   retain ~50–60% water while still stalling. If no such point exists
   inside the prior, T1 (Fickian vs Flory–Rehner) is a modeling goal to
   spec — still not to implement — before the first weighed cook. If a
   point exists, the scale experiment has a predicted target.

7. **Collagen: two-rate *structure*, same scale.** Split enzymatic
   (\(<60\) °C, Tornberg / Baldwin) vs thermal gelatinization as two
   states or two rates, inert at the current \(C^\star = 0.90\) pull
   anchor. Do not retune \(E_a\) against one TVWBB pull. Texture data
   waits on a cook.

8. **Output-feedback estimator on the mock pit.** The ambient KF plus
   `mock_heatermeter` is enough to build a probe-only meat-field
   estimator (and to see it fail) without hardware. Control Jacobians /
   SQP wait until the state is even approximately observed.

9. **TVWBB account, if reachable from the road (C4).** The four
   login-gated wrap/rest attachments are the only new *data* that do not
   need a personal cook. Prefer those over another temperature-only keg.
   Yield pairs still wanted; none are known on the open web.

A reasonable **non-goal** until home: MTK rewrite, blackbox Enzyme
through the 2-D closure engine, ADVI, promoting `opt_salt`, or a
Flory–Rehner moisture rewrite.

### When a cook is possible again

These stay the highest-leverage items in the project; they are just not
the next ones.

1. **Instrument package, in this order (C2).**
   (a) Open-loop characterization burn (fan steps, no meat) to break
   \(UA\)–\(\Delta H_{\mathrm{eff}}\) and \(h_c\)–\(m_{\mathrm{conv}}\).
   (b) Scale: meat in/out, fuel in/ash out — put mass loss in the
   likelihood. This is the single highest-value experiment in the repo.
   (c) Second meat probe at known depth; wet-bulb; IR surface.
   Recalibrate. If retained water is still ~4%, the Fickian slab is the
   wrong model (T1), not a wrong \(k_m\).

2. **Fly the pilot in shadow mode (C1).** Advisory wrap/pull against a
   real HeaterMeter. Score executed quality with mass and photos, not
   model juiciness.

3. **Texture or yield, not a third cooker.** Warner–Bratzler / probe
   force on a designed hold if the two-rate collagen structure is in;
   otherwise any SMOKER log with before/after weights. Do not keep
   anchoring \(C^\star = 0.90\) to one TVWBB pull.

---

## Suggested reading stack (beyond what the repo already uses)

| Topic | Starting points |
|---|---|
| Stall | Blonder / AmazingRibs; *Modernist Cuisine* oven-humidity experiment |
| Moisture in cooking meat | van der Sman 2007, 2013; Feyissa et al. 2013 |
| Tenderness | Tornberg 2005; Davey et al. 1976; Baldwin; Vaskoska et al. 2018 |
| Safety | FSIS Appendix A (2021) + Scientific Gap 6; not the consumer Danger Zone page |
| Sorption | Comaposada et al. 2001; GAB as COST-90 default |
| Smoke ring | Blonder Genuine Ideas (moisture-limited ring) |
| UQ | Vehtari et al. 2017 (PSIS); Gutenkunst et al. 2007 (sloppy models) |

---

The project already knows most of its sins. The failure mode to avoid
while travelling is shipping another differentiable decision (splits,
salt, 2-D bark) on top of a twin that finishes a 19 h brisket with 4% of
its water and a safety term that punishes the cook style the model is
named for. Honest 2-D replay, green CI, a matching equation sheet, a
safety term that does not fight low-and-slow — then, back at a pit,
scale, wet-bulb, open-loop burn, and the rest of the roadmap is earned.
