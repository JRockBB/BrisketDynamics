# Roadmap

Two horizons: what ships for the Tesseract Hackathon (due Aug 31, 2026 AoE),
and where the project goes as a long-lived "computational pitmastery" tool.
Within each list, items are in priority order; hackathon items also carry the
dependency ordering that matters.

## Hackathon (by priority; ⚑ = ordering constraint)

1. ✅ **Fuel-bed ignition state** (`937e6b0`) — third pit state (airflow-assisted
   logistic ignition spread), a0=1 provably inert for synthetic
   back-compat; k_spread/a0 sampled for the real cook. Julia 34/34,
   composed gate 15/15.
2. ✅ **Juiciness output** — retained moisture at pull as a differentiable
   output (`w_retained`) + optional control-objective term (`w_juicy`).
   Julia 36/36, app suite 18/18; docs/diagram/README updated.
3. ✅ **Re-run MAP-real** — warm-started 600-step refinement: loss 551 → 516,
   σ_pit 2.6 → 2.1, σ_core 1.4 → 0.88 (at the digitization floor); early-cook
   overshoot resolved; a0 = 0.73, k_spread ≈ 4e-4 (physical). Cold-start
   lesson: 12-latent SVI needs the warm start (`run_map_real.py --init`).
5. **NUTS on the real cook** — the calibration capstone's posterior
   (`run_nuts.py --real`, checkpointed, needs plugged-in hours). ⚑ After (3).
   On completion: corner_real + predictive-band figures, writeup paragraph.
5. ✅ **MPC re-planning demo** — hourly receding-horizon re-planning with
   full-field state handoff through the Tesseracts; a −15 K cold front the
   controller's model does not know is absorbed: doneness 0.92 (MPC) vs 0.68
   (open-loop), target 0.90. Emergent insurance: the asymmetric deadline
   objective front-loads heat before the unforeseen disturbance, and the
   cook "finishes early and lets go" — pull-when-done behavior arising from
   the asymmetry. figures/mpc_demo.png.
7. **In-container `checkgradients` sweep** — the deferred full endpoint
   verification artifact, run inside the built images
   (`scripts/checkgradients/run.sh`); flips the writeup's last TODO.
8. **Repo public + push + CI green** — user creates the GitHub repo; fix
   remote, push all commits, shepherd the first CI runs.
   ⚑ Blocks submission; user-owned start.
9. **Submission mechanics** — drafts written (`docs/submission.md`: form
   text, LinkedIn post, forum showcase — two placeholders: NUTS-real
   sentence + repo URL confirm). Remaining: final writeup pass, actual
   posting. User-owned: hackathon registration (required for prize
   eligibility).
9. *Stretch, only if 1–8 are done — in this order:*
   **Cook time as a decision variable** — the general formulation: planned
   duration (equivalently, when to light the fire) is optimized *jointly*
   with the schedule against a serving deadline, ideally chance-constrained
   under the posterior (choose start time s.t. P(done by deadline) ≥ 95% —
   unifying robust control with time planning). Doneness enters as a soft
   term so the optimizer can trade slight underdoneness against lateness,
   with the asymmetry kept (late ≫ early; earliness mildly penalized — a
   2 a.m. fire-lighting has human cost, which makes the start-time optimum
   nontrivial). Mid-cook this reduces to pull-when-done (elapsed time is
   sunk; the live time-decision is the pull), whose t_done = differentiable
   crossing of the doneness target. Fixes the fixed-horizon formulation
   error exposed by the MPC demo (a losing cook should extend, not
   surrender). Cheap first implementation: outer scan over candidate
   durations with inner schedule optimization; differentiable duration via
   s = t/t_f time-scaling later. Promoted above the other stretch items. **Differentiable wrap-time** (smooth
   Texas-crutch blend on the surface BC; "the model decides when to wrap").
   Flashy but new-physics-through-the-adjoint risk. **Extend synthetic NUTS
   chain** for better ESS (currently 7–82) from its checkpoint. **File the
   tesseract-jax partial-tangent bug** upstream with our minimal repro.
   **≤5-min demo video.**

Explicitly out of scope for the hackathon (identifiability-first ethos):
CFD smoke, 2D/3D meat, humidity dynamics without an RH/wet-bulb sensor,
bark chemistry without calibration data.

## Post-hackathon (by priority)

1. **Close the loop for real: MPC as the controller.** Productionize the
   receding-horizon re-planner; integrate with HeaterMeter (open-source fan
   controller — our planner can sit above its PID as the planning layer).
   The library stops advising and starts piloting.
2. **Monolith / Dyad exploration: move the Tesseract boundary to the
   edge.** The K=2 Picard split exists only because the pit/meat boundary
   bisects the physics; a monolithic Julia core (one ~123-state DAE, one
   adjoint, Turing/AdvancedHMC calibration, MCMCThreads parallel chains)
   is worth ~2x per gradient, trivial multi-chain, and retires the
   juliacall threading zoo entirely. Two candidate shapes to prototype
   against the two-Tesseract baseline: (a) hand-written monolith (merge
   the RHS functions, keep full sensealg control); (b) **Dyad/MTK acausal
   components** — pit, fuel bed, and future heat sources as declared
   components with typed ports (T_source in, q_load out), the PDE staying
   a hand-written component inside. Dyad's component discipline is exactly
   the architecture the meats-and-cookers generalization item wants, so
   this exploration doubles as its foundation — benchmark both for
   gradient cost, NUTS wall-clock, and adjoint controllability (we needed
   surgical sensealg choices; symbolic layers add distance to those
   knobs). Either way, keep what Tesseract is actually for post-hackathon
   by wrapping the *whole* coupled model as a single Tesseract — a
   language-agnostic, versioned deployment skin for the MPC/HeaterMeter
   service — so the boundary sits around the physics, not through it.
   Decides *where* the items below get built.
3. **The ~$150 instrument package + data flywheel.** In value order: a
   two-hour open-loop *characterization burn* (step the fan 10→30→15%, no
   meat, no controller — free, and the only way to identify the pit's fast
   fan-response dynamics, which closed-loop cook logs provably hide); a
   *scale* (brisket before/after + fuel before/ash after — mass loss is the
   single most informative missing observable; directly identifies
   D_eff/k_m and the burn ridge); a second meat probe at known shallow depth
   (probe depth as a nuisance latent); DIY wet-bulb probe (thermocouple +
   wick — measures the exact stall-relevant quantity; industrial smokehouse
   practice) to make humidity an input and retire the p_v,pit confound; IR
   spot readings of the surface (h_c/k_m separation).
4. **Hierarchical multi-cook pooling.** Pit parameters are properties of the
   smoker (pool tightly across cooks); meat parameters are per-brisket
   (partial pooling → a personal prior over "briskets I buy"). Every logged
   cook tightens the twin. Designed experiment worth running early: water
   pan in vs. out, two otherwise-identical cooks, to identify the humidity
   pathway.
5. **Wrap physics + wrap-time optimization.** Smooth surface-BC blend
   (evaporation suppression, changed h) parameterized by wrap time; then
   "when to crutch" becomes a differentiable decision alongside the vent
   schedule.
6. **Salt & seasoning physics + salting-time optimization.** Dry-brining is
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
   salt, so this ships only with the *scale* (item 2) and a designed
   experiment — identical cuts salted at −24 h / −1 h / unsalted, weighed
   before and after; mass loss vs salting time is the curve the WHC
   coupling fits. Non-salt rub effects (sugar browning, emissivity, surface
   mass transfer) belong to the bark item below, not here.
7. **Differentiable bark model.** Maillard/browning kinetics submodel driven
   by the existing surface T/W histories (bread-crust color models
   transfer); calibrate against standardized-photo colorimetry (L\*a\*b\*
   index, color card) and optionally penetration-force texture. To our
   knowledge unpublished territory.
8. **Fidelity where new sensors justify it.** Newman product-rule quasi-3D
   center-line correction (~3× cost, differentiable, cheap taste of 3D);
   full 2D axisymmetric surface model *only* once thermal-camera surface
   fields exist to calibrate it (also feeds bark distribution / burnt ends).
   Sensors first, dimensions second.
9. **Community/product shape.** Publish as a usable open-source stack
   (calibrate-from-your-log CLI, schedule/MPC frontend); write up the bark
   and pooling work properly if the data supports it.
10. **Generalize across meats and cookers.** The architecture already is the
   generalization — heat source ⇄ meat through (T_source, q_load), i.e. swap
   one Tesseract, keep the other. The model family is small: geometry mode
   (slab/cylinder/sphere — one curvature term), surface-BC mode (evaporative
   / skin-barrier for poultry / saturated-steam / wrapped), source model
   (charcoal / pellet-auger / two-zone offset / rock reservoir / electric),
   and doneness kinetics (collagen, myosin, pasteurization — all Arrhenius).
   Notable cases: pulled pork ≈ free transfer (geometry switch); ribs make
   wrap physics mandatory; a combi oven identifies meat parameters under
   controlled T+RH for transfer to any smoker; sous vide validates against
   Baldwin's tables; and the sealed earth pit (pibil/imu) is the jewel —
   saturated steam kills the stall entirely, the source is an uncontrolled
   thermal discharge, and "control" becomes design (rock mass, seal time) in
   an uninstrumented cooking tradition.
