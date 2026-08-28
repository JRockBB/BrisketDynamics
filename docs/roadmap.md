# Roadmap

What is left. Hackathon due Aug 31, 2026 AoE; everything else is
post-hackathon. Equations of what already exists: `docs/model.md`.
Gaps: `docs/model_gaps.md`.

## Hackathon

1. **Current-model real-cook NUTS** — first
   `uv run python ../scripts/run_map_real.py --init ../data/calibration/map_real.json`
   (AR(1) likelihood; writes a new MAP including \(\rho\)). Then
   `uv run python ../scripts/run_nuts.py --real` from a **fresh**
   checkpoint dir (do not resume `nuts_ckpt_real` or `7a`/`7b`). 300
   warmup + 300 samples. Fill `figures/corner_real.png` and
   `figures/posterior_predictive_real.png`. Quote ESS, split-r̂,
   divergences as-is. The identifiability claim to check: meat side
   pinned, pit side only along ridges (UA ∝ ΔH_eff, r_max vs u_leak).
2. **Submission** — post the form / LinkedIn / forum drafts in
   `docs/submission.md`. User-owned: hackathon registration
   (tally.so/r/kdx2ej, required for prize eligibility).
3. *Stretch, only if 1–2 are done — in this order:*
   1. **Differentiable duration** via \(s = t/t_f\) time-scaling. The
      outer-scan planner (`timeplan.py`, `fullplan.py`) already chooses
      duration by cook quality. Remaining: duration as a differentiable
      decision jointly with the schedule. Formulation: the perfect brisket
      defines its own time — duration by *pure cook quality* (target
      doneness, bark, juiciness), no external costs bending it. The serving
      deadline never changes the cook; it positions the *start*: light the
      fire (planned duration) before serving, where planned ≥ quality-optimal
      duration is the shortest with an empirical done-fraction over N
      posterior draws. The safety margin is one-sided because finishing
      early is nearly free (the resting hold exists for this). Mid-cook this
      reduces to pull-when-done (elapsed time is sunk; the live
      time-decision is the pull), whose t_done = differentiable crossing of
      the doneness target. A losing cook should extend, not surrender.
      The pull is a decision variable (`opt_pull`: `t_pull` optimized
      jointly with the schedule and wrap, quality at serve —
      `docs/tier1_plan.md` §1); remaining: the MPC pull-when-done state
      machine (replans keep a fixed horizon and cook to the plan's pull).
   2. **Multi-chain NUTS-real** — 2–3 short chains from dispersed inits
      (separate processes); real split-r̂ across chains and a test of
      whether the pit-ridge mass location is reproducible.
   3. **Extend synthetic NUTS chain** for better ESS (currently 7–82) from
      its checkpoint.
   4. **≤5-min demo video.**

Out of scope for the hackathon (identifiability-first ethos): CFD smoke,
2D/3D meat, humidity dynamics without an RH/wet-bulb sensor, bark chemistry
without calibration data.

## Post-hackathon (by priority)

1. **Close the loop for real: MPC as the controller.** Productionize the
   receding-horizon re-planner; integrate with HeaterMeter (open-source fan
   controller — our planner can sit above its PID as the planning layer).
   The library stops advising and starts piloting. Includes
   **MPC-with-wrap**: closed-loop replanning of the wrap moment (the crutch
   is in the leisure-optimal plan at 3.5 h / 17 h — the closed-loop question
   is re-timing it under disturbances, plus irreversibility once executed).
   Remaining: wrap irreversibility (`t_wrap` frozen once executed) and
   pointing the re-planner at a real pit.
2. **Discrete adjoint via Enzyme-through-the-stepper: kill the backward
   pass properly.** Measured 2026-08-21: the meat VJP costs ~1.9 s base
   + 66 ms per cotangent save point (~9 s at 49 points, ~80% of gradient
   cost on the 115-point real-cook grid) against a 0.06 s forward solve,
   and the cost is flat in tolerance. Mechanism (SciMLSensitivity
   source-confirmed): every discrete cotangent injection is a u-modifying
   callback that restarts the backward BDF at order 1 — an artifact of
   the *continuous* adjoint formulation. A **discrete adjoint** —
   Enzyme reverse-mode through the stepper iteration itself — has no
   backward ODE and no restarts: save-point cotangents simply add into
   the reverse sweep, cost ≈ small multiple of the forward. Potential
   ~10-50x on every NUTS/SVI/control gradient, dwarfing every
   configuration-level option (QuadratureAdjoint failed our FD gate;
   KenCarp/checkpointing/sparsity all measured out). Price: Enzyme
   cannot eat adaptive FBDF, so this means a hand-rolled fixed-step
   stiff stepper (TRBDF2-class) with a custom Enzyme rule for the
   Newton/LU linear solve, fixed-step accuracy validation against the
   adaptive reference, and a full re-run of the gradient-verification
   doctrine. Week-class; the single highest-leverage performance item
   on this list. Composes with the monolith item below (one stepper,
   one discrete adjoint for the whole coupled system).
   Measured dead end (2026-08-22): routing through the direct
   `adjoint_sensitivities` API is NOT a shortcut — its infrastructure
   overhead (26.2 s FBDF-backward vs the rrule path's 11.2 s on the 19 h
   fixture) swamps the genuine one-step-backward win (KenCarp47 backward:
   17.6 s, confirming restarts cost ~1.5x). The production Zygote-rrule
   path is the fastest continuous-adjoint configuration for this model;
   only the full discrete adjoint beats it.
3. **Monolith / Dyad exploration: move the Tesseract boundary to the
   edge.** Shape (a) exists: `src/monolith.jl`, numbers in
   `docs/monolith.md`. The K=2 Picard split exists only because the pit/meat boundary
   bisects the physics; a monolithic Julia core (one coupled DAE, one
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
   knobs). A third shape to benchmark: (c) **Reactant.jl + EnzymeMLIR** —
   trace the smooth parts (pit ODE, orchestration, quality objective) to
   StableHLO/XLA with Enzyme differentiating through the compiled graph.
   Caveat measured 2026-08-21: the cost center is the stiff BDF *backward*
   solve (adaptive stepping, Newton/LU, injection callbacks — the
   XLA-hostile case, and the reason this project is polyglot at all), so
   Reactant helps the edges, not the core; viable only paired with a
   Julia-side stiff adjoint kernel, i.e. the same boundary redrawn with
   MLIR tracing instead of Tesseract. Either way, keep what Tesseract is
   actually for post-hackathon by wrapping the *whole* coupled model as a
   single Tesseract — a language-agnostic, versioned deployment skin for
   the MPC/HeaterMeter service — so the boundary sits around the physics,
   not through it. Decides *where* the items below get built.
4. **The ~$150 instrument package + data flywheel.** In value order: a
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
5. **Hierarchical multi-cook pooling.** Pit parameters are properties of the
   smoker (pool tightly across cooks); meat parameters are per-brisket
   (partial pooling → a personal prior over "briskets I buy"). Every logged
   cook tightens the twin. Designed experiment worth running early: water
   pan in vs. out, two otherwise-identical cooks, to identify the humidity
   pathway.
6. **Wrap-constant calibration.** Calibrate η_evap / ε_wrap against a
   wrapped cook log — posts document the wrap moment, making the wrap event
   a known-input step-response experiment — via the two-cook joint fit
   (pooling, above). Log acquisition is user-owned, like the 2013 log. A
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
8. **Bark model calibration.** Calibrate against standardized-photo
   colorimetry (hourly lid-open photos + color card; wrapped/unwrapped pair
   separates ΔL_S from ΔL_B — plan in `docs/bark_plan.md`). Optionally
   penetration-force texture. Add **smoke quality** (the "thin blue smoke"
   variable): clean vs smoldering combustion changes deposition chemistry at
   the same mass — plausibly a differentiable function of airflow/burn
   state, making bitterness a controllable output. To our knowledge
   unpublished territory. (The two-sided mahogany band is already in the
   objective.)
9. **Fidelity where new sensors justify it.** Newman product-rule quasi-3D
   center-line correction (~3× cost, differentiable, cheap taste of 3D);
   full 2D axisymmetric surface model *only* once thermal-camera surface
   fields exist to calibrate it (also feeds bark distribution / burnt ends).
   Sensors first, dimensions second.
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
    Notable cases: pulled pork ≈ free transfer (geometry switch); ribs make
    wrap physics mandatory; a combi oven identifies meat parameters under
    controlled T+RH for transfer to any smoker; sous vide validates against
    Baldwin's tables; and the sealed earth pit (pibil/imu) is the jewel —
    saturated steam kills the stall entirely, the source is an uncontrolled
    thermal discharge, and "control" becomes design (rock mass, seal time) in
    an uninstrumented cooking tradition.
