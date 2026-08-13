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
3. **Re-run MAP-real** after (1)+(2) — minutes; refreshes `map_real.json` and
   the fit figure.
4. **NUTS on the real cook** — the calibration capstone's posterior
   (`run_nuts.py --real`, checkpointed, needs plugged-in hours). ⚑ After (3).
   On completion: corner_real + predictive-band figures, writeup paragraph.
5. **MPC re-planning demo** — receding-horizon re-optimization from mid-cook
   state with a simulated disturbance (wind gust / fuel sag) visibly absorbed.
   Uses existing pieces (ICs as inputs + control loop); the best new figure
   available. Build while (4) samples.
6. **In-container `checkgradients` sweep** — the deferred full endpoint
   verification artifact, run inside the built images
   (`scripts/checkgradients/run.sh`); flips the writeup's last TODO.
7. **Repo public + push + CI green** — user creates the GitHub repo; fix
   remote, push all commits, shepherd the first CI runs.
   ⚑ Blocks submission; user-owned start.
8. **Submission mechanics** — final writeup pass (NUTS-real paragraph, repo
   URL), submission form, LinkedIn post (tag Pasteur Labs & ISI, Tesseract),
   forum showcase post. User-owned: hackathon registration (required for
   prize eligibility).
9. *Stretch, only if 1–8 are done:* **differentiable wrap-time** (smooth
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
2. **The ~$150 instrument package + data flywheel.** In value order: a
   *scale* (brisket before/after + fuel before/ash after — mass loss is the
   single most informative missing observable; directly identifies
   D_eff/k_m and the burn ridge); a second meat probe at known shallow depth
   (probe depth as a nuisance latent); DIY wet-bulb probe (thermocouple +
   wick — measures the exact stall-relevant quantity; industrial smokehouse
   practice) to make humidity an input and retire the p_v,pit confound; IR
   spot readings of the surface (h_c/k_m separation).
3. **Hierarchical multi-cook pooling.** Pit parameters are properties of the
   smoker (pool tightly across cooks); meat parameters are per-brisket
   (partial pooling → a personal prior over "briskets I buy"). Every logged
   cook tightens the twin. Designed experiment worth running early: water
   pan in vs. out, two otherwise-identical cooks, to identify the humidity
   pathway.
4. **Wrap physics + wrap-time optimization.** Smooth surface-BC blend
   (evaporation suppression, changed h) parameterized by wrap time; then
   "when to crutch" becomes a differentiable decision alongside the vent
   schedule.
5. **Differentiable bark model.** Maillard/browning kinetics submodel driven
   by the existing surface T/W histories (bread-crust color models
   transfer); calibrate against standardized-photo colorimetry (L\*a\*b\*
   index, color card) and optionally penetration-force texture. To our
   knowledge unpublished territory.
6. **Fidelity where new sensors justify it.** Newman product-rule quasi-3D
   center-line correction (~3× cost, differentiable, cheap taste of 3D);
   full 2D axisymmetric surface model *only* once thermal-camera surface
   fields exist to calibrate it (also feeds bark distribution / burnt ends).
   Sensors first, dimensions second.
7. **Community/product shape.** Publish as a usable open-source stack
   (calibrate-from-your-log CLI, schedule/MPC frontend); write up the bark
   and pooling work properly if the data supports it.
8. **Generalize across meats and cookers.** The architecture already is the
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
