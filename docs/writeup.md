# Differentiable BBQ: Bayesian Calibration and Optimal Control of a Coupled Smoker–Brisket System

**Track 04 — Differentiable inference & UQ** (cross-track: 02 — Multi-physics & coupled systems)
**Repo:** https://github.com/JRock95/BrisketDynamics · Apache 2.0 · all code authored Aug 3–31, 2026

Real barbecue is a coupled multi-physics control problem. A charcoal pit with
its own thermal dynamics drives heat into a brisket whose internal physics —
heat conduction and moisture transport with surface evaporation — produce the
famous **stall**: a multi-hour plateau in internal temperature caused by
evaporative cooling. We built both solvers in Julia, composed them through
Tesseract into a single end-to-end differentiable function under JAX, and used
the gradients seven ways: MAP and full NUTS Bayesian calibration (synthetic
recovery study **and** a real 19-hour cook log), damper-schedule optimization,
schedule optimization *in expectation over the posterior*, receding-horizon
MPC with disturbance rejection, cook-time planning — duration chosen by cook
quality, start time interrogated against the posterior — and a mechanistic
**bark model** (Maillard + smoke deposition) whose color gradients flow
through both components and complete the quality objective. This document follows the six judging criteria in
order.

## 1. Composition across a real boundary

Two Julia Tesseracts with **deliberately different differentiation
strategies**, composed in JAX via tesseract-jax:

| Component | Physics | Differentiation |
|---|---|---|
| `brisket-meat` | 1D slab heat+moisture PDE + surface bark states (Maillard, smoke deposition), 122 stiff ODEs (method of lines, FBDF), smoothed evaporative surface BC, collagen kinetics | **SciMLSensitivity continuous adjoint** (`InterpolatingAdjoint` + `EnzymeVJP`), exposed as the Tesseract `vector_jacobian_product` |
| `brisket-pit` | lumped charcoal pit, 3 ODEs (energy balance, fuel burn with smooth burnout taper, Minion-method ignition spread) | **exact dense ForwardDiff Jacobian**, transposed to a VJP |

![Composition diagram](../figures/composition_diagram.png)

The components are coupled **two-way** (pit temperature → meat boundary
condition; meat evaporative + sensible load → pit energy balance) by a
whole-horizon Picard iteration with K=2 fixed passes, unrolled in the JAX
orchestrator:

$$
\big(T_{pit}^{(k)},\, m^{(k)}\big) = P_\varphi\big(u,\, q^{(k-1)}\big),
\qquad
\big(T_{core},\, j_{evap},\, q^{(k)}\big) = M_\theta\big(T_{pit}^{(k)}\big),
\qquad k = 1 \ldots K .
$$

The model is *defined* as the k=K output of this unrolled map, so
reverse-mode AD through the composed VJPs is the exact gradient of the
simulated system — no implicit-function asterisk. One `jax.grad` of the
composed log-density crosses the Julia↔JAX boundary four times (two adjoint
solves, two forward-mode Jacobian transposes).

The stall mechanism — and the place where differentiability was deliberately
engineered — is the meat's surface boundary condition. Evaporative flux is
mass-transfer-limited with a *smoothed* moisture-availability switch and a
*smoothly clamped* vapor-pressure deficit,

$$
j_{evap} \;=\; k_m \,\underbrace{\sigma\!\Big(\tfrac{W_s - W_{crit}}{\delta_W}\Big)}_{\text{availability}}\,
\frac{\operatorname{softplus}_{\delta_p}\!\big(p_{sat}(T_s) - p_{pit}\big)}{R_v\, T_s},
\qquad
-k\,\partial_x T\big|_{s} \;=\; h_c\,(T_{pit}-T_s) + \varepsilon\sigma_{SB}\,(T_{pit}^4-T_s^4) - \lambda_{vap}\, j_{evap},
$$

so the plateau emerges (and breaks) smoothly: no branch ever enters the
adjoint, and NUTS's likelihood surface stays differentiable through the very
feature that carries the information. Calibration uses a masked Gaussian
likelihood with a digitization noise floor,
$y^{pit}_i \sim \mathcal N\big(T_{pit}(t_i;\theta,\varphi),\, (\sigma_0+\sigma_{pit})^2\big)$
over non-lid-event observations (likewise for the core probe).

A three-level gradient verification gate backs this: Julia adjoints vs central
finite differences (rel. err ≤1e-3, most components ≤1e-5), per-endpoint
`tesseract-runtime check-gradients`, and a composed FD check of
`jax.grad(loss ∘ coupled_cook)` — **17/17 directions pass at 1e-10 tolerances** (every meat parameter, all 7 pit parameters incl. the ignition dynamics, damper directions, the wrap time, and the damper→burn-rate→smoke→bark-color chain through both components; the Julia-level suite runs 68 tests incl. 27 adjoint-vs-FD checks)
(`docs/verification.md` documents the methodology, including why the
compiled-ReverseDiff-tape variant is *silently wrong* for time-interpolated
forcings, and the FD conditioning required to verify a parameter that spans
twelve orders of magnitude).

To our knowledge — based on a survey of the pasteurlabs `mosaic` suite, the
official examples, and the si-tesseract forum — no public example existed of a
SciMLSensitivity continuous adjoint exposed through the Tesseract VJP
interface; the Julia precedents use Zygote or hand-written adjoints.

## 2. Gradients doing real work

Every result below pulls adjoint gradients across the container boundary; none
is solvable component-by-component because the observable (core temperature,
doneness) depends on the two-way coupling.

**Bayesian calibration, synthetic recovery** (`figures/corner.png`,
`figures/posterior_predictive.png`): 10 physical latents + 2 noise scales,
literature-centered priors. MAP (300 SVI steps, 5.4 min) recovers 7/10
parameters within 10%. Full NUTS (300+300, initialized at MAP): every true
parameter inside the posterior, and the geometry is the physics — the
**k_m–h_c ridge** *is* the stall's evaporative-cooling/convective-heating
trade-off; D_eff stays broad ("the stall informs but does not pin moisture
diffusivity"). Honest diagnostics: single chain, ESS 7–82, split-r̂ ≤ 1.14 —
quoted as-is; the chain is extendable from its checkpoint.

**The capstone — a real cook** (`figures/cooklog_fit_map.png`): we digitized a
19-hour brisket cook from a 2013 Virtual Weber Bulletin Board post (18.5″ WSM,
HeaterMeter controller; automated pixel extraction with a ±1–1.5 °F
uncertainty budget and provenance in `data/cook_log/PROVENANCE.md`). The
measured blower duty replays open-loop as the control input; initial
conditions come from the log; lid events are masked. The calibrated model
tracks the 19-hour pit hold within ~3 °C and reproduces the core's full arc —
rise, multi-hour stall at ~70 °C, **stall break at ~14 h**, climb to 90 °C.
The poster described his door as leaky; the calibration put the leak
parameter at u_leak = 0.21, and the Minion-start ignition state landed at
a0 = 0.73 with an hours-scale spread constant — both physical. Core noise
calibrates to the digitization floor (σ_core = 0.88 K). Honest residual:
±3 °C pit wobble at sub-hour timescales — a real model deficiency (see
Limitations).

**The real-cook posterior** (`figures/corner_real.png`,
`figures/posterior_predictive_real.png`): full NUTS over all 12 physical
latents + 2 noise scales — 300 warmup + 300 samples, 34 h wall-clock on two
throttled laptop threads, kernel-checkpointed through the power losses. The
posterior-predictive band holds the entire 19-hour pit trace and the core's
rise–stall–break arc. Parameter space quantifies the identifiability story
told in Limitations: the meat side is pinned (D_eff to ±13%, k_m and L to
±3% at 90% credibility — the long horizon and the stall break carry real
information), while the pit side is identified only in *combinations* — the
posterior slides along two clean ridges (UA ∝ ΔH_eff, heat loss vs heat
release; r_max vs u_leak, burn rate vs leak dilution), the fingerprint of
closed-loop data. Instructively, the MAP's u_leak = 0.21 and the
posterior's 0.70–0.79 band produce near-identical trajectories: the "leaky
door" is a property of the burn ridge, not an individually pinned number
(`figures/corner_real_widerange.png` widens the axes to overlay the MAP
point — point estimate and posterior bulk sit far apart along the flat
ridge directions while fitting the data equally well).
Diagnostics quoted as-is: single chain, ESS 4–17, split-r̂ up to 2.6 — the
chain explores the ridges but has not equilibrated; it remains extendable
from its checkpoint.

**Optimal control** (`figures/control_before_after.png`): Optax Adam on 16
sigmoid-bounded damper knots through the full coupled pipeline. Objective
falls 40.9 → 0.10 in 200 iterations; core doneness 0.26 → **0.900 on a 0.90
target** with the surface under the bark limit.

**Control under calibrated uncertainty** (`figures/control_robust.png`):
re-optimizing the expected objective over 8 posterior draws (sequential
pipeline evaluations inside one jitted graph). Evaluated on 30 held-out draws,
the MAP-optimal schedule yields doneness 0.780 ± 0.123 (worst case 0.582); the
robust schedule 0.863 ± 0.069 (worst 0.727) — **half the spread** — and the
schedule is interpretable: burn hard while fuel is plentiful, coast through
the stall, reopen for the finish.

**Receding-horizon MPC** (`figures/mpc_demo.png`): the calibrated smoker as
plant, hourly re-plans from the measured state (full T/W/C-field handoff
through the Tesseracts' state IO), and a −15 K ambient cold front at t = 6 h
that the controller's model does not know. Open-loop lands doneness 0.68;
MPC lands **0.92** on a 0.90 target. Two emergent behaviors worth noting:
the asymmetric deadline objective (undershoot ≫ overshoot; the resting hold
makes early finishes cheap) front-loads heat as *insurance against
disturbances the controller cannot foresee*, and once the target is banked
the controller closes the damper and lets the cook coast — pull-when-done
behavior arising from the objective's asymmetry rather than being
programmed.

**Cook time as a decision variable** (`figures/timeplan.png`): the perfect
brisket defines its own time — an outer scan over planned durations with an
inner schedule optimization per duration. Instructively, the answer moved
as the quality objective completed: with doneness + juiciness alone the
optimizer chose a **9 h hot-and-fast** cook (`timeplan.png`, left panel);
adding bark color flipped it to the **~17 h low-and-slow** of the full
composition below — the clearest demonstration in this project that *what
you optimize is part of the model*. The chance layer then asks the
NUTS-real posterior to certify a start time and receives **an honest
refusal**:
P(serveably done) = 0 at every duration, because the closed-loop cook log
never exercised the damper above ~0.35 and the control-relevant pit
response is unidentified — the *same* damper input yields a 202 °C pit at
the MAP and 118 °C at a posterior draw, both fitting the 19-hour log
equally well. The planner thus reproduces, in decision space, the
identifiability limit named in Limitations: open-loop guarantees await the
characterization burn; live guarantees come from closing the loop (the MPC
above). Caveat: absolute hours inherit the uncalibrated doneness scale
(fixed literature collagen kinetics; C = 0.90 as "probe tender" is not yet
anchored by texture data).

**The bark model — quality's third leg** (`figures/bark_model.png`,
`figures/fullplan.png`): two surface states inside the meat solve — Maillard
browning (first-order in lightness, Arrhenius × a water-activity bell via a
smooth sorption isotherm; bread-crust kinetics lineage) and smoke-phenolic
deposition, which requires a *moist* surface and smoke flux. The smoke flux
is the pit's **exact burn rate**, exported as a new output and passed as a
forcing to the meat — making bark color a genuinely **two-component
observable**: `d(color)/d(damper)` crosses the boundary through the pit's
ForwardDiff VJP and the meat's continuous adjoint (composed FD-verified).
Constants are literature-anchored and explicitly uncalibrated — the
designed calibration experiment (hourly photos with a color card; a
wrapped/unwrapped cook pair to separate smoke color from Maillard) is
specified in `docs/bark_plan.md`. Forward validation reproduces pitmaster
lore mechanistically: hot-and-fast Maillard-saturates, low-and-slow is
smoke-dominated, wrapping freezes bark. With color completing the quality
vector, the full composition — duration × wrap × schedule jointly optimized
— **rediscovers traditional practice**: a ~17 h low-and-slow cook, wrap
never chosen at any duration (the reference real cook: 19 h, unwrapped).
The Texas crutch drops out of the leisure-optimal plan and re-emerges as
what it actually is — a closed-loop contingency tool for cooks under
deadline pressure (the MPC-with-wrap demo is roadmapped). The objective
also offers a **judge mode**: the KCBS scoring weights (appearance 0.14 /
taste 0.57 / tenderness 0.29 — the multipliers real judges' scores carry)
applied to the subset of each category the simulation can observe — a
two-sided *mahogany* color band (judges ding black bark as burnt-looking),
a smoke-deposition band (phenolic mass as a smoke-depth correlate, with an
over-deposition/bitterness ceiling), moisture, and the collagen tenderness
band. Stated plainly: tenderness is a mechanistic proxy, taste is two
correlates (smoke mass, moisture — seasoning and smoke *quality* are
absent), appearance is color only. Absolute durations remain conditional
on these placeholder normalizations; the weights, at least, are now cited
rather than invented.

Every objective mode also carries a **food-safety constraint**: a smooth
integral of surface time in the bacterial danger zone (4.4–60 °C), capped
at the USDA-guidance 4 hours — normal smoker schedules pay nothing; the
term exists so no optimizer, ever, recommends a cold-smolder start. A
model that tells people how to cook meat owes them this term
(`docs/model_gaps.md` is the standing audit of what the quality layer
does and does not yet model).

## 3. Why Tesseract is load-bearing

The meat PDE is stiff with a moving evaporation front; the correct
differentiation strategy is a continuous adjoint via SciMLSensitivity, not
naive autodiff. The outer loops (NumPyro NUTS, Optax) live in JAX. There is no
native way to `jax.grad` through a Julia ODE solve: the *only* gradient path
between the two stacks is the Tesseract VJP interface, consumed by
tesseract-jax as a JAX primitive. Could this have been pure Julia with
Turing.jl? Not without abandoning NumPyro/Optax; pure JAX would mean
reimplementing a stiff MOL solver and its adjoint. The same `coupled_cook`
function is consumed unchanged by calibration and control — infrastructure,
not a script. Modeling scope we chose consciously: pit vapor pressure is fixed
at 2 kPa; real pit humidity varies (water pans exist precisely to manipulate
the stall), and this constant is entangled with k_m's identifiability — k_m is
calibrated *conditional on* assumed humidity.

## 4. A real application

Food engineering / culinary physics — a domain without an existing Tesseract
demo, with real data, a beloved phenomenon (the stall) whose mechanism is
still argued about on forums, and a genuinely useful artifact: a calibrated
digital twin of a specific smoker that turns "when do I open the vents?" into
a posterior-aware optimization. The physics is honest (Choi–Okos-derived
properties, smoothed wet-bulb-limited evaporation, collagen denaturation
kinetics as the doneness diagnostic), and the smoothing that keeps the stall
differentiable is exactly what makes NUTS's likelihood surface tractable.

## 5. Execution & technical depth

The war stories are documented in `docs/verification.md`; highlights:

- **Silently-wrong adjoint caught by FD gates:** compiled ReverseDiff tapes
  freeze time-dependent interpolation weights as constants, zeroing the
  forcing-knot adjoint. Our tiered FD verification caught it; the writeup of
  *why* is in the repo.
- **A NaN chain with four masks:** GaussAdjoint's quadrature emits NaN
  gradients on hot fuel-collapse trajectories with coupling-feedback
  cotangents. Downstream it masqueraded as shape corruption, "thread races,"
  and solver crashes. Root-caused by capturing the exact failing call at the
  endpoint and replaying it in Julia; fixed by switching to
  InterpolatingAdjoint (agrees with ReverseDiff to 6 digits, 4× faster there).
  The capture-and-raise guard is now a permanent endpoint feature.
- **Cross-language threading, three deadlock modes:** jitted JAX executes
  callbacks on XLA worker threads; juliacall off its init thread hangs. All
  Julia work runs on one dedicated thread; Julia exceptions are stringified
  on that thread; zero-copy views of Julia/XLA memory never cross threads
  (copy on entry and exit). In-container there is no JAX, so a
  `BRISKET_JL_DIRECT` switch bypasses the executor — juliacall's atexit hook
  otherwise hangs the container's shutdown.
- **27× adjoint speedup** (profiled sensealg matrix: 10.7 s → 0.4 s per VJP),
  making NUTS feasible on a laptop: composed gradient 19.5 s → 1.53 s.
- **Power-loss-proof sampling:** NumPyro's kernel-level `HMCState` pickles
  cleanly, so our runner checkpoints *every 5 NUTS steps, warmup included*.
  The posterior in this writeup survived four battery deaths with zero lost
  samples.

## Limitations & where this goes

The model is honest about its scope. Pit humidity is fixed (a water pan would
invalidate the calibrated evaporation coefficient — the k_m confound named in
§3); the Texas-crutch wrap and bark (Maillard + smoke deposition) are now
modeled — smooth blends and surface states with a *differentiable wrap
time*, so "when to wrap" is optimized jointly with the vents — but their
constants are literature-fixed pending calibration data (wrapped-cook log;
bark photos): our calibration cook was unwrapped, and since wrapped logs document the
wrap moment, the wrap event is a known-input step-response experiment
waiting for the two-cook joint fit; "doneness" is
collagen conversion — a good proxy for probe-tender, silent on bark and
smoke ring (juiciness now has a first proxy: retained moisture at pull is a
differentiable output); and the pit's **fast (sub-hour) dynamics are a known deficiency**: burn
rate responds instantly to fan duty (no coal-bed thermal lag) and the pit is
a single thermal node, so replaying the controller's fan moves open-loop
produces ±3 °C wobble around a hold the real controller kept flat.
Instructively, this dataset *cannot* repair that deficiency — the cook was
recorded under closed-loop control, which correlates the input with the
disturbances and suppresses exactly the excitation needed to identify the
fast dynamics (the posterior widths of the pit-side parameters quantify
this). The fix is a designed experiment, not a bigger model: a two-hour
open-loop *characterization burn* — step the fan, no meat, no controller —
would identify the fan-response dynamics directly and tell us whether a
coal-bed lag state is warranted before we add one.
Smoke itself is deliberately *not* modeled as a fluid: a single-probe cook log
cannot identify CFD-scale parameters, and flavor deposition is chemistry the
data cannot see — lumped heat delivery is what the evidence supports.

One control-side simplification is worth naming: our schedules optimize
doneness at a *fixed* final time, but the practitioner's objective is
pull-when-done — doneness as the constraint, pull time as the outcome, a
deadline as an asymmetric lateness penalty (finishing early is nearly free;
the resting hold exists for this). Fixed-horizon control cannot say "this
brisket needs another 90 minutes," which is often the right answer. The
general fix — cook time as a decision variable, chosen jointly with the
schedule and chance-constrained under the posterior ("when should I light
the fire so P(done by dinner) ≥ 95%?") — is the first control upgrade on
the roadmap.

The natural next step is closing the loop on a physical smoker. The
receding-horizon re-planner is demonstrated in simulation (§2); state — full
temperature/moisture/doneness fields — is first-class Tesseract IO, so
pointing it at a real pit is an estimator plus a fan controller away
(HeaterMeter integration is first on the post-hackathon roadmap). Pit
parameters are
properties of the smoker and transfer across cooks; meat parameters are
per-brisket, which makes hierarchical pooling across a user's logged cooks
the right statistical structure — every cook tightens a personal prior. The
workflow this enables for a backyard cook: one instrumented calibration cook,
a posterior for *your* smoker, and a vent schedule — re-planned live — that
lands tender, on time, with fuel to spare.

## 6. Reproducibility & communication

Everything is pinned (`Manifest.toml`, `uv.lock`) and layered so each claim is
one command (`README.md`): Julia physics + gradient tests, boundary tests, the
composed FD gate, each figure's generating script. The native no-Docker dev
loop (`Tesseract.from_tesseract_api`) runs the full science pipeline on
Windows; both containers build and solve (`scripts/build_containers.sh` —
Julia toolchain and precompiled package baked at build time via
`package_data`; `brisket-meat:0.1.0` and `brisket-pit:0.1.0` verified with
in-container applies **and a full in-container check-gradients sweep** —
523 checks over every trajectory input, zero real mismatches; the two
benign FD-artifact classes at measure-zero kink/onset elements are
dissected, with an eps-dependence experiment proving the adjoint right, in
`docs/verification.md`). Figures: the animated calibrated cook
(`figures/cook_animation.gif`), corner plots (synthetic + real), 
posterior-predictive bands (synthetic + real), control before/after,
robust-control comparison, MPC demo, real-cook fit.

---

*Data attribution: cook log digitized from a public post by user "Mache" on
The Virtual Weber Bulletin Board (2013), used as a small factual data
extraction with attribution; provenance, method, and uncertainty budget in
`data/cook_log/PROVENANCE.md`.*
