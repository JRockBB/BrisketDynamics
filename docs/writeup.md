# Differentiable BBQ: Bayesian Calibration and Optimal Control of a Coupled Smoker–Brisket System

**Track 04 — Differentiable inference & UQ** (cross-track: 02 — Multi-physics & coupled systems)
**Repo:** https://github.com/JRock95/BrisketDynamics · Apache 2.0 · all code authored Aug 3–31, 2026

Real barbecue is a coupled multi-physics control problem. A charcoal pit with
its own thermal dynamics drives heat into a brisket whose internal physics —
heat conduction and moisture transport with surface evaporation — produce the
famous **stall**: a multi-hour plateau in internal temperature caused by
evaporative cooling. We built both solvers in Julia, composed them through
Tesseract into a single end-to-end differentiable function under JAX, and used
the gradients seven ways: MAP and NUTS Bayesian calibration (synthetic
recovery study **and** a real 19-hour cook log), damper-schedule optimization,
schedule optimization *in expectation over the posterior*, receding-horizon
MPC with full-field state handoff, cook-time planning —
duration chosen by cook quality, start time interrogated against the
posterior — and a mechanistic **bark model** (Maillard + smoke deposition)
whose color gradients flow through both components and complete the quality
objective. This document follows the six judging criteria in order.

## 1. Composition across a real boundary

Two Julia Tesseracts with **deliberately different differentiation
strategies**, composed in JAX via tesseract-jax:

| Component | Physics | Differentiation |
|---|---|---|
| `brisket-meat` | 1D slab heat+moisture PDE (denaturation expulsion, fat rendering) + surface bark states (Maillard, smoke deposition), 162 stiff ODEs (method of lines, FBDF), smoothed evaporative surface BC, collagen kinetics | **SciMLSensitivity continuous adjoint** (`InterpolatingAdjoint` + `EnzymeVJP`), exposed as the Tesseract `vector_jacobian_product` |
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
k\,\partial_x T\big|_{s} \;=\; h_c\,(T_{pit}-T_s) + \varepsilon\sigma_{SB}\,(T_{pit}^4-T_s^4) - \lambda_{vap}\, j_{evap},
$$

so the plateau emerges (and breaks) smoothly: no branch ever enters the
adjoint, and NUTS's likelihood surface stays differentiable through the very
feature that carries the information. (The complete equation set — all four
meat fields with their kinetics, the bark states, the pit ODEs, the coupling
map, the KCBS objective, and the likelihood — is written out in
`docs/model.md`.) Calibration uses a masked Gaussian
likelihood with a digitization noise floor,
$y^{pit}_i \sim \mathcal N\big(T_{pit}(t_i;\theta,\varphi),\, (\sigma_0+\sigma_{pit})^2\big)$
over non-lid-event observations (likewise for the core probe).

A three-level gradient verification gate backs this: Julia adjoints vs central
finite differences (rel. err ≤1e-3, most components ≤1e-5), per-endpoint
`tesseract-runtime check-gradients`, and a composed FD check of
`jax.grad(loss ∘ coupled_cook)`. Meat adjoints cover all seven θ latents
(including expulsion \(k_e\), \(a_{\mathrm{myosin}}\)) and a mid-cook
nonuniform-IC VJP; the composed Python gate covers the five
transport/thermal meat latents, all seven pit parameters, damper
directions, wrap time, and the damper→burn-rate→smoke→bark-color chain.
Methodology, including why a compiled ReverseDiff tape is *silently wrong*
for time-interpolated forcings, is in `docs/verification.md`.

To our knowledge — based on a survey of the pasteurlabs `mosaic` suite, the
official examples, and the si-tesseract forum — no public example existed of a
SciMLSensitivity continuous adjoint exposed through the Tesseract VJP
interface; the Julia precedents use Zygote or hand-written adjoints.

## 2. Gradients doing real work

Every result below pulls adjoint gradients across the container boundary; none
is solvable component-by-component because the observable (core temperature,
doneness) depends on the two-way coupling.

**Bayesian calibration, synthetic recovery** (`figures/corner.png`,
`figures/posterior_predictive.png`): 7 meat + pit latents + 2 noise scales,
literature-centered priors. MAP (300 SVI steps) recovers most parameters
within 10%. Full NUTS (300+300, initialized at MAP): every true
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
rise, multi-hour plateau at ~70 °C, **stall break at ~14 h**, climb to 90 °C.
The poster described his door as leaky; the calibration puts the leak
parameter at u_leak = 0.22, and the Minion-start ignition state lands at
a0 = 0.86 with an hours-scale spread constant — both physical. Expulsion
kinetics (\(k_e\), \(a_{\mathrm{myosin}}\)) are calibrated jointly with the
transport/thermal latents — the stall entry carries water-holding
information a thermometer can see. Honest residuals:
σ_core = 1.30 K against a ±0.8 K digitization floor. A temperature-only
fit that ignores the water budget can score better on the probes while
hiding an unphysical moisture state; this model stays accountable to a
kitchen scale. Plus the standing ±3 °C pit wobble
at sub-hour timescales (see Limitations).

**The real-cook posterior** (`figures/corner_real.png`,
`figures/posterior_predictive_real.png`):
NUTS over the physical latents plus 2 noise scales — 300 warmup + 300
samples, kernel-checkpointed through the power losses. The
posterior-predictive band holds the entire 19-hour pit trace and the core's
rise–stall–break arc. Parameter space quantifies the identifiability story
told in Limitations: the meat side is pinned (D_eff to ±13%, k_m and L to
±3% at 90% credibility — the long horizon and the stall break carry real
information), while the pit side is identified only in *combinations* — the
posterior slides along two clean ridges (UA ∝ ΔH_eff, heat loss vs heat
release; r_max vs u_leak, burn rate vs leak dilution), the fingerprint of
closed-loop data. Instructively, the MAP's u_leak and the
posterior's 0.70–0.79 band produce near-identical trajectories: the "leaky
door" is a property of the burn ridge, not an individually pinned number
(`figures/corner_real_widerange.png` widens the axes to overlay the MAP
point — point estimate and posterior bulk sit far apart along the flat
ridge directions while fitting the data equally well).

**Three chains make the ridge story rigorous**
(`data/calibration/multichain_diagnostics.json`): two additional 200+200
chains from dispersed initializations reproduce the ridge GEOMETRY almost
exactly (per-chain correlations +0.98 to +0.998 on UA–ΔH_eff, −0.98 to
−0.99 on r_max–u_leak) while cross-chain split-r̂ runs 1.13–4.6 on every
parameter: independent chains agree on the identified *combinations* and
park at different positions *along* them. That pair of facts — ridges
reproduce, positions don't — is the cleanest statement of the closed-loop
identifiability limit this data permits.

**Optimal control** (`figures/control_before_after.png`): Optax Adam on 16
sigmoid-bounded damper knots through the full coupled pipeline. Objective
falls 40.9 → 0.10 in 200 iterations; core doneness 0.26 → **0.900 on a 0.90
target** with the surface under the bark limit.

**Control under calibrated uncertainty** (`figures/control_robust.png`):
re-optimizing the expected objective over 8 posterior draws
(sequential pipeline evaluations inside one jitted graph).
Evaluated on 30 held-out draws,
the MAP-optimal schedule yields doneness 0.780 ± 0.123 (worst case 0.582); the
robust schedule 0.863 ± 0.069 (worst 0.727) — **half the spread** — and the
schedule is interpretable: burn hard while fuel is plentiful, coast through
the stall, reopen for the finish.

**Receding-horizon MPC** (`figures/mpc_demo.png`): the calibrated smoker as
plant, hourly re-plans from the measured state (full T/W/C/F/B/S-field
handoff through the Tesseracts' state IO). A mid-cook finite-difference
gate covers the adjoint through those handed-off fields. Food-safety terms
in the objective are soft surface penalties under an intact-muscle
assumption (`safety_report` prints the raw lethality/dwell residuals);
they are not a certification.

**Cook time as a decision variable** (`figures/timeplan.png`): the perfect
brisket defines its own time — an outer scan over planned durations with an
inner schedule optimization per duration. Doneness + juiciness alone prefer
a **9 h hot-and-fast** cook (`timeplan.png`, left panel); adding bark color
prefers ~17 h, and that 17 h plan wraps — *what you optimize (and what
physics backs it) is part of the model*. The chance layer asks posterior
draws for an **empirical done-fraction** (N=16; not a 95% certification).
On this posterior P(serveably done) = 0 at every duration: the closed-loop
cook log never exercised the damper above ~0.35 and the control-relevant pit
response is unidentified — the *same* damper input yields a 202 °C pit at
the MAP and 118 °C at a posterior draw, both fitting the 19-hour log
equally well. The planner thus reproduces, in decision space, the
identifiability limit named in Limitations: open-loop guarantees await the
characterization burn; live guarantees come from closing the loop. Caveat:
absolute hours inherit the uncalibrated doneness scale
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
smoke-dominated, wrapping freezes bark. With the full quality
vector in place — color, smoke, tenderness, and mechanistic juiciness —
the composition (duration × wrap × schedule jointly optimized) lands on a
**17 h cook that wraps at 3.5 h**: smoke bare while the bark and
smoke-band develop, then crutch and hold near 90 °C, ending at full
doneness with 51% perceptual juiciness. The wrap is how the plan protects
juiciness; its moment scales with horizon (a 10 h cook wraps at ~7 h, the
17 h optimum at 3.5 h — the longer you can cook, the sooner you stop paying
evaporative losses). The optimum is duration-flat past ~16 h (gains halve
per added hour; 17 h is the 2%-plateau's left edge). The **default
quality objective** is KCBS-weighted: appearance 0.14 / taste 0.57 /
tenderness 0.29 — the multipliers real judges' scores carry — applied to
the subset of each category the simulation observes: a two-sided
*mahogany* color band (judges ding black bark as burnt-looking), a
smoke-deposition band (phenolic mass as a smoke-depth correlate with an
over-deposition/bitterness ceiling), perceptual juiciness, and the
two-sided collagen tenderness band. Every weight and band is an
overridable config field — a different preference is a different config,
not a different mode. Juiciness is mechanistic on both channels:
retained water under denaturation-driven expulsion (myosin/actin
water-holding-capacity steps) plus retained rendered fat (the
fat-rendering state), with the cook-loss budget — evaporation, purge,
drained tallow — closed algebraically and testable against a kitchen
scale. Stated plainly: tenderness is a mechanistic proxy, taste is two
correlates (smoke mass, juiciness — seasoning and smoke *quality* are
absent; the fat/water sensory weighting β_fat is a placeholder),
appearance is color only. Absolute durations remain conditional on the
placeholder band normalizations; the weights are cited.

The objective also carries **food-safety soft penalties** (not a
certification): pasteurization is a time–temperature curve, so the terms
integrate **surface** lethality F = ∫10^((T_s−T_ref)/z) dt toward the
FSIS-guidance ~7-log equivalent (60 °C reference, z ≈ 6 °C) plus a
danger-zone dwell cap. The model assumes **intact muscle** (surface
pathogens); the cold-spot field is not evaluated, there is no hard
feasibility constraint, and an optimizer can trade these penalties against
quality. `safety_report` prints the raw residuals. Normal smoker
schedules typically pay little; the terms exist so a cold-smolder start is
expensive. `docs/model_gaps.md` lists what the quality layer still does
not model.

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
  The capture-and-raise guard is a permanent endpoint feature.
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
§3); the Texas-crutch wrap and bark (Maillard + smoke deposition) are
modeled — smooth blends and surface states with a *differentiable wrap
time*, so "when to wrap" is optimized jointly with the vents — but their
constants are literature-fixed pending calibration data (wrapped-cook log;
bark photos): our calibration cook was unwrapped, and since wrapped logs document the
wrap moment, the wrap event is a known-input step-response experiment
waiting for the two-cook joint fit. "Doneness" is collagen conversion at the
center node — a good proxy for probe-tender; bark and smoke ring are *not*
inside that scalar (bark *is* in the quality objective as \(L^*\) and
deposition \(S\); smoke ring is unmodeled). Juiciness is mechanistic on both
channels: finite-volume retained water under denaturation-driven expulsion
plus retained rendered fat, with \(\beta_{\mathrm{fat}}\) still a sensory
placeholder. Geometry is a fixed-\(L\) slab: 20–30% shrinkage is unmodeled,
so late-cook conduction paths stay too long (`docs/model_gaps.md` shrinkage). The
pit's **fast (sub-hour) dynamics are a known deficiency**: burn
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
brisket needs another 90 minutes," which is often the right answer. Cook
time as a decision variable is already implemented as an outer duration
scan with inner schedule optimization (`timeplan.py`, `fullplan.py`).
What remains is differentiable duration (\(s = t/t_f\)); the chance layer
is an empirical done-fraction over \(N\) draws, not a 95% certification.

The natural next step is closing the loop on a physical smoker. The
receding-horizon re-planner is demonstrated in simulation (§2); state — full
T/W/C/F/B/S fields — is first-class Tesseract IO, so
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
