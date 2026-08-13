# Differentiable BBQ: Bayesian Calibration and Optimal Control of a Coupled Smoker–Brisket System

**Track 04 — Differentiable inference & UQ** (cross-track: 02 — Multi-physics & coupled systems)
**Repo:** https://github.com/JRock95/BrisketDynamics · Apache 2.0 · all code authored Aug 3–31, 2026

Real barbecue is a coupled multi-physics control problem. A charcoal pit with
its own thermal dynamics drives heat into a brisket whose internal physics —
heat conduction and moisture transport with surface evaporation — produce the
famous **stall**: a multi-hour plateau in internal temperature caused by
evaporative cooling. We built both solvers in Julia, composed them through
Tesseract into a single end-to-end differentiable function under JAX, and used
the gradients four ways: MAP and full NUTS Bayesian calibration (synthetic
recovery study **and** a real 19-hour cook log), damper-schedule optimization,
and schedule optimization *in expectation over the posterior* — control under
calibrated uncertainty. This document follows the six judging criteria in
order.

## 1. Composition across a real boundary

Two Julia Tesseracts with **deliberately different differentiation
strategies**, composed in JAX via tesseract-jax:

| Component | Physics | Differentiation |
|---|---|---|
| `brisket-meat` | 1D slab heat+moisture PDE, 120 stiff ODEs (method of lines, FBDF), smoothed evaporative surface BC, collagen kinetics | **SciMLSensitivity continuous adjoint** (`InterpolatingAdjoint` + `EnzymeVJP`), exposed as the Tesseract `vector_jacobian_product` |
| `brisket-pit` | lumped charcoal pit, 2 ODEs (energy balance, fuel burn with smooth burnout taper) | **exact dense ForwardDiff Jacobian**, transposed to a VJP |

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
`jax.grad(loss ∘ coupled_cook)` — **13/13 pass at 1e-10 tolerances**
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
The poster described his door as leaky; the posterior put the leak parameter
at u_leak = 0.22. Honest residual: a ~3 °C pit overshoot in hours 2.5–5.5.

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
- Upstream finding to be filed against tesseract-jax: the JVP-rule input
  flattening misaligns when only a subset of differentiable inputs carries
  tangents (minimal repro in repo history).

## 6. Reproducibility & communication

Everything is pinned (`Manifest.toml`, `uv.lock`) and layered so each claim is
one command (`README.md`): Julia physics + gradient tests, boundary tests, the
composed FD gate, each figure's generating script. The native no-Docker dev
loop (`Tesseract.from_tesseract_api`) runs the full science pipeline on
Windows; containerized builds (`scripts/build_containers.sh`, Julia toolchain
baked at build time via `package_data`) provide the reproducible artifact
[TODO: confirm container build + in-container checkgradients before
submission]. Figures: the animated calibrated cook
(`figures/cook_animation.gif`), corner plot, posterior-predictive band,
control before/after, robust-control comparison, real-cook fit.

---

*Data attribution: cook log digitized from a public post by user "Mache" on
The Virtual Weber Bulletin Board (2013), used as a small factual data
extraction with attribution; provenance, method, and uncertainty budget in
`data/cook_log/PROVENANCE.md`.*
