# Submission package (drafts)

Status: READY except one item — confirm the repo URL once the public repo
exists at github.com/JRock95/BrisketDynamics.

User-owned prerequisites: hackathon registration (tally.so/r/kdx2ej,
required for prize eligibility) and creating the public GitHub repo.

---

## 1. Submission form

**Project name:** Differentiable BBQ — BrisketDynamics

**Track:** 04 — Differentiable inference & UQ (cross-track: 02 —
Multi-physics & coupled systems)

**Repo:** https://github.com/JRock95/BrisketDynamics

**One-line description:**
A charcoal smoker and a brisket as two Julia Tesseracts — a stiff
heat–moisture PDE with a SciMLSensitivity continuous adjoint and a lumped
pit ODE with a ForwardDiff Jacobian — composed into one end-to-end
differentiable JAX function and used five ways: MAP + NUTS Bayesian
calibration (synthetic and a real 19-hour cook log), damper-schedule
optimization, robust control in expectation over the posterior, and
receding-horizon MPC that absorbs an unmodeled cold front.

**Longer description (if the form has a field for it):**
Real barbecue is a coupled multi-physics control problem: a charcoal pit
with its own thermal and fuel dynamics drives a brisket whose internal
heat conduction and moisture transport produce the famous "stall" — a
multi-hour plateau caused by evaporative cooling. We built both solvers in
Julia with deliberately different differentiation strategies (continuous
adjoint for the 120-ODE stiff PDE; exact forward-mode Jacobian for the
3-state pit), exposed each through the Tesseract VJP interface, and coupled
them two-way in JAX via tesseract-jax with an unrolled Picard iteration, so
reverse-mode AD through the composed VJPs is the exact gradient of the
simulated system.

The gradients then do real work. NUTS calibration recovers synthetic truth
and — the capstone — fits a digitized real 19-hour overnight cook (18.5"
WSM), tracking the pit hold within ~3 °C and reproducing the core's full
arc: rise, multi-hour stall, break at ~14 h. The full NUTS posterior over all 12 physical
parameters shows exactly which physics the data pins: meat-side parameters
to within a few percent, pit-side parameters only in combinations — two
clean ridges (heat loss vs heat release, burn rate vs leak) that are the
fingerprint of closed-loop cook data — while the posterior-predictive band
holds both 19-hour traces. The same coupled function
drives Optax damper-schedule optimization (doneness 0.900 on a 0.90
target), robust control over posterior draws (half the doneness spread of
the MAP-optimal schedule on held-out draws), and a receding-horizon MPC
demo with full-field state handoff through the Tesseracts: against an
unmodeled −15 K cold front, MPC lands doneness 0.92 vs open-loop's 0.68.

A three-tier gradient verification gate (Julia adjoint vs FD,
per-endpoint check-gradients, composed FD through the full pipeline —
15/15) backs every result, and the writeup names the model's deficiencies
honestly, including why closed-loop cook logs provably cannot identify the
pit's fast dynamics. To our knowledge — based on a survey of the mosaic
suite, the official examples, and the forum — no public example existed of
a SciMLSensitivity continuous adjoint exposed through the Tesseract VJP
interface.

**Key artifacts:** `docs/writeup.md` (six-criteria writeup) ·
`figures/cook_animation.gif` · `figures/cooklog_fit_map.png` ·
`figures/corner.png` · `figures/mpc_demo.png` ·
`figures/composition_diagram.png` · `docs/verification.md`

---

## 2. LinkedIn post

Can you take the gradient of a brisket?

For the Tesseract Hackathon by Pasteur Labs & ISI, I built
BrisketDynamics: a charcoal smoker and a brisket as two differentiable
physics components — a stiff heat–moisture PDE (the physics behind the
infamous BBQ "stall") and a charcoal-pit ODE — written in Julia, composed
through Tesseract into one end-to-end differentiable function under JAX.

What the gradients bought, flowing across the Julia↔JAX container boundary
on every step:

🔥 Bayesian calibration (NumPyro NUTS) against a real 19-hour overnight
cook digitized from a 2013 forum post — the model reproduces the full arc:
rise, five-hour stall, break at hour 14. The calibration even found the
poster's leaky smoker door.

🔥 An optimized vent schedule that hits target doneness on deadline —
and a robust version optimized in expectation over the posterior.

🔥 Receding-horizon MPC: hit the simulated cook with a cold front the
controller has never seen, and it re-plans hourly — doneness 0.92 vs 0.68
for the open-loop schedule. My favorite part: the asymmetric deadline
objective makes the controller bank heat early as insurance, then ease off
once the cook is won — nobody programmed that.

The interesting engineering: a SciMLSensitivity continuous adjoint exposed
as a Tesseract VJP (to my knowledge a first in public code), two-way
coupling by an unrolled Picard iteration so the composed gradient is exact,
and a 15/15 finite-difference verification gate on every parameter.

Repo: https://github.com/JRock95/BrisketDynamics

#TesseractHackathon #DifferentiableProgramming #JuliaLang #JAX
#BayesianInference #BBQ

[Attach: figures/mpc_demo.png or figures/cook_animation.gif]
[Tag: Pasteur Labs & ISI; Tesseract]

---

## 3. Forum showcase post (si-tesseract discourse)

**Title:** BrisketDynamics — Bayesian calibration & MPC of a coupled
smoker–brisket system (two Julia Tesseracts, SciMLSensitivity adjoint as
VJP)

Hi all — my hackathon entry composes two Julia Tesseracts into an
end-to-end differentiable BBQ simulator, then uses the gradients for
Bayesian calibration on a real 19-hour cook log and for optimal control.

**The composition.** `brisket-meat` is a 1D heat–moisture PDE (method of
lines, 120 stiff ODEs, FBDF) whose smoothed evaporative surface BC
produces the stall; its `vector_jacobian_product` endpoint is a
SciMLSensitivity continuous adjoint (`InterpolatingAdjoint` + `EnzymeVJP`).
`brisket-pit` is a 3-state charcoal-pit ODE (energy balance, fuel burn,
Minion-method ignition spread) differentiated with an exact ForwardDiff
Jacobian, transposed to a VJP. A JAX orchestrator couples them two-way
(pit temperature → boundary condition; evaporative + sensible load → pit
energy balance) with a K=2 Picard iteration unrolled in the graph, so
`jax.grad` through the composed VJPs is the exact gradient of the
simulated system. One gradient crosses the Julia↔JAX boundary four times.

**What the gradients do.** NumPyro NUTS calibration (12 latents) against a
digitized real overnight WSM cook — pit hold tracked within ~3 °C, the
stall and its break reproduced, and physically interpretable parameters
(the poster mentioned his door leaked; the leak parameter calibrated to
0.21). The full posterior pins the meat-side physics to a few percent but
identifies the pit side only along two clean ridges (heat loss vs heat
release, burn rate vs leak) — the identifiability fingerprint of
closed-loop data, quantified. Optax damper-schedule
optimization, a robust variant in expectation over posterior draws, and a
receding-horizon MPC demo — full T/W/C-field state handoff through the
Tesseracts' IO — that absorbs an unmodeled −15 K cold front (doneness 0.92
vs 0.68 open-loop).

**Notes that might be useful to others building Julia Tesseracts:**
- To our knowledge there was no public SciMLSensitivity-adjoint-as-VJP
  example (the mosaic Julia precedents use Zygote or hand-written
  adjoints); ours is documented in `docs/verification.md`, including a
  gotcha where compiled ReverseDiff tapes silently freeze time-interpolated
  forcings, and a GaussAdjoint NaN-quadrature failure mode.
- juliacall + jitted JAX callbacks: all Julia work must run on one
  dedicated thread; three distinct deadlock modes and their fixes are
  documented in the repo.
- Checkpointing NumPyro at the kernel level (`HMCState` pickles cleanly)
  made sampling survive four battery deaths with zero lost samples.

Repo: https://github.com/JRock95/BrisketDynamics — writeup in
`docs/writeup.md`, every figure regenerable by script. Feedback welcome!
