# Submission package (drafts)

Author notes (not posted): fill the NUTS-on-monolith diagnostics from
`scripts/merge_nuts_mono.py` and the robust-control numbers from
`data/control/schedule_robust.json` where marked. Repo:
https://github.com/JRock95/BrisketDynamics. User-owned: hackathon
registration (tally.so/r/kdx2ej, required for prize eligibility).

---

## 1. Submission form

**Project name:** Differentiable BBQ — BrisketDynamics

**Track:** 04 — Differentiable inference & UQ (cross-track: 02 —
Multi-physics & coupled systems)

**Repo:** https://github.com/JRock95/BrisketDynamics

**One-line description:**
A charcoal smoker and a brisket — a lumped pit ODE, a stiff 1D heat–moisture
PDE with the evaporative stall, and the pit's humidity — as one Julia
Tesseract with forward-mode `jvp` / `vjp` / `jacobian` endpoints, driven from
JAX for Gauss–Newton and NUTS calibration on a real 19-hour cook log, a
duration × wrap × pull × chimney × spritz × vent plan scored at serve, robust
control over the posterior, and receding-horizon MPC that re-times the wrap,
the pull, the chimney and the spritz policy.

**Longer description (if the form has a field for it):**
Real barbecue is a coupled multi-physics control problem: a charcoal pit with
its own thermal and fuel dynamics drives a brisket whose internal heat
conduction and moisture transport produce the famous "stall" — a multi-hour
plateau caused by evaporative cooling. The physics is Julia: a 166-ODE stiff
system (method of lines, FBDF) integrating pit, meat and pit humidity
together, with smoothed switches so the stall emerges and breaks
differentiably. It is exposed as one Tesseract whose derivative endpoints are
forward-mode through the solve — a JVP is one Dual solve, a VJP or full
Jacobian one chunked pass — so tesseract-jax gives JAX both reverse and forward
mode through the same interface.

The derivatives do real work. A Gauss–Newton MAP on a digitized real 19-hour
overnight cook (18.5" WSM) converges from the prior medians in 17 iterations
and about two minutes, tracking the pit hold to ~1.3 K rms and reproducing
the core's rise, stall and break; the Fisher information from the same
Jacobian — the curvature a finite-difference Hessian cannot resolve at that
optimum — shows which physics the data pins (the meat's) and which it identifies only along
ridges (the pit's: heat loss vs heat release, burn rate vs leak). The plan —
cook length, wrap time, pull time and vent schedule optimized jointly against
a KCBS-weighted quality objective scored at serve — lands on a 20 h cook that
wraps at 8.1 h and pulls at 18.4 h, with no fuel added and no spritzing. Receding-horizon MPC replans the
schedule, the wrap, the pull, the chimney and the spritz policy hourly from
the full meat field; hit with a
15 K cold front it has never seen, it re-times the crutch and the pull and
serves at doneness 0.964 where the open-loop plan ends at 0.54.

Everything — the three-level finite-difference verification, the deadlocks
found and fixed in the Julia↔JAX threading, the model's deficiencies
(closed-loop cook logs cannot identify the pit's fast dynamics) — is
documented in the repo.

**Key artifacts:** `docs/writeup.md` (six-criteria writeup) ·
`docs/monolith.md` (the seam, measured) · `figures/cooklog_fit_map.png` ·
`figures/corner_laplace_real.png` · `figures/fullplan.png` ·
`figures/mpc_demo.png` · `figures/control_robust.png` ·
`figures/bark_model.png` · `docs/verification.md`

---

## 2. LinkedIn post

Can you take the gradient of a brisket?

For the Tesseract Hackathon by Pasteur Labs & ISI, I built BrisketDynamics:
a charcoal smoker and a brisket — a pit ODE, a stiff heat–moisture PDE (the
physics behind the infamous BBQ "stall"), and the pit's humidity — as one
differentiable Julia Tesseract driven from JAX.

What the derivatives bought:

🔥 Calibration against a real 19-hour overnight cook digitized from a 2013
forum post — the model reproduces the full arc: rise, five-hour stall, break,
climb to 90 °C. A Gauss–Newton fit from the prior medians takes two minutes;
the posterior says which physics a cook log pins (the meat's) and which it
only constrains in combinations (the pit's).

🔥 A plan that picks its own cook length, wrap time, pull time and vent
schedule, scored at serve — the rest is part of the cook.

🔥 Receding-horizon MPC that re-plans the vents, the crutch and the pull
hourly from the full meat field. Hit the simulated cook with a cold front the
controller has never seen: the open-loop plan serves a tough brisket, MPC
wraps earlier, pulls earlier, and lands it.

Repo: https://github.com/JRock95/BrisketDynamics

#TesseractHackathon #DifferentiableProgramming #JuliaLang #JAX
#BayesianInference #BBQ

[Attach: figures/mpc_demo.png]
[Tag: Pasteur Labs & ISI; Tesseract]

---

## 3. Forum showcase post (si-tesseract discourse)

**Title:** BrisketDynamics — calibration & MPC of a coupled smoker–brisket
system as one Julia Tesseract with forward-mode jvp / vjp / jacobian endpoints

Hi all — my hackathon entry wraps a coupled BBQ simulator (charcoal pit ODE +
1D heat–moisture PDE with the evaporative stall + pit humidity, 165 stiff
ODEs) as one Tesseract and uses the derivatives from JAX for Bayesian
calibration on a real 19-hour cook log and for planning and MPC.

**The boundary, and why it is where it is.** The Tesseract sits at the edge
of the physics: the pit ODE, the meat PDE and the humidity balance are one
Julia solve, and forward mode inside is cheap because the parameter count is
small (a 15-column Jacobian on the real cook in 2.5 s). The monolith
Tesseract exposes `jacobian_vector_product` (one Dual solve),
`vector_jacobian_product` and `jacobian` (one chunked pass), so tesseract-jax
gives `jax.grad` for Optax/NUTS and `jacfwd` for Gauss–Newton through the same
interface.

**What the derivatives do.** Gauss–Newton MAP on the real cook from the prior
medians in 17 iterations / 136 s; a Laplace on the Gauss–Newton curvature
(a finite-difference Hessian is indefinite there); a duration × wrap × pull ×
chimney × spritz × vent plan scored at serve (13 cook
lengths in 59 min); receding-horizon MPC replanning vents, crutch, pull, chimney and spritz
hourly (4 cooks in 43 min) that absorbs a 15 K cold front the open-loop plan
does not survive; robust control over posterior draws.

**Notes that might be useful to others building Julia Tesseracts:**
- If your parameter count is small and the forward solve is cheap, forward
  mode makes the endpoint discipline free: `jvp`, `vjp` and `jacobian` are
  each one Julia call, and tesseract-jax already has the JVP rule.
- juliacall + jitted JAX callbacks: one executor thread per process for all
  Julia work, and every Julia→numpy array copied on that thread — a
  zero-copy view released from an XLA thread deadlocks the GC; a second
  executor deadlocks the moment two modules share the runtime. Both are
  documented in the repo.
- numpyro derives a `TransformedDistribution`'s support from the transform's
  codomain; an `AffineTransform` of a Beta needs `domain=unit_interval` or
  the guide samples outside the support.

Repo: https://github.com/JRock95/BrisketDynamics — writeup in
`docs/writeup.md`, the seam measurements in `docs/monolith.md`, every figure
regenerable by script. Feedback welcome!
