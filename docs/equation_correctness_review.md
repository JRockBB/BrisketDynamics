# Equation and correctness review

Audit of the working tree, 30 Aug 2026, verified finding by finding against
the code the same day; each finding below carries its verdict and what was
done. The coupled monolith is the production path (166 ODEs, FBDF,
ForwardDiff). Picard is the coupling oracle only.

| Core equation forms | Confirmed | Overstated | Wrong | Fixed in code / docs |
|---|---|---|---|---|
| 10 / 10 match | 11 | 3 | 3 | 9 (rest are physics decisions or accepted residues) |

## Verdict

The governing equations that the monolith actually integrates — slab method of
lines with shrinkage, two-sided evaporative BC, pit energy/fuel/ignition,
quasi-steady humidity, AR(1) likelihood, and the KCBS-style objective — match
the intended mathematics and the tests that exercise them. There is no
discretization or sign error that would silently give the wrong stall.

The problems were elsewhere: an equation sheet that disagreed with the
optimizer on the tenderness overshoot weight and with the code on the state
count, a conservation claim that was an identity, an oracle gate looser than
the number the writeup quotes, and two physics closures (water activity in
the evaporative driving force; frozen thermophysical properties) that are
choices the docs described more strongly than the code implements. The VI
finding did not survive verification. Doc, test and dead-code items are
fixed; the two physics closures are decisions, listed at the end.

## Findings

| ID | Sev | Kind | Finding | Where |
|---|---|---|---|---|
| H1 | ~~High~~ Low | Bug → not a bug | Soft-fail is a null step, counted, and never fired in the shipped runs | `scripts/run_vi.py`, `tesseracts/monolith/tesseract_api.py` |
| H2 | High | Physics | Evaporation ignores water activity; drying is a moisture switch | `src/meat.jl` `evap_flux`, `docs/model.md` §1 |
| H3 | High | Docs — fixed | `model.md` stated \(\rho_{ov} = 0.5\); the optimizer uses 1.0 | `docs/model.md` §4, `app/src/brisket_app/control.py` `over_rel` |
| H4 | High | Test — fixed | Cook-loss “conservation to \(10^{-8}\)” was an algebraic identity | `docs/model.md` §1, `test/test_meat.jl` |
| H5 | High | Physics | Frozen \(k\), \(\rho\), \(c_p\) while the writeup says Choi–Okos | `src/thermophysics.jl`, `docs/writeup.md` §4 |
| H6 | ~~High~~ Low | Label — fixed | 720 s is the FSIS 6.5-log line, now labelled so; the dwell term is inactive on every published plan | `app/src/brisket_app/control.py`, `docs/model.md` §4 |
| M1 | ~~Medium~~ — | Wrong | \(t_k = t_{\mathrm{sp}} + k^*\Delta_{\mathrm{sp}}\), so \(\partial t_k/\partial\Delta_{\mathrm{sp}} = k^*\): the interval gradient is real and piecewise smooth | `src/meat.jl` `spritz_pulse_time` |
| M2 | Medium | Physics | Softplus leak: expulsion does not stop at \(W_{\mathrm{eq}}\) | `src/meat.jl` `expulsion_rate`, `docs/roadmap.md` |
| M3 | ~~Medium~~ — | Closure, not a bug — doc fixed | Draft-set air flow is the right closure for a charcoal pit; AFR is now defined as the ratio at full burn | `src/monolith.jl` `_pit_algebra`, `docs/model.md` |
| M4 | Medium | Docs — fixed | State-dimension and \(\theta\)-length comments disagreed with the code | `docs/model.md` §1, README, `coupled.py`, Tesseract schema |
| M5 | ~~Medium~~ — | Wrong | Both displays are the same balance, once solved for the pit input and once for the conduction | `docs/model.md` §1 |
| M6 | Medium | Test — fixed | The K = 8 gate now runs on a 60 s exchange grid at 0.02 K (measured 1.3 / 2.6 mK); the oracle's lid gate uses the monolith's blend width | `test/test_monolith.jl`, `src/pit.jl` |
| M7 | Medium | Physics | Collagen \(E_a\) is low versus cooking literature; one pull anchors the scale | `src/meat.jl` `MeatConfig`, `docs/writeup.md` |
| M8 | Medium | Physics | NUTS is still running; quoted UQ is Laplace / ADVI | `docs/writeup.md` §2, `data/calibration/nuts_mono_real/` |
| L1 | Low | Dead code — removed | Unused `_meat_u0(cfg)` dropped the smoke-ring state | `src/meat.jl` |
| L2 | Low | Physics | Wrap cuts convection by only 20% | `MeatConfig.η_h = 0.2` |
| L3 | Low | Physics | No enthalpy for purge, fat melt, or collagen | `src/meat.jl` interior RHS |
| L4 | Low | Docs — fixed | Section 1 first wrote the no-shrinkage heat equation without saying so | `docs/model.md` §1 |

## What the equations get right

Checked symbol-for-symbol against `src/meat.jl`, `src/pit.jl`,
`src/monolith.jl`, `src/thermophysics.jl`, `app/src/brisket_app/control.py`,
and `calibrate.py`.

| Item | Status | Notes |
|---|---|---|
| Magnus \(p_{\mathrm{sat}}\) | Match | \(610.94\exp(17.625\,T_C/(T_C+243.04))\); ~2.6% high at 100 °C, <1% in the stall band |
| Two-sided evap / Newton humidity | Match | \(j = F(\Delta p - c j)\) with unique root \(1 + c F' > 0\); two Newton steps tested vs bisection to \(10^{-9}\) |
| Shrinkage MOL (center / interior / surface) | Match | Flux-form faces, half-cell surface, \(A/A_0\) on water only. Heat area factor cancels as claimed |
| Wrap / pull / spritz / lid pulses | Match | \(\sigma'\) pulses integrate to the dose; `lid_open` integrates to \(\tau\). `*_NEVER` saturates to 0 |
| Collagen Arrhenius, fat logistic, \(W_{\mathrm{eq}}\) | Match | Referenced Arrhenius at 60 °C; render switch at 55 °C; myosin/actin steps as written. Scale is the issue, not the form |
| Pit 3-state ODE + Minion spread | Match | \(\dot m = r_{\max}\,g(u)\,a\,\tanh(m/m_t)\); \(a_0 = 1\) recovers the two-state model exactly |
| AR(1) likelihood | Match | Stationary AR(1) on contiguous non-lid runs; \(\sigma = 1.2 + \sigma_{\mathrm{extra}}\); \(\rho = 0.95\beta\). \(\rho = 0\) recovers iid |
| Quality \(J\) vs `control.py` | Match* | Bands, KCBS weights, fuel-at-pull, in-pit bark-burn, dual-node \(F\) and dwell match the code. \(\rho_{ov}\) does not (H3) |
| Draft convection \((g/g_{\mathrm{ref}})^{m_{\mathrm{conv}}}\) | Match | Same exponent on \(h_c\) and \(k_m\). \(m_{\mathrm{conv}} = 0\) disables the lever (tested) |
| Forward-mode \(J\) / JVP / VJP | Match | Julia FD gates, endpoint algebra (\(\mathrm{jvp} = Jv\), \(\mathrm{vjp} = J^\top w\)), composed `jax.grad`. Adaptive-step jitter is measured, not hidden |
| System type | Stiff ODE, not a DAE | `ODEProblem` + `FBDF`; no mass matrix, no `DAEProblem`. Algebraic closures are eliminated in the RHS (see below) |

## The monolith is a stiff ODE, not a DAE

A differential-algebraic equation keeps differential states and algebraic
constraints in the system,

\[
\dot x = f(x,y,t),\qquad 0 = g(x,y,t)
\]

or \(M\dot z = f(z,t)\) with a singular mass matrix. The solver then treats
\(y\) (or the null space of \(M\)) as algebraic.

The monolith is posed as a standard `ODEProblem` and integrated with `FBDF`.
The state is only

\[
x = [T,\,W,\,C,\,F,\,B,\,S,\,R,\,T_p,\,m,\,a]
\]

(166 components). There is no mass matrix, no `DAEProblem`, and no algebraic
unknown in \(x\).

What looks algebraic is **reduced inside the right-hand side**, not left as a
constraint:

- Pit humidity is quasi-steady: \(p_{v,\mathrm{pit}} = p_{v,\mathrm{base}} + c\,j\),
  and \(j = F(p_{\mathrm{sat}} - p_{v,\mathrm{base}} - c j)\) is a scalar
  fixed point, taken with a linearized start plus two Newton steps
  (`evap_flux_fb`).
- Burn rate, airflow, and \(p_{v,\mathrm{base}}\) are explicit functions of
  \((u, m, a)\) (`_pit_algebra`).

Those are index-0 closures: the fast algebraic variables are eliminated so
the integrator only sees \(\dot x = f(x,t;p)\). That is why the docs call
humidity “quasi-steady” and the pit quantities “algebraic” — they mean
“computed from the current state,” not “DAE algebraic states.”

A true DAE version would keep \(p_v\) (or \(j\)) as a state with
\(0 = j - F(\Delta p - c j)\) and hand a singular \(M\) to the solver. The
code does not do that.

## High findings

### H1 — Soft-fail is a null step, not a wrong step (verified: not a bug)

**Kind:** Bug → withdrawn · `scripts/run_vi.py`, `tesseracts/monolith/tesseract_api.py`

`run_vi.py` and the NUTS launcher both set `BRISKET_SOFT_FAIL=1` (the NUTS
claim in the original text was wrong). All three endpoints go through one
`_maybe_soft_fail`, which counts every event and prints it. A sentinel value
is a constant, so a zero tangent is its exact derivative: under SVI the draw
moves nothing, under NUTS the trajectory is rejected. Neither walks in a
wrong direction. The shipped VI log and both NUTS logs contain zero
soft-fail events, so the published figures were computed on the true
likelihood throughout. No change.

### H2 — Evaporation ignores water activity; drying is a moisture switch (confirmed; decision)

**Kind:** Physics · `src/meat.jl` `evap_flux`, `docs/model.md` §1

The driving force is \((p_{\mathrm{sat}}(T_s) - p_v)/(R_v T_s)\). \(a_w\) from
the Oswin isotherm is used only for bark and the smoke ring. Surface dry-out
is a logistic on \(W_s\) at \(W_{\mathrm{crit}} = 0.40\), where \(a_w\) is
still \(\approx 0.89\). Thermodynamic drying
(\(j \propto a_w p_{\mathrm{sat}} - p_v\)) and the availability gate are
different mechanisms.

During the wet stall this is nearly harmless (\(a_w \approx 1\)). After the
surface dries, evaporation shuts off while the isotherm still predicts a large
vapor pressure. Stall break and late-cook moisture are shaped by a switch, not
by sorption physics. \(k_m\) and \(D_{\mathrm{eff}}\) absorb the error.

### H3 — `model.md` stated \(\rho_{ov} = 0.5\); the optimizer uses 1.0 (fixed)

**Kind:** Docs · `docs/model.md` §4, `ControlConfig.over_rel` · the equation sheet now says \(\rho_{ov} = 1\)

The documented tenderness overshoot weight is 0.5.
`ControlConfig.over_rel` defaults to 1.0, and the writeup says undershoot and
overshoot are weighted equally. The published 20 h plan used the code default.

Anyone implementing \(J\) from the equation sheet will under-penalize
pot-roast overshoot by \(2\times\) versus the figures. The code and writeup
agree with each other; the equation document does not.

### H4 — Cook-loss “conservation to \(10^{-8}\)” was an algebraic identity (fixed)

**Kind:** Test · `docs/model.md` §1, `test/test_meat.jl` · `test_meat.jl` now
integrates the expulsion sink over the saved fields (trapezoid in x and t) and
compares it with the drip term: 0.3690 vs 0.3696 and 0.6344 vs 0.6354 (dry
basis) on the two fixtures, 0.15–0.17 % — quadrature, with the PDE balance
closed. `model.md` no longer claims \(10^{-8}\).

`drip_w` is defined as \(W_0(1 - w_{\mathrm{retained}}) + \mathrm{spritz}_w -
\mathrm{evap}_w\). The test \(w_{\mathrm{lost}} \approx \mathrm{evap}_w +
\mathrm{drip}_w\) is then tautological. The test file says so; `model.md`
still calls it an “algebraically closed cook-loss budget (tested to
\(10^{-8}\)).” There is no check that `drip_w` equals the integrated
expulsion sink.

A discretization bug in the surface water balance or the expulsion term would
not fail this test. The \(10^{-8}\) figure is not evidence that the PDE
conserves water.

### H5 — Frozen \(k\), \(\rho\), \(c_p\) while the writeup said Choi–Okos (confirmed; wording fixed, physics a decision)

**Kind:** Physics · `src/thermophysics.jl`, `docs/writeup.md` §4 · the writeup
now says "Choi–Okos properties at the initial composition, held fixed"

`K_MEAT = 0.45`, `RHO_MEAT = 1060`, `CP_MEAT = 3400` are Choi–Okos evaluations
at one composition and 60 °C, then held fixed. Heat capacity per physical
volume does not fall as water leaves; shrinkage only rescales capacity through
\(\varsigma\). The founding brief asked for T/composition-dependent
properties.

A drying slab stays as thermally massive as wet meat at the current volume.
Post-stall climb is slower than a composition-aware energy equation.
`k_mult` can hide the bias; it cannot represent the missing T-dependence.

### H6 — Food-safety label (fixed) and stiffness (inactive in practice)

**Kind:** Label · `app/src/brisket_app/control.py`, `docs/model.md` §4 · both now
call 720 s the FSIS Appendix A 6.5-log line. On every published plan the
core dwell is 1.3–1.7 h against the 4 h cap, so the softplus-squared dwell
term is zero; it only bites a cook that actually sits in the danger zone.

\(F_{\mathrm{req}} = 720\) s at 60 °C with \(z = 6\) K matches FSIS Appendix A
for a 6.5-log *Salmonella* reduction in roast beef at 140 °F, not the claimed
~7-log. The 4 h danger-zone cap is the USDA 40–140 °F rule. Both terms enter
\(J\) as softplus-squared penalties with weight 500.

A long cool cook spends hours with the core below 60 °C; `w_dwell = 500` can
dominate appearance/taste/tenderness. The fullplan scan rejects raw residuals;
MPC only penalizes. Neither is a certification, and the 1D-slab cold spot is
not a packer point.

## Medium findings

### M1 — Spritz interval uses `round()` (wrong)

\(k^* = \max(\mathrm{round}((t - t_{\mathrm{sp}})/\Delta_{\mathrm{sp}}), 0)\)
picks the nearest pulse and its derivative is zero almost everywhere — but
the pulse time is \(t_k = t_{\mathrm{sp}} + k^*\Delta_{\mathrm{sp}}\), so
\(\partial t_k/\partial\Delta_{\mathrm{sp}} = k^*\): every pulse after the
first shifts in proportion to its index, and \(\partial J/\partial
\Delta_{\mathrm{sp}}\) is the piecewise-smooth sum of those shifts. The
Tesseract JVP test carries a \(\Delta_{\mathrm{sp}}\) tangent and matches
the Jacobian at 1e-8 tolerances. The interval's gradient was zero in the
published plan because no pulse fired, not because of `round`.

### M2 — Softplus leak at \(W_{\mathrm{eq}}\)

\(\mathrm{sp}_\delta(W - W_{\mathrm{eq}}) = \delta\log(1+e^{(W-W_{\mathrm{eq}})/\delta})\)
equals \(\delta\ln 2\) at equality, so the sink is \(\sim k_e\delta_e\ln 2\)
even when \(W = W_{\mathrm{eq}}\). Acknowledged in the roadmap; calibrated
\(k_e\) absorbs the offset. Long holds slowly dry below the stated
water-holding equilibrium.

### M3 — Air mass flow sized at \(r_{\max}\), not \(\dot m_b\) (closure, not a bug)

Ventilation is \(\mathrm{AFR}\cdot r_{\max}\cdot g(u)\): the draft through
a charcoal pit is set by the damper and the chimney, not by how much of the
bed is lit, so a spreading or tapering bed burns leaner in the same air.
That is the physical picture; AFR is the air-to-fuel ratio at full burn and
`model.md` now says so. Combustion water uses \(\dot m_b\) as it should.
Early-cook humidity being lower than a burn-proportional ventilation would
give is the consequence of that closure, not an error in it.

### M4 — State-dimension and \(\theta\)-length comments disagreed with the code (fixed)

Code integrates \(4\cdot 40 + B + S + R + 3\) pit \(= 166\) ODEs, \(\theta\)
length 8. `model.md` said 162 meat ODEs (no ring), README listed \(B\), \(S\)
but not \(R\), `coupled.py` said “Meat (7)”, the Tesseract `theta` Field
omitted `m_conv`. All four now match the code (163 meat + 3 pit, \(\theta\)
of 8 including `m_conv`). The length checks were always correct
(`MEAT_NP = 8`).

### M5 — Rest-section \(q_{\mathrm{surf}}\) is the same symbol (wrong)

\(q_{\mathrm{surf}} = k\partial_x T|_s + \lambda j\) is the surface energy
balance solved for the pit's convective + radiative input (the glossary
defines \(q_{\mathrm{surf}}\) as exactly that, `surface_q_pit_blended`);
\(k\partial_x T|_s = (1-p)\,q_{\mathrm{surf}} + p\,q_{\mathrm{rest}} -
\lambda j\) is the same balance solved for the conduction with the rest
blend applied to the input. One symbol, one meaning, consistent with the
code.

### M6 — Picard–monolith test was \(50\times\) looser than the writeup (fixed)

The 0.04 K in the writeup is the real-cook-grid benchmark; the test's 2 K was
the interpolation floor of a 10 min exchange grid on a cold-start fixture,
not a coupling tolerance. The test now also runs K = 8 on a 60 s exchange
grid, where the oracle matches the monolith to 1.3 mK (pit) and 2.6 mK
(core), gated at 0.02 K — a coupling regression of hundredths of a kelvin
fails it. The oracle's spritz-lift stop now blends over the same width as the
monolith's wrap / pull blend (`δ_spritz_stop`), so the two gates coincide
whenever the wrap and the pull are more than a few blend widths apart.

### M7 — Collagen \(E_a\) is low versus cooking literature

\(E_a = 173\) kJ/mol, \(k_{\mathrm{ref}} = 10^{-6}\) s\(^{-1}\) at 60 °C.
Cooking-collagen hydrolysis is often quoted at 300–600 kJ/mol. At these
numbers, \(\tau \approx 8\) h at 80 °C and \(\approx 0.7\) h at 95 °C. The
real-cook pull (19.0 h, 89.5 °C) lands at \(C = 0.874\) against a 0.90
target. The band \(C \in [0.90, 0.95]\) is a literature-shaped prior plus one
pitmaster judgment, not a texture calibration.

### M8 — NUTS is still running; quoted UQ is Laplace / ADVI

The writeup brackets \(\hat r\) / ESS / divergences as to-be-quoted.
Untracked `chain0`/`chain1` files are in the working tree. Robust control and
the plan used Fisher / Laplace draws, not NUTS. The identifiability claim
(meat pinned, pit along \(UA \propto \Delta H_{\mathrm{eff}}\) and
\(r_{\max}\) vs \(u_{\mathrm{leak}}\)) is a curvature statement, not a sampled
posterior. ADVI is known to be under-dispersed on those ridges.

## Low findings

### L1 — Unused `_meat_u0(cfg)` dropped the smoke-ring state (removed)

The 1-arg constructor built \(4n_x+2\) entries while `meat_rhs_core!` reads
`u[4nx+3]`; it had no caller. Deleted; the field constructor (which includes
\(R\)) is the only one left.

### L2 — Wrap cuts convection by only 20%

\(h_c(1 - \eta_h s) \to 0.8\,h_c\) when fully wrapped. Evaporation is cut 90%
and \(\varepsilon\) drops \(0.85 \to 0.10\). The wrap is almost entirely an
evaporation/radiation event. The second plateau near 86 °C in the plan is set
by \(\eta_{\mathrm{evap}}\) (literature-fixed). This cook log is unwrapped
end-to-end.

### L3 — No enthalpy for purge, fat melt, or collagen

Interior water leaves as liquid drip (no \(\lambda\)). Fat rendering and
collagen conversion carry no latent heat. Radiation is
\(\varepsilon\sigma(T_p^4 - T_s^4)\): black enclosure at gas temperature.
Small energy errors relative to the stall’s evaporative sink.

### L4 — Section 1 first wrote the no-shrinkage heat equation (fixed)

\(\partial_t T = \alpha\partial_x^2 T\) was printed as the interior model
without saying that the code integrates the Lagrangian form
\(\rho c_p\varsigma\,\partial_t T = \partial_\xi((k/\varsigma)\partial_\xi T)\)
with \(n_{\mathrm{shrink}} = 1/3\). The display now says it is the fixed-slab
form, which the shrinkage section's equation reduces to at \(A/A_0 = 1\).

## Equation notes that are not bugs

**Surface energy and coupling.** The pit load is the convective plus radiative
part only. Latent heat leaves with vapor and does not return to the gas-energy
equation. That is the right closure for a lumped pit:
\(q_{\mathrm{meat}} = A(t)\,q_{\mathrm{pit}}\),
\(q_{\mathrm{net}} = q_{\mathrm{in}} - \lambda j\) on the meat.

**Humidity algebra.** \(p_v \approx (p_{\mathrm{atm}}/0.622)\,\omega\) is the
small-\(\omega\) mixing-ratio inversion.
\(c = (p_{\mathrm{atm}}/0.622)\,A/\dot m_{\mathrm{air}}\) is the correct
linearization of the meat’s own vapor. The Newton start is the exact
linearization; two iterations are overkill in the cooking range (tested).

**FSIS F-value form.**
\(F = \int 10^{(T-60^\circ\mathrm{C})/6\,\mathrm{K}}\,dt\) on both \(T_s\) and
\(T_{\mathrm{core}}\) is the standard z-value equivalent. 720 s at 60 °C is
the FSIS 140 °F / 12 min 6.5-log line, not 7-log. The integral is
curve-faithful; the label is not.

**Real-cook likelihood.** Masked AR(1) on 10 min samples with a 1.2 K
digitization floor matches the provenance budget (\(\pm 1\)–1.5 °F). Treating
HeaterMeter fan duty as an exact exogenous input is correct for “parameters
given this \(u(t)\)” and is not a generative closed-loop model. Ambient is a
constant 300 K; the log’s ambient channel is unusable (black case in the sun,
up to +35 °F).

## Acknowledged residues (already in `docs/roadmap.md`)

These are real limitations. The project names them. They are not new findings;
they bound what the figures mean.

| Residue | What it implies |
|---|---|
| 1D homogeneous slab | No point/flat split, no edges, no grain. Core node is this geometry’s cold spot only |
| One unwrapped 2013 cook log | Wrap, rest, spritz, bark color, smoke ring, and \(\eta_{\mathrm{evap}}\) are not identified. The log is a flat probe on a packer, unwrapped 19 h |
| Closed-loop fan data | \(UA \propto \Delta H_{\mathrm{eff}}\) and \(r_{\max}\) vs \(u_{\mathrm{leak}}\) are ridges. A 2 h open-loop characterization burn is the fix, not a bigger model |
| Bark / ring / \(\beta_{\mathrm{fat}}\) | Literature-scale constants. \(L^*\) and juiciness are not judge scores. Ring is an output, not in \(J\) |
| One-state pit | Cannot show a 25 K one-minute dome transient. Lid loss is a lumped ~7 K dip of \(C_{\mathrm{th}}\) |
| Quasi-steady humidity | No RH/wet-bulb sensor. AFR, \(y_w\), \(\omega_{\mathrm{amb}}\) are literature constants. Standalone meat uses 2 kPa |
| ForwardDiff jitter | ~\(4\times 10^{-5}\) relative Jacobian move under a last-bit input change at tol \(10^{-6}\). Near an optimum, two exact paths differ by ~0.5% |

## Done, and what is left to decide

Done: H3, H4 (expulsion-integral test), H6 (label), M4, M6 (fine-grid gate,
shared lid blend), L1, L4, the H5 wording, and the M3 definition of AFR.
Withdrawn after verification: H1, M1, M5. Accepted residues already in
`docs/roadmap.md`: M2, M7, L2, L3; M8 resolves when the running chains
finish.

Both physics closures were then implemented (branch `tier2`, to be merged
and recalibrated once the running chains finish):

1. **Water activity in the evaporative driving force** (H2):
   \(j = k_m\,(a_w(W_s)\,p_{\mathrm{sat}}(T_s) - p_v)/(R_v T_s)\) with the
   Oswin isotherm, linear and two-sided, so the pit-feedback root is closed
   form; \(W_{\mathrm{crit}}\), \(\delta_W\), \(\delta_p\) and the clamp
   are gone. The stall break is now set by the interior water supply
   (\(D_{\mathrm{eff}}\)), not a surface switch: the default
   \(D_{\mathrm{eff}}\) moved 7.0e-9 → 3.8e-9 to keep a stall-then-break
   on the test cook. A consequence surfaced in the bark model: the
   symmetric Maillard bell shut browning off on a dried crust, so hot
   cooks browned less than low-and-slow; the bell now has a dry-side floor
   \(g_{\mathrm{dry}} = 0.5\) (Purlis: crust browning starts because the
   crust dries) — a placeholder like the other bark constants.
2. **Composition-dependent \(ho c_p\) and \(k\)** (H5): per node, from
   the Choi–Okos component properties at the node's temperature on its
   water / fat / non-fat-solids composition (ash lumped into protein,
   component densities the shrinkage constants); \(k_{\mathrm{mult}}\)
   still scales \(k\). The frozen \(K_{\mathrm{meat}}\),
   \(ho_{\mathrm{meat}}\), \(c_{p,\mathrm{meat}}\) constants are gone.

## Sources

`src/meat.jl`, `src/pit.jl`, `src/monolith.jl`, `src/thermophysics.jl`,
`src/interpolation.jl`, `test/test_meat.jl`, `test/test_monolith.jl`,
`app/src/brisket_app/control.py`, `calibrate.py`, `coupled.py`, `mpc.py`,
`tesseracts/monolith/tesseract_api.py`, `docs/model.md`, `writeup.md`,
`verification.md`, `roadmap.md`, `data/cook_log/PROVENANCE.md`.

Literature cross-checks: August–Roche–Magnus; FSIS Appendix A (140 °F / 12 min);
Choi & Okos 1986 (frozen evaluation); Purlis Maillard \(E_a\).

Scope: equations and correctness, not a performance review.
