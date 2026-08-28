# Rest / carryover dynamics — review

> Historical review of the rest/pull implementation on the two-Tesseract pipeline; `t_rest`, `check_picard` and the adjoint it discusses no longer exist (docs/monolith.md).

**Date:** 2026-08-27. Read-only review of the working tree. No code changes.
**Scope:** Pull blend, rest boundary condition, quality-at-serve, and how those
reach the Tesseract, coupled cook, objective, and planners.
**Equations of record:** `docs/model.md` (Rest / carryover). **Plan:**
`docs/tier1_plan.md` §1. **Standing residues:** `docs/model_gaps.md`
Limitations → Rest.

Categorization matches the Julia review skill: **Critical** (wrong answer
or silent control mismatch), **Performance**, **Suggestion**, **Nitpick**.

---

## Executive summary

The rest model is the right shape for this codebase. After `t_pull` only
the surface environment changes; carryover, rewetting, continued collagen
conversion, and the board puddle are existing interior physics. The pull
is a smooth p-tail parameter like the wrap, the adjoint and composed FD
gates exist, and `PULL_NEVER` is bitwise the old in-pit cook.

The large quality effect is **doneness at serve ≠ doneness at pull**, not
the temperature bump. That is documented (`model_gaps.md`: modelled
carryover < 0.5 K; collagen scale `k_ref` is the binding uncalibrated
constant) and is why the deadline formulation leans on the hold.

The implementation is not finished at the control boundary:

1. Open-loop planners score quality **at serve** with a 1 h faux-cambro
   rest by default. MPC and `run_mpc.py` still score quality **at the
   pull**, with no rest BC and no pull executor. A plan that pulls early
   so the hold can finish collagen is then *defended* as if doneness were
   due at pull.
2. `opt_pull=True` is the `ControlConfig` default. `--rest-h 0` does
   **not** turn rest off; it only skips extending the grid. Pull-when-done
   still fires inside the cook horizon (~90 % of `t_end` at the initial
   logit).
3. Physics tests check shape, not the magnitudes the plan asked for.
   Carryover is never required to be positive.

Nothing here is a discretization bug in the MOL RHS. The risk is using
the rest-aware plans as if the whole stack consumed them.

---

## What is implemented

### Physics (Julia)

`MeatConfig` rest constants (`src/meat.jl` ~40–53), defaults = faux
cambro:

| symbol | field | cambro default | board (comments / tests) |
|---|---|---|---|
| \(T_r\) | `T_rest` | 293.15 K | same (room) |
| \(h_r\) | `h_rest` | 1.5 W/m²K | 7 |
| \(\varepsilon_r\) | `ε_rest` | 0.10 (foil) | 0.85 |
| \(\eta_r\) | `η_rest` | 0.9 | 0 |
| \(p_{v,r}\) | `p_v_rest` | 1200 Pa | (not overridden) |
| \(\delta_p\) | `δ_pull` | 300 s | — |

Pull blend \(p(t)=\sigma((t-t_{\mathrm{pull}})/\delta_p)\) with the same
`logistic` as the wrap (`(1+tanh(x/2))/2`). `PULL_NEVER = 1e9` saturates
to exact 0 on any cook horizon.

Surface (RHS and observables share helpers, so they cannot disagree):

```
q_in = (1-p) * q_pit_wrapped(Tpit, Ts, h_c, s) + p * q_rest(Ts)
j    = (1-p) * j_pit_fb(...) + p * (1-η_r) * evap_flux(Ts, Ws, k_m·h_r/h_c, p_v_rest)
q_net = q_in - λ j
q_meat = A_meat · A(t)/A0 · (1-p) · q_pit_wrapped   # pit load; zero once pulled
dS × (1-s)(1-p)                                     # no smoke on the board
```

Interior equations are untouched: conduction → carryover; diffusion with
\(j\approx 0\) → surface rewet; one-sided expulsion while \(T>T_{\mathrm{myosin}}\)
→ board puddle; Arrhenius collagen → hold conversion.

`p = [θ | Tpit | smoke | pv_base | imair | t_wrap | t_pull]`. `meat_vjp`
returns `dt_pull` from an explicit tail index (the wrap off-by-one
template).

### Python / control

- Meat Tesseract: `t_pull` differentiable; `rest_t/rest_h/rest_eps/rest_eta`
  plain floats. No `p_v_rest` / `δ_pull` inputs.
- `CookGrids.t_rest`: exchange grid runs to `t_end + t_rest`; damper knots
  span the cook only. Pit interpolant holds the last damper; `q_meat→0`
  so the fire is inert as a load.
- `control.objective`: quality at the last save (serve); fuel at
  `t_pull`; bark-burn cap weighted by `(1-p)`; safety integrated through
  the rest.
- `opt_pull`: `t_pull = t_end · σ(z_p)` with `PULL_Z_INIT = 2.2`
  (σ ≈ 0.90). Default **on**.
- `timeplan.plan_duration`: default `t_rest = 1 h`. `run_fullplan` /
  `run_timeplan` default `--rest-h 1`.
- Calibration (`calibrate.py`, `synth.py`) never sets `t_pull` — correct;
  the cook log ends at the pull, so rest is calibration-neutral.

---

## Findings

### Critical

**C1. MPC defends a rest-aware plan as if quality were due at the pull.**

`docs/tier1_plan.md` §1 v2: rest is the final MPC segment (`t_pull_seg`
mirrors `t_wrap_seg`). That is not built.

- `mpc.plan` forces `opt_pull=False` and never passes `t_pull` /
  `t_rest` / `rest` (`app/src/brisket_app/mpc.py` ~272–281).
  `advance_plant` likewise has no pull.
- `scripts/run_mpc.py` reads `best_t_pull_h` from the fullplan JSON and
  **shortens** `T_F` to that pull, then comments that rest is not
  simulated and quality is at the pull for all four cooks (~320–325).
- `executed_objective` bark-burn term is `mean(surf_over²)` with no
  `(1-p)` weight, and `fuel_frac` is `(m0_start - state.m0) / m0_start`
  over the whole executed horizon (`mpc.py` ~204–225, 541). The plan’s
  own risk note (`tier1_plan.md` Risks) was: any report that reads
  `m_fuel[-1]` must read at the pull.

Consequence: fullplan / timeplan pull early so \(C_{\mathrm{serve}}=0.90\)
after the hold. MPC then tries to put \(C=0.90\) **at that earlier
time**, i.e. it cooks through the hold the planner was counting on. The
two stacks answer different questions. Until a pull executor exists,
`run_mpc` should at least score the same functional as the plan (append
`t_rest`, apply the rest BC after the committed pull) even if it cannot
re-time the pull.

**C2. `--rest-h 0` / `t_rest=0` does not disable rest.**

`ControlConfig.opt_pull` defaults True. `objective` then sets
`t_pull = grids.t_end * sigmoid(z_p)` whenever `z_p` is passed — and
`optimize_schedule` always traces `z_p`, initializing it at 90 % of
`t_end` (`PULL_Z_INIT`). The `t_pull=` argument is ignored in that mode
(`control.py` ~255–256, 486, 562).

So:

| knobs | what actually happens |
|---|---|
| `t_rest=1 h`, `opt_pull=True` (timeplan / fullplan defaults) | pull inside the cook, then rest until `t_end + 1 h` |
| `t_rest=0`, `opt_pull=True` (`run_control.py`, `run_wraptime.py`) | pull at ~90 % of `t_end`, rest on the remaining **cook** grid with the pit ODE still running (`q_meat=0`) |
| `t_rest=0`, `opt_pull=False`, `t_pull=1e9` | true in-pit cook to the deadline |

`timeplan`’s docstring (“every candidate cook is PULLED at its duration”)
describes v1. v2 is on by default, and the `t_pull = t_f` it passes is
dead code under `opt_pull`.

`run_control.py` constructs `ControlConfig(n_iter=...)` with no rest grid
and default `opt_pull`. That script now optimizes a pull-when-done cook
whether or not the operator asked for a hold.

Fix shape: either default `opt_pull=False` and turn it on only in the
planners that append a rest, or make `t_rest=0` force `PULL_NEVER` unless
the caller opts in. Do not leave “no rest” as a three-flag conjunction.

---

### Suggestion (correctness / physics)

**S1. Carryover test never asserts a rise.**

`test/test_meat.jl` “rest / carryover (pull blend) physics” (~234–297):

- `PULL_NEVER` bitwise identity — good.
- Pre-blend core matches — good.
- `rise_hf >= rise_lo` — true even if both are ≤ 0.
- Core eventually cooler; board cooler than cambro by 5 K after 4 h —
  cooling, not carryover.
- Collagen `+0.02` over a **4 h** rest from a 10 h / 110 °C-pit cook,
  vs the plan’s `+0.05` for **1 h** from 88 °C.
- Rewetting and `q_meat≈0` after `t_pull+8δ` — good.
- Planned `drip_w` still growing while the interior is above myosin —
  not tested.

`model_gaps.md` already records modelled carryover < 0.5 K. That is
plausible for a stalled, nearly isothermal brisket (Blonder’s steak-scale
carryover does not apply). The test should assert the sign and a loose
band (e.g. cambro rise ∈ [0, 2] K on a hot-fast pull; board rise smaller)
so a BC that only cools cannot go green. The 4 h / +0.02 collagen gate
does not protect the “1 h hold is why the deadline formulation works”
claim.

**S2. Cambro heat loss likely double-counts radiation to the room.**

`surface_heat_rest` is convection to `T_rest` plus radiation to
`T_rest` with foil ε = 0.10. `h_rest = 1.5` is already an *effective*
coefficient through towel + cooler to room air. The foil does not see
20 °C sky; it sees warmer cooler walls.

Order-of-magnitude at \(T_s=363\) K, \(T_r=293\) K:

- \(q_{\mathrm{conv}} \approx 1.5\times(-70) = -105\) W/m²
- \(q_{\mathrm{rad}} \approx 0.10\,\sigma(T_r^4-T_s^4) \approx -57\) W/m²

Radiation is ~35 % of the cambro sink. Folding it into the effective \(h\)
(or radiating to a wall temperature closer to the meat) would slow the
hold, raise carryover a little, and leave more of the collagen window
open. This is a literature-preset residue, not a coding error; it is a
plausible contributor to “carryover < 0.5 K”. Profile `ε_rest ∈ {0, 0.10}`
on a hot-fast pull before treating the number as physics.

**S3. Rest humidity is room air, not a closed cooler, and is not overridable.**

`p_v_rest = 1200 Pa` (~50 % RH at 20 °C). The Tesseract rest dict exposes
T, h, ε, η but not \(p_{v,r}\). A sealed cambro heads toward saturation;
residual evaporation `(1-η_r)·k_m·h_r/h_c` into 1.2 kPa room air is a
small extra dry-out. Directionally conservative for juiciness at serve.
Add `rest_pv` as a plain input so a warming-cabinet / cambro / board
preset is a complete tuple. Do not promote it to θ (no observable).

**S4. Heat–mass analogy divides by \(h_c\) with no floor.**

`k_m * h_rest / h_c` (`meat.jl` ~291). Correct Chilton–Colburn scaling of
the *calibrated* \(k_m\) — keep it. \(h_c=\theta_3\) is unconstrained in
the posterior. A near-zero draw (or a Dual at 0) blows up rest
evaporation. Softclamp or `h_c + ε` on this ratio only.

**S5. Rest BC replaces wrap; it does not compose with it.**

After `p→1`, wrap blend `s` no longer affects heat or evaporation. That
is the intended “rest preset is the environment.” It means:

- Unwrapped cook + cambro rest: foil/towel appear at the pull (the
  300 s blend is documented as “pull + wrap for the hold”). Good.
- Wrapped cook + board rest (`η_rest=0`): foil vanishes at the pull even
  if `s=1`. Callers must set the rest dict to how the meat is actually
  held, not inherit wrap state.

No named presets in code — only comments. A small `REST_CAMBRO` /
`REST_BOARD` / `REST_CABINET` dict in Python would stop the board/cambro
constants drifting between tests and scripts.

**S6. `T_rest` is not the cook’s ambient.**

Default 293.15 K. Coupled ICS typically use `t_amb = 300 K`. ~7 K extra
driving force on a 1.5 W/m²K cambro is small (~10 W/m²) but the rest
environment should default to the same ambient the pit used, unless the
preset is a warming cabinet.

**S7. Diagnostic `pv_pit` ignores the pull.**

`meat_solve` builds `pv_pit` from `evap_flux_fb` and pit `pv_base` with
no `pl` (`meat.jl` ~1000–1004). After the pull the surface sees
`p_v_rest`. Plots of “what the surface saw” over a rest-extended horizon
are wrong in the tail. Gate with `(1-p)` or splice `p_v_rest`.

**S8. Maillard is not gated by `(1-p)`; smoke is.**

Smoke deposition × `(1-s)(1-p)` matches the plan. `dB = bark_rate(Ts, aw)`
keeps running. A cambro rest that rewets toward the Maillard aw peak
while the surface is still ~80–90 °C will keep dropping \(L^*\).
Physically arguable (the meat is still hot) and unstated. Either gate it
or one sentence in `model.md` that appearance at serve includes hold
browning.

**S9. Pull smear is ~20 min of mixed BCs.**

Gaps doc: blend smears over \(\pm 4\delta_p \approx 20\) min. For that
window the meat still takes a fraction of pit heat while already leaking
to the room. That cuts the already-small carryover. Same philosophy as
the wrap (δ = 600 s). Fine for AD; do not read `T_core(t_pull)` as
“the instant it left the pit”.

**S10. No composed / objective tests for the rest *grid* or fuel-at-pull.**

`test_composed_grad.py` `test_grad_tpull` pulls at 4 h on a 6 h grid
with `t_rest=0`. That is the rest *BC* through Picard, not
`CookGrids.t_rest`, not `opt_pull`, not `jnp.interp` fuel, not the
`(1-p)` bark-burn weight. A cheap gate: `objective` with `t_rest>0` and
a pull inside the cook, FD on `z_p`, plus an assertion that
`m_fuel[-1] ≠ m_at_pull` when the pit keeps burning after the pull.

**S11. `over_rel = 0.5` is still the “early finishes rest” fudge.**

The plan’s prediction was that a modelled hold makes `over_rel` stop
being a stand-in. It is still 0.5, and MPC still needs it more than
fullplan does. Once MPC consumes rest, revisit whether the overshoot
discount is still doing the hold’s job.

---

### Performance

**P1. Pit ODE runs through the rest with a frozen damper and `q_meat≈0`.**

Documented and load-correct. Extra pit integration over `t_rest` (and
over `t_end - t_pull` when `opt_pull` fires) is cheap next to the meat
solve. Not worth a split-horizon unless rest becomes hours of warming
cabinet. Do not read `m_fuel[-1]` or `tpit[-1]` as cook-end quantities
(`tier1_plan.md` Risks — still live).

No type-stability or allocation issue specific to the pull tail; it is
two extra `p` entries and a logistic.

---

### Nitpick

**N1.** `coupled_cook` docstring: “default = never pull — the meat then
rests under the rest environment” (`coupled.py` ~100–102). Never-pull
means never rest. The next clause is the one that is true.

**N2.** `timeplan.plan_duration` seed path builds `grids_prev` without
`t_rest`. Harmless today (`u_tknots` span `t_end` only) but it will
bit-rot if damper knots ever cover the hold.

**N3.** Python `DELTA_PULL_S = 300.0` duplicates `MeatConfig.δ_pull`.
Same pattern as wrap. A Julia/Python drift mis-weights the bark-burn cap
only, not the physics.

**N4.** `jax.nn.sigmoid` and Julia `logistic` are the same function.
Bark-burn `(1-p)` in the objective matches the meat blend. Good.

**N5.** Warming-cabinet preset (335 K, h = 5, ε = 0.85, η = 0.9) exists
only in comments (`meat.jl`, `tier1_plan.md` table). Fine until a demo
wants it.

**N6.** `chance_of_done` docstring still says “at `t_f`”; the body
replays pull + rest and scores serve. Comment vs code.

---

## What is solid

- **Bitwise off switch.** `PULL_NEVER` / omitted `t_pull` recover the
  in-pit cook exactly (Julia `==` tests; Python
  `test_pull_default_backward_compat`). Calibration is untouched.
- **Shared surface helpers.** RHS, `_meat_core` observables, and
  `meat_solve` all call `surface_j_blended` / `surface_q_pit_blended`.
  The wrap lesson (unblended `j_evap` silently poisoning `t_wrap`
  gradients) was applied.
- **p-tail hygiene.** Explicit `MEAT_NP + 4nk + 2` indexing; FD on
  `t_pull` with off-by-one guards on `dt_wrap`, θ, and a mid knot
  (`test/test_gradients.jl` “pull-time (rest) adjoint vs FD”, relerr
  < 2e-3 at a 22 s step inside `δ_pull`).
- **Composed VJP.** `t_pull` through K=2 Picard, pit second pass seeing
  `q_meat→0` (`test_grad_tpull`).
- **Interior reuse.** No new states, no events, no rest-specific
  collagen/moisture ODEs. The hold’s collagen leverage is the existing
  Arrhenius evaluated at hold temperatures — the point of the design.
- **Smoke off the board; pit unloaded.** Both tested after `t_pull+8δ`.
- **Objective split (when rest is actually on).** Fuel at pull,
  bark-burn in the pit, quality and safety through the hold. That split
  is the right one; MPC simply does not use it.
- **Config, not θ.** Rest constants are literature presets. Correct
  given the log ends at the pull.

---

## Test plan vs tests (tier1_plan.md §1)

| planned | present | gap |
|---|---|---|
| 1. `PULL_NEVER` ⇒ bitwise identity | yes | — |
| 2. Carryover rise > 0.5 K, peak 15–45 min, board faster fall | shape only; no positive rise; no peak-time | S1 |
| 3. `ΔC > 0.05` for 1 h cambro from 88 °C | `ΔC > 0.02` over 4 h from an unspecified core | weaker than the claim |
| 4. Rewetting; `drip_w` still grows | rewetting only | drip untested |
| 5. `q_meat ≈ 0` after `t_pull+4δ` | after `+8δ`, 1 % of in-pit max | ok |
| 6. FD `t_pull` + p-tail guards + composed `test_grad_tpull` | yes | rest *grid* / objective untested (S10) |

---

## Control-stack map (who actually rests)

| entry point | rest grid | `opt_pull` | quality evaluated at |
|---|---|---|---|
| `run_fullplan` / `run_timeplan` | 1 h default | True (default) | serve |
| `run_robust_control` | from plan JSON `rest_h` | True | serve (expected J) |
| `run_control` / `run_wraptime` | none | **True** | last cook knot, after an in-horizon pull |
| `mpc.plan` / `run_mpc` | none; horizon cut to pull | **False** | pull |
| calibration / synth | none | n/a | n/a (log ends at pull) |

The interesting engineering is done on the meat side. The remaining
defect is that “quality at serve” is a property of some scripts, not of
the model-plus-controller as a whole.

---

## Recommended order (no work in this pass)

1. Make “no rest” a single switch (`opt_pull=False` default, or
   `t_rest=0` ⇒ `PULL_NEVER`). Stop `run_control` from silently
   pull-when-done.
2. Either implement the MPC pull segment or stop truncating `run_mpc` to
   `best_t_pull_h` without appending the hold the plan optimized for.
   Today those two programs disagree about when \(C=0.90\) is due.
3. Tighten the Julia physics test to the plan’s carryover sign/peak and
   1 h collagen ΔC, and add `drip_w`.
4. Expose `rest_pv`, named presets, and `T_rest` defaulting to ambient.
   Profile cambro `ε_rest` if carryover remaining < 0.5 K still bothers
   the writeup — collagen, not the bump, is the control-relevant term.
5. One composed gate on `objective` with `t_rest>0` (fuel-at-pull +
   `z_p`).

Do not add rest constants to θ. Do not treat optimizer output as a USDA
cook; safety through the rest is still a surface softplus
(`model_gaps.md` #4).
