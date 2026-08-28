# Tier-1 gaps: rest / carryover, shrinkage, humidity feedback

> Design + implementation plan (2026-08-26) for `docs/model_gaps.md` Tier 1
> (gap numbers below are as of that date: #1 shrinkage, #2 rest, #3
> humidity). Equations of what exists: `docs/model.md`. Same shape as
> `docs/bark_plan.md` / `docs/moisture_plan.md`: physics first, then where
> it lands, then the tests, then what it changes.
>
> **Status.** §1–§3 are built (Julia core, meat Tesseract, `coupled.py`,
> `control.objective` with `opt_pull`, planners; Julia + composed FD
> gates). MAP-real refit under the full model with ρ capped at 0.95. Open:
> the MPC pull-when-done segment; VI-driven artifacts regenerating.

## 0. Ground rules

1. No compatibility switches: each feature is part of the model. The
   calibrated artifacts (`map_real.json`, VI, NUTS, `data/control/*`) are
   refit in §4; the literature default θ is re-tuned so the test fixtures
   still produce the stall and its break.
2. **Work in a git worktree, not the main checkout.** NUTS-real is live
   (`data/calibration/nuts_ckpt_real/{0,1}`: 2 chains, `done: false`,
   `state.pkl` touched 20:54 today, two ~1 GB python processes). The
   checkpoint runner *re-imports `src/meat.jl` on resume*; a resumed chain
   in an edited checkout would continue under different physics. The
   Tesseract APIs `Pkg.activate` their own `REPO_ROOT`, so a worktree is
   self-contained (`uv sync` once for its `app/.venv`). No fullplan scans
   or refits on this laptop while the chains sample.
3. **Refit blast radius.** Shrinkage (#1) and humidity (#3) change the
   forward model *at the calibration point*; the rest (#2) does not (the log
   ends at the pull). Hence the order **rest → shrinkage → humidity** and
   ONE calibration campaign at the end (§4), not three.
4. Verification doctrine unchanged (`docs/verification.md`): Julia FD per
   new parameter direction with a relative step; composed gate per new
   cross-boundary path; no full check-gradients sweeps until the container
   rebuild. Ask before any run > 10 min.
5. **Config constants, not θ.** No new latents in this pass. Each new
   constant is *profiled* (a few MAP fits at fixed values), not sampled:
   θ 7 → 8 would invalidate every checkpoint and npz again (the 5 → 7
   lesson). Promotion to θ waits for the observable that identifies it
   (ruler/scale for `n_shrink`, wet-bulb probe for humidity).

## 1. Rest / carryover (#2) — smallest, calibration-neutral, moves the control answers most

### Physics

After the pull only the boundary condition changes. Pull blend
$p(t) = \sigma\big((t - t_{\mathrm{pull}})/\delta_{\mathrm{pull}}\big)$,
$\delta_{\mathrm{pull}} = 300$ s (pull + foil + towel takes minutes; also
keeps FD across the switch well-posed, as for the wrap). Surface:

$$
q_{\mathrm{surf}} = (1-p)\,q_{\mathrm{pit}}(T_p, T_s, h_c, s)
+ p\,\big[h_r\,(T_r - T_s) + \varepsilon_r\,\sigma_{SB}\,(T_r^4 - T_s^4)\big]
- \lambda\, j,
$$

$$
j = (1-p)\,(1-\eta_e s)\,k_m\,\varphi\,\Delta\rho_v(T_s;\,p_{v,\mathrm{pit}})
+ p\,(1-\eta_r)\,k_m \tfrac{h_r}{h_c}\,\varphi\,\Delta\rho_v(T_s;\,p_{v,\mathrm{amb}}) .
$$

Rest presets (plain inputs; the practitioner's hold is the cambro):

| preset | $T_r$ | $h_r$ [W/m²K] | $\varepsilon_r$ | $\eta_r$ |
|---|---|---|---|---|
| board (unwrapped, room) | $T_{\mathrm{amb}}$ | 7 (natural convection) | 0.85 | 0 |
| faux cambro (foil + towel + cooler) | $T_{\mathrm{amb}}$ | 1.5 (effective, through insulation) | 0.10 | 0.9 |
| warming oven / holding cabinet | 335 K | 5 | 0.85 | 0.9 (wrapped) |

The rest-phase mass-transfer coefficient scales with $h_r/h_c$ (heat–mass
analogy) so the calibrated $k_m$ keeps its meaning. Smoke deposition
$\times(1-p)$ (no smoke on the board). Interior equations are unchanged and
already do the right things: conduction gives the **carryover** rise;
diffusion with $j \approx 0$ rewets the surface from the interior ("juice
redistributes"); the one-sided expulsion sink stops as $T$ falls
($W_{\mathrm{eq}}(T)$ is a floor, not a memory — the one-sidedness is what
makes cooling well-posed) but keeps purging while the interior is above
$T_{\mathrm{myosin}}$ (the board puddle); **collagen keeps converting** —
at 90 °C, $k_C$ is 174× its 60 °C value ($E_a/R = 20808$ K), so a 1 h hold
converts ~45 % of the remaining collagen (~12 % at 80 °C). That last term is
why "the deadline formulation leans on the hold": doneness at serve ≠
doneness at pull, by a lot.

Observables apply the same blend (`_meat_core` and `meat_solve`, the wrap
lesson): `j_evap`, and `q_meat × (1 − p)` — the pit sees no load after the
pull. `t_pull` is a differentiable p-tail entry exactly like `t_wrap`, so
it rides the verified continuous adjoint with no new VJP machinery.

### Where it lands

- `src/meat.jl`: `MeatConfig` gains `δ_pull, T_rest, h_rest, ε_rest, η_rest,
  p_v_rest`; `p = [θ | Tpit | smoke | t_wrap | t_pull]` (new tail — every
  `p[end]` / `(end-1)` slice audited; the t_wrap off-by-one guard test is
  the template); `pull_blend`; surface BC + `dS × (1−p)`; observables;
  `meat_vjp` returns `dt_pull`; `PULL_NEVER = 1.0e9`.
- `tesseracts/meat/tesseract_api.py`: `t_pull` Differentiable scalar
  (default 1e9), `rest_t / rest_h / rest_eps / rest_eta` plain floats
  (defaults = cambro), VJP `dt_pull`; `abstract_eval` unchanged (no new
  outputs).
- `coupled.py`: the horizon `t_end` is the cook, start to serve; after
  `t_pull` the meat integrates under the rest BC while the pit keeps
  burning (`q_meat = 0`); `t_pull` passthrough (default never) inside the
  `optimization_barrier` tuple.
- `control.py`: `opt_pull` flag; `t_pull = t_end · σ(z_p)`; **fuel at the
  pull** `jnp.interp(t_pull, tk, m_fuel)` (else the rest counts fuel);
  surface cap weighted by $(1-p)$; quality at serve — already the last
  index the code reads; safety through the rest (lethality keeps banking;
  a long board rest re-enters the danger zone and the model now says so).
- `timeplan.py` / `run_fullplan.py`: the scan is over cook lengths (start
  to serve); `opt_pull` inside each — the pull, and so the rest, is a
  decision variable.
- `mpc.py`: every replan optimizes schedule, wrap and pull over the
  remaining time to serve; the pull executes like the wrap (candidate →
  `wrap_commit` → irreversible, `t_pull_seg` mirrors `t_wrap_seg`) and the
  plant rests to serve; `executed_objective` prices fuel at the pull and
  the bark-burn cap in the pit only.

### Tests (Julia, minutes)

1. `t_pull = PULL_NEVER` ⇒ every output `==` the unwrapped solve.
2. Carryover: pull at core 85 °C from a 110 °C pit into the cambro preset —
   core rises (> 0.5 K), peaks after 15–45 min, then falls; board preset:
   smaller rise, faster fall. (Expected magnitude 1–5 K; assert the shape,
   not the number.)
3. Doneness continues: `C_core(serve) − C_core(pull) > 0.05` for a 1 h
   cambro rest from 88 °C.
4. Rewetting: `W_s(serve) > W_s(pull)` in the cambro preset; `drip_w` still
   grows while the interior is above $T_{\mathrm{myosin}}$.
5. `q_meat ≈ 0` after `t_pull + 4 δ_pull`.
6. FD on `t_pull` (clone of the wrap-time testset; step ≪ δ_pull), θ and
   last-knot off-by-one guards with the pull active; composed
   `test_grad_tpull` in `test_composed_grad.py`.

### What it changes

Prediction: the optimizer pulls earlier ($C_{\mathrm{pull}} \approx
0.80$–$0.85 \to C_{\mathrm{serve}} = 0.90$); "THE COOK" shortens from 17 h;
`over_rel` is 1 (the 0.5 discount stood in for the hold).
Flag, don't tune: with the literature collagen scale ($C = 0.90$ at
$k_{\mathrm{ref}}$, a Tier-2 limitation) a 1–2 h hold may push every plan
into the overdone band — that makes `k_ref` the binding uncalibrated
constant and is worth one sentence in the writeup. The chance layer
(`chance_of_done`) now evaluates P(done at *serve*) with the hold's actual
conversion instead of a one-sided assumption.

Cost: ~1 day; demo reruns (fullplan / timeplan) ≈ 1–2 h compute each at
60 inner iterations, no refit.

## 2. Shrinkage (#1) — material coordinates, no new state, parameter-free by default

### Physics

Mass is conserved, volume is not: the fixed-$L$ slab lets the wet density
fall (1060 → ~650 kg/m³ at 40 % loss) where real meat keeps $\rho_{\mathrm{wet}}
\approx$ const and shrinks. Solve on the **dry-matter (Lagrangian)
coordinate** $\xi \in [0, L_0]$: dry mass per unit $\xi$ is constant, the
grid never moves, and the existing `fv_mean` in $\xi$ *is* dry-mass
weighting. Specific volume per kg dry matter (F is fat per kg dry matter,
as in the code):

$$
v(W, F) = \frac{1-F}{\rho_{\mathrm{sol}}} + \frac{F}{\rho_f} + \frac{W}{\rho_w},
\qquad \rho_{\mathrm{sol}} \approx 1350,\ \rho_f \approx 900,\ \rho_w \approx 980\ \mathrm{kg/m^3}.
$$

Check against the existing constants: $v_0$ gives $\rho_{\mathrm{wet}} =
1071 \approx$ `RHO_MEAT` and $1/v_0 = 302 \approx$ `RHO_DRY` — nothing is
being re-tuned. Thickness stretch and geometry:

$$
s(W,F) = \big(v/v_0\big)^{n_{sh}}, \qquad
x(\xi,t) = \int_0^{\xi} s\,d\xi', \qquad
L(t) = \int_0^{L_0} s\,d\xi, \qquad
A(t) = A_0\,\big(\bar v/v_0\big)^{1-n_{sh}} .
$$

$n_{sh} = 1/3$ isotropic (default — volumetric shrinkage is density
bookkeeping, not a free parameter; only the *anisotropy* is), $n_{sh} = 1$
pure thickness, $n_{sh} = 0$ off (bitwise). The volume change not taken by
thickness is lateral, hence $A(t)$, which multiplies the surface fluxes
in the material balance and $q_{\mathrm{meat}}$ to the pit. Meat shrinks
transverse to the fibre at the myosin step and along it at the actin step
(Tornberg 2005); a flat's thickness is transverse, so $n_{sh} \in [1/3,
2/3]$ is the plausible band — profile it (§4). Equations in $\xi$,
conservation form, per dry mass:

$$
\rho c_p\, s\,\partial_t T = \partial_\xi\!\Big(\frac{k}{s}\,\partial_\xi T\Big),
\qquad
\partial_t W = \partial_\xi\!\Big(\frac{D_{\mathrm{eff}}}{s^{2}}\,\partial_\xi W\Big)
- k_e\,\mathrm{sp}_{\delta_e}\!\big(W - W_{\mathrm{eq}}(T)\big),
$$

(the $s^{-2}$ in the water equation: dry mass per $\xi$ is conserved, so
the dry-matter concentration per physical volume rises as the matrix
contracts). Surface $k\,\partial_x T = (k/s)\,\partial_\xi T$; the water
balance sees the shrunken surface, $(A/A_0)\,j$, while the heat balance's
area factor cancels. On the MOL grid: face stretch = arithmetic mean of
neighbours, cell width $s_i\,\Delta\xi$, the surface half-cell $s_n
\Delta\xi/2$; rolling window, allocation-free; $W$ enters $T$'s stencil
through $s$ (Enzyme differentiates it; sparsity pattern unchanged).
$\theta$'s $L$ is the raw half-thickness $L_0$. `evap_w` per dry mass is
$\int j\,A(t)/A_0\,dt / (\rho_d L_0)$.

Not in v1 (goes to gaps): composition-dependent $c_p(W,F)$ per dry mass
(Choi–Okos in material form) — a natural follow-on once the coordinate
exists.

### Where it lands

- `src/meat.jl`: `n_shrink, ρ_sol, ρ_fat, ρ_water` in `MeatConfig`;
  `stretch(W, F, cfg)`; metric factors in the RHS; outputs `l_thick`
  (trajectory, `L(t)`) and `shrink_final = L(t_f)/L_0` — **a ruler measures
  this**: a free calibration observable next to the scale.
- Tesseract: `n_shrink` plain input (default 0), `l_thick` output (plain).
  No new differentiable IO, no VJP change.
- `mpc.py` handoff: fields live in $\xi$ → unchanged.

### Tests

1. `n_shrink = 0` ⇒ bitwise identical.
2. $n_{sh} = 1/3$: `shrink_final` ∈ [0.82, 0.92] on the 16 h stall fixture;
   $n_{sh} = 1$: ∈ [0.60, 0.75]; `l_thick` non-increasing (tolerance for
   local diffusion).
3. Late-cook acceleration: time to core 85 °C is shorter with shrinkage on
   than off at the same θ (the gap's claim, as a number).
4. Mass closure and `w_retained` unchanged in meaning (dry-mass weighted).
5. FD with $n_{sh} = 1/3$: `D_eff`, `L`, `k_mult`, a mid `Tpit` knot (the
   stretch couples $W \to T$); the mid-cook nonuniform-IC VJP with the
   stretch on.

### What it changes

At the MAP, `k_mult = 1.46` (2.09 in the θ=7 refit) — 1.5–2× the
Choi–Okos conductivity — is the fixed-$L$ model buying the late climb the
data shows. `figures/cooklog_fit_map.png` already *overshoots* that climb
(+2–3 K at 18–19 h; the data plateaus at ~90 °C before the pull), so
shrinkage will not "fix a residual". The falsifiable prediction is that
the refit returns `k_mult → ~1` with $L_0$ slightly larger (~2.2–2.4 cm
half-thickness). If `k_mult` stays at 1.5–2, shrinkage was not what it was
absorbing. Conduction time scales as $L^2$: a 16 % thinner slab (isotropic
at ~40 % mass loss) conducts 29 % faster late in the cook, which is where
the doneness timing lives.

Cost: ~1 day + its share of §4.

## 3. Humidity feedback (#3) — quasi-steady vapour balance, closed-form self-consistent evaporation

### Magnitudes first (why this is first-order, and what is small)

Pit residence time: $V \approx 0.07$ m³ (18.5" WSM), $\dot m_{\mathrm{air}}
\approx 2.5\times10^{-3}$ kg/s → ~30 s ≪ the 10-min exchange grid, so
$p_{v,\mathrm{pit}}$ is **algebraic, not a state**. At the log's hold (MAP
$r_{\max}$, $g(u) = 0.26 + 0.74 \cdot 0.2 = 0.41$):

| term | estimate | $\Delta p_v$ |
|---|---|---|
| air flow $\dot m_{\mathrm{air}} = \mathrm{AFR}\cdot r_{\max}\, g(u)$, AFR ≈ 20 | $2.5\times10^{-3}$ kg/s (Guru 10 CFM blower at 10–30 % duty: $0.6$–$1.7\times10^{-3}$ + leak — same order) | — |
| ambient, 20 °C / 50 % RH | $\omega_{\mathrm{amb}} \approx 0.007$ | 1.2 kPa |
| combustion water $y_w \dot m_b$ (briquettes ≈ 2.5 wt% H → 0.22 kg/kg, + 6 % moisture) | $0.28 \times 1.25\times10^{-4}$ | +2.3 kPa |
| meat vapour $A\,j$ at peak | $3\times10^{-5}$ kg/s | +2.0 kPa |
| water pan (≈ 0.4 L/h, WSM refill folklore) | $1.1\times10^{-4}$ kg/s | +7 kPa |

So $p_{v,\mathrm{pit}} \approx 5$–6 kPa without a pan, ~12 kPa with; the fixed
2 kPa sits at the bottom of the plausible range. At the stall the surface
sits where $p_{\mathrm{sat}}(T_s) = p_v + j/(k_m\varphi)$, and
$dp_{\mathrm{sat}}/dT \approx 1.1$ kPa/K near 65 °C, so +3.5 kPa is ~+3 K on the
plateau at fixed $k_m$ — larger than the 1.7 K core residual scale. The
*self*-humidification term is smaller: with $a = k_m\varphi/(R_v T_s)$ and
$c = (p_{\mathrm{atm}}/0.622)\,A/\dot m_{\mathrm{air}}$, the feedback factor
$ac \approx 0.05$ at the MAP $k_m$ (0.25 at the physical $k_m$ below) —
~10 % of the deficit. The base humidity (combustion, ambient, pan) is the
first-order correction; the self-term is cheap once the plumbing exists.

Consistency check available *without* a sensor: the Chilton–Colburn analogy
gives $k_m/h_c = 1/(\rho_{\mathrm{air}} c_{p,\mathrm{air}} \mathrm{Le}^{2/3})
\approx 1.2\times10^{-3}$ (m/s)/(W/m²K) at 380 K. The MAP has
$k_m/h_c = 1.07\times10^{-3}/4.52 = 2.4\times10^{-4}$ — **5× below the
analogy**. The calibrated $k_m$ is absorbing the missing humidity, exactly
as the gaps doc says.

### Balance

Absolute humidity $\omega$, small-$\omega$ linearisation
$p_v \approx (p_{\mathrm{atm}}/0.622)\,\omega$:

$$
p_{v,\mathrm{pit}}(t) =
\underbrace{\frac{p_{\mathrm{atm}}}{0.622}\Big(\omega_{\mathrm{amb}}
+ \frac{y_w\,\dot m_b(t) + \dot m_{\mathrm{pan}}}{\dot m_{\mathrm{air}}(t)}\Big)}_{p_{v,\mathrm{base}}(t)}
\;+\;
\underbrace{\frac{p_{\mathrm{atm}}}{0.622}\,\frac{A}{\dot m_{\mathrm{air}}(t)}}_{c(t)}\; j,
\qquad
\dot m_{\mathrm{air}} = \mathrm{AFR}\, r_{\max}\, g(u) .
$$

Air flow follows the draft $g(u)$, the burn follows draft × ignition ×
fuel, so the air–fuel ratio is $\mathrm{AFR}/(a\tanh(m/m_t)) \ge
\mathrm{AFR}$ — lean while the bed is partly lit, as it should be. The
surface flux is then a scalar fixed point solved **exactly in the RHS**:

$$
j = a\,\mathrm{sp}_{\delta_p}\!\big(p_{\mathrm{sat}}(T_s) - p_{v,\mathrm{base}} - c\,j\big),
\qquad
f(j) = j - a\,\mathrm{sp}_{\delta_p}(D - cj),\quad f' = 1 + ac\,\sigma\big((D-cj)/\delta_p\big) > 0 .
$$

Monotone, unique root; start from the clamp-active closed form
$j_0 = a\,\mathrm{sp}_{\delta_p}(D)/(1+ac)$ and take **two fixed Newton
steps** (branch-free, smooth, no `while` in the adjoint). No Picard lag, no
new state, **no pit code**: $\dot m_b$ is already exported, $g(u)$ is
algebraic in `u_knots` and `u_leak` (φ[4]), so the orchestrator assembles
the two knot vectors in JAX. Empty knots ⇒ constant `p_v_pit` (bitwise).
New output `pv_pit(t)`: the observable a wet-bulb probe measures — the
likelihood hook for the roadmap's sensor.

Condensation (#7) stays clamped, but becomes a one-line change once
$p_{v,\mathrm{pit}}$ is dynamic (two-sided flux when
$p_{\mathrm{sat}}(T_s) < p_{v,\mathrm{pit}}$ — cold meat in a steamy pit).

### Where it lands

- `src/meat.jl`: `p = [θ | Tpit | smoke | pv_base | pv_mix | t_wrap | t_pull]`;
  `evap_flux(Ts, Ws, k_m, pv_base, c, cfg)` with the Newton closed form;
  `_meat_core` / `meat_solve` observables use the same function; `meat_vjp`
  returns `dpv_base_knots, dpv_mix_knots`; output `pv_pit`.
- Tesseract meat: two Differentiable knot inputs (default empty), `pv_pit`
  output, VJP rows, `abstract_eval`.
- `coupled.py`: `HumidityConfig(afr=20.0, y_w=0.28, pan_kg_s=0.0,
  omega_amb=0.007, p_atm=101325.0)`; `humidity_knots(u_knots, utk, tk,
  u_leak, mdot, hc)` — the damper must be interpolated to `tk` with the
  **same hat basis** as the pit's `lininterp` (write a JAX `hat_interp`;
  a mismatch would silently decouple the forcing from the pit's own
  airflow); the pair goes through the `optimization_barrier` tuple;
  `humidity=None` ⇒ off.
- `calibrate.py`: `bound_model_real(..., humidity=…)`; the log's conditions
  as plain inputs: $\omega_{\mathrm{amb}}$ from the (qualitative) ambient
  channel plus an assumed RH; **water pan present? — user-owned: check the
  Mache post** (a full pan is the WSM default; foiled/empty is common).

### Tests

1. Empty knots ⇒ bitwise identical to `p_v_pit = 2000`.
2. Unit: two-step Newton $j$ vs a converged fixed-point iteration at 5
   $(T_s, W_s, c)$ points, < 1e-10 relative.
3. Monotone: $\int j$ falls and the stall plateau rises with `pv_base`; with
   $c > 0$, $j < j(c = 0)$ and `pv_pit ≥ pv_base`.
4. Composed: $\partial\,\mathrm{tcore}/\partial u$ through the humidity path
   alone (other paths zeroed by construction: constant Tpit + zero smoke
   via a meat-only call) is finite, nonzero, FD-consistent — the
   `test_bark_smoke_path` pattern; Julia FD on one `pv_base` and one
   `pv_mix` knot; θ/last-knot off-by-one guards with the new tail.
5. Diagnostic (printed, not asserted): `pv_pit` at the log's hold ∈
   [4, 8] kPa without a pan.

### What it changes

- Damper authority gains a second channel: airflow dilutes humidity →
  deeper evaporative cooling at high fan duty, on top of the thermal
  effect. Control plans and the wrap timing will move; the MPC ambient
  Kalman filter acquires a confound (humidity ≈ weather) — record in gaps.
- Refit prediction: $k_m$ rises toward the Chilton–Colburn ratio and the
  $k_m$–$h_c$ ridge rotates. If $k_m$ does not move, AFR / $y_w$ / pan are
  wrong or the fixed 2 kPa happened to be right for this cooker — either
  way it is information the sensor-less log can give.
- Still cannot be *validated* without a wet-bulb probe (the gaps doc is
  right); it can be checked for *consistency* ($k_m$ physicality, plateau
  level at the log's fan duty).

Cost: ~1.5 days + its share of §4.

## 4. Calibration campaign (after all three; every step user-approved)

1. **Forward-only first** (minutes): at the current MAP, four coupled solves
   on the log's fan schedule — off / +shrink / +humidity / +both — residual
   vs time next to `cooklog_fit_map.png`. Sizes each effect *before* an
   evening of refits.
2. `run_map_real.py --init map_real.json` with both on (300 steps).
   Profile $n_{sh} \in \{1/3, 1/2, 2/3\}$ × pan $\in \{0, 0.4\ \mathrm{L/h}\}$:
   6 MAP fits, overnight-class in total. Record loss, `k_mult`,
   $k_m/h_c$, $L_0$, `sigma_core`, `rho_core` per cell → one table for
   gaps/writeup.
3. Decision point: adopt the $(n_{sh}, \mathrm{pan})$ cell by loss *and*
   physicality (`k_mult ≈ 1`, $k_m/h_c \approx 10^{-3}$). Re-centre a prior
   only if the MAP moved > 1 prior-σ ($k_m$ will).
4. VI (evening) → NUTS-real (overnight, **fresh** checkpoint dir) → control
   artifacts in order: `run_control → run_robust_control → run_fullplan`
   (rest on) `→ run_timeplan → run_wraptime → run_mpc → render_animation`.
   Every figure through the zone-by-zone inspection rule.
5. Synthetic dataset: untouched (no θ site changes). `THETA_DEFAULT` /
   `MEAT_θ_DEFAULT` stay in sync (no change).

## 5. Docs & verification sync (per feature, at merge)

- `docs/model.md`: §1 gains the pull blend + rest BC, the $\xi$-coordinate
  equations, the vapour balance; new outputs (`l_thick`, `pv_pit`);
  notation table entries ($p$, $s$, $v$, $c$).
- `docs/model_gaps.md`: #1–#3 leave Tier 1 for "Limitations of what is
  already implemented" with their honest residues — rest constants
  literature-fixed; $n_{sh}$ profiled, not identified; humidity
  structural and unvalidated (sensor); $A(t)$ lateral-scaling assumption;
  $c_p(W,F)$ absent; condensation (#7) still clamped; humidity as an MPC
  confound.
- `docs/verification.md`: new FD directions (`t_pull`, `pv_*` knots, the
  shrink-on directions) and the composed gate count.
- `README.md` test counts; writeup Limitations paragraph (post-submission).
- Containers: rebuild + re-sweep once at the end (payloads gain the new
  inputs); `scripts/checkgradients/meat_payload.json` updated then.

## 6. Schedule and decision points

- **Now → Aug 31** (NUTS-real sampling on this machine): the rest model in
  a worktree, bitwise-off, tests green, merged without touching any
  calibration artifact. *If* the chains finish early and the user approves
  the compute: re-run fullplan / timeplan at the current MAP with quality
  at serve — a pre-submission result that needs no refit ("the hold is
  modelled, not assumed; pull-when-done is a differentiable decision").
  Otherwise the rest ships as implemented + tested, demos deferred.
- **Post-submission week 1**: shrinkage (1 day), humidity (1.5 days), the
  forward-effect diagnostic (§4.1).
- **Week 2**: §4 campaign, artifacts, docs.

User-owned decisions: (a) water pan in the Mache cook (read the post);
(b) rest preset + duration for the demos (1 h faux cambro is the
practitioner default); (c) whether `n_shrink` joins θ later — only with
ruler/scale data.

## Risks

- **p-tail relayout** is the landmine (the state-relayout lesson from the
  moisture build): `dsmoke_knots = dp[(MEAT_NP+nk+1):(end-1)]` and
  `dt_wrap = dp[end]` must become explicit index arithmetic; the
  off-by-one guard tests exist for exactly this.
- Pit runs through the rest: any objective or report that reads
  `m_fuel[-1]` (control, `executed_objective`, fullplan JSON) must read at
  the pull. Grep for `m_fuel` at merge.
- Shrinkage changes the stiff stencil's coefficients mid-solve; FBDF has
  handled the wrap and expulsion switches, but the FD baselines will drift
  (they did at every state-count change) — recompute, do not loosen.
- Humidity forcing vs pit airflow mismatch (hat basis) — silent; covered
  by the composed test (4) which would show an FD disagreement.
- Newton-in-RHS for $j$: two steps is exact to ~1e-12 in the tested band;
  add a Julia test at extreme $c$ (pan on, low fan) where $ac \to 1$.
- Compute contention with the live chains; laptop must stay plugged in for
  anything in §4.

## Sources

- Crank, *The Mathematics of Diffusion* (2nd ed., 1975), moving-boundary
  chapter: material (Lagrangian) coordinate transformation for shrinking
  media.
- Mayor & Sereno, "Modelling shrinkage during convective drying of food
  materials: a review", *J. Food Eng.* 61 (2004) — volume ∝ water content
  (density bookkeeping) as the baseline shrinkage law.
- Tornberg, "Effects of heat on meat proteins — implications on structure
  and quality of meat products", *Meat Sci.* 70 (2005) — transverse vs
  longitudinal shrinkage at the myosin / actin–collagen steps.
- Incropera & DeWitt, *Fundamentals of Heat and Mass Transfer*, ch. 6 —
  Chilton–Colburn heat–mass analogy ($k_m/h_c$ check).
- Baldwin, "Sous vide cooking: a review", *Int. J. Gastronomy Food Sci.* 1
  (2012); Blonder, genuineideas.com carryover-cooking measurements — rest /
  carryover magnitudes.
- Charcoal briquette composition (H ≈ 2–3 wt %, moisture 5–8 wt %) and
  stoichiometric air (≈ 11.5 kg/kg C): manufacturer data / FAO charcoal
  reports — cite the specific source when the constants are set.
- WSM water-pan refill rates and the Guru Pit Viper 10 CFM rating: forum /
  product data (order-of-magnitude only; the pan rate is a plain input).
