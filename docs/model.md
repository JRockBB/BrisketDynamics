# The model, mathematically

![The coupled equations](../figures/math_diagram.png)

Every equation the code integrates, in one place, in the same notation as
the diagram above. Calibrated latents are marked **θ** (meat) or **φ**
(pit); everything else is a fixed constant from `MeatConfig`/`PitConfig`
with its provenance commented in the source (`src/meat.jl`, `src/pit.jl`,
`app/src/brisket_app/control.py`).

## 0. Notation

Three σ's appear: $\sigma(\cdot)$ is always the logistic sigmoid,
$\sigma_{SB}$ the Stefan–Boltzmann constant, and $\sigma_{aw}$ the width
of the Maillard water-activity bell. All switches are smooth — no branch
ever enters the adjoint.

**Smooth primitives**

| symbol | definition | code |
|---|---|---|
| $\sigma(x)$ | logistic $\tfrac12(1+\tanh(x/2))$ | `logistic` |
| $\mathrm{sp}_\delta(x)$ | softplus $\delta\log(1+e^{x/\delta})$ (smooth $\max(0,x)$, width $\delta$) | `softclamp` / `log1pexp` |

**States and evaluations**

| symbol | meaning |
|---|---|
| $T, W, C, F$ | meat fields: temperature [K], dry-basis moisture [kg/kg], collagen conversion [0–1], un-rendered fat [kg/kg] |
| $B, S$ | surface bark states: Maillard browning, smoke deposition (each in [0,1)) |
| $T_s, W_s$ | field values at the surface node $x = L$ |
| $C_0$ | collagen conversion at the center node (the probe location) — "doneness" |
| $T_p, m, a$ | pit states: gas temperature [K], fuel mass [kg], ignited bed fraction |

**Latents.** θ = $[D_{\mathrm{eff}}, k_m, h_c, L, k_{\mathrm{mult}},
k_e, a_{\mathrm{myo}}]$ (moisture transport, evaporative mass transfer,
convection, half-thickness, conductivity multiplier, expulsion rate,
myosin water-holding amplitude). φ = $[C_{th}, UA, r_{\max},
\Delta H_{e\!f\!f}, u_{\mathrm{leak}}, k_{\mathrm{spr}}, a_0]$ (thermal
mass, loss coefficient, max burn rate, effective heat of combustion, leak
floor, ignition spread rate, initial ignited fraction). Code names:
`a_myosin`, `k_spread`, `dH_eff`.

**Derived functions** (each defined where it first appears below)

| symbol | meaning | code |
|---|---|---|
| $\varphi(W_s)$ | moisture-availability switch | in `evap_flux` |
| $\Delta\rho_v(T_s)$ | clamped vapor-density deficit | in `evap_flux` |
| $s(t)$ | Texas-crutch wrap blend | `wrap_blend` |
| $\varepsilon(s)$ | wrap-blended emissivity | in RHS |
| $q_{\mathrm{surf}}$ | surface heat drawn from the pit [W/m²] | `surface_heat_in_wrapped` |
| $k_C(T), k_f(T), k_B(T_s)$ | collagen / fat-render / Maillard rates | `collagen_rate`, `fat_rate`, `bark_rate` |
| $W_{\mathrm{eq}}(T)$ | water-holding equilibrium | `w_eq` |
| $a_w(W_s)$ | surface water activity (Oswin isotherm) | `a_w_oswin` |
| $g(a_w)$ | Maillard water-activity bell | `g_aw` |
| $\mathrm{wet}(a_w)$ | smoke-deposition (tacky-surface) gate | `wet_blend` |
| $g(u)$ | pit airflow from damper + leak | in pit RHS |
| $\dot m_b$ | fuel burn rate [kg/s] (the smoke-flux forcing) | `mdot` |

**Key constants** (defaults and provenance in the config structs):
$W_0, F_0$ initial moisture/fat; $\rho_d$ dry-matter density; $\lambda$
latent heat (`λ_vap`); $R_v$ water-vapor gas constant; wrap constants
$\eta_e$ (`η_evap`), $\eta_h$, $\varepsilon_{\mathrm{wrap}}$, $\delta_w$
(`δ_wrap`); bark color map $L^*_0, \Delta_B, \Delta_S$ (`L0_bark`,
`dL_B`, `dL_S`); deposition rate $k_{\mathrm{dep}}$; fuel taper scale
$m_t$ (`m_taper`); expulsion widths $\delta_e, \delta_{\mathrm{myo}},
\delta_{\mathrm{act}}$; drain split $\eta_{\mathrm{drain}}$; sensory
weight $\beta_{\mathrm{fat}}$.

## 1. Meat: 1-D heat–moisture–kinetics PDE (162 ODEs after MOL)

Half-slab $x \in [0, L]$, symmetry at the center $x=0$ (probe location),
convective/radiative/evaporative boundary at the surface $x=L$. Method of
lines on $n_x = 40$ uniform nodes → $4n_x + 2 = 162$ stiff ODEs (FBDF).

**Interior fields** (θ: $D_{\mathrm{eff}}$, $k_{\mathrm{mult}}$, $k_e$;
with $\alpha = k_{\mathrm{mult}} K_{\mathrm{meat}} / (\rho c_p)$):

$$
\partial_t T = \alpha\,\partial_x^2 T,
\qquad
\partial_t W = D_{\mathrm{eff}}\,\partial_x^2 W
\;-\; k_e\,\mathrm{sp}_{\delta_e}\!\left(W - W_{\mathrm{eq}}(T)\right),
$$

$$
\partial_t C = k_C(T)\,(1 - C),
\qquad
\partial_t F = -\,k_f(T)\,F,
$$

with symmetry ($\partial_x T = \partial_x W = 0$) at $x = 0$ and the rate
functions

$$
k_C(T) = k_{\mathrm{ref}}\,
\exp\!\Big[-\frac{E_a}{R}\Big(\frac{1}{T} - \frac{1}{T_{\mathrm{ref}}}\Big)\Big],
\qquad
k_f(T) = k_{f0}\;\sigma\!\Big(\frac{T - T_{\mathrm{render}}}{\delta_{\mathrm{render}}}\Big)
$$

(collagen: Arrhenius referenced to $T_{\mathrm{ref}} = 60$ °C; fat: render
switch ~55 °C). The expulsion sink (the purge; θ: $k_e$,
$a_{\mathrm{myo}}$) relaxes $W$ one-sidedly toward a
temperature-dependent water-holding equilibrium with two smooth
denaturation steps (myosin ~58 °C, actin ~72 °C):

$$
W_{\mathrm{eq}}(T) = W_0\Big(1
- a_{\mathrm{myo}}\,\sigma\!\big(\tfrac{T - T_{\mathrm{myo}}}{\delta_{\mathrm{myo}}}\big)
- a_{\mathrm{act}}\,\sigma\!\big(\tfrac{T - T_{\mathrm{act}}}{\delta_{\mathrm{act}}}\big)\Big).
$$

**Surface boundary** ($x = L$) — the stall mechanism, with the wrap blend
$s$. Evaporative flux is mass-transfer-limited (θ: $k_m$, $h_c$):

$$
j = (1 - \eta_e\, s)\;k_m\,\varphi(W_s)\,\Delta\rho_v(T_s),
\qquad
-\rho_d D_{\mathrm{eff}}\,\partial_x W|_s = j,
$$

$$
\varphi(W_s) = \sigma\!\Big(\frac{W_s - W_{\mathrm{crit}}}{\delta_W}\Big)
\;\;\text{(availability)},
\qquad
\Delta\rho_v(T_s)
= \frac{\mathrm{sp}_{\delta_p}\!\left(p_{\mathrm{sat}}(T_s) - p_{v,\mathrm{pit}}\right)}{R_v\,T_s}
\;\;\text{(clamped deficit)},
$$

$$
-k\,\partial_x T|_s
= h_c\,(1 - \eta_h\, s)\,(T_p - T_s)
+ \varepsilon(s)\,\sigma_{SB}\,(T_p^4 - T_s^4)
- \lambda\, j,
\qquad
\varepsilon(s) = \varepsilon\,(1-s) + \varepsilon_{\mathrm{wrap}}\,s .
$$

The heat drawn from the pit (the coupling output) is the
convective + radiative part:

$$
q_{\mathrm{surf}} = -k\,\partial_x T|_s + \lambda\, j .
$$

**Texas crutch (wrap).** The smooth blend
$s(t) = \sigma\!\left((t - t_{\mathrm{wrap}})/\delta_w\right)$ carries the
wrap: it suppresses evaporation ($\eta_e$), reduces convection
($\eta_h$), swaps emissivity toward foil
($\varepsilon_{\mathrm{wrap}}$), and blocks smoke contact via $(1-s)$ in
the deposition state. $t_{\mathrm{wrap}}$ is a differentiable decision
variable; $t_{\mathrm{wrap}} = 10^9$ makes $s \equiv 0$ bitwise (the
unwrapped model recovered exactly).

**Bark states** (surface only), with the Oswin water activity
$a_w(W_s) = \left(1 + (k_{ox}/W_s)^{n_{ox}}\right)^{-1}$:

$$
\dot B = k_B(T_s)\; g(a_w)\,(1 - B),
\qquad
k_B(T_s) = k_{0,\mathrm{bark}}\, e^{-E_{a,\mathrm{bark}}/(R T_s)},
\qquad
g(a_w) = \exp\!\Big[-\frac{(a_w - a_{pk})^2}{2\sigma_{aw}^2}\Big],
$$

$$
\dot S = k_{\mathrm{dep}}\;\mathrm{wet}(a_w)\;\dot m_b\;(1 - s)\,(1 - S),
\qquad
\mathrm{wet}(a_w) = \sigma\!\Big(\frac{a_w - a_{\mathrm{dry}}}{\delta_{\mathrm{dry}}}\Big),
$$

$$
L^* = L^*_0 - \Delta_B\, B - \Delta_S\, S .
$$

($\dot m_b \ge 0$ by construction; the code clamps defensively.) The
split-cook continuity experiment separates the two channels'
time-weighting empirically: $S$ is early-weighted (needs the wet
surface), $B$ is late-weighted (hot dry surface).

**Quality summaries** (dry-basis means; $\eta_{\mathrm{drain}}$ = rendered
fat fraction that drips):

$$
w_{\mathrm{ret}} = \frac{\overline W(t_f)}{W_0},\qquad
f_{\mathrm{rend}} = 1 - \frac{\overline F(t_f)}{F_0},\qquad
\mathrm{juiciness}(W, F) = w_{\mathrm{ret}}
+ \beta_{\mathrm{fat}}\,\frac{(1-\eta_{\mathrm{drain}})\,F_0\, f_{\mathrm{rend}}}{W_0},
$$

with the algebraically closed cook-loss budget (tested to $10^{-8}$):
water lost $= \int j\,dt/(\rho_d L)$ (evaporated) $+$ drip (purge), fat
lost $= \eta_{\mathrm{drain}} F_0 f_{\mathrm{rend}}$ (tallow catch).

## 2. Pit: lumped 3-state ODE

With damper $u(t) \in [0,1]$ and airflow
$g(u) = u_{\mathrm{leak}} + (1 - u_{\mathrm{leak}})\,u$ (the leak floor
means closing the damper never kills the fire):

$$
\dot m = -\,\dot m_b,\qquad
\dot m_b = r_{\max}\, g(u)\, a\,
\tanh\!\Big(\frac{\max(m,0)}{m_t}\Big),
$$

$$
C_{th}\,\dot T_p = \Delta H_{e\!f\!f}\,\dot m_b
- UA\,(T_p - T_{\mathrm{amb}}) - q_{\mathrm{meat}},
\qquad
\dot a = k_{\mathrm{spr}}\, a\,(1-a)\, g(u),\quad a(0) = a_0 .
$$

At $a_0 = 1$ the spread term vanishes and the two-state model is
recovered exactly. $\dot m_b$ is exported (exact algebraic recompute at
save times) as the smoke-flux forcing to the bark model.

## 3. Coupling: unrolled Picard iteration

The components exchange whole trajectories on a shared knot grid, $K = 2$
fixed passes, with the meat heat load
$q_{\mathrm{meat}} = A_{\mathrm{meat}} \cdot q_{\mathrm{surf}}$
($A_{\mathrm{meat}}$ the exposed surface area):

$$
\big(T_p^{(k)},\, \dot m_b^{(k)}\big)
= P_\varphi\big(u,\, q_{\mathrm{meat}}^{(k-1)}\big),
\qquad
\big(T_{\mathrm{core}},\, j,\, q_{\mathrm{meat}}^{(k)}\big)
= M_\theta\big(T_p^{(k)},\, \dot m_b^{(k)}\big),
\qquad k = 1 \ldots K .
$$

The model is *defined* as the $k = K$ output of this unrolled map, so
reverse-mode AD through the composed Tesseract VJPs (continuous adjoint on
the meat side, transposed ForwardDiff Jacobian on the pit side) is the
exact gradient of the simulated system — no implicit-function
approximation.

## 4. The quality objective (KCBS-weighted; every term overridable)

Decision variables: damper knots $z \mapsto u = \sigma(z)$ and wrap
$z_w \mapsto t_{\mathrm{wrap}} = t_f\,\sigma(z_w)$. Below
$\mathrm{sp}(x) = \mathrm{sp}_1(x) = \log(1+e^x)$, and the category
weights are $(w_{\mathrm{app}}, w_{\mathrm{taste}}, w_{\mathrm{tend}}) =
(0.14,\, 0.57,\, 0.29)$ — the normalized multipliers KCBS judging
carries.

**Tenderness** $\mathcal T(C_0)$ — two-sided collagen band at the center
node, each side normalized to its band width (under: $c^\star = 0.90$,
width 0.05; over: $c_{od} = 0.95$, discounted by $\rho_{ov} = 0.5$ since
early finishes rest):

$$
\mathcal T =
\Big[\tfrac{\mathrm{sp}(20(c^\star - C_0))/20}{0.05}\Big]^2
+ \rho_{ov}\Big[\tfrac{\mathrm{sp}(k_{ov}(C_0 - c_{od}))/k_{ov}}{1 - c_{od}}\Big]^2,
\qquad k_{ov} = \tfrac{4}{1 - c_{od}} .
$$

**Appearance** $\mathcal A(L^*)$ — two-sided mahogany band on final $L^*$
($[L^*_{lo}, L^*_{hi}] = [28, 34]$, softness $\delta_L = 2$):

$$
\mathcal A =
\mathrm{sp}\Big(\tfrac{L^* - L^*_{hi}}{\delta_L}\Big)^2
+ \mathrm{sp}\Big(\tfrac{L^*_{lo} - L^*}{\delta_L}\Big)^2 .
$$

**Taste** $\mathcal G_{\mathrm{smoke}}(S) + w_d\,(1-\mathrm{juiciness})^2$
— smoke-deposition band $[S_{lo}, S_{hi}] = [0.55, 0.90]$ (depth without
acrid over-deposition; $S_f$ = final deposition) plus dryness on
mechanistic juiciness ($w_d$: `w_dry`):

$$
\mathcal G_{\mathrm{smoke}} =
\mathrm{sp}\Big(\tfrac{S_{lo} - S_f}{\delta_S}\Big)^2
+ \mathrm{sp}\Big(\tfrac{S_f - S_{hi}}{\delta_S}\Big)^2 .
$$

**Food safety** $\mathcal S_{\mathrm{safety}}$, curve-faithful and
unconditional. Surface lethality via the z-value equivalence (FSIS ~7-log
at 60 °C reference, $z = 6$ K, requirement $F_{\mathrm{req}} = 720$
equivalent seconds), penalized on the log shortfall, plus a danger-zone
dwell cap ($h_{\mathrm{zone}}$ = hours with $T_s$ in 4.4–60 °C, cap 4 h):

$$
F_{\mathrm{leth}} = \int_0^{t_f} 10^{(T_s - T_{\mathrm{ref}})/z}\, dt,
\qquad
\mathcal S = w_F\,\mathrm{sp}\Big(\tfrac{\log_{10} F_{\mathrm{req}} - \log_{10} F_{\mathrm{leth}}}{0.2}\Big)^2
+ w_D\,\mathrm{sp}\Big(\tfrac{h_{\mathrm{zone}} - 4\,\mathrm{h}}{0.5}\Big)^2 .
$$

**Total** — the diagram's $\mathrm{ops}(T_s, m, u)$ collects the three
operational terms (bark-burn surface cap, fuel fraction, schedule
smoothness):

$$
J = w_{\mathrm{app}}\,\mathcal A
+ w_{\mathrm{taste}}\,\big[\mathcal G_{\mathrm{smoke}} + w_d\,(1-\mathrm{juiciness})^2\big]
+ w_{\mathrm{tend}}\,\mathcal T
+ \underbrace{w_s\,\overline{\mathrm{sp}\big(\tfrac{T_s - T_s^{\max}}{\delta_T}\big)^2}
+ w_f\,\Delta m_{\mathrm{fuel}}
+ w_u \textstyle\sum_i (\Delta u_i)^2}_{\mathrm{ops}(T_s,\,m,\,u)}
+ \mathcal S .
$$

## 5. Calibration likelihood

Masked Gaussian on both probes with the digitization noise floor
$\sigma_0 = 1.2$ K (real-cook model; $\sigma_{\mathrm{pit}},
\sigma_{\mathrm{core}}$ are sampled noise scales):

$$
y^{\mathrm{pit}}_i \sim \mathcal N\big(T_p(t_i;\theta,\varphi),\,
(\sigma_0 + \sigma_{\mathrm{pit}})^2\big),
\qquad
y^{\mathrm{core}}_i \sim \mathcal N\big(T_{\mathrm{core}}(t_i;\theta,\varphi),\,
(\sigma_0 + \sigma_{\mathrm{core}})^2\big),
$$

over non-lid-event observations; LogNormal priors centered on the model
defaults (Beta for the (0,1)-bounded $u_{\mathrm{leak}}, a_0$). MAP by
SVI/AutoDelta; posterior by NUTS, every leapfrog step pulling the composed
VJPs across the Julia↔JAX boundary.
