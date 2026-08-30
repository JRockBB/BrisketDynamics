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
ever enters the derivatives.

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
k_e, a_{\mathrm{myo}}, m_{\mathrm{conv}}]$ (moisture transport, evaporative
mass transfer, convection, half-thickness, conductivity multiplier,
expulsion rate, myosin water-holding amplitude, draft exponent). φ =
$[C_{th}, UA, r_{\max}, \Delta H_{e\!f\!f}, u_{\mathrm{leak}},
k_{\mathrm{spr}}, a_0]$ (thermal mass, loss coefficient, max burn rate,
effective heat of combustion, leak floor, ignition spread rate, initial
ignited fraction). Code names: `a_myosin`, `m_conv`, `k_spread`, `dH_eff`.

**Draft convection.** The damper moves air as well as heat: the surface
transfer coefficients are quoted at a reference draft $g_{\mathrm{ref}}$
(mid-damper, 0.5) and scale with the pit's airflow $g(u)$ as

$$
h_c(u) = h_c\,\Big(\frac{g(u)}{g_{\mathrm{ref}}}\Big)^{m_{\mathrm{conv}}},
\qquad
k_m(u) = k_m\,\Big(\frac{g(u)}{g_{\mathrm{ref}}}\Big)^{m_{\mathrm{conv}}},
$$

the same exponent for heat and mass transfer (Chilton–Colburn analogy);
forced convection over a slab runs $m \approx 0.5$ (laminar) to $0.8$
(turbulent), prior Beta(6, 4). A closed-loop cook log varies the fan duty
all night, so the data has some say in $m_{\mathrm{conv}}$; expect a ridge
with $h_c$.

**Derived functions** (each defined where it first appears below)

| symbol | meaning | code |
|---|---|---|
| $\varphi(W_s)$ | moisture-availability switch | in `evap_flux` |
| $\Delta\rho_v(T_s)$ | clamped vapor-density deficit | in `evap_flux` |
| $s(t)$ | Texas-crutch wrap blend | `wrap_blend` |
| $\varepsilon(s)$ | wrap-blended emissivity | in RHS |
| $p(t)$ | pull blend (in the pit → resting) | `pull_blend` |
| $v(W,F)$, $\varsigma$ | specific volume per kg dry matter; thickness stretch $(v/v_0)^{n_{sh}}$ | `specific_volume`, in RHS |
| $p_{v,\mathrm{base}}$, $c$ | pit base vapor pressure [Pa]; vapor-mixing coefficient | in `coupled.py`, `mixing_coefficient` |
| $\dot m_{\mathrm{air}}$ | pit air mass flow [kg/s] | in `coupled.py` |
| $A(t)/A_0$ | exposed-area factor $(\bar v/v_0)^{1-n_{sh}}$ | `a_area` in RHS |
| $q_{\mathrm{surf}}$ | surface heat drawn from the pit [W/m²] | `surface_q_pit_blended` |
| $q_{\mathrm{rest}}$ | surface heat from the rest environment [W/m²] | `surface_heat_rest` |
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
weight $\beta_{\mathrm{fat}}$; rest constants $T_r, h_r, \varepsilon_r,
\eta_r, p_{v,r}, \delta_p$ (`T_rest`, `h_rest`, `ε_rest`, `η_rest`,
`p_v_rest`, `δ_pull`); shrinkage exponent $n_{sh}$ (`n_shrink`, 0 =
fixed slab) and component densities $\rho_{\mathrm{sol}}, \rho_f,
\rho_w$.

## 1. Meat: 1-D heat–moisture–kinetics PDE (163 ODEs after MOL)

Half-slab $x \in [0, L]$, symmetry at the center $x=0$ (probe location),
convective/radiative/evaporative boundary at the surface $x=L$. Method of
lines on $n_x = 40$ uniform nodes → $4n_x + 3 = 163$ meat ODEs (the
fields, two bark states and the smoke-ring depth), 166 with the three
pit states, one stiff system (FBDF).

**Interior fields** (θ: $D_{\mathrm{eff}}$, $k_{\mathrm{mult}}$, $k_e$;
with $\alpha = k_{\mathrm{mult}} K_{\mathrm{meat}} / (\rho c_p)$), written
here on a fixed slab — the code integrates the Lagrangian dry-matter form
of the shrinkage section below, which reduces to this at $A/A_0 = 1$:

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
$s$. The surface mass flux is mass-transfer-limited (θ: $k_m(u)$, $h_c(u)$) and
two-sided: evaporation, gated by moisture availability, when the surface's
saturation pressure exceeds the pit's vapor pressure, condensation —
ungated: a cold surface below the dew point condenses whatever its water
content — when it does not. Fridge-cold meat in a humid pit condenses for
its first half hour; the latent heat warms the surface and the wet surface
is where smoke adheres:

$$
j = (1 - \eta_e\, s)\;k_m\,\frac{\varphi(W_s)\,\mathrm{sp}_{\delta_p}(\Delta p) - \mathrm{sp}_{\delta_p}(-\Delta p)}{R_v T_s},
\qquad \Delta p = p_{\mathrm{sat}}(T_s) - p_v,
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

where the pit vapor pressure is the pit's own quasi-steady balance plus
the meat's vapor (the pit residence time is ~30 s, far below the exchange
grid):

$$
p_{v,\mathrm{pit}} = p_{v,\mathrm{base}} + c\, j,
\qquad
p_{v,\mathrm{base}} = \frac{p_{\mathrm{atm}}}{0.622}\Big(\omega_{\mathrm{amb}}
+ \frac{y_w\,\dot m_b + \dot m_{\mathrm{pan}}}{\dot m_{\mathrm{air}}}\Big),
\qquad
c = \frac{p_{\mathrm{atm}}}{0.622}\,\frac{A(t)}{\dot m_{\mathrm{air}}},
\qquad
\dot m_{\mathrm{air}} = \mathrm{AFR}\, r_{\max}\, g(u)
$$

(ambient absolute humidity $\omega_{\mathrm{amb}}$, combustion water
yield $y_w$, water-pan rate $\dot m_{\mathrm{pan}}$, air–fuel ratio of the
draft AFR — plain constants in `coupled.Humidity`). AFR is the
air-to-fuel ratio at full burn, so the air flow is set by the draft, not
by the burn: a spreading or tapering bed burns leaner in the same air.
$j$ then solves
$j = F(p_{\mathrm{sat}} - p_{v,\mathrm{base}} - c j)$ with
$F(\Delta p) = a_e\,\mathrm{sp}_{\delta_p}(\Delta p) - a_c\,\mathrm{sp}_{\delta_p}(-\Delta p)$,
$a_e = k_m\varphi/(R_v T_s)$, $a_c = k_m/(R_v T_s)$ — a scalar fixed
point with a unique root ($1 + c F' > 0$; condensation lowers the pit
humidity and is self-limiting), taken from the linearized start plus two
Newton steps inside the RHS (`evap_flux_fb`). The pit exports $\dot m_b$; the orchestrator forms
$p_{v,\mathrm{base}}$ and $1/\dot m_{\mathrm{air}}$ on the exchange grid as
two more differentiable meat forcings; `pv_pit` is an output.

$$
k\,\partial_x T|_s
= h_c(u)\,(1 - \eta_h\, s)\,(T_p - T_s)
+ \varepsilon(s)\,\sigma_{SB}\,(T_p^4 - T_s^4)
- \lambda\, j,
\qquad
\varepsilon(s) = \varepsilon\,(1-s) + \varepsilon_{\mathrm{wrap}}\,s .
$$

The heat drawn from the pit (the coupling output) is the
convective + radiative part (the pit supplies surface heat; latent heat
leaves with vapor):

$$
q_{\mathrm{surf}} = k\,\partial_x T|_s + \lambda\, j .
$$

**Texas crutch (wrap).** The smooth blend
$s(t) = \sigma\!\left((t - t_{\mathrm{wrap}})/\delta_w\right)$ carries the
wrap: it suppresses evaporation ($\eta_e$), reduces convection
($\eta_h$), swaps emissivity toward foil
($\varepsilon_{\mathrm{wrap}}$), and blocks smoke contact via $(1-s)$ in
the deposition state. $t_{\mathrm{wrap}}$ is a differentiable decision
variable; $t_{\mathrm{wrap}} = 10^9$ puts the logistic blend at numerical
zero over any cook horizon (Python no-wrap is `WRAP_NEVER = 1e9 + 0\cdot z_w`
so the unused wrap logit stays in the tape).

**Spritz.** A spritz policy — first spritz $t_{\mathrm{sp}}$, interval
$\Delta_{\mathrm{sp}}$, dose $m_{\mathrm{sp}}$ (0.02–0.10 kg/m²: a light
to a heavy spray) — lands water at $T_{\mathrm{sp}}$ (20 °C) on the surface
over $\tau_{\mathrm{sp}}$ (1 min) at $t_{\mathrm{sp}} + k\Delta_{\mathrm{sp}}$
while the meat is in the pit and unwrapped:

$$
j_{\mathrm{sp}} = m_{\mathrm{sp}}\,\frac{\sigma'\big((t - t_{\mathrm{sp}} - k^*\Delta_{\mathrm{sp}})/\tau_{\mathrm{sp}}\big)}{\tau_{\mathrm{sp}}}\,(1-s)\,(1-p),
\qquad k^* = \max\big(\mathrm{round}((t - t_{\mathrm{sp}})/\Delta_{\mathrm{sp}}),\,0\big),
$$

A fraction $f_{\mathrm{abs}}$ (0.2, assumed) soaks into the crust and
enters the surface water balance, $-\rho_d D_{\mathrm{eff}}\,\partial_x
W|_s = j - f_{\mathrm{abs}}\,j_{\mathrm{sp}}$, leaving again through the
two-sided flux like condensate; the rest flashes off the hot surface
within the pulse. The energy balance carries both,
$-\lambda\,(1 - f_{\mathrm{abs}})\,j_{\mathrm{sp}} - c_w\,(T_s -
T_{\mathrm{sp}})\,j_{\mathrm{sp}}$ (the flash and the warming of the
water) — so a spritz is mostly evaporative cooling: the surface drops,
the stall lengthens, the briefly wet surface takes smoke, and each spritz
lifts the lid ($q_{\mathrm{lid}}$ in the pit balance). Only the absorbed
fraction counts toward retained water, which is what keeps the planner
from spraying its way to juiciness. All three policy parameters are
differentiable decisions.

**Rest / carryover (pull).** A second blend of the same construction,
$p(t) = \sigma\!\left((t - t_{\mathrm{pull}})/\delta_p\right)$
($\delta_p = 300$ s), swaps the surface environment from the pit to the
rest; the interior equations are untouched, so carryover (conduction),
juice redistribution (diffusion with the surface flux blocked), and the
hold's continued collagen conversion follow with no new physics:

$$
k\,\partial_x T|_s
= (1-p)\,q_{\mathrm{surf}}
+ p\,\big[h_r\,(T_r - T_s) + \varepsilon_r\,\sigma_{SB}\,(T_r^4 - T_s^4)\big]
- \lambda\, j,
$$

$$
j = (1-p)\,(1-\eta_e s)\,j_{\mathrm{2s}}(T_s, W_s;\,k_m,\,p_{v,\mathrm{pit}})
+ p\,(1-\eta_r)\,j_{\mathrm{2s}}(T_s, W_s;\,k_m\tfrac{h_r}{h_c},\,p_{v,r}),
$$

with $j_{\mathrm{2s}}$ the two-sided flux above,

with the rest-phase mass-transfer coefficient scaled by the heat–mass
analogy $h_r/h_c$, smoke deposition gated by $(1-p)$, and the coupling
output $q_{\mathrm{meat}} = A_{\mathrm{meat}}\,(1-p)\,q_{\mathrm{surf}}$
(the pit sees no load once the meat is out). The rest constants are
literature-fixed presets (faux cambro: $T_r = T_{\mathrm{amb}}$, $h_r =
1.5$, $\varepsilon_r = 0.1$, $\eta_r = 0.9$; bare board: $h_r = 7$,
$\varepsilon_r = 0.85$, $\eta_r = 0$) — the cook log ends at the pull, so
nothing here is calibrated. $t_{\mathrm{pull}}$ is a differentiable
input like $t_{\mathrm{wrap}}$ (`PULL_NEVER = 1e9` puts the blend at
numerical zero). With a rest appended to the horizon, every quality
summary below is evaluated at serve; the objective prices fuel at the pull
and the bark-burn cap in the pit only.

**Shrinkage (material coordinate).** The fields above are integrated on
the dry-matter (Lagrangian) coordinate $\xi \in [0, L]$ — $L$ is the raw
half-thickness, dry mass per unit $\xi$ is constant, and the grid never
moves. Density bookkeeping gives each node's specific volume per kg dry
matter and its thickness stretch,

$$
v(W,F) = \frac{1-F_0}{\rho_{\mathrm{sol}}}
+ \frac{F + (1-\eta_{\mathrm{drain}})(F_0 - F)}{\rho_f}
+ \frac{W}{\rho_w},
\qquad
\varsigma = \big(v/v_0\big)^{n_{sh}},
\qquad
\frac{A(t)}{A_0} = \big(\bar v/v_0\big)^{1-n_{sh}},
$$

($v_0 = v(W_0,F_0)$; the drained fat's volume has left, the retained
rendered share stays). Volumetric shrinkage is therefore not a free
parameter — only its split between thickness ($n_{sh}$: $1/3$ isotropic,
$1$ all-thickness) and the lateral area is. The interior equations in
$\xi$ are, with heat per physical volume ($\rho c_p$ fixed per volume) and
water per dry mass (the dry-matter concentration rises as the matrix
contracts, hence $\varsigma^{-2}$):

$$
\rho c_p\,\varsigma\,\partial_t T = \partial_\xi\!\Big(\frac{k}{\varsigma}\,\partial_\xi T\Big),
\qquad
\partial_t W = \partial_\xi\!\Big(\frac{D_{\mathrm{eff}}}{\varsigma^{2}}\,\partial_\xi W\Big)
- k_e\,\mathrm{sp}_{\delta_e}\!\big(W - W_{\mathrm{eq}}(T)\big),
$$

discretized in flux form with face stretches (mean of the neighbours). At
the surface the heat balance is unchanged per physical area (the area
factor cancels between capacity and intake) while the water balance sees
the shrunken exposed surface, $-\rho_d D_{\mathrm{eff}}\,\varsigma^{-2}\,
\partial_\xi W|_s = (A/A_0)\, j$; the pit coupling becomes
$q_{\mathrm{meat}} = A_0\,(A/A_0)\,q_{\mathrm{surf}}$, and the physical
thickness $L(t) = \int_0^{L}\varsigma\,d\xi$ is an output (`l_thick`; a
ruler measures it). $n_{sh} = 1/3$ (isotropic).

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

**Quality summaries.** Endpoint cells have half volume on the uniform slab,
so field means use the finite-volume weight
\(\overline{v} = \bigl(\tfrac12 v_1 + \sum_{i=2}^{n-1} v_i + \tfrac12 v_n\bigr)/(n-1)\)
(`fv_mean` in `src/meat.jl`). \(\eta_{\mathrm{drain}}\) is the rendered-fat
fraction that drips:

$$
w_{\mathrm{ret}} = \frac{\overline W(t_f)}{W_0},\qquad
f_{\mathrm{rend}} = 1 - \frac{\overline F(t_f)}{F_0},\qquad
\mathrm{juiciness}(W, F) = w_{\mathrm{ret}}
+ \beta_{\mathrm{fat}}\,\frac{(1-\eta_{\mathrm{drain}})\,F_0\, f_{\mathrm{rend}}}{W_0},
$$

with the cook-loss budget (drip is defined by the closure; the
discretized water balance is checked against the integrated expulsion
sink in `test/test_meat.jl`):
water lost $= \int j\,dt/(\rho_d L)$ (evaporated) $+$ drip (purge), fat
lost $= \eta_{\mathrm{drain}} F_0 f_{\mathrm{rend}}$ (tallow catch).

**Smoke ring** (output only; not in the objective). NO from the fire
dissolves into the wet surface and penetrates inward; it binds myoglobin as
the pink nitrosyl complex only where the meat has not yet passed the
myoglobin denaturation temperature, so the ring is a diffusion front that
stops when it meets the denaturation front. One surface-anchored state,
the ring depth $r$ in the material coordinate:

$$
\frac{dr}{dt} = \frac{D_{\mathrm{ring}}\,\Theta(t)}{r + r_0}\;
\sigma\!\Big(\frac{T_{\mathrm{den}} - T(L - r, t)}{\delta_{\mathrm{den}}}\Big),
\qquad
\Theta = \mathrm{wet}(a_w)\,\frac{\dot m_b}{\dot m_b + \dot m_0}\,(1 - s)\,(1 - p),
$$

($r\,dr = D\,\Theta\,dt$ is the penetration depth of a diffusing species
under a gated surface source; the pit-gas NO concentration is set by the
air–fuel ratio, so the fire enters only as "lit"; foil and the board stop
uptake). $D_{\mathrm{ring}} = 1.5\cdot10^{-9}$ m²/s (effective, NO-assisted
— nitrite alone diffuses at $2$–$3\cdot10^{-10}$), $T_{\mathrm{den}} = 62$ °C,
$r_0 = 0.5$ mm. The physical depth is $r$ times the final stretch
$l(t_f)/L$. Literature-scale constants with no ring measurement behind
them (`docs/roadmap.md`, *Residues of the implemented physics*).

## 2. Pit: lumped 3-state ODE

With damper $u(t) \in [0,1]$ and airflow
$g(u) = u_{\mathrm{leak}} + (1 - u_{\mathrm{leak}})\,u$ (the leak floor
means closing the damper never kills the fire):

$$
\dot m = -\,\dot m_b + \dot m_{\mathrm{add}}(t),\qquad
\dot m_b = r_{\max}\, g(u)\, a\,
\tanh\!\Big(\frac{\max(m,0)}{m_t}\Big),
$$

$$
C_{th}\,\dot T_p = \Delta H_{e\!f\!f}\,\dot m_b
- UA\,(T_p - T_{\mathrm{amb}}) - q_{\mathrm{meat}} - q_{\mathrm{lid}},
\qquad
\dot a = k_{\mathrm{spr}}\, a\,(1-a)\, g(u)
+ (1-a)\,\frac{\dot m_{\mathrm{add}}}{\max(m, m_t)},\quad a(0) = a_0 .
$$

**Lid-open loss.** Every lid lift costs the pit: $q_{\mathrm{lid}} =
UA_{\mathrm{open}}\,(T_p - T_{\mathrm{amb}})\,\mathrm{open}(t)$ with
$\mathrm{open}$ a unit-height smooth window — two minutes to dump a chimney,
thirty seconds per spritz while the policy runs (in the pit, unwrapped).
$UA_{\mathrm{open}} = 110$ W/K is what a lift loses to convective exchange
through the open top plus radiation from the bed (≈ 0.3 MJ per 30 s at a
90 K pit–ambient difference); in a lumped model that is a ~7 K dip of the
effective thermal mass recovering on $C_{th}/UA$, not the 25 K dome-gas
transient a probe reads for a minute. It is the cost that makes opening
the lid a real decision.

**Fuel addition.** One lit chimney of mass $m_{\mathrm{ch}}$ (1.4 kg) is
dumped over $\tau_{\mathrm{add}}$ (5 min) at $t_{\mathrm{add}}$ as a
smooth unit-area pulse, $\dot m_{\mathrm{add}} = m_{\mathrm{ch}}\,
\sigma'\big((t - t_{\mathrm{add}})/\tau_{\mathrm{add}}\big)/\tau_{\mathrm{add}}$;
lit coals also move the ignited fraction of the (now larger) bed toward 1.
$t_{\mathrm{add}}$ is a differentiable decision (never = beyond the
horizon, exactly zero); the lid lift it costs is $q_{\mathrm{lid}}$ below.

At $a_0 = 1$ the spread term vanishes and the two-state model is
recovered exactly. $\dot m_b$ is exported (exact algebraic recompute at
save times) as the smoke-flux forcing to the bark model.

## 3. Coupling: one solve

Pit, meat and the quasi-steady humidity balance are integrated together as
one stiff system (`src/monolith.jl`): the pit sees the meat's instantaneous
heat load $q_{\mathrm{meat}} = n_{\mathrm{meat}}\,A(t)\, q_{\mathrm{surf}}$
($n_{\mathrm{meat}}$ identical briskets sharing the pit; their vapor scales
the humidity balance the same way) and the meat sees
the instantaneous $T_p$, $\dot m_b$ and $p_{v,\mathrm{base}}$. There is no
exchange grid and no truncation; derivatives are forward-mode
(ForwardDiff) through the solve and exposed as the monolith Tesseract's
`jacobian_vector_product`, `vector_jacobian_product` and `jacobian`
endpoints. The Picard composition of the two component solvers
(`picard_cook`) is kept as the test oracle: it converges to the monolith
with the pass count (docs/monolith.md).

## 4. The quality objective (KCBS-weighted; every term overridable)

Decision variables: damper knots $z \mapsto u = \sigma(z)$, wrap
$z_w \mapsto t_{\mathrm{wrap}} = t_f\,\sigma(z_w)$, pull
$z_p \mapsto t_{\mathrm{pull}} = t_f\,\sigma(z_p)$ (the meat rests from the
pull to serve, $t_f + t_{\mathrm{rest}}$, where the quality terms are
evaluated), the chimney $z_a \mapsto t_{\mathrm{add}} = t_f\,\sigma(z_a)$
and the spritz policy $z_s \mapsto t_{\mathrm{sp}} = t_f\,\sigma(z_s)$,
$z_d \mapsto \Delta_{\mathrm{sp}} = \Delta_{\min} + (\Delta_{\max} -
\Delta_{\min})\,\sigma(z_d)$ (20 min – 2 h), $z_m \mapsto m_{\mathrm{sp}}$
(0.02–0.10 kg/m²). Fuel is priced at the pull as load + chimney − remaining. Below
$\mathrm{sp}(x) = \mathrm{sp}_1(x) = \log(1+e^x)$, and the category
weights are $(w_{\mathrm{app}}, w_{\mathrm{taste}}, w_{\mathrm{tend}}) =
(0.14,\, 0.57,\, 0.29)$ — the normalized multipliers KCBS judging
carries.

**Tenderness** $\mathcal T(C_0)$ — two-sided collagen band at the center
node, each side normalized to its band width (under: $c^\star = 0.90$,
width 0.05; over: $c_{od} = 0.95$, weighted $\rho_{ov} = 1$ — pot roast
counts like tough; `over_rel` in `ControlConfig`):

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
unconditional, evaluated on the surface (intact-muscle contamination)
*and* on the core node — the cold spot of the symmetric slab, the last
point to heat, since the condition must hold in all parts of the product.
Lethality via the z-value equivalence (the FSIS Appendix A 6.5-log line:
720 s at the 60 °C reference,
$z = 6$ K, requirement $F_{\mathrm{req}} = 720$ equivalent seconds),
penalized on the log shortfall, plus a danger-zone dwell cap
($h_{\mathrm{zone}}$ = hours with $T$ in 4.4–60 °C, cap 4 h), each for
$T \in \{T_s, T_{\mathrm{core}}\}$:

$$
F_{\mathrm{leth}}[T] = \int_0^{t_f} 10^{(T - T_{\mathrm{ref}})/z}\, dt,
\qquad
\mathcal S = \sum_{T} w_F\,\mathrm{sp}\Big(\tfrac{\log_{10} F_{\mathrm{req}} - \log_{10} F_{\mathrm{leth}}[T]}{0.2}\Big)^2
+ w_D\,\mathrm{sp}\Big(\tfrac{h_{\mathrm{zone}}[T] - 4\,\mathrm{h}}{0.5}\Big)^2 .
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

**Synthetic** data is generated iid, so the synthetic model is iid
Gaussian on both probes.

**Real cook** — masked AR(1) Gaussian. Lid events and probe-settling
holes break the chain (contiguous usable runs only). $\sigma_0 = 1.2$ K
is the digitization floor; $\sigma_{\mathrm{pit}},\sigma_{\mathrm{core}}$
are sampled extras on the *marginal* residual scale; $\rho$ is a
per-probe lag-1 correlation on the 10-min grid, $\rho = 0.95\,\beta$ with
$\beta \sim$ Beta$(2,2)$ — a correlation time of at most ~3 h, so the
discrepancy process cannot become a free offset:

$$
\sigma = \sigma_0 + \sigma_{\cdot},\qquad
y_{t_0} \sim \mathcal N(\mu_{t_0},\,\sigma^2),\qquad
y_t \mid y_{t-1}
\sim \mathcal N\big(\mu_t + \rho\,(y_{t-1}-\mu_{t-1}),\,
\sigma^2(1-\rho^2)\big)
$$

with $\mu^{\mathrm{pit}} = T_p(t;\theta,\varphi)$ and
$\mu^{\mathrm{core}} = T_{\mathrm{core}}(t;\theta,\varphi)$, independently
for each probe. $\rho = 0$ recovers the iid model. LogNormal priors
centered on the model defaults (Beta for the (0,1)-bounded
$u_{\mathrm{leak}}, a_0, \rho$). MAP by Gauss–Newton on the monolith's
Jacobian (`scripts/run_gn_map.py`). Evening-scale posterior by full-rank
ADVI (`scripts/run_vi.py`). Exact posterior by NUTS
(`scripts/run_nuts_mono.py`), every leapfrog step pulling one VJP through
the Tesseract.
