# 1D slab heat–moisture model of a brisket flat with smoothed surface
# evaporation (the stall mechanism) and collagen-denaturation kinetics.
#
# Geometry: half-slab x ∈ [0, L] with symmetry at x = 0 (center, where the
# temperature probe sits) and the convective/radiative/evaporative boundary at
# x = L (surface). Method of lines on nx uniform nodes; states are the fields
# T [K], W [kg water / kg dry matter], collagen conversion C ∈ [0, 1], and
# un-rendered fat F [kg fat / kg dry matter], plus the two surface bark
# states — 4·nx + 3 ODEs.
#
# Differentiation: forward mode (ForwardDiff) through the coupled monolith
# (src/monolith.jl). The RHS is written branch-free (see interpolation.jl)
# so it is smooth in every input.

Base.@kwdef struct MeatConfig
    nx::Int = 40                # spatial nodes
    A_meat::Float64 = 0.12      # exposed surface area [m²] (heat-exchange scaling to the pit)
    T0::Float64 = 281.15        # initial meat temperature [K] (~8 °C)
    W0::Float64 = 2.33          # initial dry-basis moisture (≈70% wet basis)
    ε::Float64 = 0.85           # surface emissivity
    λ_vap::Float64 = 2.3e6      # latent heat of vaporization [J kg⁻¹]
    p_v_pit::Float64 = 2000.0   # pit vapor pressure [Pa] when no humidity forcing is supplied
    g_ref::Float64 = 0.5        # draft airflow at which h_c and k_m are quoted (mid-damper)
    n_meat::Float64 = 1.0       # identical briskets sharing the pit (heat load and vapor scale)
    # spritz: one dose of water at surface temperature T_spritz, sprayed over
    # τ_spritz, at t_spritz + k·dt_spritz while in the pit and unwrapped
    m_spritz::Float64 = 0.05    # water per spritz per exposed area [kg/m²] (~8 mL on a brisket)
    τ_spritz::Float64 = 60.0    # spray duration [s]
    T_spritz::Float64 = 293.15  # spritz liquid temperature [K]
    f_abs::Float64 = 0.2        # fraction of a spritz that soaks into the crust; the rest flashes off (assumed)
    # Doneness kinetics: first-order Arrhenius conversion of the doneness
    # state C. The DEFAULTS are beef collagen (probe-tender); other cuts are
    # PRESETS of these three constants, not new code — e.g. a poultry/myosin
    # doneness or a fish doneness is a different (Ea, k_ref, T_ref) triple
    # from the literature, and pasteurization is not C at all but the
    # lethality integral (control.py lethality_dwell / safety_report).
    Ea::Float64 = 173e3         # collagen denaturation activation energy [J mol⁻¹]
    k_ref::Float64 = 1e-6       # collagen rate at T_ref [s⁻¹]
    T_ref::Float64 = 333.15     # collagen reference temperature [K] (60 °C)
    # Second route to tenderness: ENZYMATIC (collagenase / cathepsin)
    # weakening below the thermal window — the sous-vide mechanism
    # (Tornberg 2005; Bouton & Harris 1981; Baldwin: tough cuts tender
    # after 24–48 h at 55 °C). A bell in T, rising from ~45 °C and killed
    # by enzyme inactivation above ~60 °C, ADDED to the thermal Arrhenius
    # rate: dC/dt = [k_C(T) + k_enz(T)] (1 − C). Sized so 24 h at the
    # peak converts 90 %; on a BBQ come-up the core crosses the window in
    # 1–3 h, so the route contributes a few % of C that the later thermal
    # conversion mostly overwrites (docs/results.md, two-rate collagen).
    # Literature scale, not texture-calibrated (roadmap C6).
    # k_enz0 = 0 is bitwise the single Arrhenius (fish / myosin presets).
    k_enz0::Float64 = 2.7e-5    # peak enzymatic conversion rate [s⁻¹] (ln 10 / 24 h)
    T_enz_on::Float64 = 318.15  # enzymatic onset [K] (45 °C)
    T_enz_off::Float64 = 333.15 # enzyme inactivation [K] (60 °C)
    δ_enz::Float64 = 3.0        # width of both edges [K]
    R_gas::Float64 = 8.314
    # State-triggered wrap (the event-sensitivity mechanism): a smooth
    # IRREVERSIBLE latch state l with dl/dt = k_latch·σ((T_core −
    # T_trig)/δ_trig)·(1 − l), driving the wrap blend as the smooth OR
    # s = 1 − (1 − s_time)(1 − l). Because the latch is an ordinary
    # smooth state, d(anything)/d(T_trig) exists through plain AD — no
    # jump conditions. T_trig itself is a meat_solve ARGUMENT
    # (T_wrap_trigger; Inf = disabled, exactly inert); these two are the
    # latch dynamics constants.
    # δ_trig sets BOTH the softness at crossing and the pre-trigger
    # accumulation: the latch integrates k_latch·σ((T−T_trig)/δ) over the
    # whole approach, so a wide δ pre-charges it hours early (a 2 K width
    # was ~97 % committed before the crossing). 0.5 K keeps the tail
    # negligible until minutes before the trigger reading.
    δ_trig::Float64 = 0.5       # trigger softness [K]
    k_latch::Float64 = 1.0 / 60.0  # latch rate [1/s] (~1 min to commit)
    # Bone region (bone-in cuts: butts, drumsticks, ribs near the bone).
    # A central bone out to bone_frac of the half-thickness/radius, as a
    # smooth per-node THERMAL property blend: conductivity and volumetric
    # heat capacity interpolate to bone values over a logistic of width
    # δ_bone_frac. v1 is thermal-only — the moisture/collagen states in
    # the bone region still evolve as meat (a small, documented error);
    # excluding bone from the water bookkeeping is the follow-up.
    # bone_frac = 0 is the boneless bitwise default (weights exactly 0).
    # Cortical bone conducts slightly better than meat but stores far
    # less heat, so a bone-in center heats FASTER — the folklore "cold
    # near the bone" is about joints and mass, not the material.
    bone_frac::Float64 = 0.0    # bone radius as a fraction of L (0 = boneless)
    δ_bone_frac::Float64 = 0.08 # blend width, fraction of L
    k_bone::Float64 = 0.55      # bone conductivity [W/mK] (cortical, literature)
    ρc_bone::Float64 = 2.2e6    # bone volumetric heat capacity [J/m³K]
    # Salt physics (dry brine; the structural half of the salting item —
    # literature constants, calibration gated on the scale + designed
    # experiment). The pre-cook brine of duration t_salt leaves the
    # analytic semi-infinite profile s0 = S_surf·erfc(depth/(2√(D_salt t)))
    # (per face, material depth); during the cook s diffuses with a
    # zero-flux surface (salt does not evaporate) and couples through
    # (a) water-holding capacity — W_eq → W_eq·(1 + β_whc·s) in the purge
    # sink — and (b) Raoult-scale water-activity depression of surface
    # evaporation, a factor 1/(1 + k_aw·s). t_salt = 0 is EXACTLY
    # unsalted: s ≡ 0 and both couplings are the identity. t_salt lives in
    # the config for now; promoting it to a differentiable decision
    # variable ("when to salt" beside "when to light") is the planner
    # step. Salt–bark interaction is not modelled.
    t_salt::Float64 = 0.0       # dry-brine duration before the cook [s]
    D_salt::Float64 = 1.0e-10   # NaCl diffusivity in muscle [m²/s] (ham-curing literature; fixed, not calibrated)
    S_surf::Float64 = 0.08      # surface salt held by the rub, dry basis [kg/kg] (placeholder scale)
    β_whc::Float64 = 3.0        # water-holding gain per unit s (placeholder scale)
    k_aw::Float64 = 0.55        # water-activity depression per unit s (Raoult scale)
    # Skin barrier (poultry-class surface): a finite skin mass-transfer
    # conductance in SERIES with the boundary layer, k_eff = 1/(1/k_m +
    # 1/k_skin) — heat still passes, evaporation is throttled. Inf (the
    # default) is bitwise no-barrier; the skin does not scale with draft.
    k_skin::Float64 = Inf       # skin mass-transfer conductance [kg m⁻² s⁻¹ Pa⁻¹ scale of k_m]
    # Texas-crutch wrap constants (foil defaults; butcher paper ≈ η_evap 0.5,
    # ε_wrap 0.3). Literature-fixed pending a wrapped-cook log — the wrap
    # moment is documented in such logs, making the wrap event a known-input
    # step-response experiment for these constants (see roadmap).
    δ_wrap::Float64 = 600.0     # wrap blend width [s] (~wrapping takes minutes)
    η_evap::Float64 = 0.9       # evaporation suppression when wrapped (foil)
    ε_wrap::Float64 = 0.10      # wrapped surface emissivity (foil)
    η_h::Float64 = 0.2          # convective-coefficient reduction when wrapped
    # Rest / carryover (docs/model.md, rest): after t_pull the surface sees
    # the rest environment instead of the pit — carryover, juice
    # redistribution and continued collagen conversion then follow from the
    # unchanged interior equations. Defaults = faux cambro (foil + towel +
    # cooler): slow conductive loss to room air through insulation, foil
    # emissivity, evaporation mostly blocked. Board rest ≈ (h 7, ε 0.85,
    # η 0); warming cabinet ≈ (T 335 K, h 5, ε 0.85, η 0.9). Literature-fixed:
    # the cook log ends at the pull, so nothing here is calibrated.
    δ_pull::Float64 = 300.0     # pull blend width [s] (pull + wrap for the hold)
    T_rest::Float64 = 293.15    # rest environment temperature [K]
    h_rest::Float64 = 1.5       # effective convective coefficient at rest [W m⁻² K⁻¹]
    ε_rest::Float64 = 0.10      # effective emissivity at rest (foil)
    η_rest::Float64 = 0.9       # evaporation suppression at rest (wrapped)
    p_v_rest::Float64 = 1200.0  # rest-environment vapor pressure [Pa] (room air)
    # Shrinkage (docs/model.md, shrinkage): the PDE is solved on the dry-matter
    # (Lagrangian) coordinate, so the grid never moves; the local thickness
    # stretch st = (v/v0)^n_shrink enters as a face metric, with v the
    # specific volume per kg dry matter (density bookkeeping — volumetric
    # shrinkage is not a free parameter, only how much of it shows up in
    # thickness: 1/3 isotropic, 1 all-thickness). The lateral remainder
    # scales the exposed area A(t)/A0 = (v̄/v0)^(1 - n_shrink). L is the raw
    # half-thickness.
    n_shrink::Float64 = 1 / 3
    # Geometry mode: curvature exponent m of the 1-D conduction/diffusion
    # operator (1/r^m)·∂_r(r^m·…) — 0 slab (default; the brisket), 1 infinite
    # cylinder, 2 sphere. θ's L is the half-thickness (slab) or the radius.
    # Curvature is applied in the material coordinate: exact at uniform
    # shrink stretch (and st ≡ 1), first-order otherwise (docs/roadmap.md).
    geom_m::Float64 = 0.0
    ρ_sol::Float64 = 1350.0     # non-fat solids (protein + ash) density [kg m⁻³]
    ρ_fat::Float64 = 900.0      # fat density [kg m⁻³]
    ρ_water::Float64 = 980.0    # water density near cooking temperature [kg m⁻³]
    # Bark model constants (literature-anchored, uncalibrated pending photo
    # data; docs/roadmap.md item 8 carries the calibration plan).
    # k0 is the bread-crust value (Purlis, 4.2e4) scaled x5: a rubbed meat
    # surface browns faster than lean bread crust (protein-rich, rub sugars),
    # and the bread constant leaves Maillard invisible at smoker temperatures.
    # Effective value; pending photo calibration like everything bark.
    k0_bark::Float64 = 2.0e5    # Maillard pre-exponential [1/s]
    Ea_bark::Float64 = 64151.0  # Maillard activation energy [J/mol] (Purlis)
    a_pk::Float64 = 0.65        # Maillard water-activity bell peak
    σ_aw::Float64 = 0.15        # bell width
    g_dry::Float64 = 0.5        # dry-side floor of the bell: a dry crust keeps browning
    δ_g::Float64 = 0.05         # width of the dry-side switch
    k_ox::Float64 = 0.10        # Oswin isotherm scale (a_w = 0.5 at W = k_ox)
    n_ox::Float64 = 1.5         # Oswin isotherm exponent
    k_dep::Float64 = 0.5        # deposition-saturation rate per kg fuel burned while wet [1/kg]
    a_dry::Float64 = 0.85       # deposition needs a_w above this (moist surface)
    δ_dry::Float64 = 0.05       # smoothing width of the wet switch
    L0_bark::Float64 = 52.0     # initial (rubbed, raw) surface lightness L*
    dL_B::Float64 = 20.0        # max L* drop from Maillard browning
    dL_S::Float64 = 10.0        # max L* drop from smoke deposition
    # smoke ring (docs/model.md): NO from the fire dissolves into the wet
    # surface and penetrates as a diffusion front, fixing myoglobin (the
    # pink nitrosyl complex) only where the meat has not yet passed the
    # myoglobin denaturation temperature. Literature-scale constants, no
    # ring measurement to check them against (docs/roadmap.md).
    D_ring::Float64 = 1.5e-9    # effective NO penetration diffusivity [m²/s]
    T_ring::Float64 = 335.15    # myoglobin denaturation (~62 C): fixation stops above
    δ_ring::Float64 = 3.0       # width of the denaturation gate [K]
    r_ring0::Float64 = 5e-4     # surface layer NO enters directly [m] (front regularization)
    mdot_ring::Float64 = 2e-5   # burn rate at which the fire counts as lit [kg/s]
    # Moisture physics (docs/model.md).
    # Fat rendering: tallow melts ~32-49 C, drains (cell rupture) from
    # ~54 C; k_f0 sets an hours-scale render time in the BBQ band.
    F0::Float64 = 0.22          # initial fat fraction, dry basis (marbling)
    T_render::Float64 = 328.15  # render switch center [K] (55 C)
    δ_render::Float64 = 4.0     # switch width [K]
    k_f0::Float64 = 4.6e-5      # render rate when hot [1/s] (~1/6h)
    η_drain::Float64 = 0.5      # rendered fat fraction that drains (drips)
    β_fat::Float64 = 1.8        # perceived-juiciness weight of retained
    # rendered fat vs water (sensory placeholder)
    # Denaturation-driven water expulsion (purge): relaxation toward the
    # temperature-dependent water-holding equilibrium W_eq(T), which drops
    # in two literature-shaped steps (myosin ~58 C, actin ~72 C).
    k_e::Float64 = 5.6e-4       # expulsion relaxation rate [1/s] (~1/30min)
    δ_e::Float64 = 0.02         # softplus width of the one-sided relaxation
    a_myosin::Float64 = 0.20    # W_eq drop at the myosin step
    T_myosin::Float64 = 331.15  # 58 C
    δ_myosin::Float64 = 4.0
    a_actin::Float64 = 0.25     # W_eq drop at the actin step
    T_actin::Float64 = 345.15   # 72 C
    δ_actin::Float64 = 4.0
end

# θ = [D_eff, k_m, h_c, L, k_mult, k_e, a_myosin, m_conv] — the meat-side
# latent parameter vector. m_conv is the draft exponent: h_c and k_m scale
# with the pit airflow as (g/g_ref)^m_conv (forced convection, heat–mass
# analogy), so the damper is a convection lever as well as a heat lever. k_e (expulsion relaxation rate) and a_myosin (W_eq drop
# at the myosin denaturation step) joined θ 2026-08-21: with them frozen at
# literature values the real-cook refit plateaued at loss 495 vs 245 — the
# five original knobs were compensating for expulsion kinetics the log can
# see (they shape the stall entry). cfg.k_e / cfg.a_myosin remain as the
# uncalibrated defaults for cfg-only call paths.
const MEAT_NP = 8

"""
    with_config(cfg::MeatConfig; overrides...)

A copy of `cfg` with the given fields replaced (kwdef reconstruction) —
the preset files build cuts and cooking environments with it.
"""
with_config(cfg::MeatConfig; kw...) = MeatConfig(;
    merge(
        NamedTuple{fieldnames(MeatConfig)}(
            ntuple(i -> getfield(cfg, i), fieldcount(MeatConfig))
        ),
        values(kw),
    )...,
)

# Default wrap time: far beyond any horizon. logistic((t − 1e9)/δ_wrap) is
# EXACTLY 0.0 (tanh saturates), so the unwrapped model is recovered bitwise
# and the reverse-mode derivative underflows to a clean zero.
const WRAP_NEVER = 1.0e9
# Default pull time: same construction — the pull blend is exactly 0.0 over
# any horizon.
const PULL_NEVER = 1.0e9
# Default spritz start: beyond any horizon (no spritz).
const SPRITZ_NEVER = 1.0e9

"""
    spritz_pulse_time(t, t_spritz, dt_spritz)

Centre of the spritz pulse nearest to `t`: `t_spritz + k·dt_spritz` with
`k = max(round((t − t_spritz)/dt_spritz), 0)` — pulses are far apart
compared with their width, so one pulse is all that matters at any `t`.
"""
spritz_pulse_time(t, t_spritz, dt_spritz) =
    t_spritz + max(round((t - t_spritz) / dt_spritz), 0.0) * dt_spritz

"""
    spritz_rate(t, t_spritz, dt_spritz, m_spritz, cfg::MeatConfig)

Water arriving at the surface from a spritz policy [kg m⁻² s⁻¹]: the dose
`m_spritz` [kg/m²] as a smooth unit-area pulse of width `cfg.τ_spritz` at
`t_spritz + k·dt_spritz`, `k = 0, 1, …` ([`spritz_pulse_time`](@ref)).
`t_spritz = SPRITZ_NEVER` is exactly zero everywhere.
"""
function spritz_rate(t, t_spritz, dt_spritz, m_spritz, cfg::MeatConfig)
    s = logistic((t - spritz_pulse_time(t, t_spritz, dt_spritz)) / cfg.τ_spritz)
    return m_spritz * s * (1 - s) / cfg.τ_spritz
end

"""
    evap_flux(Ts::Number, Ws::Number, k_m::Number, [p_v::Number,] cfg::MeatConfig)

Surface mass flux, two-sided and linear in the difference between the
surface's own equilibrium vapor pressure — the Oswin isotherm `a_w(Ws)`
times saturation — and the ambient: evaporation while `a_w p_sat > p_v`,
condensation onto the hygroscopic surface when it is not. Smooth in every
input. As the surface dries the isotherm lowers its vapor pressure and the
flux becomes limited by what the interior supplies (the falling-rate
period); condensation is what a cold brisket does in a humid pit for its
first half hour, and the wet surface it leaves is where smoke adheres.

# Arguments
- `Ts::Number`: Surface temperature [K].
- `Ws::Number`: Surface moisture, dry basis [kg water / kg dry].
- `k_m::Number`: Mass-transfer coefficient [m/s] (θ latent).
- `p_v::Number`: Ambient vapor pressure [Pa]; the 4-argument form uses the
  fixed pit value `cfg.p_v_pit`.
- `cfg::MeatConfig`: Physical configuration.

# Returns
- `j::Number`: Mass flux [kg m⁻² s⁻¹]; > 0 evaporation, < 0 condensation.
"""
function evap_flux(Ts, Ws, k_m, p_v, cfg::MeatConfig, f_aw=1.0)
    return k_m * (f_aw * a_w_oswin(Ws, cfg) * p_sat(Ts) - p_v) / (R_V * Ts)
end
evap_flux(Ts, Ws, k_m, cfg::MeatConfig) = evap_flux(Ts, Ws, k_m, cfg.p_v_pit, cfg)

"""
    evap_flux_fb(Ts, Ws, k_m, pv_base, c, cfg::MeatConfig)

The two-sided surface flux of [`evap_flux`](@ref) with the pit humidity
feedback (docs/model.md, pit humidity): the pit vapor pressure is `pv_base + c·j`
— the pit's own vapor balance (ambient, combustion water, pan) plus the
meat's vapor diluted by the air mass flow, `c` the mixing coefficient
[Pa per kg m⁻² s⁻¹] — so `j = a (a_w p_sat(Ts) − pv_base − c j)`, `a = k_m/(R_v Ts)`: linear,
`j = a (a_w p_sat − pv_base)/(1 + c a)` in closed form (condensation
lowers the pit humidity and is self-limiting). `c = 0` is `evap_flux`.

# Returns
- `j::Number`: Mass flux [kg m⁻² s⁻¹]; > 0 evaporation, < 0 condensation.
"""
function evap_flux_fb(Ts, Ws, k_m, pv_base, c, cfg::MeatConfig, f_aw=1.0)
    a = k_m / (R_V * Ts)
    return a * (f_aw * a_w_oswin(Ws, cfg) * p_sat(Ts) - pv_base) / (1 + c * a)
end

"""
    surface_heat_in(Tpit::Number, Ts::Number, h_c::Number, cfg::MeatConfig)

Unwrapped surface heat flux density into the meat (convective + radiative),
before the evaporative sink.

# Arguments
- `Tpit::Number`: Pit gas temperature [K].
- `Ts::Number`: Meat surface temperature [K].
- `h_c::Number`: Convective coefficient [W m⁻² K⁻¹] (θ latent).
- `cfg::MeatConfig`: Physical configuration (emissivity).

# Returns
- `q::Number`: Heat flux density [W m⁻²].
"""
function surface_heat_in(Tpit, Ts, h_c, cfg::MeatConfig)
    return h_c * (Tpit - Ts) + cfg.ε * σ_SB * (Tpit^4 - Ts^4)
end

"""
    wrap_blend(t::Number, t_wrap::Number, cfg::MeatConfig)

Smooth Texas-crutch switch: 0 unwrapped, → 1 after `t_wrap` over width
`δ_wrap`. Branch-free (tanh logistic), so it is safe in every AD path.
A smooth blend — not a DiffEq event — keeps t_wrap an ordinary parameter
for forward-mode differentiation through the solve (event times need jump
conditions, and FD across a discrete switch is ill-posed); it also smears
toward reality,
since wrapping a brisket takes minutes, not an instant.

# Arguments
- `t::Number`: Current time [s].
- `t_wrap::Number`: Wrap moment [s]; `WRAP_NEVER` saturates to exactly 0.
- `cfg::MeatConfig`: Physical configuration (`δ_wrap`).

# Returns
- `s::Number`: Wrap blend ∈ [0, 1].
"""
wrap_blend(t, t_wrap, cfg::MeatConfig) = logistic((t - t_wrap) / cfg.δ_wrap)

"""
    surface_heat_in_wrapped(Tpit::Number, Ts::Number, h_c::Number, s::Number,
                            cfg::MeatConfig)

Wrap-blended surface heat flux density: convection reduced by `η_h·s`,
emissivity blended toward `ε_wrap`. At `s = 0` this is exactly
[`surface_heat_in`](@ref).

# Arguments
- `Tpit::Number`: Pit gas temperature [K].
- `Ts::Number`: Meat surface temperature [K].
- `h_c::Number`: Convective coefficient [W m⁻² K⁻¹] (θ latent).
- `s::Number`: Wrap blend ∈ [0, 1] from [`wrap_blend`](@ref).
- `cfg::MeatConfig`: Physical configuration.

# Returns
- `q::Number`: Heat flux density [W m⁻²].
"""
function surface_heat_in_wrapped(Tpit, Ts, h_c, s, cfg::MeatConfig)
    return h_c * (1 - cfg.η_h * s) * (Tpit - Ts) +
           (cfg.ε * (1 - s) + cfg.ε_wrap * s) * σ_SB * (Tpit^4 - Ts^4)
end

# --- Rest / carryover (docs/model.md, rest) -------------------------------
"""
    pull_blend(t::Number, t_pull::Number, cfg::MeatConfig)

Smooth pull switch: 0 in the pit, → 1 resting after `t_pull` over width
`δ_pull`. Same construction as [`wrap_blend`](@ref) for the same reasons
(an ordinary differentiable parameter, not an event);
`PULL_NEVER` saturates to exactly 0.

# Arguments
- `t::Number`: Current time [s].
- `t_pull::Number`: Pull moment [s].
- `cfg::MeatConfig`: Physical configuration (`δ_pull`).

# Returns
- `pl::Number`: Pull blend ∈ [0, 1].
"""
pull_blend(t, t_pull, cfg::MeatConfig) = logistic((t - t_pull) / cfg.δ_pull)

"""
    surface_heat_rest(Ts::Number, cfg::MeatConfig)

Surface heat flux density into the meat from the REST environment
(convective + radiative at `T_rest`, `h_rest`, `ε_rest`) — negative while
the meat is hotter than the room.

# Arguments
- `Ts::Number`: Meat surface temperature [K].
- `cfg::MeatConfig`: Physical configuration (rest constants).

# Returns
- `q::Number`: Heat flux density [W m⁻²].
"""
function surface_heat_rest(Ts, cfg::MeatConfig)
    return cfg.h_rest * (cfg.T_rest - Ts) + cfg.ε_rest * σ_SB * (cfg.T_rest^4 - Ts^4)
end

"""
    surface_j_blended(Ts, Ws, h_c, k_m, s, pl, pv_base, c, cfg::MeatConfig)

Pull-blended evaporative mass flux: in the pit (`pl = 0`) the wrap-blended
[`evap_flux_fb`](@ref) into the pit's humidity (`pv_base`, mixing `c`); at
rest (`pl = 1`) the mostly blocked rest evaporation into room air, whose
mass-transfer coefficient follows the heat–mass analogy `k_m·h_rest/h_c`.
Shared by the RHS and the observables so the two can never disagree.

# Returns
- `j::Number`: Evaporative mass flux [kg m⁻² s⁻¹].
"""
function surface_j_blended(Ts, Ws, h_c, k_m, s, pl, pv_base, c, cfg::MeatConfig, f_aw=1.0)
    # skin barrier in series with the boundary layer (both branches); the
    # RHS and every observable pass through here, so they cannot disagree
    k_m = isinf(cfg.k_skin) ? k_m : 1 / (1 / k_m + 1 / cfg.k_skin)
    return (1 - pl) *
           evap_flux_fb(Ts, Ws, (1 - cfg.η_evap * s) * k_m, pv_base, c, cfg, f_aw) +
           pl * (1 - cfg.η_rest) *
           evap_flux(Ts, Ws, k_m * cfg.h_rest / h_c, cfg.p_v_rest, cfg, f_aw)
end

"""
    mixing_coefficient(a_area, imair, cfg::MeatConfig)

Vapor-balance mixing coefficient `c = (p_atm/0.622)·A(t)/ṁ_air`
[Pa per kg m⁻² s⁻¹] from the exposed-area factor and the inverse air mass
flow `imair` [s/kg].
"""
mixing_coefficient(a_area, imair, cfg::MeatConfig) =
    P_ATM / EPS_W * cfg.A_meat * cfg.n_meat * a_area * imair

"""
    surface_q_pit_blended(Tpit, Ts, h_c, s, pl, cfg::MeatConfig)

Heat flux density drawn FROM THE PIT (the coupling output): the wrap-blended
convective + radiative input, gated by the pull blend — zero once pulled.

# Returns
- `q::Number`: Heat flux density [W m⁻²].
"""
function surface_q_pit_blended(Tpit, Ts, h_c, s, pl, cfg::MeatConfig)
    return (1 - pl) * surface_heat_in_wrapped(Tpit, Ts, h_c, s, cfg)
end

# --- Shrinkage (docs/model.md, shrinkage) -------------------------------------
"""
    specific_volume(W, F, cfg::MeatConfig)

Volume per kg of (initial) dry matter [m³/kg]: non-fat solids, fat — the
un-rendered fat plus the retained rendered share (the drained share has
left the meat) — and water. Density bookkeeping, not a model: at the raw
state it closes with the raw density (1060 kg/m³) and `RHO_DRY` to ~1 %.

# Arguments
- `W::Number`: Dry-basis moisture [kg/kg].
- `F::Number`: Un-rendered fat, dry basis [kg/kg].
- `cfg::MeatConfig`: Physical configuration (`ρ_sol`, `ρ_fat`, `ρ_water`,
  `F0`, `η_drain`).

# Returns
- `v::Number`: Specific volume [m³ per kg dry matter].
"""
function specific_volume(W, F, cfg::MeatConfig)
    fat = F + (1 - cfg.η_drain) * (cfg.F0 - F)
    return (1 - cfg.F0) / cfg.ρ_sol + fat / cfg.ρ_fat + max(W, zero(W)) / cfg.ρ_water
end

"""
    heat_capacity_dry(W, F, T, cfg::MeatConfig)

Heat capacity per kg of dry matter [J kg⁻¹ K⁻¹] from the local composition
— water `W`, fat (`F` plus the retained rendered share) and the non-fat
solids `1 - F0` — with the Choi–Okos component specific heats at `T` (ash
lumped into protein). Divided by [`specific_volume`](@ref) it is the
volumetric heat capacity ρc_p of the node.
"""
function heat_capacity_dry(W, F, T, cfg::MeatConfig)
    fat = F + (1 - cfg.η_drain) * (cfg.F0 - F)
    return max(W, zero(W)) * cp_water_co(T) + fat * cp_fat_co(T) +
           (1 - cfg.F0) * cp_protein_co(T)
end

"""
    conductivity_mix(W, F, T, v, cfg::MeatConfig)

Thermal conductivity [W m⁻¹ K⁻¹] of the local mixture: the Choi–Okos
parallel model on the volume fractions of water, fat and non-fat solids
(`v` is the node's specific volume per kg dry matter,
[`specific_volume`](@ref)); `k_mult` (θ) scales it in the RHS.
"""
function conductivity_mix(W, F, T, v, cfg::MeatConfig)
    fat = F + (1 - cfg.η_drain) * (cfg.F0 - F)
    xw = max(W, zero(W)) / cfg.ρ_water / v
    xf = fat / cfg.ρ_fat / v
    xs = (1 - cfg.F0) / cfg.ρ_sol / v
    return xw * k_water_co(T) + xf * k_fat_co(T) + xs * k_protein_co(T)
end

"""
    shrink_exponents(cfg::MeatConfig)

`(n_s, n_a, v0)`: the thickness-stretch exponent, the lateral (area)
exponent `1 - n_s` (0 at `n_s = 0`), and the raw-state specific volume the
stretch is normalized by. Evaluated once per closure so the RHS stays
branch-free.
"""
function shrink_exponents(cfg::MeatConfig)
    n_s = cfg.n_shrink
    n_a = n_s > 0 ? 1 - n_s : 0.0
    return n_s, n_a, specific_volume(cfg.W0, cfg.F0, cfg)
end

"""
    fv_weights(n::Int)

Finite-volume node weights on the uniform slab mesh (endpoint cells have
half volume): `[½, 1, …, 1, ½]`, so `fv_weights(n)' * v == fv_mean(v) * (n - 1)`.
"""
fv_weights(n::Int) = vcat(0.5, ones(n - 2), 0.5)

"""
    fv_mean(v)

Finite-volume mean over the node values `v` (endpoint cells have half
volume): `(v[1] + v[end] + 2·Σ interior) / (2·(n − 1))`.
"""
fv_mean(v) = (first(v) + last(v) + 2 * sum(v[2:(end - 1)])) / (2 * (length(v) - 1))

# --- Bark model (surface states; docs/model.md) ---------------------------
"""
    a_w_oswin(W::Number, cfg::MeatConfig)

Smooth Oswin sorption isotherm: surface water activity from dry-basis
moisture (`a_w = 0.5` at `W = k_ox`).

# Arguments
- `W::Number`: Surface moisture, dry basis [kg/kg].
- `cfg::MeatConfig`: Physical configuration (`k_ox`, `n_ox`).

# Returns
- `aw::Number`: Water activity ∈ (0, 1).
"""
a_w_oswin(W, cfg::MeatConfig) = 1 / (1 + (cfg.k_ox / max(W, 1e-6))^cfg.n_ox)

"""
    g_aw(aw::Number, cfg::MeatConfig)

Maillard water-activity bell (peak `a_pk`, width `σ_aw`) with a dry-side
floor `g_dry` (switch width `δ_g`): browning is dilution-limited on a wet
surface, and a crust that has dried past the peak keeps browning at the
rate its temperature sets (Purlis's bread-crust browning starts because
the crust dries) — `g_dry` is a placeholder pending the photo calibration.

# Arguments
- `aw::Number`: Surface water activity.
- `cfg::MeatConfig`: Physical configuration.

# Returns
- `g::Number`: Rate multiplier ∈ (0, 1].
"""
function g_aw(aw, cfg::MeatConfig)
    bell = exp(-((aw - cfg.a_pk)^2) / (2 * cfg.σ_aw^2))
    return bell + (1 - bell) * cfg.g_dry * logistic((cfg.a_pk - aw) / cfg.δ_g)
end

"""
    wet_blend(aw::Number, cfg::MeatConfig)

Smoke deposition needs a moist (tacky) surface: smooth switch that turns
deposition on above water activity `a_dry`.

# Arguments
- `aw::Number`: Surface water activity.
- `cfg::MeatConfig`: Physical configuration (`a_dry`, `δ_dry`).

# Returns
- `w::Number`: Deposition gate ∈ (0, 1).
"""
wet_blend(aw, cfg::MeatConfig) = logistic((aw - cfg.a_dry) / cfg.δ_dry)

"""
    bark_rate(Ts::Number, aw::Number, cfg::MeatConfig)

Maillard browning rate: Arrhenius in surface temperature × the
water-activity bell.

# Arguments
- `Ts::Number`: Surface temperature [K].
- `aw::Number`: Surface water activity.
- `cfg::MeatConfig`: Physical configuration (bark kinetics constants).

# Returns
- `r::Number`: Browning rate [1/s].
"""
function bark_rate(Ts, aw, cfg::MeatConfig)
    return cfg.k0_bark * exp(-cfg.Ea_bark / (cfg.R_gas * Ts)) * g_aw(aw, cfg)
end

"""
    l_star_bark(B::Number, S::Number, cfg::MeatConfig)

Predicted surface lightness from the two bark states (affine map to CIELAB
L*): the raw rubbed surface darkens with Maillard browning `B` and smoke
deposition `S`.

# Arguments
- `B::Number`: Maillard browning state ∈ [0, 1).
- `S::Number`: Smoke deposition state ∈ [0, 1).
- `cfg::MeatConfig`: Physical configuration (`L0_bark`, `dL_B`, `dL_S`).

# Returns
- `L::Number`: Predicted CIELAB lightness L*.
"""
l_star_bark(B, S, cfg::MeatConfig) = cfg.L0_bark - cfg.dL_B * B - cfg.dL_S * S

# --- Moisture physics (docs/model.md) --------------------------------------
"""
    w_eq(T, a_myosin, cfg)

Temperature-dependent water-holding equilibrium (dry basis): W₀ scaled down
by two smooth denaturation steps (myosin ~58 °C amplitude `a_myosin`, actin
~72 °C amplitude `cfg.a_actin`). `a_myosin` is a θ latent (calibrated); the
actin amplitude stays a config constant.
"""
function w_eq(T, a_myosin, cfg::MeatConfig)
    w = cfg.W0 * (
        1 - a_myosin * logistic((T - cfg.T_myosin) / cfg.δ_myosin) -
        cfg.a_actin * logistic((T - cfg.T_actin) / cfg.δ_actin)
    )
    return max(w, zero(w))  # water-holding capacity cannot go negative
end
w_eq(T, cfg::MeatConfig) = w_eq(T, cfg.a_myosin, cfg)

"""
    expulsion_rate(W, T, k_e, a_myosin, cfg)

Smooth one-sided expulsion sink [1/s]: relax `W` toward `w_eq(T)` from above
only (softplus of the excess, width `cfg.δ_e`). `k_e` and `a_myosin` are θ
latents; the cfg-only method uses the uncalibrated defaults.
"""
function expulsion_rate(W, T, k_e, a_myosin, cfg::MeatConfig)
    return k_e * cfg.δ_e * log1pexp((W - w_eq(T, a_myosin, cfg)) / cfg.δ_e)
end

"""
    expulsion_rate_salted(W, T, s, k_e, a_myosin, cfg::MeatConfig)

[`expulsion_rate`](@ref) with the salt water-holding coupling: the
equilibrium is raised to `w_eq·(1 + β_whc·s)` (salted muscle holds more
water, so less purges). `s = 0` reproduces `expulsion_rate` exactly.
"""
function expulsion_rate_salted(W, T, s, k_e, a_myosin, cfg::MeatConfig)
    weq = w_eq(T, a_myosin, cfg) * (1 + cfg.β_whc * max(s, zero(s)))
    return k_e * cfg.δ_e * log1pexp((W - weq) / cfg.δ_e)
end

"""
    bone_weights(cfg::MeatConfig)

Per-node bone blend weights on the material grid (node 1 = center):
`w = logistic((bone_frac − ξ̂)/δ_bone_frac)`, exactly zero everywhere for
the boneless default. Static (fractions of L), so the RHS closure can
hoist them once.
"""
function bone_weights(cfg::MeatConfig)
    cfg.bone_frac <= 0 && return zeros(cfg.nx)
    return [
        logistic((cfg.bone_frac - (i - 1) / (cfg.nx - 1)) / cfg.δ_bone_frac)
        for i in 1:(cfg.nx)
    ]
end

"""
    erfc_smooth(x)

Smooth complementary-error-function approximation
`1 − tanh(2x/√π + 0.088x³)` (max error ~2e-3 for x ≥ 0): exact enough for
a literature-scale brine profile, dependency-free and ForwardDiff-safe.
"""
erfc_smooth(x) = 1 - tanh(1.1283791670955126 * x + 0.088 * x^3)

"""
    salt_profile(L, cfg::MeatConfig)

Dry-brine initial salt profile on the material grid (dry basis, kg/kg):
the semi-infinite constant-surface-concentration solution
`S_surf·erfc(d/(2√(D_salt·t_salt)))` at depth `d = L(1 − ξ̂)` from the
surface face. `t_salt ≤ 0` returns exact zeros (unsalted).
"""
_primal(x) = x isa ForwardDiff.Dual ? _primal(ForwardDiff.value(x)) : x

function salt_profile(L, cfg::MeatConfig, t_salt=cfg.t_salt)
    # branch on the PRIMAL value: ForwardDiff compares eps-aware, so a
    # Dual seeded exactly at the unsalted sentinel (Dual(0,1) <= 0 is
    # false -- 0+eps > 0) would fall into the sqrt(t) singularity and
    # return NaN partials, killing the whole Dual solve; the disabled
    # column must be exact zeros instead
    _primal(t_salt) <= 0 && return zeros(typeof(float(t_salt)), cfg.nx)
    pen = 2 * sqrt(cfg.D_salt * t_salt)
    return [
        cfg.S_surf * erfc_smooth(L * (1 - (i - 1) / (cfg.nx - 1)) / pen) for
        i in 1:(cfg.nx)
    ]
end
expulsion_rate(W, T, cfg::MeatConfig) = expulsion_rate(W, T, cfg.k_e, cfg.a_myosin, cfg)

"""
    fat_rate(T::Number, cfg::MeatConfig)

Fat render rate: smooth melt/drain switch (center `T_render`, width
`δ_render`) times the base rate `k_f0`.

# Arguments
- `T::Number`: Local temperature [K].
- `cfg::MeatConfig`: Physical configuration.

# Returns
- `r::Number`: Render rate [1/s].
"""
fat_rate(T, cfg::MeatConfig) = cfg.k_f0 * logistic((T - cfg.T_render) / cfg.δ_render)

"""
    log1pexp(x::Number)

Numerically stable softplus `log(1 + eˣ)` for the expulsion sink
(overflow-safe; `max`/`abs` are scalar AD primitives, so it is Enzyme-safe
and branch-free).

# Arguments
- `x::Number`: Input.

# Returns
- `y::Number`: `log(1 + exp(x))`, ≥ 0.
"""
log1pexp(x) = max(x, zero(x)) + log1p(exp(-abs(x)))

"""
    collagen_rate(T::Number, cfg::MeatConfig)

Collagen conversion (doneness) rate: the thermal Arrhenius route
referenced to `k_ref` at `T_ref` PLUS the enzymatic (collagenase) route
`k_enz0·σ((T − T_on)/δ)·σ((T_off − T)/δ)`, active only in the 45–60 °C
window (sous-vide tenderization; gone above the inactivation edge).
`k_enz0 = 0` recovers the single Arrhenius exactly.

# Arguments
- `T::Number`: Local temperature [K].
- `cfg::MeatConfig`: Physical configuration (`Ea`, `k_ref`, `T_ref`,
  `k_enz0`, `T_enz_on`, `T_enz_off`, `δ_enz`).

# Returns
- `r::Number`: Conversion rate [1/s].
"""
function collagen_rate(T, cfg::MeatConfig)
    k_th = cfg.k_ref * exp(-(cfg.Ea / cfg.R_gas) * (1 / T - 1 / cfg.T_ref))
    k_enz = cfg.k_enz0 * logistic((T - cfg.T_enz_on) / cfg.δ_enz) *
            logistic((cfg.T_enz_off - T) / cfg.δ_enz)
    return k_th + k_enz
end

"""
    meat_rhs_core!(du, u, t, θ, Tpit, smoke, pv_base, imair, airflow, t_wrap,
                   t_pull, t_spritz, dt_spritz, m_spritz, cfg, n_s, n_a, v0)

The meat right-hand side under the INSTANTANEOUS forcings at time `t` —
pit temperature, burn rate (smoke), base vapor pressure, inverse air mass
flow and draft airflow — shared by the knot-forced Tesseract path ([`make_meat_rhs`](@ref))
and the coupled monolith ([`make_monolith_rhs`](@ref)). States:
`[T(nx) | W(nx) | C(nx) | F(nx) | B | S | R]` (R the smoke-ring depth in
the material coordinate). Bark and ring states are pure
integrators; fat F is a pure sink; the ONE quality-physics feedback is the
expulsion sink in dW (denaturation purge).

# Arguments
- `du, u`: Derivative and state vectors (length 4·nx + 3).
- `t::Number`: Time [s].
- `θ`: The meat latents as an 8-tuple
  `(D_eff, k_m, h_c, L, k_mult, k_e, a_myosin, m_conv)`.
- `Tpit, smoke, pv_base, imair, airflow::Number`: Forcings at `t`; `airflow`
  is the pit's normalized draft `g(u) = u_leak + (1 − u_leak) u`.
- `t_wrap, t_pull::Number`: Wrap and pull times [s].
- `t_spritz, dt_spritz, m_spritz::Number`: Spritz policy — first spritz [s],
  interval [s] and dose [kg/m²] (see [`spritz_rate`](@ref)); spritzing stops
  at the wrap and the pull.
- `cfg::MeatConfig`: Physical configuration.
- `n_s, n_a, v0`: [`shrink_exponents`](@ref) of `cfg` (hoisted by the caller).

# Returns
- `q_meat::Number`: Heat the meat draws from the pit [W] (surface-area
  scaled, pit part only) — the pit's coupling input.
"""
function meat_rhs_core!(
    du, u, t, θ, Tpit, smoke, pv_base, imair, airflow, t_wrap, t_pull, t_spritz,
    dt_spritz, m_spritz, cfg::MeatConfig, n_s, n_a, v0,
    wb::AbstractVector=bone_weights(cfg), T_trig=Inf,
)
    nx = cfg.nx
    D_eff, k_m, h_c, L, k_mult, k_e, a_myosin, m_conv = θ
    # draft convection: h_c and k_m are quoted at cfg.g_ref and scale with
    # the airflow as (g/g_ref)^m_conv — the same exponent for heat and mass
    # transfer (Chilton–Colburn analogy)
    g_fac = (airflow / cfg.g_ref)^m_conv
    h_c = h_c * g_fac
    k_m = k_m * g_fac

    # material (dry-matter) coordinate: Δξ is the raw node spacing; the
    # physical spacing is st·Δξ with the local thickness stretch
    # st = (v/v0)^n_shrink
    Δξ = L / (nx - 1)

    T = @view u[1:nx]
    W = @view u[(nx + 1):2nx]
    C = @view u[(2nx + 1):3nx]
    F = @view u[(3nx + 1):4nx]
    Sl = @view u[(4nx + 1):5nx]  # salt, dry basis [kg/kg]
    dT = @view du[1:nx]
    dW = @view du[(nx + 1):2nx]
    dC = @view du[(2nx + 1):3nx]
    dF = @view du[(3nx + 1):4nx]
    dSl = @view du[(4nx + 1):5nx]
    # bone blend: node conductivity and volumetric heat capacity
    # interpolate to bone values by the hoisted static weights wb
    # (defined AFTER the views so the closures capture assigned locals)
    kmix(i, v) = (1 - wb[i]) * conductivity_mix(W[i], F[i], T[i], v, cfg) +
                 wb[i] * cfg.k_bone
    ρcmix(i, v) = (1 - wb[i]) * heat_capacity_dry(W[i], F[i], T[i], cfg) / v +
                  wb[i] * cfg.ρc_bone

    # node stretches from the node's own water + fat, in a rolling
    # window (allocation-free); the fv-weighted running sum of v gives
    # the slab-mean specific volume for the exposed-area factor
    # ... and the node's own ρc_p and k from its composition and temperature
    # (Choi–Okos), the conductivity at a face being the mean of the two nodes
    v_c = specific_volume(W[1], F[1], cfg)
    st_m = (v_c / v0)^n_s
    vsum = v_c / 2
    kc_m = k_mult * kmix(1, v_c)
    ρc_m = ρcmix(1, v_c)
    v_c = specific_volume(W[2], F[2], cfg)
    st_c = (v_c / v0)^n_s
    vsum += v_c
    kc_c = k_mult * kmix(2, v_c)
    ρc_c = ρcmix(2, v_c)
    # center: symmetry (zero-flux) via mirror ghost node — a full cell
    # of width st_1·Δξ with two identical faces. Curvature: near r = 0 the
    # operator (1/r^m)·∂_r(r^m k ∂_r T) → (1 + m)·k·∂²T — the finite-volume
    # face/volume factors reduce to exactly this limit.
    m_g = cfg.geom_m
    Δξ̂ = 1 / (nx - 1)  # normalized material spacing (ξ̂ ∈ [0, 1])
    c_geo = 1 + m_g
    f_c = (st_m + st_c) / 2
    dT[1] = c_geo * (kc_m + kc_c) * (T[2] - T[1]) / (f_c * st_m * Δξ^2 * ρc_m)
    dW[1] = c_geo * 2D_eff * (W[2] - W[1]) / (f_c^2 * Δξ^2)
    dSl[1] = c_geo * 2 * cfg.D_salt * (Sl[2] - Sl[1]) / (f_c^2 * Δξ^2)
    # interior: flux form with face metrics (st at the face = mean of
    # the neighbours). Heat is per PHYSICAL volume (cell capacity ∝ st
    # times the local ρc_p, flux ∝ k/st); water is per DRY MASS (cell dry
    # mass fixed in ξ, flux ∝ 1/st² since the dry-matter concentration
    # rises as the matrix contracts)
    @inbounds for i in 2:(nx - 1)
        v_p = specific_volume(W[i + 1], F[i + 1], cfg)
        st_p = (v_p / v0)^n_s
        vsum += v_p
        kc_p = k_mult * kmix(i + 1, v_p)
        f_m = (st_m + st_c) / 2
        f_p = (st_c + st_p) / 2
        # curvature: face areas ξ̂^m and the cell's mean r^m volume factor
        # (all exactly 1 at m = 0, keeping the slab bit-identical)
        if iszero(m_g)
            a_m = a_p = g_i = 1.0
        else
            ξ̂m = (i - 1.5) * Δξ̂
            ξ̂p = (i - 0.5) * Δξ̂
            a_m = ξ̂m^m_g
            a_p = ξ̂p^m_g
            g_i = (ξ̂p^(m_g + 1) - ξ̂m^(m_g + 1)) / ((m_g + 1) * Δξ̂)
        end
        dT[i] =
            (a_p * (kc_c + kc_p) * (T[i + 1] - T[i]) / f_p -
             a_m * (kc_m + kc_c) * (T[i] - T[i - 1]) / f_m) /
            (2 * g_i * st_c * Δξ^2 * ρc_c)
        dW[i] =
            D_eff * (a_p * (W[i + 1] - W[i]) / f_p^2 -
                     a_m * (W[i] - W[i - 1]) / f_m^2) / (g_i * Δξ^2)
        dSl[i] =
            cfg.D_salt * (a_p * (Sl[i + 1] - Sl[i]) / f_p^2 -
                          a_m * (Sl[i] - Sl[i - 1]) / f_m^2) / (g_i * Δξ^2)
        st_m, st_c, v_c = st_c, st_p, v_p
        kc_m, kc_c = kc_c, kc_p
        ρc_c = ρcmix(i + 1, v_p)
    end
    # the loop counted the surface node fully: halve it (fv weight),
    # then the exposed-area factor A(t)/A0 from the slab mean; st_c is
    # now the surface node's stretch and f_s its inner face
    vsum -= v_c / 2
    a_area = (vsum / (nx - 1) / v0)^n_a
    f_s = (st_m + st_c) / 2

    # surface: half-cell finite-volume balance, with the smooth
    # Texas-crutch blend (s = 0 recovers the unwrapped BC exactly) and
    # the pull blend (pl = 0 recovers the in-pit BC exactly; pl = 1 is
    # the rest environment — carryover, rewetting and continued collagen
    # conversion all follow from the unchanged interior equations)
    Ts, Ws = T[nx], W[nx]
    # wrap: the time trigger OR the state latch (smooth, irreversible)
    latch = clamp(u[5nx + 4], zero(u[5nx + 4]), one(u[5nx + 4]))
    du[5nx + 4] = cfg.k_latch * logistic((T[1] - T_trig) / cfg.δ_trig) * (1 - latch)
    s = wrap_blend(t, t_wrap, cfg)
    s = 1 - (1 - s) * (1 - latch)
    pl = pull_blend(t, t_pull, cfg)
    # pit humidity: the pit's base vapor pressure plus the meat's own
    # vapor, p_v = pv_base + c·j with c ∝ exposed area / air mass flow
    c_mix = mixing_coefficient(a_area, imair, cfg)
    f_aw = 1 / (1 + cfg.k_aw * max(Sl[nx], zero(Sl[nx])))
    j = surface_j_blended(Ts, Ws, h_c, k_m, s, pl, pv_base, c_mix, cfg, f_aw)
    q_pit = surface_q_pit_blended(Tpit, Ts, h_c, s, pl, cfg)
    q_in = q_pit + pl * surface_heat_rest(Ts, cfg)
    # spritz: water lands on the surface (in the pit, unwrapped) and is
    # warmed from the bottle to the surface; a fraction f_abs soaks into
    # the crust (and leaves again through the two-sided flux j like
    # condensate), the rest flashes off the hot surface within the pulse
    # and takes its latent heat from the meat
    j_sp = spritz_rate(t, t_spritz, dt_spritz, m_spritz, cfg) * (1 - s) * (1 - pl)
    q_net = q_in - cfg.λ_vap * (j + (1 - cfg.f_abs) * j_sp) -
            CP_WATER * (Ts - cfg.T_spritz) * j_sp
    # half-cell balances: heat per physical volume (the area factor
    # cancels between capacity and intake); water per dry mass — the
    # physical surface j leaves through has shrunk by a_area
    if iszero(m_g)
        a_in = g_s = 1.0
    else
        ξ̂in = 1 - Δξ̂ / 2
        a_in = ξ̂in^m_g
        g_s = (1 - ξ̂in^(m_g + 1)) / ((m_g + 1) * (Δξ̂ / 2))
    end
    dT[nx] =
        (a_in * (kc_m + kc_c) / 2 * (T[nx - 1] - Ts) / (f_s * Δξ) + q_net) /
        (g_s * ρc_c * st_c * Δξ / 2)
    dW[nx] =
        (a_in * RHO_DRY * D_eff * (W[nx - 1] - Ws) / (f_s^2 * Δξ) -
         a_area * (j - cfg.f_abs * j_sp)) /
        (g_s * RHO_DRY * Δξ / 2)
    # salt: zero-flux surface (nothing evaporates), inner-face diffusion only
    dSl[nx] =
        a_in * cfg.D_salt * (Sl[nx - 1] - Sl[nx]) / (f_s^2 * Δξ) / (g_s * Δξ / 2)

    # collagen conversion (doneness diagnostic as state — adjoint for free)
    # + denaturation-driven water expulsion (purge; drips out of the
    #   node budget) + fat rendering
    @inbounds for i in 1:nx
        dC[i] = collagen_rate(T[i], cfg) * (1 - C[i])
        dW[i] -= expulsion_rate_salted(W[i], T[i], Sl[i], k_e, a_myosin, cfg)
        dF[i] = -fat_rate(T[i], cfg) * F[i]
    end

    # bark states: Maillard browning B and smoke deposit S at the surface
    aw = a_w_oswin(Ws, cfg)
    B = u[5nx + 1]
    du[5nx + 1] = bark_rate(Ts, aw, cfg) * (1 - B)
    S = u[5nx + 2]
    # (1 - S): deposition sites saturate, so S ∈ [0, 1) and dL_S is the
    # max smoke darkening; (1 - s): foil blocks smoke contact; (1 - pl):
    # no smoke on the board
    du[5nx + 2] =
        cfg.k_dep * wet_blend(aw, cfg) * max(smoke, zero(smoke)) * (1 - s) * (1 - pl) *
        (1 - S)
    # smoke ring: NO uptake needs a wet surface, a lit fire, no foil and
    # the pit (the pit gas NO concentration is set by the air–fuel ratio,
    # so "lit" is the whole dependence on the burn rate); the front
    # advances by diffusion, r·dr = D Θ dt, and only where the meat at the
    # front is still below the myoglobin denaturation temperature
    R = u[5nx + 3]
    fire = max(smoke, zero(smoke)) / (max(smoke, zero(smoke)) + cfg.mdot_ring)
    Θ = wet_blend(aw, cfg) * fire * (1 - s) * (1 - pl)
    T_front = temperature_at_depth(T, R, Δξ, nx)
    du[5nx + 3] =
        cfg.D_ring * Θ * logistic((cfg.T_ring - T_front) / cfg.δ_ring) / (R + cfg.r_ring0)
    return cfg.A_meat * cfg.n_meat * a_area * q_pit
end

"""
    temperature_at_depth(T, r, Δξ, nx)

Temperature at material depth `r` below the surface node `nx`, by linear
interpolation between grid nodes (piecewise-linear in `r`, clamped to the
slab).
"""
function temperature_at_depth(T, r, Δξ, nx)
    η = clamp(r / Δξ, zero(r / Δξ), (nx - 1) * one(r / Δξ))
    k = min(floor(Int, η), nx - 2)
    f = η - k
    return (1 - f) * T[nx - k] + f * T[nx - k - 1]
end

"""
    make_meat_rhs(cfg, tk0, tkdt, nk)

Build the in-place MOL right-hand side. The parameter vector is
`p = vcat(θ, Tpit_knots)` with the pit-temperature forcing interpolated from
`nk` knots on the uniform grid `tk0 + (i-1)·tkdt` — putting the forcing knots
in `p` is what exposes the VJP w.r.t. the boundary-condition trajectory.
"""
function make_meat_rhs(cfg::MeatConfig, tk0::Float64, tkdt::Float64, nk::Int)
    # p = [θ(8) | Tpit_knots(nk) | smoke_knots(nk) | pv_base_knots(nk) |
    # imair_knots(nk) | airflow_knots(nk) | t_wrap | t_pull | t_spritz |
    # dt_spritz | m_spritz]: the pit forcings (temperature, burn rate for smoke, base
    # vapor pressure, inverse air mass flow, draft airflow) share one knot grid; the two scalar switch times are the p-tail (explicit
    # index arithmetic everywhere — see meat_vjp). The body lives in
    # meat_rhs_core!, shared with the coupled monolith.
    n_s, n_a, v0 = shrink_exponents(cfg)
    wb = bone_weights(cfg)
    return function meat_rhs!(du, u, p, t)
        θ = (p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8])
        knots = @view p[(MEAT_NP + 1):(MEAT_NP + nk)]
        smoke_knots = @view p[(MEAT_NP + nk + 1):(MEAT_NP + 2nk)]
        pvb_knots = @view p[(MEAT_NP + 2nk + 1):(MEAT_NP + 3nk)]
        imair_knots = @view p[(MEAT_NP + 3nk + 1):(MEAT_NP + 4nk)]
        airflow_knots = @view p[(MEAT_NP + 4nk + 1):(MEAT_NP + 5nk)]
        meat_rhs_core!(
            du,
            u,
            t,
            θ,
            lininterp(t, tk0, tkdt, knots),
            lininterp(t, tk0, tkdt, smoke_knots),
            lininterp(t, tk0, tkdt, pvb_knots),
            lininterp(t, tk0, tkdt, imair_knots),
            lininterp(t, tk0, tkdt, airflow_knots),
            p[MEAT_NP + 5nk + 1],
            p[MEAT_NP + 5nk + 2],
            p[MEAT_NP + 5nk + 3],
            p[MEAT_NP + 5nk + 4],
            p[MEAT_NP + 5nk + 5],
            cfg,
            n_s,
            n_a,
            v0,
            wb,
            p[MEAT_NP + 5nk + 6],
        )
        return nothing
    end
end

"""
    _meat_u0(cfg, T0f, W0f, C0f; F0f, B0, S0, R0)

Initial state from full fields (receding-horizon state handoff). Empty
vectors / NaN scalars fall back to the uniform defaults.

Since the juiciness build, the fat field and bark states MUST hand off
too: F restarting at full marbling would re-render fat every replanned
segment and mis-score the juiciness term the default objective carries;
bark restarting at zero would zero the appearance/smoke terms mid-cook.

# Arguments
- `cfg::MeatConfig`: Physical configuration.
- `T0f, W0f, C0f::AbstractVector`: Field ICs (length nx) or empty for the
  uniform defaults.

# Keyword Arguments
- `F0f::AbstractVector`: Un-rendered fat field IC. [Default=empty ⇒ F0]
- `B0::Real`: Maillard bark state IC. [Default=NaN ⇒ 0]
- `S0::Real`: Smoke deposition state IC. [Default=NaN ⇒ 0]

# Returns
- `u0::Vector{Float64}`: Initial state vector (length 4·nx + 3).
"""
function _meat_u0(
    cfg::MeatConfig, T0f::AbstractVector, W0f::AbstractVector, C0f::AbstractVector;
    F0f::AbstractVector = Float64[], B0::Real = NaN, S0::Real = NaN, R0::Real = NaN,
    Ssalt0f::AbstractVector = Float64[], s0 = nothing, LATCH0::Real = NaN,
)
    T = isempty(T0f) ? fill(cfg.T0, cfg.nx) : collect(float.(T0f))
    W = isempty(W0f) ? fill(cfg.W0, cfg.nx) : collect(float.(W0f))
    C = isempty(C0f) ? zeros(cfg.nx) : collect(float.(C0f))
    F = isempty(F0f) ? fill(cfg.F0, cfg.nx) : collect(float.(F0f))
    # salt: an explicit restart field wins; else the brine profile (s0,
    # possibly Dual-valued through L); else unsalted zeros
    Sl = isempty(Ssalt0f) ? (s0 === nothing ? zeros(cfg.nx) : s0) :
         collect(float.(Ssalt0f))
    (
        length(T) == cfg.nx && length(W) == cfg.nx && length(C) == cfg.nx &&
        length(F) == cfg.nx && length(Sl) == cfg.nx
    ) || throw(ArgumentError("field ICs must have length nx=$(cfg.nx)"))
    return vcat(
        T, W, C, F, Sl,
        [isnan(B0) ? 0.0 : float(B0), isnan(S0) ? 0.0 : float(S0),
         isnan(R0) ? 0.0 : float(R0), isnan(LATCH0) ? 0.0 : float(LATCH0)],
    )
end

"""
    meat_solve(θ, Tpit_knots, tknots, tsave; cfg, abstol, reltol, T0f, W0f,
               C0f, t_wrap, smoke_knots, t_pull)

Forward solve of the meat PDE under a pit-temperature forcing trajectory,
optionally continuing through the rest after `t_pull` (docs/model.md, rest).

# Arguments
- `θ::AbstractVector`: Meat latents `[D_eff m²/s, k_m m/s, h_c W/m²K, L m,
  k_mult, k_e 1/s, a_myosin]` (length `MEAT_NP`).
- `Tpit_knots::AbstractVector`: Pit temperature forcing [K] on `tknots`.
- `tknots::AbstractVector`: Uniform forcing knot times [s].
- `tsave::AbstractVector`: Output save times [s].

# Keyword Arguments
- `cfg::MeatConfig`: Fixed physical configuration. [Default=MeatConfig()]
- `abstol::Float64`: ODE absolute tolerance. [Default=1e-6]
- `reltol::Float64`: ODE relative tolerance. [Default=1e-6]
- `T0f, W0f, C0f, F0f::AbstractVector`: Optional full-field initial
  conditions (receding-horizon handoff); empty ⇒ uniform defaults.
  [Default=empty]
- `B0, S0, R0::Real`: Bark-state and smoke-ring ICs for the handoff; NaN ⇒ 0.
  [Default=NaN]
- `t_wrap::Real`: Texas-crutch wrap time [s]; `WRAP_NEVER` ⇒ bitwise
  unwrapped. [Default=WRAP_NEVER]
- `smoke_knots::AbstractVector`: Smoke-flux forcing [kg/s] on the `tknots`
  grid; empty ⇒ zeros (no deposition). [Default=empty]
- `t_pull::Real`: Pull time [s]; the surface sees the rest environment
  (`cfg` rest constants) after it. [Default=PULL_NEVER]
- `pv_base_knots::AbstractVector`: Pit base vapor pressure [Pa] on the
  `tknots` grid (ambient + combustion water + pan, before the meat's own
  vapor); empty ⇒ constant `cfg.p_v_pit`. [Default=empty]
- `imair_knots::AbstractVector`: Inverse pit air mass flow [s/kg] on the
  `tknots` grid — the meat's vapor is diluted by it; empty ⇒ zeros (no
  self-humidification). [Default=empty]
- `airflow_knots::AbstractVector`: Normalized draft airflow `g(u)` on the
  `tknots` grid — scales `h_c`, `k_m` as `(g/g_ref)^m_conv`; empty ⇒
  `cfg.g_ref` (no scaling). [Default=empty]
- `t_spritz, dt_spritz::Real`: Spritz policy, first spritz and interval [s]
  ([`spritz_rate`](@ref)); `SPRITZ_NEVER` = none. [Default=SPRITZ_NEVER, 3600]
- `m_spritz::Real`: Spritz dose [kg/m²]; NaN ⇒ `cfg.m_spritz`. [Default=NaN]

# Returns
- `out::NamedTuple`: Tesseract outputs (`Tsurf`, `Tcore`, `bark_b`, `bark_s`,
  `l_star`, `j_evap`, `q_meat`, `C_field_final`, `w_retained`, `juiciness`,
  `fat_rendered`, `evap_w`, `drip_w`, `drip_f`, `l_thick` — the physical
  half-thickness trajectory [m], `pv_pit` — the pit vapor pressure the
  surface saw [Pa]) plus the full field
  trajectories (`T_field`/`W_field`/`C_field`/`F_field`, nx × nt) and `t`,
  for tests and visualization.
"""
function meat_solve(
    θ::AbstractVector,
    Tpit_knots::AbstractVector,
    tknots::AbstractVector,
    tsave::AbstractVector;
    cfg::MeatConfig=MeatConfig(),
    abstol=1e-6,
    reltol=1e-6,
    T0f::AbstractVector=Float64[],
    W0f::AbstractVector=Float64[],
    C0f::AbstractVector=Float64[],
    F0f::AbstractVector=Float64[],
    Ssalt0f::AbstractVector=Float64[],
    B0::Real=NaN,
    S0::Real=NaN,
    R0::Real=NaN,
    t_wrap::Real=WRAP_NEVER,
    smoke_knots::AbstractVector=Float64[],
    t_pull::Real=PULL_NEVER,
    pv_base_knots::AbstractVector=Float64[],
    imair_knots::AbstractVector=Float64[],
    airflow_knots::AbstractVector=Float64[],
    t_spritz::Real=SPRITZ_NEVER,
    dt_spritz::Real=3600.0,
    m_spritz::Real=NaN,
    T_wrap_trigger::Real=Inf,
    LATCH0::Real=NaN,
    t_salt::Real=NaN,
)
    nx = cfg.nx
    msp = isnan(m_spritz) ? cfg.m_spritz : float(m_spritz)
    sk, pvb, im, af = _forcing_knots(
        cfg, tknots, smoke_knots, pv_base_knots, imair_knots, airflow_knots
    )
    length(θ) == MEAT_NP || throw(
        ArgumentError(
            "θ must have length $MEAT_NP = [D_eff, k_m, h_c, L, k_mult, k_e, " *
            "a_myosin]; got $(length(θ))",
        ),
    )
    p = vcat(
        θ, Tpit_knots, sk, pvb, im, af, float(t_wrap), float(t_pull), float(t_spritz),
        float(dt_spritz), msp, float(T_wrap_trigger),
    )
    any(isnan, p) && throw(
        ArgumentError(
            "meat solve got NaN inputs at indices $(findall(isnan, p)) of p " *
            "(θ|Tpit|smoke|pv_base|imair|airflow|t_wrap|t_pull|t_spritz|" *
            "dt_spritz|m_spritz|T_wrap_trigger)",
        ),
    )
    tk0, tkdt = float(tknots[1]), float(tknots[2] - tknots[1])
    rhs! = make_meat_rhs(cfg, tk0, tkdt, length(tknots))
    # FullSpecialize: 3.6x faster forward solve (measured 0.23 -> 0.06 s,
    # 8 h/49-knot fixture); backward unaffected
    prob = ODEProblem{true,SciMLBase.FullSpecialize}(
        rhs!,
        # the brine duration is a differentiable input: NaN defers to the
        # config, and a Dual t_salt flows into the state through the
        # analytic profile (the promotion below keeps it alive)
        promote_type(eltype(p), typeof(float(t_salt)), Float64).(_meat_u0(
            cfg, T0f, W0f, C0f; F0f, B0, S0, R0, Ssalt0f,
            s0=salt_profile(θ[4], cfg, isnan(t_salt) ? cfg.t_salt : t_salt),
            LATCH0,
        )),
        (float(tsave[1]), float(tsave[end])),
        p,
    )
    sol = solve(prob, FBDF(); saveat=tsave, abstol=abstol, reltol=reltol)
    A = Array(sol)
    (SciMLBase.successful_retcode(sol) && size(A, 2) == length(tsave)) || throw(
        ErrorException(
            "meat solve failed: retcode=$(sol.retcode), " *
            "saved $(size(A, 2))/$(length(tsave)) points",
        ),
    )
    Tsurf = A[nx, :]
    Wsurf = A[2nx, :]
    Mts = hat_matrix(tsave, tk0, tkdt, length(tknots))
    Tpit_ts = Mts * Tpit_knots
    latch_ts = clamp.(A[5nx + 4, :], 0.0, 1.0)
    s_ts = 1 .- (1 .- wrap_blend.(tsave, float(t_wrap), Ref(cfg))) .* (1 .- latch_ts)
    pl_ts = pull_blend.(tsave, float(t_pull), Ref(cfg))
    bark_b = A[5nx + 1, :]
    bark_s = A[5nx + 2, :]
    ring = A[5nx + 3, :]
    w_retained = fv_mean(A[(nx + 1):2nx, end]) / cfg.W0
    F_mean = fv_mean(A[(3nx + 1):4nx, end])
    R_ret = (1 - cfg.η_drain) * (cfg.F0 - F_mean)
    n_s, n_a, v0 = shrink_exponents(cfg)
    wfv = fv_weights(nx)
    Vmat = specific_volume.(A[(nx + 1):2nx, :], A[(3nx + 1):4nx, :], Ref(cfg))
    l_thick = (θ[4] / (nx - 1)) .* vec(wfv' * ((Vmat ./ v0) .^ n_s))
    a_ts = (vec(wfv' * Vmat) ./ ((nx - 1) * v0)) .^ n_a
    pv_base_ts = Mts * pvb
    c_ts = mixing_coefficient.(a_ts, Mts * im, Ref(cfg))
    g_fac = ((Mts * af) ./ cfg.g_ref) .^ θ[8]
    h_ts, km_ts = θ[3] .* g_fac, θ[2] .* g_fac
    f_aw_ts = 1 ./ (1 .+ cfg.k_aw .* max.(A[5nx, :], 0.0))
    j_evap = surface_j_blended.(
        Tsurf, Wsurf, h_ts, km_ts, s_ts, pl_ts, pv_base_ts, c_ts, Ref(cfg), f_aw_ts
    )
    # the pit vapor pressure the surface saw: base plus the meat's own vapor
    pv_pit =
        pv_base_ts .+
        c_ts .*
        evap_flux_fb.(Tsurf, Wsurf, (1 .- cfg.η_evap .* s_ts) .* km_ts, pv_base_ts, c_ts, Ref(cfg))
    dts = diff(collect(float.(tsave)))
    ja = j_evap .* a_ts
    evap_w = sum((ja[1:(end - 1)] .+ ja[2:end]) ./ 2 .* dts) / (RHO_DRY * θ[4])
    jsp = spritz_rate.(tsave, t_spritz, dt_spritz, msp, Ref(cfg)) .* (1 .- s_ts) .* (1 .- pl_ts) .* a_ts
    spritz_w = cfg.f_abs * sum((jsp[1:(end - 1)] .+ jsp[2:end]) ./ 2 .* dts) / (RHO_DRY * θ[4])
    return (;
        Tsurf,
        Tcore=A[1, :],
        bark_b,
        bark_s,
        ring,
        l_star=l_star_bark.(bark_b, bark_s, Ref(cfg)),
        j_evap,
        q_meat=cfg.A_meat * cfg.n_meat .* a_ts .*
               surface_q_pit_blended.(Tpit_ts, Tsurf, h_ts, s_ts, pl_ts, Ref(cfg)),
        C_field_final=A[(2nx + 1):3nx, end],
        w_retained,
        juiciness=w_retained + cfg.β_fat * R_ret / cfg.W0,
        fat_rendered=(cfg.F0 - F_mean) / cfg.F0,
        evap_w,
        spritz_w,
        drip_w=(cfg.W0 - w_retained * cfg.W0) + spritz_w - evap_w,
        drip_f=cfg.η_drain * (cfg.F0 - F_mean),
        l_thick,
        pv_pit,
        T_field=A[1:nx, :],
        W_field=A[(nx + 1):2nx, :],
        C_field=A[(2nx + 1):3nx, :],
        F_field=A[(3nx + 1):4nx, :],
        salt_field=A[(4nx + 1):5nx, :],
        salt_mean=vec(fv_weights(nx)' * A[(4nx + 1):5nx, :]) ./ (nx - 1),
        wrap_latch=A[5nx + 4, :],
        t=collect(float.(tsave)),
    )
end

"""
    _forcing_knots(cfg, tknots, smoke_knots, pv_base_knots, imair_knots)

Materialize the optional forcing knot vectors on the `tknots` grid: empty
smoke ⇒ zeros, empty base vapor pressure ⇒ `cfg.p_v_pit`, empty inverse
air flow ⇒ zeros (no dilution term). Length mismatches fail loudly.
"""
function _forcing_knots(
    cfg::MeatConfig, tknots, smoke_knots, pv_base_knots, imair_knots, airflow_knots
)
    nk = length(tknots)
    fillk(v, default, name) = begin
        k = isempty(v) ? fill(default, nk) : collect(float.(v))
        length(k) == nk || throw(
            ArgumentError("$name must share the tknots grid (got $(length(k)) vs $nk)"),
        )
        k
    end
    return (
        fillk(smoke_knots, 0.0, "smoke_knots"),
        fillk(pv_base_knots, cfg.p_v_pit, "pv_base_knots"),
        fillk(imair_knots, 0.0, "imair_knots"),
        fillk(airflow_knots, cfg.g_ref, "airflow_knots"),
    )
end
