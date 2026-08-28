# 1D slab heat–moisture model of a brisket flat with smoothed surface
# evaporation (the stall mechanism) and collagen-denaturation kinetics.
#
# Geometry: half-slab x ∈ [0, L] with symmetry at x = 0 (center, where the
# temperature probe sits) and the convective/radiative/evaporative boundary at
# x = L (surface). Method of lines on nx uniform nodes; states are the fields
# T [K], W [kg water / kg dry matter], collagen conversion C ∈ [0, 1], and
# un-rendered fat F [kg fat / kg dry matter], plus the two surface bark
# states — 4·nx + 2 ODEs.
#
# Differentiation: continuous adjoint via SciMLSensitivity
# (InterpolatingAdjoint + ReverseDiffVJP compiled tape), pulled through
# Zygote so that the algebraic observables (j_evap, q_meat) and the ODE solve
# share one pullback. The RHS is written branch-free (see interpolation.jl,
# softclamp) so the compiled tape is valid at every (u, p, t).

Base.@kwdef struct MeatConfig
    nx::Int = 40                # spatial nodes
    A_meat::Float64 = 0.12      # exposed surface area [m²] (heat-exchange scaling to the pit)
    T0::Float64 = 281.15        # initial meat temperature [K] (~8 °C)
    W0::Float64 = 2.33          # initial dry-basis moisture (≈70% wet basis)
    ε::Float64 = 0.85           # surface emissivity
    λ_vap::Float64 = 2.3e6      # latent heat of vaporization [J kg⁻¹]
    W_crit::Float64 = 0.40      # moisture availability midpoint (dry basis)
    δW::Float64 = 0.05          # smoothing width of the availability sigmoid
    p_v_pit::Float64 = 2000.0   # pit vapor pressure [Pa] when no humidity forcing is supplied
    δp::Float64 = 100.0         # softclamp scale on the vapor-pressure deficit [Pa]
    Ea::Float64 = 173e3         # collagen denaturation activation energy [J mol⁻¹]
    k_ref::Float64 = 1e-6       # collagen rate at T_ref [s⁻¹]
    T_ref::Float64 = 333.15     # collagen reference temperature [K] (60 °C)
    R_gas::Float64 = 8.314
    # Texas-crutch wrap constants (foil defaults; butcher paper ≈ η_evap 0.5,
    # ε_wrap 0.3). Literature-fixed pending a wrapped-cook log — the wrap
    # moment is documented in such logs, making the wrap event a known-input
    # step-response experiment for these constants (see roadmap).
    δ_wrap::Float64 = 600.0     # wrap blend width [s] (~wrapping takes minutes)
    η_evap::Float64 = 0.9       # evaporation suppression when wrapped (foil)
    ε_wrap::Float64 = 0.10      # wrapped surface emissivity (foil)
    η_h::Float64 = 0.2          # convective-coefficient reduction when wrapped
    # Rest / carryover (docs/tier1_plan.md §1): after t_pull the surface sees
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
    # Shrinkage (docs/tier1_plan.md §2): the PDE is solved on the dry-matter
    # (Lagrangian) coordinate, so the grid never moves; the local thickness
    # stretch st = (v/v0)^n_shrink enters as a face metric, with v the
    # specific volume per kg dry matter (density bookkeeping — volumetric
    # shrinkage is not a free parameter, only how much of it shows up in
    # thickness: 1/3 isotropic, 1 all-thickness). The lateral remainder
    # scales the exposed area A(t)/A0 = (v̄/v0)^(1 - n_shrink). L is the raw
    # half-thickness.
    n_shrink::Float64 = 1 / 3
    ρ_sol::Float64 = 1350.0     # non-fat solids (protein + ash) density [kg m⁻³]
    ρ_fat::Float64 = 900.0      # fat density [kg m⁻³]
    ρ_water::Float64 = 980.0    # water density near cooking temperature [kg m⁻³]
    # Bark model constants (literature-anchored, uncalibrated pending photo
    # data; docs/bark_plan.md carries sources and the identifiability plan).
    # k0 is the bread-crust value (Purlis, 4.2e4) scaled x5: a rubbed meat
    # surface browns faster than lean bread crust (protein-rich, rub sugars),
    # and the bread constant leaves Maillard invisible at smoker temperatures.
    # Effective value; pending photo calibration like everything bark.
    k0_bark::Float64 = 2.0e5    # Maillard pre-exponential [1/s]
    Ea_bark::Float64 = 64151.0  # Maillard activation energy [J/mol] (Purlis)
    a_pk::Float64 = 0.65        # Maillard water-activity bell peak
    σ_aw::Float64 = 0.15        # bell width
    k_ox::Float64 = 0.10        # Oswin isotherm scale (a_w = 0.5 at W = k_ox)
    n_ox::Float64 = 1.5         # Oswin isotherm exponent
    k_dep::Float64 = 0.5        # deposition-saturation rate per kg fuel burned while wet [1/kg]
    a_dry::Float64 = 0.85       # deposition needs a_w above this (moist surface)
    δ_dry::Float64 = 0.05       # smoothing width of the wet switch
    L0_bark::Float64 = 52.0     # initial (rubbed, raw) surface lightness L*
    dL_B::Float64 = 20.0        # max L* drop from Maillard browning
    dL_S::Float64 = 10.0        # max L* drop from smoke deposition
    # Moisture physics (docs/moisture_plan.md; gap audit Tier 1 #1/#2).
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

# θ = [D_eff, k_m, h_c, L, k_mult, k_e, a_myosin] — the meat-side latent
# parameter vector. k_e (expulsion relaxation rate) and a_myosin (W_eq drop
# at the myosin denaturation step) joined θ 2026-08-21: with them frozen at
# literature values the real-cook refit plateaued at loss 495 vs 245 — the
# five original knobs were compensating for expulsion kinetics the log can
# see (they shape the stall entry). cfg.k_e / cfg.a_myosin remain as the
# uncalibrated defaults for cfg-only call paths.
const MEAT_NP = 7

# Default wrap time: far beyond any horizon. logistic((t − 1e9)/δ_wrap) is
# EXACTLY 0.0 (tanh saturates), so the unwrapped model is recovered bitwise
# and the reverse-mode derivative underflows to a clean zero.
const WRAP_NEVER = 1.0e9
# Default pull time: same construction — the pull blend is exactly 0.0 over
# any horizon.
const PULL_NEVER = 1.0e9

"""
    evap_flux(Ts::Number, Ws::Number, k_m::Number, [p_v::Number,] cfg::MeatConfig)

Evaporative mass flux at the meat surface: mass-transfer coefficient ×
smooth moisture availability × vapor-density deficit, with the deficit
clamped smoothly at 0 (no condensation branch). Every factor is smooth —
this is what keeps the adjoint and the NUTS posterior surface well-behaved
through the stall.

# Arguments
- `Ts::Number`: Surface temperature [K].
- `Ws::Number`: Surface moisture, dry basis [kg water / kg dry].
- `k_m::Number`: Mass-transfer coefficient [m/s] (θ latent).
- `p_v::Number`: Ambient vapor pressure [Pa]; the 4-argument form uses the
  fixed pit value `cfg.p_v_pit`.
- `cfg::MeatConfig`: Physical configuration.

# Returns
- `j::Number`: Evaporative mass flux [kg m⁻² s⁻¹], ≥ 0.
"""
function evap_flux(Ts, Ws, k_m, p_v, cfg::MeatConfig)
    φ = logistic((Ws - cfg.W_crit) / cfg.δW)
    Δρv = softclamp(p_sat(Ts) - p_v, cfg.δp) / (R_V * Ts)
    return k_m * φ * Δρv
end
evap_flux(Ts, Ws, k_m, cfg::MeatConfig) = evap_flux(Ts, Ws, k_m, cfg.p_v_pit, cfg)

"""
    evap_flux_fb(Ts, Ws, k_m, pv_base, c, cfg::MeatConfig)

Evaporative flux with the pit humidity feedback (docs/tier1_plan.md §3):
the pit vapor pressure is `pv_base + c·j` — the pit's own vapor balance
(ambient, combustion water, pan) plus the meat's vapor diluted by the air
mass flow, `c` the mixing coefficient [Pa per kg m⁻² s⁻¹] — so
`j = a·sp_δp(p_sat(Ts) − pv_base − c j)` with `a = k_m φ/(R_v Ts)`. The
root is unique (`f′ = 1 + a c σ > 0`); two Newton steps from the
clamp-active closed form are exact to rounding over the cooking range.
`c = 0` is [`evap_flux`](@ref).

# Returns
- `j::Number`: Evaporative mass flux [kg m⁻² s⁻¹], ≥ 0.
"""
function evap_flux_fb(Ts, Ws, k_m, pv_base, c, cfg::MeatConfig)
    a = k_m * logistic((Ws - cfg.W_crit) / cfg.δW) / (R_V * Ts)
    D = p_sat(Ts) - pv_base
    j = a * softclamp(D, cfg.δp) / (1 + a * c)
    for _ in 1:2
        r = D - c * j
        j -= (j - a * softclamp(r, cfg.δp)) / (1 + a * c * logistic(r / cfg.δp))
    end
    return j
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
A smooth blend — not a DiffEq event — keeps t_wrap an ordinary parameter of
the verified continuous adjoint (event-time adjoints need jump conditions,
and FD across a discrete switch is ill-posed); it also smears toward reality,
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

# --- Rest / carryover (docs/tier1_plan.md §1) -------------------------------
"""
    pull_blend(t::Number, t_pull::Number, cfg::MeatConfig)

Smooth pull switch: 0 in the pit, → 1 resting after `t_pull` over width
`δ_pull`. Same construction as [`wrap_blend`](@ref) for the same reasons
(an ordinary parameter of the continuous adjoint, not an event);
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
function surface_j_blended(Ts, Ws, h_c, k_m, s, pl, pv_base, c, cfg::MeatConfig)
    return (1 - pl) * evap_flux_fb(Ts, Ws, (1 - cfg.η_evap * s) * k_m, pv_base, c, cfg) +
           pl * (1 - cfg.η_rest) *
           evap_flux(Ts, Ws, k_m * cfg.h_rest / h_c, cfg.p_v_rest, cfg)
end

"""
    mixing_coefficient(a_area, imair, cfg::MeatConfig)

Vapor-balance mixing coefficient `c = (p_atm/0.622)·A(t)/ṁ_air`
[Pa per kg m⁻² s⁻¹] from the exposed-area factor and the inverse air mass
flow `imair` [s/kg].
"""
mixing_coefficient(a_area, imair, cfg::MeatConfig) = P_ATM / EPS_W * cfg.A_meat * a_area * imair

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

# --- Shrinkage (docs/tier1_plan.md §2) -------------------------------------
"""
    specific_volume(W, F, cfg::MeatConfig)

Volume per kg of (initial) dry matter [m³/kg]: non-fat solids, fat — the
un-rendered fat plus the retained rendered share (the drained share has
left the meat) — and water. Density bookkeeping, not a model: at the raw
state it closes with `RHO_MEAT` / `RHO_DRY` to ~1 %.

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
    return (1 - cfg.F0) / cfg.ρ_sol + fat / cfg.ρ_fat + W / cfg.ρ_water
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

# --- Bark model (surface states; docs/bark_plan.md) -----------------------
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

Maillard water-activity bell (peak `a_pk`, width `σ_aw`): browning is
mobility-limited when too dry and dilution-limited when too wet.

# Arguments
- `aw::Number`: Surface water activity.
- `cfg::MeatConfig`: Physical configuration.

# Returns
- `g::Number`: Rate multiplier ∈ (0, 1].
"""
g_aw(aw, cfg::MeatConfig) = exp(-((aw - cfg.a_pk)^2) / (2 * cfg.σ_aw^2))

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

# --- Moisture physics (docs/moisture_plan.md) ------------------------------
"""
    w_eq(T, a_myosin, cfg)

Temperature-dependent water-holding equilibrium (dry basis): W₀ scaled down
by two smooth denaturation steps (myosin ~58 °C amplitude `a_myosin`, actin
~72 °C amplitude `cfg.a_actin`). `a_myosin` is a θ latent (calibrated); the
actin amplitude stays a config constant.
"""
function w_eq(T, a_myosin, cfg::MeatConfig)
    return cfg.W0 * (
        1 - a_myosin * logistic((T - cfg.T_myosin) / cfg.δ_myosin) -
        cfg.a_actin * logistic((T - cfg.T_actin) / cfg.δ_actin)
    )
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
and branch-free like [`softclamp`](@ref)).

# Arguments
- `x::Number`: Input.

# Returns
- `y::Number`: `log(1 + exp(x))`, ≥ 0.
"""
log1pexp(x) = max(x, zero(x)) + log1p(exp(-abs(x)))

"""
    collagen_rate(T::Number, cfg::MeatConfig)

Arrhenius collagen-denaturation (doneness) rate, referenced to `k_ref` at
`T_ref`.

# Arguments
- `T::Number`: Local temperature [K].
- `cfg::MeatConfig`: Physical configuration (`Ea`, `k_ref`, `T_ref`).

# Returns
- `r::Number`: Conversion rate [1/s].
"""
function collagen_rate(T, cfg::MeatConfig)
    return cfg.k_ref * exp(-(cfg.Ea / cfg.R_gas) * (1 / T - 1 / cfg.T_ref))
end

"""
    meat_rhs_core!(du, u, t, θ, Tpit, smoke, pv_base, imair, t_wrap, t_pull,
                   cfg, n_s, n_a, v0)

The meat right-hand side under the INSTANTANEOUS forcings at time `t` —
pit temperature, burn rate (smoke), base vapor pressure and inverse air
mass flow — shared by the knot-forced Tesseract path ([`make_meat_rhs`](@ref))
and the coupled monolith ([`make_monolith_rhs`](@ref)). States:
`[T(nx) | W(nx) | C(nx) | F(nx) | B | S]`. Bark states are pure
integrators; fat F is a pure sink; the ONE quality-physics feedback is the
expulsion sink in dW (denaturation purge).

# Arguments
- `du, u`: Derivative and state vectors (length 4·nx + 2).
- `t::Number`: Time [s].
- `θ`: The meat latents as a 7-tuple `(D_eff, k_m, h_c, L, k_mult, k_e, a_myosin)`.
- `Tpit, smoke, pv_base, imair::Number`: Forcings at `t`.
- `t_wrap, t_pull::Number`: Wrap and pull times [s].
- `cfg::MeatConfig`: Physical configuration.
- `n_s, n_a, v0`: [`shrink_exponents`](@ref) of `cfg` (hoisted by the caller).

# Returns
- `q_meat::Number`: Heat the meat draws from the pit [W] (surface-area
  scaled, pit part only) — the pit's coupling input.
"""
function meat_rhs_core!(
    du, u, t, θ, Tpit, smoke, pv_base, imair, t_wrap, t_pull, cfg::MeatConfig, n_s, n_a, v0
)
    nx = cfg.nx
    D_eff, k_m, h_c, L, k_mult, k_e, a_myosin = θ

    # material (dry-matter) coordinate: Δξ is the raw node spacing; the
    # physical spacing is st·Δξ with the local thickness stretch
    # st = (v/v0)^n_shrink
    Δξ = L / (nx - 1)
    k_th = k_mult * K_MEAT
    α = k_th / (RHO_MEAT * CP_MEAT)

    T = @view u[1:nx]
    W = @view u[(nx + 1):2nx]
    C = @view u[(2nx + 1):3nx]
    F = @view u[(3nx + 1):4nx]
    dT = @view du[1:nx]
    dW = @view du[(nx + 1):2nx]
    dC = @view du[(2nx + 1):3nx]
    dF = @view du[(3nx + 1):4nx]

    # node stretches from the node's own water + fat, in a rolling
    # window (allocation-free); the fv-weighted running sum of v gives
    # the slab-mean specific volume for the exposed-area factor
    v_c = specific_volume(W[1], F[1], cfg)
    st_m = (v_c / v0)^n_s
    vsum = v_c / 2
    v_c = specific_volume(W[2], F[2], cfg)
    st_c = (v_c / v0)^n_s
    vsum += v_c
    # center: symmetry (zero-flux) via mirror ghost node — a full cell
    # of width st_1·Δξ with two identical faces
    f_c = (st_m + st_c) / 2
    dT[1] = 2α * (T[2] - T[1]) / (f_c * st_m * Δξ^2)
    dW[1] = 2D_eff * (W[2] - W[1]) / (f_c^2 * Δξ^2)
    # interior: flux form with face metrics (st at the face = mean of
    # the neighbours). Heat is per PHYSICAL volume (cell capacity ∝ st,
    # flux ∝ 1/st); water is per DRY MASS (cell dry mass fixed in ξ,
    # flux ∝ 1/st² since the dry-matter concentration rises as the
    # matrix contracts)
    @inbounds for i in 2:(nx - 1)
        v_p = specific_volume(W[i + 1], F[i + 1], cfg)
        st_p = (v_p / v0)^n_s
        vsum += v_p
        f_m = (st_m + st_c) / 2
        f_p = (st_c + st_p) / 2
        dT[i] =
            α * ((T[i + 1] - T[i]) / f_p - (T[i] - T[i - 1]) / f_m) / (st_c * Δξ^2)
        dW[i] =
            D_eff * ((W[i + 1] - W[i]) / f_p^2 - (W[i] - W[i - 1]) / f_m^2) / Δξ^2
        st_m, st_c, v_c = st_c, st_p, v_p
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
    s = wrap_blend(t, t_wrap, cfg)
    pl = pull_blend(t, t_pull, cfg)
    # pit humidity: the pit's base vapor pressure plus the meat's own
    # vapor, p_v = pv_base + c·j with c ∝ exposed area / air mass flow
    c_mix = mixing_coefficient(a_area, imair, cfg)
    j = surface_j_blended(Ts, Ws, h_c, k_m, s, pl, pv_base, c_mix, cfg)
    q_pit = surface_q_pit_blended(Tpit, Ts, h_c, s, pl, cfg)
    q_in = q_pit + pl * surface_heat_rest(Ts, cfg)
    q_net = q_in - cfg.λ_vap * j
    # half-cell balances: heat per physical volume (the area factor
    # cancels between capacity and intake); water per dry mass — the
    # physical surface j leaves through has shrunk by a_area
    dT[nx] =
        (k_th * (T[nx - 1] - Ts) / (f_s * Δξ) + q_net) /
        (RHO_MEAT * CP_MEAT * st_c * Δξ / 2)
    dW[nx] =
        (RHO_DRY * D_eff * (W[nx - 1] - Ws) / (f_s^2 * Δξ) - a_area * j) /
        (RHO_DRY * Δξ / 2)

    # collagen conversion (doneness diagnostic as state — adjoint for free)
    # + denaturation-driven water expulsion (purge; drips out of the
    #   node budget) + fat rendering
    @inbounds for i in 1:nx
        dC[i] = collagen_rate(T[i], cfg) * (1 - C[i])
        dW[i] -= expulsion_rate(W[i], T[i], k_e, a_myosin, cfg)
        dF[i] = -fat_rate(T[i], cfg) * F[i]
    end

    # bark states: Maillard browning B and smoke deposit S at the surface
    aw = a_w_oswin(Ws, cfg)
    B = u[4nx + 1]
    du[4nx + 1] = bark_rate(Ts, aw, cfg) * (1 - B)
    S = u[4nx + 2]
    # (1 - S): deposition sites saturate, so S ∈ [0, 1) and dL_S is the
    # max smoke darkening; (1 - s): foil blocks smoke contact; (1 - pl):
    # no smoke on the board
    du[4nx + 2] =
        cfg.k_dep * wet_blend(aw, cfg) * max(smoke, zero(smoke)) * (1 - s) * (1 - pl) *
        (1 - S)
    return cfg.A_meat * a_area * q_pit
end

"""
    make_meat_rhs(cfg, tk0, tkdt, nk)

Build the in-place MOL right-hand side. The parameter vector is
`p = vcat(θ, Tpit_knots)` with the pit-temperature forcing interpolated from
`nk` knots on the uniform grid `tk0 + (i-1)·tkdt` — putting the forcing knots
in `p` is what exposes the VJP w.r.t. the boundary-condition trajectory.
"""
function make_meat_rhs(cfg::MeatConfig, tk0::Float64, tkdt::Float64, nk::Int)
    # p = [θ(7) | Tpit_knots(nk) | smoke_knots(nk) | pv_base_knots(nk) |
    # imair_knots(nk) | t_wrap | t_pull]: the pit forcings (temperature,
    # burn rate for smoke, base vapor pressure, inverse air mass flow) share
    # one knot grid; the two scalar switch times are the p-tail (explicit
    # index arithmetic everywhere — see meat_vjp). The body lives in
    # meat_rhs_core!, shared with the coupled monolith.
    n_s, n_a, v0 = shrink_exponents(cfg)
    return function meat_rhs!(du, u, p, t)
        θ = (p[1], p[2], p[3], p[4], p[5], p[6], p[7])
        knots = @view p[(MEAT_NP + 1):(MEAT_NP + nk)]
        smoke_knots = @view p[(MEAT_NP + nk + 1):(MEAT_NP + 2nk)]
        pvb_knots = @view p[(MEAT_NP + 2nk + 1):(MEAT_NP + 3nk)]
        imair_knots = @view p[(MEAT_NP + 3nk + 1):(MEAT_NP + 4nk)]
        meat_rhs_core!(
            du,
            u,
            t,
            θ,
            lininterp(t, tk0, tkdt, knots),
            lininterp(t, tk0, tkdt, smoke_knots),
            lininterp(t, tk0, tkdt, pvb_knots),
            lininterp(t, tk0, tkdt, imair_knots),
            p[MEAT_NP + 4nk + 1],
            p[MEAT_NP + 4nk + 2],
            cfg,
            n_s,
            n_a,
            v0,
        )
        return nothing
    end
end

"""
    _meat_u0(cfg::MeatConfig)

Uniform initial state `[T | W | C | F | B, S]` (length 4·nx + 2) at the
config defaults: cold meat, full moisture, zero conversion, full fat, no
bark.

# Arguments
- `cfg::MeatConfig`: Physical configuration.

# Returns
- `u0::Vector{Float64}`: Initial state vector.
"""
function _meat_u0(cfg::MeatConfig)
    return vcat(
        fill(cfg.T0, cfg.nx),
        fill(cfg.W0, cfg.nx),
        zeros(cfg.nx),
        fill(cfg.F0, cfg.nx),
        zeros(2),
    )
end

"""
    _meat_u0(cfg, T0f, W0f, C0f; F0f, B0, S0)

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
- `u0::Vector{Float64}`: Initial state vector (length 4·nx + 2).
"""
function _meat_u0(
    cfg::MeatConfig, T0f::AbstractVector, W0f::AbstractVector, C0f::AbstractVector;
    F0f::AbstractVector = Float64[], B0::Real = NaN, S0::Real = NaN,
)
    T = isempty(T0f) ? fill(cfg.T0, cfg.nx) : collect(float.(T0f))
    W = isempty(W0f) ? fill(cfg.W0, cfg.nx) : collect(float.(W0f))
    C = isempty(C0f) ? zeros(cfg.nx) : collect(float.(C0f))
    F = isempty(F0f) ? fill(cfg.F0, cfg.nx) : collect(float.(F0f))
    (
        length(T) == cfg.nx && length(W) == cfg.nx && length(C) == cfg.nx &&
        length(F) == cfg.nx
    ) || throw(ArgumentError("field ICs must have length nx=$(cfg.nx)"))
    return vcat(T, W, C, F, [isnan(B0) ? 0.0 : float(B0), isnan(S0) ? 0.0 : float(S0)])
end

# InterpolatingAdjoint + EnzymeVJP. Both choices are load-bearing:
#
# EnzymeVJP: ~10-25x faster than ReverseDiff retaping, and immune to the
# compiled-tape trap (ReverseDiffVJP(true) freezes the time-dependent
# hat-basis interpolation weights as constants, silently zeroing the
# forcing-knot adjoint). InterpolatingAdjoint(ReverseDiffVJP(false)) is the
# DIAGNOSTIC ALTERNATIVE — an independent differentiator for cross-checking
# suspected adjoint bugs (it settled the 2026-08-20 silent-zero hunt by
# agreeing with Enzyme, proving the bug upstream of the backend). It is
# never auto-substituted; production is Enzyme, always. Never (true).
#
# InterpolatingAdjoint, NOT GaussAdjoint: GaussAdjoint's quadrature goes NaN
# on this problem when the backward pass crosses a hot fuel-collapse pit
# trajectory with a q_meat (coupling-feedback) cotangent — deterministic
# reproduction in scripts/ history, captured 2026-08-10; Interpolating agrees
# with ReverseDiff to 6 digits on the same data and is 4x faster than Gauss
# there anyway. Symptoms of Gauss NaN: dt_NaN warnings mid-optimization, NaN
# damper schedules, truncated pit solves.
const MEAT_SENSEALG = InterpolatingAdjoint(; autojacvec=EnzymeVJP())

# Absolute-tolerance floor for the adjoint solve. The backward system
# carries one accumulator per parameter that starts at zero with an O(1e3)
# integrand when a cotangent injects at the surface; below ~1e-8 the first
# backward step cannot satisfy the absolute tolerance at any dt and FBDF
# aborts (silently: the algebraic part of the gradient survives, the
# solution part is lost). The forward solve keeps the caller's tolerances.
const ADJOINT_ABSTOL_FLOOR = 1e-8

# Watches the logging stream during the backward solve: SciMLBase reports a
# first-step abort ("dt was forced below floating point epsilon") or a
# maxiters stop ("Interrupted") only as a warning, and neither raises.
struct _AdjointAbortWatch <: Base.CoreLogging.AbstractLogger
    inner::Base.CoreLogging.AbstractLogger
    hit::Base.RefValue{Bool}
end
function Base.CoreLogging.min_enabled_level(l::_AdjointAbortWatch)
    return min(Base.CoreLogging.Warn, Base.CoreLogging.min_enabled_level(l.inner))
end
Base.CoreLogging.shouldlog(l::_AdjointAbortWatch, args...) = true
Base.CoreLogging.catch_exceptions(l::_AdjointAbortWatch) = Base.CoreLogging.catch_exceptions(l.inner)
function Base.CoreLogging.handle_message(
    l::_AdjointAbortWatch, level, message, _module, group, id, file, line; kwargs...
)
    if level >= Base.CoreLogging.Warn
        msg = string(message)
        if occursin("dt was forced below", msg) || occursin("Larger maxiters", msg)
            l.hit[] = true
        end
    end
    # forward only what the wrapped logger would itself have accepted
    if level >= Base.CoreLogging.min_enabled_level(l.inner) &&
        Base.CoreLogging.shouldlog(l.inner, level, _module, group, id)
        Base.CoreLogging.handle_message(
            l.inner, level, message, _module, group, id, file, line; kwargs...
        )
    end
    return nothing
end

"""
    fv_mean(v)

Finite-volume mean on the uniform slab mesh: endpoint cells have half
volume, so ``\\bar v = (\\tfrac12 v_1 + \\sum_{i=2}^{n-1} v_i + \\tfrac12 v_n)/(n-1)``.
"""
fv_mean(v) = (first(v) + last(v) + 2 * sum(v[2:(end - 1)])) / (2 * (length(v) - 1))

"""
    _meat_core(p, cfg, prob, tsave, Mts; abstol, reltol, sensealg)

Differentiable core: solve the PDE and compute the boundary-exchange
observables. Returns a NamedTuple of arrays. `prob` is a prototype ODEProblem
built OUTSIDE the pullback; parameters are injected via `solve(...; p = p)`,
the canonical pattern SciMLSensitivity's adjoint rrule intercepts (building
the problem inside the pullback silently drops the solve-path gradient).
`Mts` is the (plain Float64) hat-basis matrix interpolating the pit knots to
`tsave`.
"""
function _meat_core(
    p, cfg::MeatConfig, prob, tsave, Mts; abstol=1e-6, reltol=1e-6, sensealg=MEAT_SENSEALG
)
    nx = cfg.nx
    sol = solve(
        prob, FBDF(); p=p, saveat=tsave, abstol=abstol, reltol=reltol, sensealg=sensealg
    )
    A = Array(sol)                       # 3nx × nt
    # Fail loudly on a truncated/failed solve — a short solution array
    # silently propagates wrong-shaped outputs through the composed pipeline.
    (SciMLBase.successful_retcode(sol) && size(A, 2) == length(tsave)) || throw(
        ErrorException(
            "meat solve failed: retcode=$(sol.retcode), " *
            "saved $(size(A, 2))/$(length(tsave)) points",
        ),
    )
    Tcore = A[1, :]
    Tsurf = A[nx, :]
    Wsurf = A[2nx, :]
    C_field_final = A[(2nx + 1):3nx, end]
    # water retained (water-only; kept for continuity + scale calibration).
    # Endpoint cells have half volume — arithmetic mean(W) is mesh-biased.
    w_retained = fv_mean(A[(nx + 1):2nx, end]) / cfg.W0
    nk = size(Mts, 2)
    Tpit_ts = Mts * p[(MEAT_NP + 1):(MEAT_NP + nk)]
    bark_b = A[4nx + 1, :]
    bark_s = A[4nx + 2, :]
    l_star = l_star_bark.(bark_b, bark_s, Ref(cfg))
    # moisture physics observables (docs/moisture_plan.md): rendered fat,
    # perceptual juiciness (water + β·retained rendered fat), and the
    # cook-loss decomposition the scale experiment measures. All dry-basis
    # mean units; mass closure is algebraic (dripped water = water lost
    # minus evaporated), assuming uniform ICs (handoff caveat documented).
    F_mean = fv_mean(A[(3nx + 1):4nx, end])
    fat_rendered = (cfg.F0 - F_mean) / cfg.F0
    R_ret = (1 - cfg.η_drain) * (cfg.F0 - F_mean)
    juiciness = w_retained + cfg.β_fat * R_ret / cfg.W0
    # observables MUST apply the same wrap AND pull blends as the RHS: the
    # bark integral (∫ j_evap dt) and the pit coupling consume these, and an
    # unblended observable would carry a silently wrong t_wrap / t_pull
    # gradient (same shared helpers as the RHS; explicit tail indices)
    t_wrap = p[MEAT_NP + 4nk + 1]
    t_pull = p[MEAT_NP + 4nk + 2]
    s_ts = wrap_blend.(tsave, t_wrap, Ref(cfg))
    pl_ts = pull_blend.(tsave, t_pull, Ref(cfg))
    # shrinkage observables from the saved W/F fields (same exponents and
    # reference volume as the RHS): thickness L(t) = Δξ·Σ_fv st, and the
    # exposed-area factor A(t)/A0 that scales the pit coupling and the
    # evaporated-water total
    n_s, n_a, v0 = shrink_exponents(cfg)
    wfv = fv_weights(nx)
    Vmat = specific_volume.(A[(nx + 1):2nx, :], A[(3nx + 1):4nx, :], Ref(cfg))
    l_thick = (p[4] / (nx - 1)) .* vec(wfv' * ((Vmat ./ v0) .^ n_s))
    a_ts = (vec(wfv' * Vmat) ./ ((nx - 1) * v0)) .^ n_a
    # pit humidity on the save grid (same balance as the RHS)
    pv_base_ts = Mts * p[(MEAT_NP + 2nk + 1):(MEAT_NP + 3nk)]
    c_ts = mixing_coefficient.(a_ts, Mts * p[(MEAT_NP + 3nk + 1):(MEAT_NP + 4nk)], Ref(cfg))
    j_evap = surface_j_blended.(
        Tsurf, Wsurf, p[3], p[2], s_ts, pl_ts, pv_base_ts, c_ts, Ref(cfg)
    )
    q_meat =
        cfg.A_meat .* a_ts .*
        surface_q_pit_blended.(Tpit_ts, Tsurf, p[3], s_ts, pl_ts, Ref(cfg))
    # evaporated water in dry-basis mean units: ∫ j·(A/A0) dt / (ρ_dry · L)
    dts = tsave[2:end] .- tsave[1:(end - 1)]
    ja = j_evap .* a_ts
    evap_w = sum((ja[1:(end - 1)] .+ ja[2:end]) ./ 2 .* dts) / (RHO_DRY * p[4])
    drip_w = (cfg.W0 - w_retained * cfg.W0) - evap_w
    drip_f = cfg.η_drain * (cfg.F0 - F_mean)
    return (;
        Tsurf,
        Tcore,
        j_evap,
        q_meat,
        C_field_final,
        w_retained,
        bark_b,
        bark_s,
        l_star,
        juiciness,
        fat_rendered,
        evap_w,
        drip_w,
        drip_f,
        l_thick,
    )
end

"""
    meat_solve(θ, Tpit_knots, tknots, tsave; cfg, abstol, reltol, T0f, W0f,
               C0f, t_wrap, smoke_knots, t_pull)

Forward solve of the meat PDE under a pit-temperature forcing trajectory,
optionally continuing through the rest after `t_pull` (docs/tier1_plan.md).

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
- `B0, S0::Real`: Bark-state ICs for the handoff; NaN ⇒ 0. [Default=NaN]
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
    B0::Real=NaN,
    S0::Real=NaN,
    t_wrap::Real=WRAP_NEVER,
    smoke_knots::AbstractVector=Float64[],
    t_pull::Real=PULL_NEVER,
    pv_base_knots::AbstractVector=Float64[],
    imair_knots::AbstractVector=Float64[],
)
    nx = cfg.nx
    sk, pvb, im = _forcing_knots(cfg, tknots, smoke_knots, pv_base_knots, imair_knots)
    length(θ) == MEAT_NP || throw(
        ArgumentError(
            "θ must have length $MEAT_NP = [D_eff, k_m, h_c, L, k_mult, k_e, " *
            "a_myosin]; got $(length(θ))",
        ),
    )
    p = vcat(θ, Tpit_knots, sk, pvb, im, float(t_wrap), float(t_pull))
    any(isnan, p) && throw(
        ArgumentError(
            "meat solve got NaN inputs at indices $(findall(isnan, p)) of p " *
            "(θ|Tpit|smoke|pv_base|imair|t_wrap|t_pull)",
        ),
    )
    tk0, tkdt = float(tknots[1]), float(tknots[2] - tknots[1])
    rhs! = make_meat_rhs(cfg, tk0, tkdt, length(tknots))
    # FullSpecialize: 3.6x faster forward solve (measured 0.23 -> 0.06 s,
    # 8 h/49-knot fixture); backward unaffected
    prob = ODEProblem{true,SciMLBase.FullSpecialize}(
        rhs!,
        _meat_u0(cfg, T0f, W0f, C0f; F0f, B0, S0),
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
    s_ts = wrap_blend.(tsave, float(t_wrap), Ref(cfg))
    pl_ts = pull_blend.(tsave, float(t_pull), Ref(cfg))
    bark_b = A[4nx + 1, :]
    bark_s = A[4nx + 2, :]
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
    j_evap = surface_j_blended.(
        Tsurf, Wsurf, θ[3], θ[2], s_ts, pl_ts, pv_base_ts, c_ts, Ref(cfg)
    )
    # the pit vapor pressure the surface saw: base plus the meat's own vapor
    pv_pit =
        pv_base_ts .+
        c_ts .*
        evap_flux_fb.(Tsurf, Wsurf, (1 .- cfg.η_evap .* s_ts) .* θ[2], pv_base_ts, c_ts, Ref(cfg))
    dts = diff(collect(float.(tsave)))
    ja = j_evap .* a_ts
    evap_w = sum((ja[1:(end - 1)] .+ ja[2:end]) ./ 2 .* dts) / (RHO_DRY * θ[4])
    return (;
        Tsurf,
        Tcore=A[1, :],
        bark_b,
        bark_s,
        l_star=l_star_bark.(bark_b, bark_s, Ref(cfg)),
        j_evap,
        q_meat=cfg.A_meat .* a_ts .*
               surface_q_pit_blended.(Tpit_ts, Tsurf, θ[3], s_ts, pl_ts, Ref(cfg)),
        C_field_final=A[(2nx + 1):3nx, end],
        w_retained,
        juiciness=w_retained + cfg.β_fat * R_ret / cfg.W0,
        fat_rendered=(cfg.F0 - F_mean) / cfg.F0,
        evap_w,
        drip_w=(cfg.W0 - w_retained * cfg.W0) - evap_w,
        drip_f=cfg.η_drain * (cfg.F0 - F_mean),
        l_thick,
        pv_pit,
        T_field=A[1:nx, :],
        W_field=A[(nx + 1):2nx, :],
        C_field=A[(2nx + 1):3nx, :],
        F_field=A[(3nx + 1):4nx, :],
        t=collect(float.(tsave)),
    )
end

"""
    meat_vjp(θ, Tpit_knots, tknots, tsave, cot; cfg, abstol, reltol,
             sensealg, t_wrap, smoke_knots, t_pull, pv_base_knots,
             imair_knots, T0f, W0f, C0f, F0f, B0, S0)

Vector–Jacobian product of the meat Tesseract outputs w.r.t.
`(θ, Tpit_knots, smoke_knots, pv_base_knots, imair_knots, t_wrap, t_pull)`.
The gradient flows through the
continuous SciMLSensitivity adjoint of the stiff solve, with cotangents
normalized to unit magnitude on entry and rescaled on exit (exact by
linearity; prevents a large-seed backward-solve abort that returns silent
zeros — see the inline comment) and a loud guard that throws if the
backward solve aborted anyway.

The optional field ICs (`T0f`/`W0f`/`C0f`/`F0f`/`B0`/`S0`) MUST match the
primal solve's: the adjoint problem is built from the SAME initial state
(receding-horizon replans differentiate their own forecast, not a fresh
default brisket).

# Arguments
- `θ::AbstractVector`: Meat latents (length `MEAT_NP`; see [`meat_solve`](@ref)).
- `Tpit_knots::AbstractVector`: Pit temperature forcing [K] on `tknots`.
- `tknots::AbstractVector`: Uniform forcing knot times [s].
- `tsave::AbstractVector`: Save times the cotangents live on [s].
- `cot`: NamedTuple of output cotangents with any subset of the keys
  `:Tsurf, :Tcore, :j_evap, :q_meat, :C_field_final, :w_retained, :bark_b,
  :bark_s, :l_star, :juiciness, :fat_rendered, :evap_w, :drip_w, :drip_f,
  :l_thick`; missing keys are zero.

# Keyword Arguments
- `cfg::MeatConfig`: Fixed physical configuration. [Default=MeatConfig()]
- `abstol::Float64`: ODE absolute tolerance. [Default=1e-6]
- `reltol::Float64`: ODE relative tolerance. [Default=1e-6]
- `sensealg`: SciMLSensitivity adjoint choice. [Default=MEAT_SENSEALG]
- `t_wrap::Real`: Wrap time [s]. [Default=WRAP_NEVER]
- `smoke_knots::AbstractVector`: Smoke forcing on `tknots`; empty ⇒ zeros.
  [Default=empty]
- `t_pull::Real`: Pull time [s]. [Default=PULL_NEVER]
- `pv_base_knots, imair_knots::AbstractVector`: Pit humidity forcings on
  `tknots` (see [`meat_solve`](@ref)); empty ⇒ constant `cfg.p_v_pit`,
  no dilution term. [Default=empty]
- `T0f, W0f, C0f, F0f::AbstractVector`: Optional full-field initial
  conditions — must match the primal's; empty ⇒ uniform defaults.
  [Default=empty]
- `B0, S0::Real`: Bark-state ICs; NaN ⇒ 0. [Default=NaN]

# Returns
- `g::NamedTuple`: `(; dθ, dTpit_knots, dsmoke_knots, dpv_base_knots,
  dimair_knots, dt_wrap, dt_pull)` gradients.
"""
function meat_vjp(
    θ::AbstractVector,
    Tpit_knots::AbstractVector,
    tknots::AbstractVector,
    tsave::AbstractVector,
    cot;
    cfg::MeatConfig=MeatConfig(),
    abstol=1e-6,
    reltol=1e-6,
    sensealg=MEAT_SENSEALG,
    t_wrap::Real=WRAP_NEVER,
    smoke_knots::AbstractVector=Float64[],
    t_pull::Real=PULL_NEVER,
    pv_base_knots::AbstractVector=Float64[],
    imair_knots::AbstractVector=Float64[],
    T0f::AbstractVector=Float64[],
    W0f::AbstractVector=Float64[],
    C0f::AbstractVector=Float64[],
    F0f::AbstractVector=Float64[],
    B0::Real=NaN,
    S0::Real=NaN,
)
    length(θ) == MEAT_NP || throw(
        ArgumentError(
            "θ must have length $MEAT_NP = [D_eff, k_m, h_c, L, k_mult, k_e, " *
            "a_myosin]; got $(length(θ))",
        ),
    )
    sk, pvb, im = _forcing_knots(cfg, tknots, smoke_knots, pv_base_knots, imair_knots)
    p = vcat(θ, Tpit_knots, sk, pvb, im, float(t_wrap), float(t_pull))
    tk0, tkdt = float(tknots[1]), float(tknots[2] - tknots[1])
    Mts = hat_matrix(tsave, tk0, tkdt, length(tknots))
    rhs! = make_meat_rhs(cfg, tk0, tkdt, length(tknots))
    # The adjoint problem starts from the SAME initial state as the primal:
    # building it from cfg defaults here silently differentiated a fresh
    # brisket for every mid-cook (receding-horizon) gradient.
    prob = ODEProblem{true,SciMLBase.FullSpecialize}(
        rhs!,
        _meat_u0(cfg, T0f, W0f, C0f; F0f, B0, S0),
        (float(tsave[1]), float(tsave[end])),
        p,
    )
    nt = length(tsave)
    z_t = zeros(nt)
    z_x = zeros(cfg.nx)
    cotfull = (
        Tsurf=get(cot, :Tsurf, z_t),
        Tcore=get(cot, :Tcore, z_t),
        j_evap=get(cot, :j_evap, z_t),
        q_meat=get(cot, :q_meat, z_t),
        C_field_final=get(cot, :C_field_final, z_x),
        w_retained=get(cot, :w_retained, 0.0),
        bark_b=get(cot, :bark_b, z_t),
        bark_s=get(cot, :bark_s, z_t),
        l_star=get(cot, :l_star, z_t),
        juiciness=get(cot, :juiciness, 0.0),
        fat_rendered=get(cot, :fat_rendered, 0.0),
        evap_w=get(cot, :evap_w, 0.0),
        drip_w=get(cot, :drip_w, 0.0),
        drip_f=get(cot, :drip_f, 0.0),
        l_thick=get(cot, :l_thick, z_t),
    )
    # Cotangent normalization: the backward adjoint solve can fail its
    # FIRST step (dt < eps abort -> SILENT ZERO gradients) when terminal
    # lambda injections are large — observed at ||cot|| ~ 1e3 with the
    # moisture RHS while the identical seed at unit norm is exact to 2e-6.
    # The VJP is linear in the cotangent, so normalizing in and rescaling
    # out is EXACT and removes the magnitude trigger entirely.
    cscale = maximum(
        x -> maximum(abs, x; init=0.0),
        (
            cotfull.Tsurf,
            cotfull.Tcore,
            cotfull.j_evap,
            cotfull.q_meat,
            cotfull.C_field_final,
            [cotfull.w_retained],
            cotfull.bark_b,
            cotfull.bark_s,
            cotfull.l_star,
            [cotfull.juiciness],
            [cotfull.fat_rendered],
            [cotfull.evap_w],
            [cotfull.drip_w],
            [cotfull.drip_f],
            cotfull.l_thick,
        ),
    )
    cscale = cscale > 0 ? cscale : 1.0
    cotn = map(c -> c isa AbstractArray ? c ./ cscale : c / cscale, cotfull)
    abstol_adj = max(abstol, ADJOINT_ABSTOL_FLOOR)
    _, back = Zygote.pullback(
        pp -> _meat_core(pp, cfg, prob, tsave, Mts; abstol=abstol_adj, reltol, sensealg), p
    )
    aborted = Ref(false)
    dp = Base.CoreLogging.with_logger(
        _AdjointAbortWatch(Base.CoreLogging.current_logger(), aborted)
    ) do
        back(cotn)[1]
    end
    aborted[] && throw(
        ErrorException(
            "meat adjoint backward solve aborted (dt < eps or maxiters) — " *
            "gradient invalid; do not trust it",
        ),
    )
    dp = dp === nothing ? zeros(length(p)) : dp .* cscale
    # Loud guard against the silent-zero failure mode: an identically zero
    # gradient with nonzero seeds means the backward solve aborted.
    if cscale > 0 &&
        all(iszero, dp) &&
        any(c -> any(!iszero, c isa AbstractArray ? c : [c]), cotfull)
        throw(
            ErrorException(
                "meat adjoint returned identically zero gradients for nonzero " *
                "cotangents — backward solve likely aborted (dt_epsilon); " *
                "do not trust this gradient",
            ),
        )
    end
    nk = length(Tpit_knots)
    # explicit p-tail slots (a silent shift is not a crash)
    return (
        dθ=dp[1:MEAT_NP],
        dTpit_knots=dp[(MEAT_NP + 1):(MEAT_NP + nk)],
        dsmoke_knots=dp[(MEAT_NP + nk + 1):(MEAT_NP + 2nk)],
        dpv_base_knots=dp[(MEAT_NP + 2nk + 1):(MEAT_NP + 3nk)],
        dimair_knots=dp[(MEAT_NP + 3nk + 1):(MEAT_NP + 4nk)],
        dt_wrap=dp[MEAT_NP + 4nk + 1],
        dt_pull=dp[MEAT_NP + 4nk + 2],
    )
end

"""
    _forcing_knots(cfg, tknots, smoke_knots, pv_base_knots, imair_knots)

Materialize the optional forcing knot vectors on the `tknots` grid: empty
smoke ⇒ zeros, empty base vapor pressure ⇒ `cfg.p_v_pit`, empty inverse
air flow ⇒ zeros (no dilution term). Length mismatches fail loudly.
"""
function _forcing_knots(cfg::MeatConfig, tknots, smoke_knots, pv_base_knots, imair_knots)
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
    )
end
