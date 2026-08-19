# 1D slab heat–moisture model of a brisket flat with smoothed surface
# evaporation (the stall mechanism) and collagen-denaturation kinetics.
#
# Geometry: half-slab x ∈ [0, L] with symmetry at x = 0 (center, where the
# temperature probe sits) and the convective/radiative/evaporative boundary at
# x = L (surface). Method of lines on nx uniform nodes; states are the fields
# T [K], W [kg water / kg dry matter], and collagen conversion C ∈ [0, 1],
# giving 3·nx ODEs.
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
    p_v_pit::Float64 = 2000.0   # pit vapor pressure [Pa] — FIXED; entangled with k_m (see writeup)
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
end

# θ = [D_eff, k_m, h_c, L, k_mult] — the meat-side latent parameter vector.
const MEAT_NP = 5

# Default wrap time: far beyond any horizon. logistic((t − 1e9)/δ_wrap) is
# EXACTLY 0.0 (tanh saturates), so the unwrapped model is recovered bitwise
# and the reverse-mode derivative underflows to a clean zero.
const WRAP_NEVER = 1.0e9

"""
    evap_flux(Ts, Ws, k_m, cfg)

Evaporative mass flux [kg m⁻² s⁻¹] at surface temperature `Ts` and surface
moisture `Ws`: mass-transfer coefficient × smooth moisture availability ×
vapor-density deficit, with the deficit clamped smoothly at 0 (no condensation
branch). Every factor is smooth — this is what keeps the adjoint and the NUTS
posterior surface well-behaved through the stall.
"""
function evap_flux(Ts, Ws, k_m, cfg::MeatConfig)
    φ = logistic((Ws - cfg.W_crit) / cfg.δW)
    Δρv = softclamp(p_sat(Ts) - cfg.p_v_pit, cfg.δp) / (R_V * Ts)
    k_m * φ * Δρv
end

"Surface heat flux density into the meat [W m⁻²] (convective + radiative), before evaporation."
surface_heat_in(Tpit, Ts, h_c, cfg::MeatConfig) =
    h_c * (Tpit - Ts) + cfg.ε * σ_SB * (Tpit^4 - Ts^4)

"""
    wrap_blend(t, t_wrap, cfg)

Smooth Texas-crutch switch s ∈ [0, 1]: 0 unwrapped, → 1 after `t_wrap` over
width `δ_wrap`. Branch-free (tanh logistic), so it is safe in every AD path.
A smooth blend — not a DiffEq event — keeps t_wrap an ordinary parameter of
the verified continuous adjoint (event-time adjoints need jump conditions,
and FD across a discrete switch is ill-posed); it also smears toward reality,
since wrapping a brisket takes minutes, not an instant.
"""
wrap_blend(t, t_wrap, cfg::MeatConfig) = logistic((t - t_wrap) / cfg.δ_wrap)

"Wrapped-blend surface heat flux: convection reduced by η_h·s, emissivity blended to ε_wrap."
surface_heat_in_wrapped(Tpit, Ts, h_c, s, cfg::MeatConfig) =
    h_c * (1 - cfg.η_h * s) * (Tpit - Ts) +
    (cfg.ε * (1 - s) + cfg.ε_wrap * s) * σ_SB * (Tpit^4 - Ts^4)

# --- Bark model (surface states; docs/bark_plan.md) -----------------------
"Smooth Oswin sorption isotherm: surface water activity from dry-basis moisture."
a_w_oswin(W, cfg::MeatConfig) = 1 / (1 + (cfg.k_ox / max(W, 1e-6))^cfg.n_ox)

"Maillard water-activity bell (peak a_pk, width σ_aw) — mobility- vs dilution-limited."
g_aw(aw, cfg::MeatConfig) = exp(-((aw - cfg.a_pk)^2) / (2 * cfg.σ_aw^2))

"Smoke deposition needs a moist surface: smooth switch on a_w."
wet_blend(aw, cfg::MeatConfig) = logistic((aw - cfg.a_dry) / cfg.δ_dry)

"Maillard browning rate [1/s] at surface temperature Ts and water activity aw."
bark_rate(Ts, aw, cfg::MeatConfig) =
    cfg.k0_bark * exp(-cfg.Ea_bark / (cfg.R_gas * Ts)) * g_aw(aw, cfg)

"Surface lightness from the two bark states (affine map to CIELAB L*)."
l_star_bark(B, S, cfg::MeatConfig) = cfg.L0_bark - cfg.dL_B * B - cfg.dL_S * S

"Arrhenius collagen conversion rate [s⁻¹] at temperature T."
collagen_rate(T, cfg::MeatConfig) =
    cfg.k_ref * exp(-(cfg.Ea / cfg.R_gas) * (1 / T - 1 / cfg.T_ref))

"""
    make_meat_rhs(cfg, tk0, tkdt, nk)

Build the in-place MOL right-hand side. The parameter vector is
`p = vcat(θ, Tpit_knots)` with the pit-temperature forcing interpolated from
`nk` knots on the uniform grid `tk0 + (i-1)·tkdt` — putting the forcing knots
in `p` is what exposes the VJP w.r.t. the boundary-condition trajectory.
"""
function make_meat_rhs(cfg::MeatConfig, tk0::Float64, tkdt::Float64, nk::Int)
    nx = cfg.nx
    # p = [θ(5) | Tpit_knots(nk) | smoke_knots(nk) | t_wrap]; the smoke
    # forcing (pit burn rate, bark smoke-flux proxy) shares the Tpit knot
    # grid. States: [T(nx) | W(nx) | C(nx) | B | S] — bark browning B and
    # smoke deposit S are pure integrators (no feedback into T/W/C in v1).
    function meat_rhs!(du, u, p, t)
        D_eff, k_m, h_c, L, k_mult = p[1], p[2], p[3], p[4], p[5]
        knots = @view p[MEAT_NP+1:MEAT_NP+nk]
        smoke_knots = @view p[MEAT_NP+nk+1:MEAT_NP+2nk]
        t_wrap = p[MEAT_NP+2nk+1]

        Δx = L / (nx - 1)
        k_th = k_mult * K_MEAT
        α = k_th / (RHO_MEAT * CP_MEAT)

        T = @view u[1:nx]
        W = @view u[nx+1:2nx]
        C = @view u[2nx+1:3nx]
        dT = @view du[1:nx]
        dW = @view du[nx+1:2nx]
        dC = @view du[2nx+1:3nx]

        Tpit = lininterp(t, tk0, tkdt, knots)

        # interior diffusion
        @inbounds for i in 2:nx-1
            dT[i] = α * (T[i+1] - 2T[i] + T[i-1]) / Δx^2
            dW[i] = D_eff * (W[i+1] - 2W[i] + W[i-1]) / Δx^2
        end
        # center: symmetry (zero-flux) via mirror ghost node
        dT[1] = 2α * (T[2] - T[1]) / Δx^2
        dW[1] = 2D_eff * (W[2] - W[1]) / Δx^2

        # surface: half-cell finite-volume balance, with the smooth
        # Texas-crutch blend (s = 0 recovers the unwrapped BC exactly)
        Ts, Ws = T[nx], W[nx]
        s = wrap_blend(t, t_wrap, cfg)
        j = (1 - cfg.η_evap * s) * evap_flux(Ts, Ws, k_m, cfg)
        q_net = surface_heat_in_wrapped(Tpit, Ts, h_c, s, cfg) - cfg.λ_vap * j
        dT[nx] = (k_th * (T[nx-1] - Ts) / Δx + q_net) / (RHO_MEAT * CP_MEAT * Δx / 2)
        dW[nx] = (RHO_DRY * D_eff * (W[nx-1] - Ws) / Δx - j) / (RHO_DRY * Δx / 2)

        # collagen conversion (doneness diagnostic as state — adjoint for free)
        @inbounds for i in 1:nx
            dC[i] = collagen_rate(T[i], cfg) * (1 - C[i])
        end

        # bark states: Maillard browning B and smoke deposit S at the surface
        aw = a_w_oswin(Ws, cfg)
        B = u[3nx+1]
        smoke = lininterp(t, tk0, tkdt, smoke_knots)
        du[3nx+1] = bark_rate(Ts, aw, cfg) * (1 - B)
        S = u[3nx+2]
        # (1 - S): deposition sites saturate, so S ∈ [0, 1) and dL_S is the
        # max smoke darkening; (1 - s): foil blocks smoke contact
        du[3nx+2] = cfg.k_dep * wet_blend(aw, cfg) * max(smoke, zero(smoke)) *
                    (1 - s) * (1 - S)
        nothing
    end
end

_meat_u0(cfg::MeatConfig) =
    vcat(fill(cfg.T0, cfg.nx), fill(cfg.W0, cfg.nx), zeros(cfg.nx), zeros(2))

"""
    _meat_u0(cfg, T0f, W0f, C0f)

Initial state from full fields (receding-horizon state handoff). Empty
vectors fall back to the uniform defaults.
"""
function _meat_u0(cfg::MeatConfig, T0f::AbstractVector, W0f::AbstractVector,
                  C0f::AbstractVector)
    T = isempty(T0f) ? fill(cfg.T0, cfg.nx) : collect(float.(T0f))
    W = isempty(W0f) ? fill(cfg.W0, cfg.nx) : collect(float.(W0f))
    C = isempty(C0f) ? zeros(cfg.nx) : collect(float.(C0f))
    (length(T) == cfg.nx && length(W) == cfg.nx && length(C) == cfg.nx) ||
        throw(ArgumentError("field ICs must have length nx=$(cfg.nx)"))
    # bark states restart at 0 on a state handoff — a mid-cook re-plan does
    # not need bark continuity (observable-only in v1); documented limitation
    vcat(T, W, C, zeros(2))
end

# InterpolatingAdjoint + EnzymeVJP. Both choices are load-bearing:
#
# EnzymeVJP: ~10-25x faster than ReverseDiff retaping, and immune to the
# compiled-tape trap (ReverseDiffVJP(true) freezes the time-dependent
# hat-basis interpolation weights as constants, silently zeroing the
# forcing-knot adjoint). Fallback: InterpolatingAdjoint(ReverseDiffVJP(false)),
# never (true).
#
# InterpolatingAdjoint, NOT GaussAdjoint: GaussAdjoint's quadrature goes NaN
# on this problem when the backward pass crosses a hot fuel-collapse pit
# trajectory with a q_meat (coupling-feedback) cotangent — deterministic
# reproduction in scripts/ history, captured 2026-08-10; Interpolating agrees
# with ReverseDiff to 6 digits on the same data and is 4x faster than Gauss
# there anyway. Symptoms of Gauss NaN: dt_NaN warnings mid-optimization, NaN
# damper schedules, truncated pit solves.
const MEAT_SENSEALG = InterpolatingAdjoint(autojacvec = EnzymeVJP())

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
function _meat_core(p, cfg::MeatConfig, prob, tsave, Mts;
                    abstol = 1e-6, reltol = 1e-6, sensealg = MEAT_SENSEALG)
    nx = cfg.nx
    sol = solve(prob, FBDF(); p = p, saveat = tsave, abstol = abstol,
                reltol = reltol, sensealg = sensealg)
    A = Array(sol)                       # 3nx × nt
    # Fail loudly on a truncated/failed solve — a short solution array
    # silently propagates wrong-shaped outputs through the composed pipeline.
    (SciMLBase.successful_retcode(sol) && size(A, 2) == length(tsave)) ||
        throw(ErrorException(
            "meat solve failed: retcode=$(sol.retcode), " *
            "saved $(size(A, 2))/$(length(tsave)) points"))
    Tcore = A[1, :]
    Tsurf = A[nx, :]
    Wsurf = A[2nx, :]
    C_field_final = A[2nx+1:3nx, end]
    # juiciness proxy: mean retained moisture at pull, as a fraction of W0
    w_retained = sum(A[nx+1:2nx, end]) / (nx * cfg.W0)
    nk = size(Mts, 2)
    Tpit_ts = Mts * p[MEAT_NP+1:MEAT_NP+nk]
    bark_b = A[3nx+1, :]
    bark_s = A[3nx+2, :]
    l_star = l_star_bark.(bark_b, bark_s, Ref(cfg))
    # observables MUST apply the same wrap blend as the RHS: the bark
    # integral (∫ j_evap dt) and the pit coupling consume these, and an
    # unblended observable would carry a silently wrong t_wrap gradient
    t_wrap = p[end]
    s_ts = wrap_blend.(tsave, t_wrap, Ref(cfg))
    j_evap = (1 .- cfg.η_evap .* s_ts) .* evap_flux.(Tsurf, Wsurf, p[2], Ref(cfg))
    q_meat = cfg.A_meat .*
             surface_heat_in_wrapped.(Tpit_ts, Tsurf, p[3], s_ts, Ref(cfg))
    (; Tsurf, Tcore, j_evap, q_meat, C_field_final, w_retained,
       bark_b, bark_s, l_star)
end

"""
    meat_solve(θ, Tpit_knots, tknots, tsave; cfg, abstol, reltol)

Forward solve. Returns the Tesseract outputs plus full T/W/C field
trajectories (`T_field` etc., nx × nt) for tests and visualization.
"""
function meat_solve(θ::AbstractVector, Tpit_knots::AbstractVector,
                    tknots::AbstractVector, tsave::AbstractVector;
                    cfg::MeatConfig = MeatConfig(), abstol = 1e-6, reltol = 1e-6,
                    T0f::AbstractVector = Float64[], W0f::AbstractVector = Float64[],
                    C0f::AbstractVector = Float64[], t_wrap::Real = WRAP_NEVER,
                    smoke_knots::AbstractVector = Float64[])
    nx = cfg.nx
    sk = isempty(smoke_knots) ? zeros(length(tknots)) : collect(float.(smoke_knots))
    length(sk) == length(tknots) || throw(ArgumentError(
        "smoke_knots must share the tknots grid (got $(length(sk)) vs $(length(tknots)))"))
    p = vcat(θ, Tpit_knots, sk, float(t_wrap))
    any(isnan, p) && throw(ArgumentError(
        "meat solve got NaN inputs at indices $(findall(isnan, p)) of p (θ|Tpit|smoke|t_wrap)"))
    tk0, tkdt = float(tknots[1]), float(tknots[2] - tknots[1])
    rhs! = make_meat_rhs(cfg, tk0, tkdt, length(tknots))
    prob = ODEProblem(rhs!, _meat_u0(cfg, T0f, W0f, C0f),
                      (float(tsave[1]), float(tsave[end])), p)
    sol = solve(prob, FBDF(); saveat = tsave, abstol = abstol, reltol = reltol)
    A = Array(sol)
    (SciMLBase.successful_retcode(sol) && size(A, 2) == length(tsave)) ||
        throw(ErrorException(
            "meat solve failed: retcode=$(sol.retcode), " *
            "saved $(size(A, 2))/$(length(tsave)) points"))
    Tsurf = A[nx, :]
    Wsurf = A[2nx, :]
    Mts = hat_matrix(tsave, tk0, tkdt, length(tknots))
    Tpit_ts = Mts * Tpit_knots
    s_ts = wrap_blend.(tsave, float(t_wrap), Ref(cfg))
    bark_b = A[3nx+1, :]
    bark_s = A[3nx+2, :]
    (; Tsurf,
       Tcore = A[1, :],
       bark_b, bark_s,
       l_star = l_star_bark.(bark_b, bark_s, Ref(cfg)),
       j_evap = (1 .- cfg.η_evap .* s_ts) .*
                evap_flux.(Tsurf, Wsurf, θ[2], Ref(cfg)),
       q_meat = cfg.A_meat .*
                surface_heat_in_wrapped.(Tpit_ts, Tsurf, θ[3], s_ts, Ref(cfg)),
       C_field_final = A[2nx+1:3nx, end],
       w_retained = sum(A[nx+1:2nx, end]) / (nx * cfg.W0),
       T_field = A[1:nx, :], W_field = A[nx+1:2nx, :], C_field = A[2nx+1:3nx, :],
       t = collect(float.(tsave)))
end

"""
    meat_vjp(θ, Tpit_knots, tknots, tsave, cot; cfg, abstol, reltol)

Vector–Jacobian product of the meat Tesseract outputs w.r.t. (θ, Tpit_knots).
`cot` is a NamedTuple with any subset of keys
(:Tsurf, :Tcore, :j_evap, :q_meat, :C_field_final); missing keys are treated
as zero cotangents. Gradient flows through the continuous SciMLSensitivity
adjoint of the stiff solve.
"""
function meat_vjp(θ::AbstractVector, Tpit_knots::AbstractVector,
                  tknots::AbstractVector, tsave::AbstractVector, cot;
                  cfg::MeatConfig = MeatConfig(), abstol = 1e-6, reltol = 1e-6,
                  sensealg = MEAT_SENSEALG, t_wrap::Real = WRAP_NEVER,
                  smoke_knots::AbstractVector = Float64[])
    sk = isempty(smoke_knots) ? zeros(length(tknots)) : collect(float.(smoke_knots))
    p = vcat(θ, Tpit_knots, sk, float(t_wrap))
    tk0, tkdt = float(tknots[1]), float(tknots[2] - tknots[1])
    Mts = hat_matrix(tsave, tk0, tkdt, length(tknots))
    rhs! = make_meat_rhs(cfg, tk0, tkdt, length(tknots))
    prob = ODEProblem(rhs!, _meat_u0(cfg), (float(tsave[1]), float(tsave[end])), p)
    nt = length(tsave)
    z_t = zeros(nt)
    z_x = zeros(cfg.nx)
    cotfull = (
        Tsurf = get(cot, :Tsurf, z_t),
        Tcore = get(cot, :Tcore, z_t),
        j_evap = get(cot, :j_evap, z_t),
        q_meat = get(cot, :q_meat, z_t),
        C_field_final = get(cot, :C_field_final, z_x),
        w_retained = get(cot, :w_retained, 0.0),
        bark_b = get(cot, :bark_b, z_t),
        bark_s = get(cot, :bark_s, z_t),
        l_star = get(cot, :l_star, z_t),
    )
    _, back = Zygote.pullback(pp -> _meat_core(pp, cfg, prob, tsave, Mts;
                                               abstol, reltol, sensealg), p)
    dp = back(cotfull)[1]
    dp = dp === nothing ? zeros(length(p)) : dp
    nk = length(Tpit_knots)
    (dθ = dp[1:MEAT_NP],
     dTpit_knots = dp[MEAT_NP+1:MEAT_NP+nk],
     dsmoke_knots = dp[MEAT_NP+nk+1:end-1],
     dt_wrap = dp[end])
end
