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
end

# θ = [D_eff, k_m, h_c, L, k_mult] — the meat-side latent parameter vector.
const MEAT_NP = 5

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
    function meat_rhs!(du, u, p, t)
        D_eff, k_m, h_c, L, k_mult = p[1], p[2], p[3], p[4], p[5]
        knots = @view p[MEAT_NP+1:MEAT_NP+nk]

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

        # surface: half-cell finite-volume balance
        Ts, Ws = T[nx], W[nx]
        j = evap_flux(Ts, Ws, k_m, cfg)
        q_net = surface_heat_in(Tpit, Ts, h_c, cfg) - cfg.λ_vap * j
        dT[nx] = (k_th * (T[nx-1] - Ts) / Δx + q_net) / (RHO_MEAT * CP_MEAT * Δx / 2)
        dW[nx] = (RHO_DRY * D_eff * (W[nx-1] - Ws) / Δx - j) / (RHO_DRY * Δx / 2)

        # collagen conversion (doneness diagnostic as state — adjoint for free)
        @inbounds for i in 1:nx
            dC[i] = collagen_rate(T[i], cfg) * (1 - C[i])
        end
        nothing
    end
end

_meat_u0(cfg::MeatConfig) =
    vcat(fill(cfg.T0, cfg.nx), fill(cfg.W0, cfg.nx), zeros(cfg.nx))

# ReverseDiffVJP(compile = false), deliberately: with a compiled tape the
# time-dependent hat-basis interpolation weights are evaluated once at taping
# time and frozen into the tape as constants, which silently zeroes the
# adjoint w.r.t. the forcing knots (verified empirically: dTpit ≡ 0 and wrong
# dh_c/dL under compile = true; ≤ 2e-4 relative error vs FD under false).
# Re-taping is also faster here — the RHS closure is rebuilt per call, so a
# compiled tape could never be reused anyway.
const MEAT_SENSEALG = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(false))

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
    Tcore = A[1, :]
    Tsurf = A[nx, :]
    Wsurf = A[2nx, :]
    C_field_final = A[2nx+1:3nx, end]
    Tpit_ts = Mts * p[MEAT_NP+1:end]
    j_evap = evap_flux.(Tsurf, Wsurf, p[2], Ref(cfg))
    q_meat = cfg.A_meat .* surface_heat_in.(Tpit_ts, Tsurf, p[3], Ref(cfg))
    (; Tsurf, Tcore, j_evap, q_meat, C_field_final)
end

"""
    meat_solve(θ, Tpit_knots, tknots, tsave; cfg, abstol, reltol)

Forward solve. Returns the Tesseract outputs plus full T/W/C field
trajectories (`T_field` etc., nx × nt) for tests and visualization.
"""
function meat_solve(θ::AbstractVector, Tpit_knots::AbstractVector,
                    tknots::AbstractVector, tsave::AbstractVector;
                    cfg::MeatConfig = MeatConfig(), abstol = 1e-6, reltol = 1e-6)
    nx = cfg.nx
    p = vcat(θ, Tpit_knots)
    tk0, tkdt = float(tknots[1]), float(tknots[2] - tknots[1])
    rhs! = make_meat_rhs(cfg, tk0, tkdt, length(tknots))
    prob = ODEProblem(rhs!, _meat_u0(cfg), (float(tsave[1]), float(tsave[end])), p)
    sol = solve(prob, FBDF(); saveat = tsave, abstol = abstol, reltol = reltol)
    A = Array(sol)
    Tsurf = A[nx, :]
    Wsurf = A[2nx, :]
    Mts = hat_matrix(tsave, tk0, tkdt, length(tknots))
    Tpit_ts = Mts * Tpit_knots
    (; Tsurf,
       Tcore = A[1, :],
       j_evap = evap_flux.(Tsurf, Wsurf, θ[2], Ref(cfg)),
       q_meat = cfg.A_meat .* surface_heat_in.(Tpit_ts, Tsurf, θ[3], Ref(cfg)),
       C_field_final = A[2nx+1:3nx, end],
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
                  sensealg = MEAT_SENSEALG)
    p = vcat(θ, Tpit_knots)
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
    )
    _, back = Zygote.pullback(pp -> _meat_core(pp, cfg, prob, tsave, Mts;
                                               abstol, reltol, sensealg), p)
    dp = back(cotfull)[1]
    dp = dp === nothing ? zeros(length(p)) : dp
    (dθ = dp[1:MEAT_NP], dTpit_knots = dp[MEAT_NP+1:end])
end
