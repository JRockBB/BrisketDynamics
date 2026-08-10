# Lumped-parameter smoker ("pit") model: 2 states — pit gas temperature and
# remaining fuel mass. Fuel burn rate responds to the damper schedule u(t)
# (with a leak floor, so closing the damper never fully kills the fire) and
# tapers smoothly to zero as the fuel runs out.
#
# Differentiation: deliberately a *different* strategy than the meat PDE.
# The system is tiny (2 states) and non-stiff, so the exact dense Jacobian
# over the full input vector via ForwardDiff (chunked forward mode) is cheap;
# the VJP endpoint returns Jᵀv̄. This makes the composed pipeline a genuine
# adjoint(meat) + forward(pit) composition across the Tesseract boundary.

Base.@kwdef struct PitConfig
    T_amb::Float64 = 293.15    # ambient temperature [K]
    Tpit0::Float64 = 293.15    # initial pit temperature [K] (cold start)
    m0::Float64 = 7.0          # initial charcoal load [kg] (full WSM ring)
    m_taper::Float64 = 0.5     # fuel mass scale for the smooth burn-out taper [kg]
end

# φ = [C_th, UA, r_max, ΔH_eff, u_leak]
const PIT_NP = 5

"""
    make_pit_rhs(cfg, u_t0, u_dt, u_nk, q_t0, q_dt, q_nk)

Pit RHS with parameter vector `p = vcat(φ, u_knots, qmeat_knots)`:
damper schedule (u_nk knots) and meat heat-load trajectory (q_nk knots) both
live in `p`, exposing VJPs w.r.t. the control and the coupling input.
"""
function make_pit_rhs(cfg::PitConfig, u_t0, u_dt, u_nk::Int, q_t0, q_dt, q_nk::Int)
    function pit_rhs!(dx, x, p, t)
        C_th, UA, r_max, ΔH_eff, u_leak = p[1], p[2], p[3], p[4], p[5]
        u_knots = @view p[PIT_NP+1:PIT_NP+u_nk]
        q_knots = @view p[PIT_NP+u_nk+1:PIT_NP+u_nk+q_nk]

        u = clamp(lininterp(t, u_t0, u_dt, u_knots), 0, 1)
        q_meat = lininterp(t, q_t0, q_dt, q_knots)

        Tp, m = x[1], x[2]
        ṁ = r_max * (u_leak + (1 - u_leak) * u) * tanh(max(m, zero(m)) / cfg.m_taper)
        dx[1] = (ΔH_eff * ṁ - UA * (Tp - cfg.T_amb) - q_meat) / C_th
        dx[2] = -ṁ
        nothing
    end
end

function _pit_solve_arr(p, cfg::PitConfig, u_tknots, q_tknots, tsave;
                        abstol = 1e-8, reltol = 1e-8)
    u_t0, u_dt = float(u_tknots[1]), float(u_tknots[2] - u_tknots[1])
    q_t0, q_dt = float(q_tknots[1]), float(q_tknots[2] - q_tknots[1])
    rhs! = make_pit_rhs(cfg, u_t0, u_dt, length(u_tknots), q_t0, q_dt, length(q_tknots))
    x0 = convert.(eltype(p), [cfg.Tpit0, cfg.m0])
    any(isnan, p) && throw(ArgumentError(
        "pit solve got NaN inputs at indices $(findall(isnan, p)) of p (φ|u|q)"))
    prob = ODEProblem(rhs!, x0, (float(tsave[1]), float(tsave[end])), p)
    sol = solve(prob, Tsit5(); saveat = tsave, abstol = abstol, reltol = reltol)
    A = Array(sol)                            # 2 × nt
    # A failed solve must fail LOUDLY: a truncated solution array silently
    # propagates wrong-shaped outputs through the composed pipeline.
    (SciMLBase.successful_retcode(sol) && size(A, 2) == length(tsave)) ||
        throw(ErrorException(
            "pit solve failed: retcode=$(sol.retcode), " *
            "saved $(size(A, 2))/$(length(tsave)) points"))
    A
end

"""
    pit_solve(φ, u_knots, u_tknots, qmeat_knots, q_tknots, tsave; cfg)

Forward solve. Returns `(; Tpit, m_fuel)` trajectories at `tsave`.
"""
function pit_solve(φ::AbstractVector, u_knots::AbstractVector, u_tknots::AbstractVector,
                   qmeat_knots::AbstractVector, q_tknots::AbstractVector,
                   tsave::AbstractVector; cfg::PitConfig = PitConfig(),
                   abstol = 1e-8, reltol = 1e-8)
    A = _pit_solve_arr(vcat(φ, u_knots, qmeat_knots), cfg, u_tknots, q_tknots, tsave;
                       abstol, reltol)
    (Tpit = A[1, :], m_fuel = A[2, :], t = collect(float.(tsave)))
end

"""
    pit_vjp(φ, u_knots, u_tknots, qmeat_knots, q_tknots, tsave, cot; cfg)

VJP of (Tpit, m_fuel) w.r.t. (φ, u_knots, qmeat_knots) via the full
ForwardDiff Jacobian: Jᵀv̄. `cot` is a NamedTuple with any subset of keys
(:Tpit, :m_fuel).
"""
function pit_vjp(φ::AbstractVector, u_knots::AbstractVector, u_tknots::AbstractVector,
                 qmeat_knots::AbstractVector, q_tknots::AbstractVector,
                 tsave::AbstractVector, cot; cfg::PitConfig = PitConfig(),
                 abstol = 1e-8, reltol = 1e-8)
    nu, nq, nt = length(u_knots), length(qmeat_knots), length(tsave)
    v = vcat(φ, u_knots, qmeat_knots)
    f = p -> vec(_pit_solve_arr(p, cfg, u_tknots, q_tknots, tsave; abstol, reltol))
    J = ForwardDiff.jacobian(f, v)            # (2nt) × (5+nu+nq)
    z = zeros(nt)
    v̄ = vec(transpose(hcat(get(cot, :Tpit, z), get(cot, :m_fuel, z))))
    g = transpose(J) * v̄
    (dφ = g[1:PIT_NP], du_knots = g[PIT_NP+1:PIT_NP+nu],
     dqmeat_knots = g[PIT_NP+nu+1:PIT_NP+nu+nq])
end
