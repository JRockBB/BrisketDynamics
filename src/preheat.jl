# The preheat: the pit lit and coming up empty until the meat goes on. The planner chooses the pit temperature the
# meat goes in at; the load itself stays a time (`t_load`, the monolith's and the system's input, no latch inside
# the cook): `preheat_time` finds when the empty pit first reaches that temperature under the planned damper.

# Only the six pit states evolve while the meat is absent. Use the same burn,
# humidity and pan algebra as the monolith without evaluating the meat RHS or
# factoring its frozen rows in every stiff step and every derivative chunk.
function _preheat_rhs(pit::PitConfig, hum::Humidity, u_t0, u_dt, nu::Int)
    return function empty_pit!(dx, x, p, t)
        C_th, UA, r_max, ΔH_eff, u_leak, k_spread = p[1:6]
        Tp, m, a, mp, Tpan, Tfb = x
        ud = clamp(lininterp(t, u_t0, u_dt, @view(p[(PIT_NP + 1):(PIT_NP + nu)])), 0, 1)
        airflow, ṁ, mair, pv_nopan = _pit_algebra(ud, m, a, r_max, u_leak, pit, hum)
        q_pan, ṁ_pan, _ = _pan_algebra(Tp, mp, Tpan, mair, pv_nopan, pit, hum)
        a_c = clamp(a, zero(a), one(a))
        if pit.c_duct > 0
            q_burn = pit.c_duct * airflow * (Tfb - Tp)
            dx[6] = (ΔH_eff * ṁ - pit.UA_fb * (Tfb - pit.T_amb) - q_burn) / pit.C_fb
        else
            q_burn = ΔH_eff * ṁ
            dx[6] = zero(Tp)
        end
        dx[1] = (q_burn + pit.p_heater * ud - UA * (Tp - pit.T_amb) - q_pan) / C_th
        dx[2] = -ṁ
        dx[3] = k_spread * a_c * (1 - a_c) * airflow
        dx[4] = -ṁ_pan
        dx[5] = pit.m_pan0 > 0 ?
                (q_pan - pit.λ_pan * ṁ_pan) / (CP_WATER * max(mp, zero(mp)) + pit.C_pan_dry) : zero(Tp)
        return nothing
    end
end

"""
    preheat_time(T_load, φ, u_knots, u_tknots; cfg, pit, hum, t_max=PREHEAT_MAX, Tpit0, mfuel0, ignited, mpan0, Tpan0,
                 Tfb0, abstol, reltol) -> (t_load, reached, shortfall)

When the empty pit, lit at t = 0 under the damper schedule `u_knots` on `u_tknots`, first reaches `T_load` [K]: the
monolith's own pit physics with the meat never loaded (it draws nothing), integrated until `T_pit = T_load`. The
pit starts at `Tpit0` (by default `pit.Tpit0`, the monolith's own start: ambient for a cook from lighting), `mfuel0` of fuel, the bed `ignited` lit (by default φ[7], the lit
fraction the monolith's own state starts from: a chimney start's `PitConfig.a_light` for a cook from lighting), the
firebox at `Tpit0`. Returns the load time [s], whether the pit got there before `t_max` (if not, `t_max`: not an
error), and the shortfall [K], `T_load` minus the pit at `t_max` when it did not (0 when it did), differentiable so a
planner's miss penalty has a gradient that opens the preheat damper; a pit already at or above `T_load` loads at
once (0).

Differentiable in every input by ForwardDiff: the crossing is found on the values, then the implicit-function
theorem carries the partials, `t = t* − (T_pit(t*) − T_load)/Ṫ_pit(t*)` evaluated in dual arithmetic (at the
values the correction is zero).
"""
function preheat_time(T_load, φ::AbstractVector, u_knots::AbstractVector, u_tknots::AbstractVector;
                      cfg::MeatConfig=MeatConfig(), pit::PitConfig=PitConfig(), hum::Humidity=Humidity(),
                      t_max::Real=PREHEAT_MAX, Tpit0::Real=NaN, mfuel0::Real=NaN, ignited::Real=NaN,
                      mpan0::Real=NaN, Tpan0::Real=NaN, Tfb0::Real=NaN, abstol=1e-8, reltol=1e-8)
    length(φ) == PIT_NP || throw(ArgumentError("φ must have length $PIT_NP; got $(length(φ))"))
    length(u_knots) == length(u_tknots) || throw(ArgumentError("u_knots and u_tknots differ in length"))
    nu = length(u_knots)
    a0 = isnan(_primal(ignited)) ? φ[PIT_NP] : ignited
    φl = vcat(φ[1:(PIT_NP - 1)], a0)                                  # the bed at lighting
    # the knot grid may carry partials too (duration mode: the knots at s_u·t_f): the crossing is found on its values
    u_t0, u_dt = float(u_tknots[1]), float(u_tknots[2] - u_tknots[1])
    rhs! = _preheat_rhs(pit, hum, _primal(u_t0), _primal(u_dt), nu)
    rhsd! = _preheat_rhs(pit, hum, u_t0, u_dt, nu)
    T0p = isnan(_primal(Tpit0)) ? pit.Tpit0 : Tpit0                    # the monolith's own start
    # the initial state in element type Tp (the values: `val` maps each input to its primal)
    function x0_of(p, Tp, val)
        return Tp.([val(T0p), val(isnan(_primal(mfuel0)) ? pit.m0 : mfuel0), p[PIT_NP],
                         val(isnan(_primal(mpan0)) ? pit.m_pan0 : mpan0), val(isnan(_primal(Tpan0)) ? pit.T_pan0 : Tpan0),
                         val(isnan(_primal(Tfb0)) ? T0p : Tfb0)])
    end
    # the crossing, on the values
    pv = vcat(_primal.(φl), _primal.(u_knots))
    Tl = _primal(T_load)
    _primal(T0p) < Tl || return (zero(float(Tl)), true, zero(float(Tl)))   # already there: the meat goes on now
    cond(x, t, integ) = x[1] - Tl
    cb = SciMLBase.ContinuousCallback(cond, integ -> SciMLBase.terminate!(integ); affect_neg! = nothing)
    prob = ODEProblem{true,SciMLBase.FullSpecialize}(rhs!, x0_of(pv, Float64, _primal), (0.0, float(t_max)), pv)
    # The empty pit has no stiff meat diffusion modes. The explicit pit solver
    # also resolves small crossing-time sensitivities without BDF step noise.
    sol = solve(prob, Tsit5(); callback=cb, abstol, reltol, save_everystep=false)
    sol.retcode == SciMLBase.ReturnCode.Terminated || SciMLBase.successful_retcode(sol) ||
        throw(ErrorException("preheat solve failed: retcode=$(sol.retcode)"))
    reached = sol.retcode == SciMLBase.ReturnCode.Terminated
    tstar = sol.t[end]
    # Keep the input type independent of the number of damper knots. A tuple
    # splatted from the schedule specializes the dual solve again at every MPC
    # horizon length; a vector carries exactly the same promoted scalar type.
    ins = vcat(T_load, φ, u_knots, u_t0, u_dt, ignited, Tpit0, mfuel0, mpan0, Tpan0, Tfb0)
    duals = eltype(ins) <: ForwardDiff.Dual
    # the partials (only when an input carries them): one dual solve to t* (the crossing, or t_max on a miss)
    function dual_end()
        pd = vcat(φl, u_knots)
        Td = eltype(ins)
        probd = ODEProblem{true,SciMLBase.FullSpecialize}(rhsd!, x0_of(Td.(pd), Td, identity), (0.0, tstar), Td.(pd))
        sold = solve(probd, Tsit5(); abstol, reltol, save_everystep=false, saveat=[tstar])
        SciMLBase.successful_retcode(sold) ||
            throw(ErrorException("preheat derivative solve failed: retcode=$(sold.retcode)"))
        return sold.u[end][1]
    end
    if !reached                                                        # the miss: how far short the pit stayed
        sf = duals ? T_load - dual_end() : Tl - sol.u[end][1]
        return (float(t_max), false, sf)
    end
    duals || return (tstar, true, zero(float(Tl)))
    # the crossing time by the implicit-function theorem
    Tstar = dual_end()
    dx = zeros(length(sol.u[end]))
    rhs!(dx, sol.u[end], pv, tstar)
    return (tstar - (Tstar - T_load) / dx[1], true, zero(Tstar))
end
