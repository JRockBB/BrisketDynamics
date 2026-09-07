# 2-D axisymmetric heat–moisture core (the fidelity rung beyond Newman):
# temperature and dry-basis moisture on an r–z grid with the SAME
# property functions as the 1-D model (Choi–Okos mixture conductivity
# and capacity, Oswin water activity, the smooth purge sink) and the
# same evaporative + convective/radiative surface conditions on the
# outer faces (rim and both ends; corner cells legitimately carry both).
#
# Solver: fixed-step lagged-coefficient ADI — sequential implicit
# tridiagonal sweeps in r then z (Thomas algorithm, pure loops); the
# purge reaction is explicit, convective surface heat enters the sweeps
# as an implicit Robin term (radiation and evaporative heat lagged in
# the source), and face moisture loss is linearized-implicit
# backward-Euler. Exactly the structure the adjoint work validated
# (first-order accurate, Enzyme-transparent; docs/results.md
# "Adjoint port").
#
# v1 scope (deliberate): no shrink metrics, no wrap/rest/spritz/salt/
# bone, no bark states — the quality states stay on the 1-D model until
# thermal-camera surface fields exist to calibrate a 2-D bark
# distribution (roadmap M2/C2). Fat is a uniform constant in the
# properties. Nothing here is calibrated.

"""
    meat_solve_2d(θ2, Tpit_knots, tknots, tsave; R, Hz, nr, nz, cfg, h)

Axisymmetric finite-cylinder cook: radius `R`, full height `Hz`,
`nr × nz` nodes (i = 1 on the axis; j = 1 and j = nz are the exposed end
faces). `θ2 = [D_eff, k_m, h_c, k_mult, k_e, a_myosin]`. Fixed step `h`
[s]; the pit forcing interpolates from `Tpit_knots` on `tknots`; save
times `tsave` must start at the cook start.

Layer options (all exactly inert at their defaults): `t_wrap` applies
the 1-D wrap blend to every exposed face; `k_grain` / `D_grain` scale
in-plane (along-fiber) conduction and moisture diffusion — the z
(through-thickness) direction keeps the 1-D-calibrated values;
`F_top` + `z_split` stack a second muscle of fat fraction `F_top` above
`z_split`·Hz (the point over the flat, a smooth step).

Returns `(t, Tcore, T_field, W_field, w_retained, C_field, C_core,
C_min, Tcold, bark_rim, bark_ends, bark_mean, bark_std)` — Tcore at the
axis mid-height, fields `nr × nz × nt`, `w_retained` the volume-weighted
retained-water fraction, collagen conversion per cell (`C_min` = the
last-done spot at serve), `Tcold` the true cold-spot trace (grid
minimum — the safety margin no symmetric 1-D core can show), and the
Maillard bark DISTRIBUTION over the exposed faces (area-weighted mean
and spread — burnt-ends live in the tail).
"""
# The transport engine shared by the prescribed-forcing solve and the
# pit-coupled solve: builds the grid state and returns a `step!` closure
# advancing one fixed step under (t, Tpit, p_v), returning the
# area-integrated surface heat uptake [W] and evaporation [kg/s] of ONE
# brisket (what a pit loses to it).
function _slab2d_engine(
    θ2, R, Hz, nr, nz, cfg, h, t_wrap, k_grain, D_grain, F_top, z_split, Tv,
    t_salt, t_spritz, dt_spritz, m_spritz
)
    D_eff, k_m, h_c, k_mult, k_e, a_myosin = θ2
    Δr = R / (nr - 1)
    Δz = Hz / (nz - 1)

    # point-vs-flat as stacked muscles: fat fraction F(z) steps smoothly
    # from cfg.F0 (flat) to F_top (point) at z_split·Hz (NaN F_top or
    # z_split >= 1 leaves the uniform cut exactly)
    Ftop = isnan(F_top) ? Tv(cfg.F0) : Tv(F_top)
    δ_split = Hz / 12
    Fz = [Tv(cfg.F0) + (Ftop - Tv(cfg.F0)) *
          logistic(((j - 1) * Δz - z_split * Hz) / δ_split) for j in 1:nz]

    T = fill(Tv(cfg.T0), nr, nz)
    W = fill(Tv(cfg.W0), nr, nz)
    # dry-brine initial condition: the 1-D erfc profile from EVERY rubbed
    # face as a smooth union (exactly the single-face profile away from
    # corners, capped at S_surf where fronts overlap); t_salt <= 0 is
    # exact zeros — compare the PRIMAL, the eps-aware Dual lesson
    S = zeros(Tv, nr, nz)
    ts_v = isnan(_primal(t_salt)) ? cfg.t_salt : t_salt
    if _primal(ts_v) > 0
        pen = 2 * sqrt(cfg.D_salt * ts_v)
        for j in 1:nz, i in 1:nr
            e_r = erfc_smooth((R - (i - 1) * Δr) / pen)
            e_lo = erfc_smooth((j - 1) * Δz / pen)
            e_hi = erfc_smooth((Hz - (j - 1) * Δz) / pen)
            S[i, j] = Tv(cfg.S_surf) *
                      (1 - (1 - e_r) * (1 - e_lo) * (1 - e_hi))
        end
    end
    msp = isnan(_primal(m_spritz)) ? cfg.m_spritz : m_spritz
    # quality states ride along explicitly (they never feed back into
    # the transport, exactly like the 1-D model): collagen conversion
    # per cell, Maillard browning per exposed face cell
    Cmat = zeros(Tv, nr, nz)
    B_rim = zeros(Tv, nz)
    B_end = zeros(Tv, nr, 2)
    kmat = zeros(Tv, nr, nz)
    ρcmat = zeros(Tv, nr, nz)
    # shrink metrics, mirroring the 1-D material coordinate per
    # direction: st = (v/v0)^n_shrink is the local stretch of BOTH the r
    # and z spacings (at the default n_shrink = 1/3 this is the
    # isotropic split, and both 1-D limits use the same exponent for
    # their resolved direction); the exposed-area factor (v̄/v0)^n_a
    # multiplies the dry-basis moisture flux only — for heat it cancels
    # between current-area intake and current-volume capacity, exactly
    # as in meat_rhs_core!. n_shrink = 0 makes every factor exactly 1.
    n_s, n_a, v0s = shrink_exponents(cfg)
    stmat = ones(Tv, nr, nz)
    nmax = max(nr, nz)
    aa = zeros(Tv, nmax); bb = zeros(Tv, nmax)
    cc = zeros(Tv, nmax); dd = zeros(Tv, nmax)

    # radial cell geometry: cell i owns [r − Δr/2, r + Δr/2] (half cells
    # at the axis and the rim); face area ∝ r_face, volume ∝ ∫ r dr
    rF = [(i - 0.5) * Δr for i in 1:(nr - 1)]           # outer face of cell i
    rV = [i == 1 ? Δr^2 / 8 :
          i == nr ? ((nr - 1)^2 - (nr - 1.5)^2) * Δr^2 / 2 :
          ((i - 0.5)^2 - (i - 1.5)^2) * Δr^2 / 2 for i in 1:nr]

    jr_face = zeros(Tv, nz)
    jz_face = zeros(Tv, nr, 2)
    zw = [j == 1 || j == nz ? Tv(0.5) : Tv(1.0) for j in 1:nz]

    function step!(t_now, Tpit, pv)
        # the wrap is the same smooth time blend as the 1-D model: it
        # damps face evaporation by (1 - η_evap·s), convection by
        # (1 - η_h·s) and blends the emissivity toward ε_wrap
        sw = wrap_blend(t_now, t_wrap, cfg)
        km_w = (1 - cfg.η_evap * sw) * k_m
        hc_w = h_c * (1 - cfg.η_h * sw)
        # spritz: water lands on every exposed face while unwrapped, a
        # fraction f_abs soaks in, the rest flashes off with its latent
        # heat (the 1-D q_net terms, mirrored per face)
        j_sp = spritz_rate(t_now, t_spritz, dt_spritz, msp, cfg) * (1 - sw)
        # pre-step surface heat uptake — what the pit loses to this
        # brisket this step (rim strips + both end annuli, real areas)
        q_up = zero(Tv)
        for j in 1:nz
            q_up += 2π * R * Δz * zw[j] *
                    surface_heat_in_wrapped(Tpit, T[nr, j], h_c, sw, cfg)
        end
        for i in 1:nr, jz in (1, nz)
            q_up += 2π * rV[i] * surface_heat_in_wrapped(Tpit, T[i, jz], h_c, sw, cfg)
        end
        vbar = zero(Tv)
        wsum = zero(Tv)
        for j in 1:nz, i in 1:nr
            v = specific_volume(W[i, j], Fz[j], cfg)
            kmat[i, j] = k_mult * conductivity_mix(W[i, j], Fz[j], T[i, j], v, cfg)
            ρcmat[i, j] = heat_capacity_dry(W[i, j], Fz[j], T[i, j], cfg) / v
            stmat[i, j] = (v / v0s)^n_s
            vbar += rV[i] * zw[j] * v
            wsum += rV[i] * zw[j]
        end
        a_area = (vbar / (wsum * v0s))^n_a

        # explicit: purge everywhere, evaporative WATER loss on the
        # exposed faces (mild rates); the surface HEAT flux is handled
        # inside the sweeps as a Robin condition — an explicit kick of
        # h·q/C into a millimetre half-cell biases the surface warm and
        # over-dries it (measured: +9 K and 4x the drying rate on the
        # end faces before this was made implicit)
        for j in 1:nz, i in 1:nr
            W[i, j] -= h * expulsion_rate_salted(W[i, j], T[i, j], S[i, j],
                                                 k_e, a_myosin, cfg)
            Cmat[i, j] += h * collagen_rate(T[i, j], cfg) * (1 - Cmat[i, j])
        end
        for j in 1:nz
            B_rim[j] += h * bark_rate(T[nr, j], a_w_oswin(W[nr, j], cfg), cfg) *
                        (1 - B_rim[j])
        end
        for i in 1:nr, (kz, jz) in enumerate((1, nz))
            B_end[i, kz] += h *
                            bark_rate(T[i, jz], a_w_oswin(W[i, jz], cfg), cfg) *
                            (1 - B_end[i, kz])
        end
        # face evaporation is linearized-implicit: an explicit update
        # overshoots the a_w equilibrium once the face dries out, the
        # lagged flux then flips sign every step, and the alternating
        # λ·j heat kicks rectify into net heating that diverges under
        # mesh refinement (measured: nr 24→48 pushed the cylinder core
        # 97→108 °C).  Backward-Euler on the W-linearized flux damps
        # the update, and the SAME applied flux feeds the Robin sink.
        δW = Tv(1e-3)
        for j in 1:nz
            C = RHO_DRY * Δr / 2
            fa = 1 / (1 + cfg.k_aw * max(S[nr, j], zero(Tv)))
            j0 = evap_flux(T[nr, j], W[nr, j], km_w, pv, cfg, fa)
            s = (evap_flux(T[nr, j], W[nr, j] + δW, km_w, pv, cfg, fa) - j0) / δW
            jr_face[j] = j0 / (1 + h * a_area * max(s, zero(Tv)) / C)
            W[nr, j] += h * a_area * (cfg.f_abs * j_sp - jr_face[j]) / C
        end
        for i in 1:nr, (kz, jz) in enumerate((1, nz))
            C = RHO_DRY * Δz / 2
            fa = 1 / (1 + cfg.k_aw * max(S[i, jz], zero(Tv)))
            j0 = evap_flux(T[i, jz], W[i, jz], km_w, pv, cfg, fa)
            s = (evap_flux(T[i, jz], W[i, jz] + δW, km_w, pv, cfg, fa) - j0) / δW
            jz_face[i, kz] = j0 / (1 + h * a_area * max(s, zero(Tv)) / C)
            W[i, jz] += h * a_area * (cfg.f_abs * j_sp - jz_face[i, kz]) / C
        end
        j_up = zero(Tv)
        for j in 1:nz
            j_up += 2π * R * Δz * zw[j] * jr_face[j]
        end
        for i in 1:nr, kz in 1:2
            j_up += 2π * rV[i] * jz_face[i, kz]
        end

        # implicit radial sweeps (each z-row): T with k/ρc, W with D_eff
        for j in 1:nz
            for i in 1:nr
                # k_grain scales in-plane (along-fiber) conduction; the z
                # direction is through-thickness, matching the 1-D values
                lo = i > 1 ?
                     k_grain * (kmat[i - 1, j] + kmat[i, j]) / 2 * rF[i - 1] /
                     ((stmat[i - 1, j] + stmat[i, j]) / 2 * Δr * rV[i] *
                      stmat[i, j] * ρcmat[i, j]) : zero(Tv)
                up = i < nr ?
                     k_grain * (kmat[i, j] + kmat[i + 1, j]) / 2 * rF[i] /
                     ((stmat[i, j] + stmat[i + 1, j]) / 2 * Δr * rV[i] *
                      stmat[i, j] * ρcmat[i, j]) : zero(Tv)
                aa[i] = -h * lo
                cc[i] = -h * up
                bb[i] = 1 + h * (lo + up)
                dd[i] = T[i, j]
            end
            βr = hc_w / (ρcmat[nr, j] * stmat[nr, j] * Δr / 2)
            qx = (surface_heat_in_wrapped(Tpit, T[nr, j], h_c, sw, cfg) -
                  hc_w * (Tpit - T[nr, j]) -
                  cfg.λ_vap * (jr_face[j] + (1 - cfg.f_abs) * j_sp) -
                  CP_WATER * (T[nr, j] - cfg.T_spritz) * j_sp) /
                 (ρcmat[nr, j] * stmat[nr, j] * Δr / 2)
            bb[nr] += h * βr
            dd[nr] += h * (βr * Tpit + qx)
            _thomas_row!(T, aa, bb, cc, dd, nr, j)
            for i in 1:nr
                lo = i > 1 ? D_grain * D_eff * rF[i - 1] /
                             (((stmat[i - 1, j] + stmat[i, j]) / 2)^2 * Δr *
                              rV[i]) : zero(Tv)
                up = i < nr ? D_grain * D_eff * rF[i] /
                              (((stmat[i, j] + stmat[i + 1, j]) / 2)^2 * Δr *
                               rV[i]) : zero(Tv)
                aa[i] = -h * lo
                cc[i] = -h * up
                bb[i] = 1 + h * (lo + up)
                dd[i] = W[i, j]
            end
            _thomas_row!(W, aa, bb, cc, dd, nr, j)
            if cfg.D_salt > 0
                for i in 1:nr
                    lo = i > 1 ? D_grain * cfg.D_salt * rF[i - 1] /
                                 (((stmat[i - 1, j] + stmat[i, j]) / 2)^2 * Δr *
                                  rV[i]) : zero(Tv)
                    up = i < nr ? D_grain * cfg.D_salt * rF[i] /
                                  (((stmat[i, j] + stmat[i + 1, j]) / 2)^2 * Δr *
                                   rV[i]) : zero(Tv)
                    aa[i] = -h * lo
                    cc[i] = -h * up
                    bb[i] = 1 + h * (lo + up)
                    dd[i] = S[i, j]
                end
                _thomas_row!(S, aa, bb, cc, dd, nr, j)
            end
        end
        # implicit axial sweeps (each r-column)
        for i in 1:nr
            for j in 1:nz
                # the end rows are HALF cells (capacity Δz/2): their
                # diffusion coefficient doubles, matching the flux terms
                cell = (j == 1 || j == nz) ? Δz / 2 : Δz
                lo = j > 1 ? (kmat[i, j - 1] + kmat[i, j]) / 2 /
                             ((stmat[i, j - 1] + stmat[i, j]) / 2 * Δz * cell *
                              stmat[i, j] * ρcmat[i, j]) : zero(Tv)
                up = j < nz ? (kmat[i, j] + kmat[i, j + 1]) / 2 /
                              ((stmat[i, j] + stmat[i, j + 1]) / 2 * Δz * cell *
                               stmat[i, j] * ρcmat[i, j]) : zero(Tv)
                aa[j] = -h * lo
                cc[j] = -h * up
                bb[j] = 1 + h * (lo + up)
                dd[j] = T[i, j]
            end
            for (kz, jz) in enumerate((1, nz))
                βz = hc_w / (ρcmat[i, jz] * stmat[i, jz] * Δz / 2)
                qx = (surface_heat_in_wrapped(Tpit, T[i, jz], h_c, sw, cfg) -
                      hc_w * (Tpit - T[i, jz]) -
                      cfg.λ_vap * (jz_face[i, kz] + (1 - cfg.f_abs) * j_sp) -
                      CP_WATER * (T[i, jz] - cfg.T_spritz) * j_sp) /
                     (ρcmat[i, jz] * stmat[i, jz] * Δz / 2)
                bb[jz] += h * βz
                dd[jz] += h * (βz * Tpit + qx)
            end
            _thomas_col!(T, aa, bb, cc, dd, nz, i)
            for j in 1:nz
                cell = (j == 1 || j == nz) ? Δz / 2 : Δz
                lo = j > 1 ? D_eff /
                             (((stmat[i, j - 1] + stmat[i, j]) / 2)^2 * Δz *
                              cell) : zero(Tv)
                up = j < nz ? D_eff /
                              (((stmat[i, j] + stmat[i, j + 1]) / 2)^2 * Δz *
                               cell) : zero(Tv)
                aa[j] = -h * lo
                cc[j] = -h * up
                bb[j] = 1 + h * (lo + up)
                dd[j] = W[i, j]
            end
            _thomas_col!(W, aa, bb, cc, dd, nz, i)
            if cfg.D_salt > 0
                for j in 1:nz
                    cell = (j == 1 || j == nz) ? Δz / 2 : Δz
                    lo = j > 1 ? cfg.D_salt /
                                 (((stmat[i, j - 1] + stmat[i, j]) / 2)^2 * Δz *
                                  cell) : zero(Tv)
                    up = j < nz ? cfg.D_salt /
                                  (((stmat[i, j] + stmat[i, j + 1]) / 2)^2 * Δz *
                                   cell) : zero(Tv)
                    aa[j] = -h * lo
                    cc[j] = -h * up
                    bb[j] = 1 + h * (lo + up)
                    dd[j] = S[i, j]
                end
                _thomas_col!(S, aa, bb, cc, dd, nz, i)
            end
        end
        return q_up, j_up
    end

    return (; step!, T, W, Cmat, B_rim, B_end, rV, zw, Δr, Δz, Tv, stmat, S)
end

function meat_solve_2d(
    θ2::AbstractVector,
    Tpit_knots::AbstractVector,
    tknots::AbstractVector,
    tsave::AbstractVector;
    R::Real=0.05,
    Hz::Real=0.09,
    nr::Int=20,
    nz::Int=20,
    cfg::MeatConfig=MeatConfig(),
    h::Real=30.0,
    t_wrap::Real=WRAP_NEVER,
    k_grain::Real=1.0,
    D_grain::Real=1.0,
    F_top::Real=NaN,
    z_split::Real=1.0,
    t_salt::Real=NaN,
    t_spritz::Real=SPRITZ_NEVER,
    dt_spritz::Real=3600.0,
    m_spritz::Real=NaN,
)
    Tv = promote_type(eltype(θ2), eltype(Tpit_knots), typeof(float(t_wrap)),
                      typeof(float(t_salt)), typeof(float(t_spritz)),
                      typeof(float(dt_spritz)), Float64)
    eng = _slab2d_engine(θ2, R, Hz, nr, nz, cfg, h, t_wrap, k_grain, D_grain,
                         F_top, z_split, Tv, t_salt, t_spritz, dt_spritz,
                         m_spritz)
    tk0, tkdt = float(tknots[1]), float(tknots[2] - tknots[1])
    t0 = float(tsave[1])
    nt_steps = round(Int, (float(tsave[end]) - t0) / h)
    save_at = [round(Int, (float(ts) - t0) / h) for ts in tsave]
    Tout = zeros(Tv, size(eng.T)..., length(tsave))
    Wout = zeros(Tv, size(eng.T)..., length(tsave))
    Cout = zeros(Tv, size(eng.T)..., length(tsave))
    Tcold = fill(Tv(cfg.T0), length(tsave))
    isave = 1
    while isave <= length(save_at) && save_at[isave] <= 0
        Tout[:, :, isave] .= eng.T
        Wout[:, :, isave] .= eng.W
        isave += 1
    end
    for n in 1:nt_steps
        t_now = t0 + (n - 1) * h
        Tpit = lininterp(t_now, tk0, tkdt, Tpit_knots)
        eng.step!(t_now, Tpit, cfg.p_v_pit)
        while isave <= length(save_at) && save_at[isave] <= n
            Tout[:, :, isave] .= eng.T
            Wout[:, :, isave] .= eng.W
            Cout[:, :, isave] .= eng.Cmat
            Tcold[isave] = minimum(eng.T)
            isave += 1
        end
    end
    return _slab2d_outputs(eng, R, cfg, Tout, Wout, Cout, Tcold, tsave)
end

"""
    coupled_solve_2d(θ2, φ, u_knots, u_tknots, tsave; R, Hz, nr, nz, cfg,
                     pit, hum, h, n_meat, t_wrap, k_grain, D_grain, F_top,
                     z_split)

The 2-D meat INSIDE the pit loop: fixed-step co-integration of the six
pit states (the monolith's pit algebra and energy balance, explicit —
`h` is far below the pit time constant C_th/UA) with the ADI meat step,
exchanging the area-integrated surface heat uptake and evaporation of
`n_meat` briskets; the meat's vapor feeds the pit humidity one step
lagged. v1 scope: no fuel additions, spritzes or lid lifts (the wrap
blend is carried); `Tpit_steps` returns the per-step forcing so a
prescribed-forcing replay reproduces the meat fields exactly.

Returns the [`meat_solve_2d`](@ref) outputs merged with the pit series
`(Tpit, m_fuel, pv_pit, q_meat, j_evap, Tpit_steps)`.
"""
function coupled_solve_2d(
    θ2::AbstractVector,
    φ::AbstractVector,
    u_knots::AbstractVector,
    u_tknots::AbstractVector,
    tsave::AbstractVector;
    R::Real=0.05,
    Hz::Real=0.09,
    nr::Int=20,
    nz::Int=20,
    cfg::MeatConfig=MeatConfig(),
    pit::PitConfig=PitConfig(),
    hum::Humidity=Humidity(),
    h::Real=30.0,
    n_meat::Real=1.0,
    t_wrap::Real=WRAP_NEVER,
    k_grain::Real=1.0,
    D_grain::Real=1.0,
    F_top::Real=NaN,
    z_split::Real=1.0,
    t_salt::Real=NaN,
    t_spritz::Real=SPRITZ_NEVER,
    dt_spritz::Real=3600.0,
    m_spritz::Real=NaN,
)
    length(φ) == PIT_NP ||
        throw(ArgumentError("φ must have length $PIT_NP; got $(length(φ))"))
    C_th, UA, r_max, ΔH_eff, u_leak, k_spread, a0 = φ
    Tv = promote_type(eltype(θ2), eltype(φ), eltype(u_knots),
                      typeof(float(t_wrap)), typeof(float(t_salt)),
                      typeof(float(t_spritz)), typeof(float(dt_spritz)),
                      Float64)
    eng = _slab2d_engine(θ2, R, Hz, nr, nz, cfg, h, t_wrap, k_grain, D_grain,
                         F_top, z_split, Tv, t_salt, t_spritz, dt_spritz,
                         m_spritz)
    u_t0, u_dt = float(u_tknots[1]), float(u_tknots[2] - u_tknots[1])
    t0 = float(tsave[1])
    nt_steps = round(Int, (float(tsave[end]) - t0) / h)
    save_at = [round(Int, (float(ts) - t0) / h) for ts in tsave]
    nsv = length(tsave)
    Tout = zeros(Tv, size(eng.T)..., nsv)
    Wout = zeros(Tv, size(eng.T)..., nsv)
    Cout = zeros(Tv, size(eng.T)..., nsv)
    Tcold = fill(Tv(cfg.T0), nsv)
    Tpit_s = fill(Tv(pit.Tpit0), nsv)
    mfuel_s = fill(Tv(pit.m0), nsv)
    pv_s = fill(Tv(cfg.p_v_pit), nsv)
    q_s = zeros(Tv, nsv)
    jv_s = zeros(Tv, nsv)
    Tpit_steps = zeros(Tv, nt_steps + 1)

    Tp = Tv(pit.Tpit0)
    m = Tv(pit.m0)
    a = Tv(a0)
    mp = Tv(pit.m_pan0)
    Tpan = Tv(pit.T_pan0)
    Tfb = Tv(pit.Tpit0)
    j_meat = zero(Tv)
    q_meat = zero(Tv)
    pv = Tv(cfg.p_v_pit)
    isave = 1
    while isave <= nsv && save_at[isave] <= 0
        Tout[:, :, isave] .= eng.T
        Wout[:, :, isave] .= eng.W
        isave += 1
    end
    for n in 1:nt_steps
        t_now = t0 + (n - 1) * h
        ud = lininterp(t_now, u_t0, u_dt, u_knots)
        ud = clamp(ud, zero(ud), one(ud))
        airflow, ṁ, mair, pv_nopan = _pit_algebra(ud, m, a, r_max, u_leak, pit, hum)
        q_pan, ṁ_pan, pv_base = _pan_algebra(Tp, mp, Tpan, mair, pv_nopan, pit, hum)
        pv = pv_base + (hum.p_atm / EPS_W) * n_meat * j_meat / mair
        Tpit_steps[n] = Tp
        q_meat, j_meat = eng.step!(t_now, Tp, pv)
        if pit.c_duct > 0
            q_duct = pit.c_duct * airflow * (Tfb - Tp)
            q_burn = q_duct
            Tfb += h * (ΔH_eff * ṁ - pit.UA_fb * (Tfb - pit.T_amb) - q_duct) /
                   pit.C_fb
        else
            q_burn = ΔH_eff * ṁ
        end
        Tp += h * (q_burn + pit.p_heater * ud - UA * (Tp - pit.T_amb) -
                   n_meat * q_meat - q_pan) / C_th
        m = max(m - h * ṁ, zero(Tv))
        a_c = clamp(a, zero(a), one(a))
        a += h * k_spread * a_c * (1 - a_c) * airflow
        if pit.m_pan0 > 0
            Tpan += h * (q_pan - cfg.λ_vap * ṁ_pan) /
                    (CP_WATER * max(mp, zero(Tv)) + pit.C_pan_dry)
            mp = max(mp - h * ṁ_pan, zero(Tv))
        end
        while isave <= nsv && save_at[isave] <= n
            Tout[:, :, isave] .= eng.T
            Wout[:, :, isave] .= eng.W
            Cout[:, :, isave] .= eng.Cmat
            Tcold[isave] = minimum(eng.T)
            Tpit_s[isave] = Tp
            mfuel_s[isave] = m
            pv_s[isave] = pv
            q_s[isave] = n_meat * q_meat
            jv_s[isave] = n_meat * j_meat
            isave += 1
        end
    end
    Tpit_steps[end] = Tp
    out = _slab2d_outputs(eng, R, cfg, Tout, Wout, Cout, Tcold, tsave)
    return merge(out, (; Tpit=Tpit_s, m_fuel=mfuel_s, pv_pit=pv_s,
                       q_meat=q_s, j_evap=jv_s, Tpit_steps))
end

function _slab2d_outputs(eng, R, cfg, Tout, Wout, Cout, Tcold, tsave)
    (; T, W, Cmat, B_rim, B_end, rV, zw, Δz, Tv, stmat, S) = eng
    nr, nz = size(T)
    vol = rV * zw'  # (nr, nz) volume weights (z ends are half cells)
    wret = sum(vol .* W) / (sum(vol) * cfg.W0)
    jm = div(nz, 2) + 1
    # exposed-area weights for the bark distribution: rim strips R·Δz
    # (half at the corners), end annuli rV
    w_rim = [R * Δz * zw[j] for j in 1:nz]
    aw_sum = sum(w_rim) + 2 * sum(rV)
    b_sum = sum(w_rim .* B_rim) + sum(rV .* B_end[:, 1]) + sum(rV .* B_end[:, 2])
    b2_sum = sum(w_rim .* B_rim .^ 2) + sum(rV .* B_end[:, 1] .^ 2) +
             sum(rV .* B_end[:, 2] .^ 2)
    b_mean = b_sum / aw_sum
    b_var = max(b2_sum / aw_sum - b_mean^2, zero(Tv))
    return (
        t=collect(float.(tsave)),
        Tcore=Tout[1, jm, :],
        T_field=Tout,
        W_field=Wout,
        w_retained=wret,
        C_field=Cout,
        C_core=Cout[1, jm, :],
        C_min=minimum(Cmat),
        Tcold=Tcold,
        bark_rim=B_rim,
        bark_ends=B_end,
        bark_mean=b_mean,
        bark_std=sqrt(b_var),
        salt_field=S,
        salt_mean=sum(vol .* S) / sum(vol),
        R_eff=eng.Δr * sum((i == 1 || i == nr ? 0.5 : 1.0) * stmat[i, div(nz, 2) + 1]
                           for i in 1:nr),
        Hz_eff=Δz * sum(zw[j] * stmat[1, j] for j in 1:nz),
    )
end

function _thomas_row!(X, aa, bb, cc, dd, n, j)
    for i in 2:n
        m = aa[i] / bb[i - 1]
        bb[i] -= m * cc[i - 1]
        dd[i] -= m * dd[i - 1]
    end
    X[n, j] = dd[n] / bb[n]
    for i in (n - 1):-1:1
        X[i, j] = (dd[i] - cc[i] * X[i + 1, j]) / bb[i]
    end
    return nothing
end

function _thomas_col!(X, aa, bb, cc, dd, n, i)
    for j in 2:n
        m = aa[j] / bb[j - 1]
        bb[j] -= m * cc[j - 1]
        dd[j] -= m * dd[j - 1]
    end
    X[i, n] = dd[n] / bb[n]
    for j in (n - 1):-1:1
        X[i, j] = (dd[j] - cc[j] * X[i, j + 1]) / bb[j]
    end
    return nothing
end


# --- the 2-D Tesseract's derivative endpoints --------------------------------

"""
    slab2d_stack(o)

The differentiable outputs of [`coupled_solve_2d`](@ref) as one vector:
`Tpit, Tcore, Tcold, m_fuel` (each nt) then the scalars `w_retained,
C_min, bark_mean, bark_std, salt_mean, R_eff, Hz_eff`.
"""
function slab2d_stack(o)
    return vcat(o.Tpit, o.Tcore, o.Tcold, o.m_fuel,
                [o.w_retained, o.C_min, o.bark_mean, o.bark_std, o.salt_mean,
                 o.R_eff, o.Hz_eff])
end

function _slab2d_stack_fn(
    θ2, φ, u_knots, u_tknots, tsave, t_wrap, t_salt, t_spritz, dt_spritz; kw...
)
    nθ, nφ, nu = length(θ2), length(φ), length(u_knots)
    v0 = vcat(
        float.(θ2), float.(φ), float.(u_knots), float(t_wrap), float(t_salt),
        float(t_spritz), float(dt_spritz),
    )
    g(v) = slab2d_stack(
        coupled_solve_2d(
            v[1:nθ], v[(nθ + 1):(nθ + nφ)], v[(nθ + nφ + 1):(nθ + nφ + nu)],
            u_tknots, tsave; t_wrap=v[end - 3], t_salt=v[end - 2],
            t_spritz=v[end - 1], dt_spritz=v[end], kw...,
        ),
    )
    return g, v0
end

"""
    slab2d_stack_jacobian(θ2, φ, u_knots, u_tknots, tsave, t_wrap, t_salt,
                          t_spritz, dt_spritz, sel; kw...)

Forward-mode Jacobian of [`slab2d_stack`](@ref) w.r.t. the entries `sel`
(1-based indices into `[θ2; φ; u_knots; t_wrap; t_salt; t_spritz;
dt_spritz]`); keyword arguments go to [`coupled_solve_2d`](@ref).
"""
function slab2d_stack_jacobian(
    θ2, φ, u_knots, u_tknots, tsave, t_wrap, t_salt, t_spritz, dt_spritz,
    sel::AbstractVector{<:Integer}; kw...,
)
    g, v0 = _slab2d_stack_fn(
        θ2, φ, u_knots, u_tknots, tsave, t_wrap, t_salt, t_spritz, dt_spritz;
        kw...,
    )
    sel = collect(Int, sel)
    function gs(vs)
        v = convert(Vector{eltype(vs)}, copy(v0))
        v[sel] .= vs
        return g(v)
    end
    return ForwardDiff.jacobian(gs, v0[sel])
end

"""
    slab2d_stack_jvp(θ2, φ, u_knots, u_tknots, tsave, t_wrap, t_salt,
                     t_spritz, dt_spritz, dv; kw...)

Directional derivative of [`slab2d_stack`](@ref) along the full-length
tangent `dv` — one Dual solve.
"""
function slab2d_stack_jvp(
    θ2, φ, u_knots, u_tknots, tsave, t_wrap, t_salt, t_spritz, dt_spritz,
    dv; kw...,
)
    g, v0 = _slab2d_stack_fn(
        θ2, φ, u_knots, u_tknots, tsave, t_wrap, t_salt, t_spritz, dt_spritz;
        kw...,
    )
    dv = collect(Float64, dv)
    length(dv) < length(v0) && (dv = vcat(dv, zeros(length(v0) - length(dv))))
    return ForwardDiff.derivative(t -> g(v0 .+ t .* dv), 0.0)
end
