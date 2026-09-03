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
times `tsave` must start at the cook start. Returns
`(t, Tcore, T_field, W_field, w_retained)` — Tcore at the axis
mid-height, fields `nr × nz × nt`, `w_retained` the volume-weighted
retained-water fraction at the end.
"""
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
)
    D_eff, k_m, h_c, k_mult, k_e, a_myosin = θ2
    Tv = promote_type(eltype(θ2), Float64)
    Δr = R / (nr - 1)
    Δz = Hz / (nz - 1)
    tk0, tkdt = float(tknots[1]), float(tknots[2] - tknots[1])
    t0 = float(tsave[1])

    T = fill(Tv(cfg.T0), nr, nz)
    W = fill(Tv(cfg.W0), nr, nz)
    kmat = zeros(Tv, nr, nz)
    ρcmat = zeros(Tv, nr, nz)
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

    nt_steps = round(Int, (float(tsave[end]) - t0) / h)
    save_at = [round(Int, (float(ts) - t0) / h) for ts in tsave]
    Tout = zeros(Tv, nr, nz, length(tsave))
    Wout = zeros(Tv, nr, nz, length(tsave))
    isave = 1
    while isave <= length(save_at) && save_at[isave] <= 0
        Tout[:, :, isave] .= T
        Wout[:, :, isave] .= W
        isave += 1
    end

    for n in 1:nt_steps
        Tpit = lininterp(t0 + (n - 1) * h, tk0, tkdt, Tpit_knots)
        for j in 1:nz, i in 1:nr
            v = specific_volume(W[i, j], cfg.F0, cfg)
            kmat[i, j] = k_mult * conductivity_mix(W[i, j], cfg.F0, T[i, j], v, cfg)
            ρcmat[i, j] = heat_capacity_dry(W[i, j], cfg.F0, T[i, j], cfg) / v
        end

        # explicit: purge everywhere, evaporative WATER loss on the
        # exposed faces (mild rates); the surface HEAT flux is handled
        # inside the sweeps as a Robin condition — an explicit kick of
        # h·q/C into a millimetre half-cell biases the surface warm and
        # over-dries it (measured: +9 K and 4x the drying rate on the
        # end faces before this was made implicit)
        for j in 1:nz, i in 1:nr
            W[i, j] -= h * expulsion_rate(W[i, j], T[i, j], k_e, a_myosin, cfg)
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
            j0 = evap_flux(T[nr, j], W[nr, j], k_m, cfg)
            s = (evap_flux(T[nr, j], W[nr, j] + δW, k_m, cfg) - j0) / δW
            jr_face[j] = j0 / (1 + h * max(s, zero(Tv)) / C)
            W[nr, j] -= h * jr_face[j] / C
        end
        for i in 1:nr, (kz, jz) in enumerate((1, nz))
            C = RHO_DRY * Δz / 2
            j0 = evap_flux(T[i, jz], W[i, jz], k_m, cfg)
            s = (evap_flux(T[i, jz], W[i, jz] + δW, k_m, cfg) - j0) / δW
            jz_face[i, kz] = j0 / (1 + h * max(s, zero(Tv)) / C)
            W[i, jz] -= h * jz_face[i, kz] / C
        end

        # implicit radial sweeps (each z-row): T with k/ρc, W with D_eff
        for j in 1:nz
            for i in 1:nr
                lo = i > 1 ?
                     (kmat[i - 1, j] + kmat[i, j]) / 2 * rF[i - 1] /
                     (Δr * rV[i] * ρcmat[i, j]) : zero(Tv)
                up = i < nr ?
                     (kmat[i, j] + kmat[i + 1, j]) / 2 * rF[i] /
                     (Δr * rV[i] * ρcmat[i, j]) : zero(Tv)
                aa[i] = -h * lo
                cc[i] = -h * up
                bb[i] = 1 + h * (lo + up)
                dd[i] = T[i, j]
            end
            βr = h_c / (ρcmat[nr, j] * Δr / 2)
            qx = (surface_heat_in(Tpit, T[nr, j], h_c, cfg) -
                  h_c * (Tpit - T[nr, j]) -
                  cfg.λ_vap * jr_face[j]) /
                 (ρcmat[nr, j] * Δr / 2)
            bb[nr] += h * βr
            dd[nr] += h * (βr * Tpit + qx)
            _thomas_row!(T, aa, bb, cc, dd, nr, j)
            for i in 1:nr
                lo = i > 1 ? D_eff * rF[i - 1] / (Δr * rV[i]) : zero(Tv)
                up = i < nr ? D_eff * rF[i] / (Δr * rV[i]) : zero(Tv)
                aa[i] = -h * lo
                cc[i] = -h * up
                bb[i] = 1 + h * (lo + up)
                dd[i] = W[i, j]
            end
            _thomas_row!(W, aa, bb, cc, dd, nr, j)
        end
        # implicit axial sweeps (each r-column)
        for i in 1:nr
            for j in 1:nz
                # the end rows are HALF cells (capacity Δz/2): their
                # diffusion coefficient doubles, matching the flux terms
                cell = (j == 1 || j == nz) ? Δz / 2 : Δz
                lo = j > 1 ? (kmat[i, j - 1] + kmat[i, j]) / 2 /
                             (Δz * cell * ρcmat[i, j]) : zero(Tv)
                up = j < nz ? (kmat[i, j] + kmat[i, j + 1]) / 2 /
                              (Δz * cell * ρcmat[i, j]) : zero(Tv)
                aa[j] = -h * lo
                cc[j] = -h * up
                bb[j] = 1 + h * (lo + up)
                dd[j] = T[i, j]
            end
            for (kz, jz) in enumerate((1, nz))
                βz = h_c / (ρcmat[i, jz] * Δz / 2)
                qx = (surface_heat_in(Tpit, T[i, jz], h_c, cfg) -
                      h_c * (Tpit - T[i, jz]) -
                      cfg.λ_vap * jz_face[i, kz]) /
                     (ρcmat[i, jz] * Δz / 2)
                bb[jz] += h * βz
                dd[jz] += h * (βz * Tpit + qx)
            end
            _thomas_col!(T, aa, bb, cc, dd, nz, i)
            for j in 1:nz
                cell = (j == 1 || j == nz) ? Δz / 2 : Δz
                lo = j > 1 ? D_eff / (Δz * cell) : zero(Tv)
                up = j < nz ? D_eff / (Δz * cell) : zero(Tv)
                aa[j] = -h * lo
                cc[j] = -h * up
                bb[j] = 1 + h * (lo + up)
                dd[j] = W[i, j]
            end
            _thomas_col!(W, aa, bb, cc, dd, nz, i)
        end

        while isave <= length(save_at) && save_at[isave] <= n
            Tout[:, :, isave] .= T
            Wout[:, :, isave] .= W
            isave += 1
        end
    end
    vol = rV * ones(Tv, nz)'  # (nr, nz) volume weights (Δz uniform)
    wret = sum(vol .* W) / (sum(vol) * cfg.W0)
    jm = div(nz, 2) + 1
    return (
        t=collect(float.(tsave)),
        Tcore=Tout[1, jm, :],
        T_field=Tout,
        W_field=Wout,
        w_retained=wret,
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
