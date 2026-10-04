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
# Layers on the core, each exactly inert at its default: shrink metrics
# (the 1-D material-coordinate stretch per direction), the wrap blend
# and the PULL blend on every exposed face (rest environment after
# t_pull, as in 1-D — the shadow of a plan ends at SERVE, not at the
# horizon), dry-brine salt from every rubbed face, spritz on the faces,
# grain anisotropy, a point-over-flat stack, Maillard bark on the faces,
# smoke deposition and the smoke-ring front on every exposed face cell
# (the 1-D surface states per cell, driven by the burn rate). The
# pit-coupled loop carries the monolith's fuel additions (one chimney or a
# split train) and lid lifts (the addition and every spritz pulse). The
# 1-D physics the layers do not switch: fat renders per node (its own
# field, feeding the properties and the shrink), a bone at the core
# (cfg.bone_frac, on the depth below the nearest exposed face), and the
# draft scaling of h_c and k_m (θ2[7] = m_conv). Nothing here is calibrated beyond what the 1-D constants carry; the bark and
# smoke DISTRIBUTIONS wait on thermal-camera surface fields (roadmap C2).

"""
    meat_solve_2d(θ2, Tpit_knots, tknots, tsave; R, Hz, nr, nz, cfg, h)

Axisymmetric finite-cylinder cook: radius `R`, full height `Hz`,
`nr × nz` nodes (i = 1 on the axis; j = 1 and j = nz are the exposed end
faces). `θ2 = [D_eff, k_m, h_c, k_mult, k_e, a_myosin, m_conv]` (the 1-D θ
without L; `h_c`, `k_m` scale with the draft as `(g/g_ref)^m_conv`). Fixed step `h`
[s]; the pit forcing interpolates from `Tpit_knots` on `tknots`; save
times `tsave` must start at the cook start.

Layer options (all exactly inert at their defaults): `t_wrap` applies
the 1-D wrap blend to every exposed face; `t_pull` swaps every face to
the rest environment (the `cfg` rest constants: `h_rest`, `T_rest`,
`ε_rest`, `η_rest`, `p_v_rest`) through the 1-D pull blend, so the
solve runs to SERVE with carryover and the hold's collagen conversion;
`k_grain` / `D_grain` scale
in-plane (along-fiber) conduction and moisture diffusion — the z
(through-thickness) direction keeps the 1-D-calibrated values;
`F_top` + `z_split` stack a second muscle of fat fraction `F_top` above
`z_split`·Hz (the point over the flat, a smooth step); `smoke_knots`
is the burn-rate forcing [kg/s] on `tknots` (the 1-D `meat_solve`
input; empty = no fire, so no smoke deposition and no ring);
`airflow_knots` the normalized draft on `tknots` (empty = `cfg.g_ref`, no
scaling). Fat renders per node from the initial point-over-flat profile,
and `cfg.bone_frac` places a bone at the core (both as in 1-D).

Returns `(t, Tcore, T_field, W_field, w_retained, C_field, C_core,
C_min, Tcold, bark_rim, bark_ends, bark_mean, bark_std)` — Tcore at the
axis mid-height, fields `nr × nz × nt`, `w_retained` the volume-weighted
retained-water fraction, collagen conversion per cell (`C_min` = the
last-done spot at serve), `Tcold` the true cold-spot trace (grid
minimum — the safety margin no symmetric 1-D core can show), and the
Maillard bark DISTRIBUTION over the exposed faces (area-weighted mean
and spread — burnt-ends live in the tail) — plus the smoke on the faces:
`smoke_rim, smoke_ends, smoke_mean, smoke_std` (deposition), `ring_rim,
ring_ends, ring_mean, ring_min` (ring depth [m], material coordinate),
`F_field`, `F_mean`, `fat_rendered`, `juiciness`, `drip_f` (the 1-D fat
scores), `l_star_mean` (the 1-D L* map at the face means), `bitterness` (the
area-mean dirty-smoke integral; zero without a cleanliness signal) and
the per-save face series `smoke_rim_t, smoke_ends_t, ring_rim_t,
ring_ends_t`.
"""
# The transport engine shared by the prescribed-forcing solve and the
# pit-coupled solve: builds the grid state and returns a `step!` closure
# advancing one fixed step under (t, Tpit, p_v), returning the
# area-integrated surface heat uptake [W] and evaporation [kg/s] of ONE
# brisket (what a pit loses to it).
function _slab2d_engine(
    θ2, R, Hz, nr, nz, cfg, h, t_wrap, t_pull, k_grain, D_grain, F_top, z_split,
    Tv, t_salt, t_spritz, dt_spritz, m_spritz
)
    length(θ2) == 7 || throw(ArgumentError(
        "θ2 must be [D_eff, k_m, h_c, k_mult, k_e, a_myosin, m_conv]; got $(length(θ2)) entries"))
    D_eff, k_m, h_c, k_mult, k_e, a_myosin, m_conv = θ2
    Δr = R / (nr - 1)
    Δz = Hz / (nz - 1)

    # point-vs-flat as stacked muscles: the initial fat fraction F(z)
    # steps smoothly from cfg.F0 (flat) to F_top (point) at z_split·Hz (NaN
    # F_top or z_split >= 1 leaves the uniform cut exactly); the fat then
    # renders per node at the 1-D rate (its own state, feeding conductivity,
    # capacity and the shrink through the composition)
    Ftop = isnan(F_top) ? Tv(cfg.F0) : Tv(F_top)
    δ_split = Hz / 12
    Fz = [Tv(cfg.F0) + (Ftop - Tv(cfg.F0)) *
          logistic(((j - 1) * Δz - z_split * Hz) / δ_split) for j in 1:nz]
    Fmat = [Fz[j] for i in 1:nr, j in 1:nz]
    # bone: the 1-D blend weight logistic((bone_frac − ξ̂)/δ_bone_frac) on
    # the normalized depth ξ̂ = 1 − d/d_max, d the distance to the nearest
    # exposed face (rim or either end); the core region keeps the cut's
    # shape, and both 1-D limits recover the slab's ξ̂ exactly. Static,
    # exactly zero for the boneless default
    wbone = zeros(Float64, nr, nz)
    if cfg.bone_frac > 0
        dface = [min(R - (i - 1) * Δr, (j - 1) * Δz, Hz - (j - 1) * Δz) for i in 1:nr, j in 1:nz]
        dmax = maximum(dface)
        wbone .= logistic.((cfg.bone_frac .- (1 .- dface ./ dmax)) ./ cfg.δ_bone_frac)
    end

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
    # smoke on the faces: the 1-D surface states per exposed face cell —
    # deposition D (saturating, darkens L*) and the smoke-ring front depth
    # Rg (material coordinate, along the face's inward normal); `dirty`
    # integrates the area-weighted deposition of an unclean burn (the 1-D
    # bitterness integrand)
    D_rim = zeros(Tv, nz)
    D_end = zeros(Tv, nr, 2)
    Rg_rim = zeros(Tv, nz)
    Rg_end = zeros(Tv, nr, 2)
    dirty = zeros(Tv, 1)
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
    # exposed-area weights of the face cells (2π dropped, it cancels in
    # every face mean): rim strips R·Δz (half at the corners), end annuli rV
    w_rim = [R * Δz * zw[j] for j in 1:nz]
    aw_sum = sum(w_rim) + 2 * sum(rV)

    # `smoke` is the burn rate [kg/s] (the 1-D smoke forcing) and `clean`
    # the burn cleanliness ∈ (0, 1]; both default to no fire
    function step!(t_now, Tpit, pv, smoke=zero(Tv), clean=one(Tv), airflow=cfg.g_ref)
        # draft convection, the 1-D law: h_c and k_m are quoted at g_ref and
        # scale with the airflow as (g/g_ref)^m_conv (one exponent for heat
        # and mass, Chilton–Colburn)
        g_fac = (airflow / cfg.g_ref)^m_conv
        hcg = h_c * g_fac
        kmg = k_m * g_fac
        # the wrap is the same smooth time blend as the 1-D model: it
        # damps face evaporation by (1 - η_evap·s), convection by
        # (1 - η_h·s) and blends the emissivity toward ε_wrap
        sw = wrap_blend(t_now, t_wrap, cfg)
        km_w = (1 - cfg.η_evap * sw) * kmg
        hc_w = hcg * (1 - cfg.η_h * sw)
        # the pull is the 1-D pull blend on every face: after t_pull the
        # face sees the rest environment (the cfg rest constants) instead
        # of the pit — convection h_rest at T_rest, emissivity ε_rest, the
        # mostly blocked rest evaporation into room air with the
        # heat–mass-analogy coefficient kmg·h_rest/hcg — and the pit stops
        # seeing the meat. Every rest term enters as pl·(rest − pit), so
        # pl = 0 (never pulled) is bitwise the pit-only arithmetic.
        pl = pull_blend(t_now, t_pull, cfg)
        km_r = kmg * cfg.h_rest / hcg
        # spritz: water lands on every exposed face while unwrapped and in
        # the pit, a fraction f_abs soaks in, the rest flashes off with its
        # latent heat (the 1-D q_net terms, mirrored per face)
        j_sp = spritz_rate(t_now, t_spritz, dt_spritz, msp, cfg) * (1 - sw) * (1 - pl)
        # node properties from the local composition (water, rendering fat)
        # and temperature, blended to bone where the bone sits (the 1-D
        # kmix/ρcmix: k_mult multiplies the blend)
        vbar = zero(Tv)
        wsum = zero(Tv)
        for j in 1:nz, i in 1:nr
            v = specific_volume(W[i, j], Fmat[i, j], cfg)
            wb = wbone[i, j]
            kmat[i, j] = k_mult * ((1 - wb) * conductivity_mix(W[i, j], Fmat[i, j], T[i, j], v, cfg) +
                                   wb * cfg.k_bone)
            ρcmat[i, j] = (1 - wb) * heat_capacity_dry(W[i, j], Fmat[i, j], T[i, j], cfg) / v +
                          wb * cfg.ρc_bone
            stmat[i, j] = (v / v0s)^n_s
            vbar += rV[i] * zw[j] * v
            wsum += rV[i] * zw[j]
        end
        a_area = (vbar / (wsum * v0s))^n_a
        # pre-step surface heat uptake — what the pit loses to this
        # brisket this step (rim strips + both end annuli, raw areas times
        # the exposed-area factor, as the 1-D pit sees A·a_area·q): the
        # pit share only
        q_up = zero(Tv)
        for j in 1:nz
            q_up += 2π * R * Δz * zw[j] * a_area * (1 - pl) *
                    surface_heat_in_wrapped(Tpit, T[nr, j], hcg, sw, cfg)
        end
        for i in 1:nr, jz in (1, nz)
            q_up += 2π * rV[i] * a_area * (1 - pl) *
                    surface_heat_in_wrapped(Tpit, T[i, jz], hcg, sw, cfg)
        end

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
            Fmat[i, j] -= h * fat_rate(T[i, j], cfg) * Fmat[i, j]
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
        # smoke on every exposed face cell, the 1-D surface terms per cell:
        # deposition needs a tacky face (wet blend), the burn, no foil and
        # the pit; the ring front advances by diffusion r·dr = D·Θ·dt only
        # while the meat at the front is below myoglobin denaturation. No
        # burn (smoke = 0) is exactly inert
        sm = max(smoke, zero(smoke))
        fire = sm / (sm + cfg.mdot_ring)
        gate = (1 - sw) * (1 - pl)
        for j in 1:nz
            wbl = wet_blend(a_w_oswin(W[nr, j], cfg), cfg)
            dep = cfg.k_dep * wbl * sm * gate * (1 - D_rim[j])
            dirty[1] += h * w_rim[j] * dep * (1 - clean)
            D_rim[j] += h * dep
            k, f = _depth_kf(Rg_rim[j], Δr, nr)
            Tf = (1 - f) * T[nr - k, j] + f * T[nr - k - 1, j]
            Rg_rim[j] += h * cfg.D_ring * wbl * fire * gate *
                         logistic((cfg.T_ring - Tf) / cfg.δ_ring) /
                         (Rg_rim[j] + cfg.r_ring0)
        end
        for i in 1:nr, (kz, jz) in enumerate((1, nz))
            wbl = wet_blend(a_w_oswin(W[i, jz], cfg), cfg)
            dep = cfg.k_dep * wbl * sm * gate * (1 - D_end[i, kz])
            dirty[1] += h * rV[i] * dep * (1 - clean)
            D_end[i, kz] += h * dep
            k, f = _depth_kf(Rg_end[i, kz], Δz, nz)
            s_in = kz == 1 ? 1 : -1        # inward along z from this end
            Tf = (1 - f) * T[i, jz + s_in * k] + f * T[i, jz + s_in * (k + 1)]
            Rg_end[i, kz] += h * cfg.D_ring * wbl * fire * gate *
                             logistic((cfg.T_ring - Tf) / cfg.δ_ring) /
                             (Rg_end[i, kz] + cfg.r_ring0)
        end
        # face evaporation is linearized-implicit: an explicit update
        # overshoots the a_w equilibrium once the face dries out, the
        # lagged flux then flips sign every step, and the alternating
        # λ·j heat kicks rectify into net heating that diverges under
        # mesh refinement (measured: nr 24→48 pushed the cylinder core
        # 97→108 °C).  Backward-Euler on the W-linearized flux damps
        # the update, and the SAME applied flux feeds the Robin sink.
        # the face flux is the pull blend of the pit branch (into the pit
        # humidity) and the rest branch (into room air, mostly blocked);
        # the pit receives the damped PIT share only
        δW = Tv(1e-3)
        j_up = zero(Tv)
        for j in 1:nz
            C = RHO_DRY * Δr / 2
            fa = 1 / (1 + cfg.k_aw * max(S[nr, j], zero(Tv)))
            jp0 = evap_flux(T[nr, j], W[nr, j], km_w, pv, cfg, fa)
            jpδ = evap_flux(T[nr, j], W[nr, j] + δW, km_w, pv, cfg, fa)
            jr0 = (1 - cfg.η_rest) * evap_flux(T[nr, j], W[nr, j], km_r, cfg.p_v_rest, cfg, fa)
            jrδ = (1 - cfg.η_rest) *
                  evap_flux(T[nr, j], W[nr, j] + δW, km_r, cfg.p_v_rest, cfg, fa)
            j0 = jp0 + pl * (jr0 - jp0)
            s = (jpδ + pl * (jrδ - jpδ) - j0) / δW
            den = 1 + h * a_area * max(s, zero(Tv)) / C
            jr_face[j] = j0 / den
            W[nr, j] += h * a_area * (cfg.f_abs * j_sp - jr_face[j]) / C
            j_up += 2π * R * Δz * zw[j] * a_area * ((1 - pl) * jp0 / den)
        end
        for i in 1:nr, (kz, jz) in enumerate((1, nz))
            C = RHO_DRY * Δz / 2
            fa = 1 / (1 + cfg.k_aw * max(S[i, jz], zero(Tv)))
            jp0 = evap_flux(T[i, jz], W[i, jz], km_w, pv, cfg, fa)
            jpδ = evap_flux(T[i, jz], W[i, jz] + δW, km_w, pv, cfg, fa)
            jr0 = (1 - cfg.η_rest) * evap_flux(T[i, jz], W[i, jz], km_r, cfg.p_v_rest, cfg, fa)
            jrδ = (1 - cfg.η_rest) *
                  evap_flux(T[i, jz], W[i, jz] + δW, km_r, cfg.p_v_rest, cfg, fa)
            j0 = jp0 + pl * (jr0 - jp0)
            s = (jpδ + pl * (jrδ - jpδ) - j0) / δW
            den = 1 + h * a_area * max(s, zero(Tv)) / C
            jz_face[i, kz] = j0 / den
            W[i, jz] += h * a_area * (cfg.f_abs * j_sp - jz_face[i, kz]) / C
            j_up += 2π * rV[i] * a_area * ((1 - pl) * jp0 / den)
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
            # Robin: (1−pl)·hc_w·(Tpit − T) + pl·h_rest·(T_rest − T) implicit;
            # radiation (pit or rest), the latent and spritz sinks lagged
            cap = ρcmat[nr, j] * stmat[nr, j] * Δr / 2
            βr = hc_w / cap
            q_lag = surface_heat_in_wrapped(Tpit, T[nr, j], hcg, sw, cfg) -
                    hc_w * (Tpit - T[nr, j])
            q_lag_r = cfg.ε_rest * σ_SB * (cfg.T_rest^4 - T[nr, j]^4)
            qx = (q_lag + pl * (q_lag_r - q_lag) -
                  cfg.λ_vap * (jr_face[j] + (1 - cfg.f_abs) * j_sp) -
                  CP_WATER * (T[nr, j] - cfg.T_spritz) * j_sp) / cap
            bb[nr] += h * (βr + pl * (cfg.h_rest - hc_w) / cap)
            dd[nr] += h * (βr * Tpit + pl * (cfg.h_rest * cfg.T_rest - hc_w * Tpit) / cap + qx)
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
                cap = ρcmat[i, jz] * stmat[i, jz] * Δz / 2
                βz = hc_w / cap
                q_lag = surface_heat_in_wrapped(Tpit, T[i, jz], hcg, sw, cfg) -
                        hc_w * (Tpit - T[i, jz])
                q_lag_r = cfg.ε_rest * σ_SB * (cfg.T_rest^4 - T[i, jz]^4)
                qx = (q_lag + pl * (q_lag_r - q_lag) -
                      cfg.λ_vap * (jz_face[i, kz] + (1 - cfg.f_abs) * j_sp) -
                      CP_WATER * (T[i, jz] - cfg.T_spritz) * j_sp) / cap
                bb[jz] += h * (βz + pl * (cfg.h_rest - hc_w) / cap)
                dd[jz] += h * (βz * Tpit + pl * (cfg.h_rest * cfg.T_rest - hc_w * Tpit) / cap + qx)
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

    dims() = (Δr * sum((i == 1 || i == nr ? 0.5 : 1.0) * stmat[i, div(nz, 2) + 1]
                       for i in 1:nr),
              Δz * sum(zw[j] * stmat[1, j] for j in 1:nz))
    return (; step!, T, W, Cmat, Fmat, Fz, B_rim, B_end, D_rim, D_end, Rg_rim, Rg_end,
            dirty, w_rim, aw_sum, rV, zw, Δr, Δz, Tv, stmat, S, dims)
end

# node offset k and fraction f of a depth d below a face on a line of n
# nodes spaced Δ (the 1-D `temperature_at_depth` arithmetic, clamped to
# the line); the floor is taken on the primal so Duals pass through f
function _depth_kf(d, Δ, n)
    η = clamp(d / Δ, zero(d / Δ), (n - 1) * one(d / Δ))
    k = min(floor(Int, _primal(η)), n - 2)
    return k, η - k
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
    t_pull::Real=PULL_NEVER,
    k_grain::Real=1.0,
    D_grain::Real=1.0,
    F_top::Real=NaN,
    z_split::Real=1.0,
    t_salt::Real=NaN,
    t_spritz::Real=SPRITZ_NEVER,
    dt_spritz::Real=3600.0,
    m_spritz::Real=NaN,
    smoke_knots::AbstractVector=Float64[],
    airflow_knots::AbstractVector=Float64[],
)
    Tv = promote_type(eltype(θ2), eltype(Tpit_knots), eltype(smoke_knots), eltype(airflow_knots),
                      typeof(float(t_wrap)), typeof(float(t_pull)),
                      typeof(float(t_salt)), typeof(float(t_spritz)),
                      typeof(float(dt_spritz)), Float64)
    isempty(smoke_knots) || length(smoke_knots) == length(tknots) ||
        throw(ArgumentError("smoke_knots must be empty or match tknots"))
    isempty(airflow_knots) || length(airflow_knots) == length(tknots) ||
        throw(ArgumentError("airflow_knots must be empty or match tknots"))
    eng = _slab2d_engine(θ2, R, Hz, nr, nz, cfg, h, t_wrap, t_pull, k_grain,
                         D_grain, F_top, z_split, Tv, t_salt, t_spritz,
                         dt_spritz, m_spritz)
    tk0, tkdt = float(tknots[1]), float(tknots[2] - tknots[1])
    t0 = float(tsave[1])
    nt_steps = round(Int, (float(tsave[end]) - t0) / h)
    save_at = [round(Int, (float(ts) - t0) / h) for ts in tsave]
    Tout = zeros(Tv, size(eng.T)..., length(tsave))
    Wout = zeros(Tv, size(eng.T)..., length(tsave))
    Cout = zeros(Tv, size(eng.T)..., length(tsave))
    Sout = zeros(Tv, size(eng.T)..., length(tsave))
    Tcold = fill(Tv(cfg.T0), length(tsave))
    sv = _slab2d_series(eng, length(tsave))
    isave = 1
    while isave <= length(save_at) && save_at[isave] <= 0
        Tout[:, :, isave] .= eng.T
        Wout[:, :, isave] .= eng.W
        Sout[:, :, isave] .= eng.S
        _slab2d_save!(sv, eng, isave, R, Hz)
        isave += 1
    end
    for n in 1:nt_steps
        t_now = t0 + (n - 1) * h
        Tpit = lininterp(t_now, tk0, tkdt, Tpit_knots)
        smoke = isempty(smoke_knots) ? zero(Tv) :
                lininterp(t_now, tk0, tkdt, smoke_knots)
        airflow = isempty(airflow_knots) ? Tv(cfg.g_ref) :
                  lininterp(t_now, tk0, tkdt, airflow_knots)
        eng.step!(t_now, Tpit, cfg.p_v_pit, smoke, one(Tv), airflow)
        while isave <= length(save_at) && save_at[isave] <= n
            Tout[:, :, isave] .= eng.T
            Wout[:, :, isave] .= eng.W
            Cout[:, :, isave] .= eng.Cmat
            Sout[:, :, isave] .= eng.S
            Tcold[isave] = minimum(eng.T)
            _slab2d_save!(sv, eng, isave, R, Hz)
            isave += 1
        end
    end
    return merge(
        _slab2d_outputs(eng, R, cfg, Tout, Wout, Cout, Sout, Tcold, tsave), sv
    )
end

# per-save quality/dimension series (the animation feed): bark on the
# faces and the current (shrunk) dimensions at every save time
function _slab2d_series(eng, nt)
    nr, nz = size(eng.T)
    return (; bark_rim_t=zeros(eng.Tv, nz, nt),
            bark_ends_t=zeros(eng.Tv, nr, 2, nt),
            smoke_rim_t=zeros(eng.Tv, nz, nt),
            smoke_ends_t=zeros(eng.Tv, nr, 2, nt),
            ring_rim_t=zeros(eng.Tv, nz, nt),
            ring_ends_t=zeros(eng.Tv, nr, 2, nt),
            F_field=zeros(eng.Tv, nr, nz, nt),
            R_eff_t=zeros(eng.Tv, nt), Hz_eff_t=zeros(eng.Tv, nt))
end

function _slab2d_save!(sv, eng, isave, R, Hz)
    sv.bark_rim_t[:, isave] .= eng.B_rim
    sv.bark_ends_t[:, :, isave] .= eng.B_end
    sv.smoke_rim_t[:, isave] .= eng.D_rim
    sv.smoke_ends_t[:, :, isave] .= eng.D_end
    sv.ring_rim_t[:, isave] .= eng.Rg_rim
    sv.ring_ends_t[:, :, isave] .= eng.Rg_end
    sv.F_field[:, :, isave] .= eng.Fmat
    re, he = eng.dims()
    sv.R_eff_t[isave] = re
    sv.Hz_eff_t[isave] = he
    return nothing
end

"""
    coupled_solve_2d(θ2, φ, u_knots, u_tknots, tsave; R, Hz, nr, nz, cfg,
                     pit, hum, h, n_meat, t_wrap, t_pull, k_grain, D_grain,
                     F_top, z_split, t_salt, t_spritz, dt_spritz, m_spritz)

The 2-D meat INSIDE the pit loop: fixed-step co-integration of the six
pit states (the monolith's pit algebra and energy balance, explicit —
`h` is far below the pit time constant C_th/UA) with the ADI meat step,
exchanging the area-integrated surface heat uptake and evaporation of
`n_meat` briskets; the meat's vapor feeds the pit humidity one step
lagged. Wrap, pull, salt and spritz are carried; after `t_pull` the
meat's heat and vapor exchange go to zero through the pull blend and
the pit runs on alone while the meat rests to the horizon (serve).
Time runs from lighting: before `t_load` the pit preheats empty (the meat
is not stepped and holds its raw state), and the lid lifts while the meat
goes on; `LOAD_AT_START` (the default) has it in from t = 0.
The burn rate is the meat's smoke forcing (deposition and the ring on
every face) and the burn cleanliness scores its dirty share.

Pit events, the monolith's expressions term for term: `t_add` dumps one
chimney `pit.m_chimney` (a finite `dt_add` [s] makes it a split train;
NaN reads `pit.dt_add`), feeding the fuel and lighting the unlit share
of the bed; the lid lifts for the addition and for every spritz pulse
(while unwrapped and in the pit), one opening taking the wider of
overlapping gates, at `pit.UA_open`. `t_add = ADD_NEVER` and no spritz
are exactly inert. `Tpit_steps` and `airflow_steps` return the per-step
forcing so a prescribed-forcing replay reproduces the meat fields exactly.

Returns the [`meat_solve_2d`](@ref) outputs merged with the pit series
`(Tpit, m_fuel, ignited, mdot, smoke_clean, lid, pv_pit, q_meat, j_evap,
Tpit_steps, airflow_steps)`.
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
    t_pull::Real=PULL_NEVER,
    k_grain::Real=1.0,
    D_grain::Real=1.0,
    F_top::Real=NaN,
    z_split::Real=1.0,
    t_salt::Real=NaN,
    t_spritz::Real=SPRITZ_NEVER,
    dt_spritz::Real=3600.0,
    m_spritz::Real=NaN,
    t_add::Real=ADD_NEVER,
    dt_add::Real=NaN,
    t_load::Real=LOAD_AT_START,
)
    length(φ) == PIT_NP ||
        throw(ArgumentError("φ must have length $PIT_NP; got $(length(φ))"))
    C_th, UA, r_max, ΔH_eff, u_leak, k_spread, a0 = φ
    Tv = promote_type(eltype(θ2), eltype(φ), eltype(u_knots),
                      typeof(float(t_wrap)), typeof(float(t_pull)),
                      typeof(float(t_salt)), typeof(float(t_spritz)),
                      typeof(float(dt_spritz)), typeof(float(t_add)),
                      typeof(float(dt_add)), Float64)
    # the split interval: NaN reads the config (Inf = one chimney)
    dta = isnan(_primal(dt_add)) ? pit.dt_add : dt_add
    eng = _slab2d_engine(θ2, R, Hz, nr, nz, cfg, h, t_wrap, t_pull, k_grain,
                         D_grain, F_top, z_split, Tv, t_salt, t_spritz,
                         dt_spritz, m_spritz)
    u_t0, u_dt = float(u_tknots[1]), float(u_tknots[2] - u_tknots[1])
    t0 = float(tsave[1])
    nt_steps = round(Int, (float(tsave[end]) - t0) / h)
    save_at = [round(Int, (float(ts) - t0) / h) for ts in tsave]
    nsv = length(tsave)
    # the lid as ONE physical opening (the monolith's gate): the fuel
    # addition, and each spritz pulse while unwrapped and in the pit
    lid_at(t) = max(lid_open(t, t_add, pit.τ_open_add), lid_open(t, t_load, pit.τ_open_load),
                    lid_open(t, spritz_pulse_time(t, t_spritz, dt_spritz),
                             pit.τ_open_spritz) *
                    (1 - wrap_blend(t, t_wrap, cfg)) * (1 - pull_blend(t, t_pull, cfg)))
    Tout = zeros(Tv, size(eng.T)..., nsv)
    Wout = zeros(Tv, size(eng.T)..., nsv)
    Cout = zeros(Tv, size(eng.T)..., nsv)
    Sout = zeros(Tv, size(eng.T)..., nsv)
    Tcold = fill(Tv(cfg.T0), nsv)
    Tpit_s = fill(Tv(pit.Tpit0), nsv)
    mfuel_s = fill(Tv(pit.m0), nsv)
    pv_s = fill(Tv(cfg.p_v_pit), nsv)
    q_s = zeros(Tv, nsv)
    jv_s = zeros(Tv, nsv)
    a_s = fill(Tv(a0), nsv)
    mdot_s = zeros(Tv, nsv)
    clean_s = ones(Tv, nsv)
    lid_s = zeros(Tv, nsv)
    Tpit_steps = zeros(Tv, nt_steps + 1)
    airflow_steps = zeros(Tv, nt_steps + 1)
    sv = _slab2d_series(eng, nsv)

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
        Sout[:, :, isave] .= eng.S
        _slab2d_save!(sv, eng, isave, R, Hz)
        ud0 = clamp(lininterp(t0, u_t0, u_dt, u_knots), 0, 1)
        af0, ṁ0, _, _ = _pit_algebra(ud0, m, a, r_max, u_leak, pit, hum)
        mdot_s[isave] = ṁ0
        clean_s[isave] = burn_cleanliness(af0, a, m, pit)
        lid_s[isave] = lid_at(t0)
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
        airflow_steps[n] = airflow
        # the burn is the meat's smoke; its cleanliness scores the dirty share
        clean = burn_cleanliness(airflow, a, m, pit)
        # the meat is stepped from its load on: before it the pit preheats empty and the meat holds its raw state
        # (a step boundary, so the 2-D path takes t_load as data, not as a differentiable input)
        if t_now + h / 2 >= _primal(t_load)
            q_meat, j_meat = eng.step!(t_now, Tp, pv, ṁ, clean, airflow)
        else
            q_meat, j_meat = zero(Tv), zero(Tv)
        end
        ṁ_add = fuel_add_rate(t_now, t_add, dta, pit)
        lid = lid_at(t_now)
        q_lid = pit.UA_open * (Tp - pit.T_amb) * lid
        if pit.c_duct > 0
            q_duct = pit.c_duct * airflow * (Tfb - Tp)
            q_burn = q_duct
            Tfb += h * (ΔH_eff * ṁ - pit.UA_fb * (Tfb - pit.T_amb) - q_duct) /
                   pit.C_fb
        else
            q_burn = ΔH_eff * ṁ
        end
        Tp += h * (q_burn + pit.p_heater * ud - UA * (Tp - pit.T_amb) -
                   n_meat * q_meat - q_pan - q_lid) / C_th
        # a chimney feeds the fuel and lights the unlit share of the bed
        # (the monolith's ignition source, read at the pre-step fuel mass)
        a_c = clamp(a, zero(a), one(a))
        a += h * k_spread * a_c * (1 - a_c) * airflow +
             h * (1 - a_c) * ṁ_add / max(m, pit.m_taper)
        m = max(m + h * (ṁ_add - ṁ), zero(Tv))
        if pit.m_pan0 > 0
            Tpan += h * (q_pan - pit.λ_pan * ṁ_pan) /
                    (CP_WATER * max(mp, zero(Tv)) + pit.C_pan_dry)
            mp = max(mp - h * ṁ_pan, zero(Tv))
        end
        while isave <= nsv && save_at[isave] <= n
            Tout[:, :, isave] .= eng.T
            Wout[:, :, isave] .= eng.W
            Cout[:, :, isave] .= eng.Cmat
            Sout[:, :, isave] .= eng.S
            Tcold[isave] = minimum(eng.T)
            _slab2d_save!(sv, eng, isave, R, Hz)
            Tpit_s[isave] = Tp
            mfuel_s[isave] = m
            pv_s[isave] = pv
            q_s[isave] = n_meat * q_meat
            jv_s[isave] = n_meat * j_meat
            a_s[isave] = a
            mdot_s[isave] = ṁ
            clean_s[isave] = clean
            lid_s[isave] = lid
            isave += 1
        end
    end
    Tpit_steps[end] = Tp
    airflow_steps[end] = airflow_steps[max(nt_steps, 1)]
    out = _slab2d_outputs(eng, R, cfg, Tout, Wout, Cout, Sout, Tcold, tsave)
    return merge(out, sv, (; Tpit=Tpit_s, m_fuel=mfuel_s, ignited=a_s,
                           mdot=mdot_s, smoke_clean=clean_s, lid=lid_s,
                           pv_pit=pv_s, q_meat=q_s, j_evap=jv_s, Tpit_steps, airflow_steps))
end

function _slab2d_outputs(eng, R, cfg, Tout, Wout, Cout, Sout, Tcold, tsave)
    (; T, W, Cmat, Fmat, Fz, B_rim, B_end, D_rim, D_end, Rg_rim, Rg_end, rV, zw, Δz, Tv,
       stmat, S, aw_sum) = eng
    nr, nz = size(T)
    vol = rV * zw'  # (nr, nz) volume weights (z ends are half cells)
    wret = sum(vol .* W) / (sum(vol) * cfg.W0)
    # fat: the volume means of the initial (per-row) and the rendered field;
    # juiciness is the 1-D score, retained water plus the retained rendered fat
    F_init = sum(vol .* (ones(nr) * Fz')) / sum(vol)
    F_mean = sum(vol .* Fmat) / sum(vol)
    R_ret = (1 - cfg.η_drain) * (F_init - F_mean)
    jm = div(nz, 2) + 1
    # the face distributions over the exposed area (bark, smoke, ring)
    b_mean, b_std = _face_stats(eng, B_rim, B_end)
    d_mean, d_std = _face_stats(eng, D_rim, D_end)
    r_mean, _ = _face_stats(eng, Rg_rim, Rg_end)
    return (
        t=collect(float.(tsave)),
        Tcore=Tout[1, jm, :],
        T_field=Tout,
        W_field=Wout,
        w_retained=wret,
        F_mean=F_mean,
        fat_rendered=(F_init - F_mean) / F_init,
        juiciness=wret + cfg.β_fat * R_ret / cfg.W0,
        drip_f=cfg.η_drain * (F_init - F_mean),
        C_field=Cout,
        C_core=Cout[1, jm, :],
        C_min=minimum(Cmat),
        Tcold=Tcold,
        bark_rim=B_rim,
        bark_ends=B_end,
        bark_mean=b_mean,
        bark_std=b_std,
        smoke_rim=D_rim,
        smoke_ends=D_end,
        smoke_mean=d_mean,
        smoke_std=d_std,
        ring_rim=Rg_rim,
        ring_ends=Rg_end,
        ring_mean=r_mean,
        ring_min=min(minimum(Rg_rim), minimum(Rg_end)),
        l_star_mean=l_star_bark(b_mean, d_mean, cfg),
        bitterness=eng.dirty[1] / aw_sum,
        salt_field=Sout,
        salt_mean=sum(vol .* S) / sum(vol),
        R_eff=eng.Δr * sum((i == 1 || i == nr ? 0.5 : 1.0) * stmat[i, div(nz, 2) + 1]
                           for i in 1:nr),
        Hz_eff=Δz * sum(zw[j] * stmat[1, j] for j in 1:nz),
    )
end

# area-weighted mean and spread of a per-face-cell quantity over the
# exposed faces (rim strips, both end annuli); a zero spread returns an
# exact zero (sqrt has no derivative there — a uniform face must not put
# NaN partials into a Dual solve)
function _face_stats(eng, Xr, Xe)
    (; w_rim, rV, aw_sum) = eng
    s1 = sum(w_rim .* Xr) + sum(rV .* Xe[:, 1]) + sum(rV .* Xe[:, 2])
    s2 = sum(w_rim .* Xr .^ 2) + sum(rV .* Xe[:, 1] .^ 2) + sum(rV .* Xe[:, 2] .^ 2)
    μ = s1 / aw_sum
    v = s2 / aw_sum - μ^2
    return μ, _primal(v) > 0 ? sqrt(v) : zero(v)
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
C_min, bark_mean, bark_std, salt_mean, R_eff, Hz_eff, smoke_mean,
ring_mean, l_star_mean, bitterness, fat_rendered, juiciness` (later
additions appended last, so every earlier offset is unchanged).
"""
function slab2d_stack(o)
    return vcat(o.Tpit, o.Tcore, o.Tcold, o.m_fuel,
                [o.w_retained, o.C_min, o.bark_mean, o.bark_std, o.salt_mean,
                 o.R_eff, o.Hz_eff, o.smoke_mean, o.ring_mean, o.l_star_mean,
                 o.bitterness, o.fat_rendered, o.juiciness])
end

# v = [θ2; φ; u_knots; t_wrap; t_salt; t_spritz; dt_spritz; t_pull; t_add;
# dt_add] — the pit events appended last (NaN dt_add reads the config)
function _slab2d_stack_fn(
    θ2, φ, u_knots, u_tknots, tsave, t_wrap, t_salt, t_spritz, dt_spritz, t_pull;
    t_add::Real=ADD_NEVER, dt_add::Real=NaN, kw...
)
    nθ, nφ, nu = length(θ2), length(φ), length(u_knots)
    v0 = vcat(
        float.(θ2), float.(φ), float.(u_knots), float(t_wrap), float(t_salt),
        float(t_spritz), float(dt_spritz), float(t_pull), float(t_add),
        float(dt_add),
    )
    g(v) = slab2d_stack(
        coupled_solve_2d(
            v[1:nθ], v[(nθ + 1):(nθ + nφ)], v[(nθ + nφ + 1):(nθ + nφ + nu)],
            u_tknots, tsave; t_wrap=v[end - 6], t_salt=v[end - 5],
            t_spritz=v[end - 4], dt_spritz=v[end - 3], t_pull=v[end - 2],
            t_add=v[end - 1], dt_add=v[end], kw...,
        ),
    )
    return g, v0
end

"""
    slab2d_stack_jacobian(θ2, φ, u_knots, u_tknots, tsave, t_wrap, t_salt,
                          t_spritz, dt_spritz, t_pull, sel; kw...)

Forward-mode Jacobian of [`slab2d_stack`](@ref) w.r.t. the entries `sel`
(1-based indices into `[θ2; φ; u_knots; t_wrap; t_salt; t_spritz;
dt_spritz; t_pull; t_add; dt_add]`, the last two from the `t_add` /
`dt_add` keywords); other keyword arguments go to
[`coupled_solve_2d`](@ref).
"""
function slab2d_stack_jacobian(
    θ2, φ, u_knots, u_tknots, tsave, t_wrap, t_salt, t_spritz, dt_spritz, t_pull,
    sel::AbstractVector{<:Integer}; kw...,
)
    g, v0 = _slab2d_stack_fn(
        θ2, φ, u_knots, u_tknots, tsave, t_wrap, t_salt, t_spritz, dt_spritz,
        t_pull; kw...,
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
                     t_spritz, dt_spritz, t_pull, dv; kw...)

Directional derivative of [`slab2d_stack`](@ref) along the tangent `dv`
(the `slab2d_stack_jacobian` input order; a shorter tangent is
zero-extended) — one Dual solve.
"""
function slab2d_stack_jvp(
    θ2, φ, u_knots, u_tknots, tsave, t_wrap, t_salt, t_spritz, dt_spritz, t_pull,
    dv; kw...,
)
    g, v0 = _slab2d_stack_fn(
        θ2, φ, u_knots, u_tknots, tsave, t_wrap, t_salt, t_spritz, dt_spritz,
        t_pull; kw...,
    )
    dv = collect(Float64, dv)
    length(dv) < length(v0) && (dv = vcat(dv, zeros(length(v0) - length(dv))))
    return ForwardDiff.derivative(t -> g(v0 .+ t .* dv), 0.0)
end
