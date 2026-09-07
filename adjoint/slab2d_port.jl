# M1: reverse-mode PERFORMANCE for the 2-D ADI — the explicit-argument
# stepper, the same pattern that made the 1-D full port 3.6x primal.
#
# slab2d_reverse.jl proved blackbox Enzyme through the closure engine is
# EXACT (1.3e-14 vs ForwardDiff) but 46x primal. This port isolates the
# transport core (T/W, Robin surfaces, linearized-implicit face
# evaporation — the reverse-mode structure; quality states are local
# ODEs that add nothing to it) in top-level functions over plain arrays,
# no closures, and measures the reverse/primal ratio on the same grid.
#
# Run:  julia --project=adjoint adjoint/slab2d_port.jl

using Pkg
Pkg.activate(@__DIR__)
using Enzyme
using ForwardDiff
using Printf
using BrisketDynamics
using BrisketDynamics: RHO_DRY, evap_flux, expulsion_rate, surface_heat_in,
                       conductivity_mix, heat_capacity_dry, specific_volume

const CFG = BrisketDynamics.with_config(MeatConfig(); n_shrink=0.0)

function thomas_row!(X, aa, bb, cc, dd, n, j)
    @inbounds for i in 2:n
        m = aa[i] / bb[i - 1]
        bb[i] -= m * cc[i - 1]
        dd[i] -= m * dd[i - 1]
    end
    @inbounds X[n, j] = dd[n] / bb[n]
    @inbounds for i in (n - 1):-1:1
        X[i, j] = (dd[i] - cc[i] * X[i + 1, j]) / bb[i]
    end
    return nothing
end

function thomas_col!(X, aa, bb, cc, dd, n, i)
    @inbounds for j in 2:n
        m = aa[j] / bb[j - 1]
        bb[j] -= m * cc[j - 1]
        dd[j] -= m * dd[j - 1]
    end
    @inbounds X[i, n] = dd[n] / bb[n]
    @inbounds for j in (n - 1):-1:1
        X[i, j] = (dd[j] - cc[j] * X[i, j + 1]) / bb[j]
    end
    return nothing
end

function step2d!(T, W, kmat, rcmat, aa, bb, cc, dd, jr, jz2, rF, rV,
                 D_eff, k_m, h_c, k_mult, k_e, a_myosin,
                 Tpit, h, Δr, Δz, nr, nz, cfg)
    Tv = eltype(T)
    @inbounds for j in 1:nz, i in 1:nr
        v = specific_volume(W[i, j], cfg.F0, cfg)
        kmat[i, j] = k_mult * conductivity_mix(W[i, j], cfg.F0, T[i, j], v, cfg)
        rcmat[i, j] = heat_capacity_dry(W[i, j], cfg.F0, T[i, j], cfg) / v
    end
    @inbounds for j in 1:nz, i in 1:nr
        W[i, j] -= h * expulsion_rate(W[i, j], T[i, j], k_e, a_myosin, cfg)
    end
    δW = Tv(1e-3)
    @inbounds for j in 1:nz
        C = RHO_DRY * Δr / 2
        j0 = evap_flux(T[nr, j], W[nr, j], k_m, cfg)
        s = (evap_flux(T[nr, j], W[nr, j] + δW, k_m, cfg) - j0) / δW
        jr[j] = j0 / (1 + h * max(s, zero(Tv)) / C)
        W[nr, j] -= h * jr[j] / C
    end
    @inbounds for i in 1:nr, kz in 1:2
        jz = kz == 1 ? 1 : nz
        C = RHO_DRY * Δz / 2
        j0 = evap_flux(T[i, jz], W[i, jz], k_m, cfg)
        s = (evap_flux(T[i, jz], W[i, jz] + δW, k_m, cfg) - j0) / δW
        jz2[i, kz] = j0 / (1 + h * max(s, zero(Tv)) / C)
        W[i, jz] -= h * jz2[i, kz] / C
    end
    @inbounds for j in 1:nz
        for i in 1:nr
            lo = i > 1 ? (kmat[i - 1, j] + kmat[i, j]) / 2 * rF[i - 1] /
                         (Δr * rV[i] * rcmat[i, j]) : zero(Tv)
            up = i < nr ? (kmat[i, j] + kmat[i + 1, j]) / 2 * rF[i] /
                          (Δr * rV[i] * rcmat[i, j]) : zero(Tv)
            aa[i] = -h * lo
            cc[i] = -h * up
            bb[i] = 1 + h * (lo + up)
            dd[i] = T[i, j]
        end
        βr = h_c / (rcmat[nr, j] * Δr / 2)
        qx = (surface_heat_in(Tpit, T[nr, j], h_c, cfg) -
              h_c * (Tpit - T[nr, j]) - cfg.λ_vap * jr[j]) /
             (rcmat[nr, j] * Δr / 2)
        bb[nr] += h * βr
        dd[nr] += h * (βr * Tpit + qx)
        thomas_row!(T, aa, bb, cc, dd, nr, j)
        for i in 1:nr
            lo = i > 1 ? D_eff * rF[i - 1] / (Δr * rV[i]) : zero(Tv)
            up = i < nr ? D_eff * rF[i] / (Δr * rV[i]) : zero(Tv)
            aa[i] = -h * lo
            cc[i] = -h * up
            bb[i] = 1 + h * (lo + up)
            dd[i] = W[i, j]
        end
        thomas_row!(W, aa, bb, cc, dd, nr, j)
    end
    @inbounds for i in 1:nr
        for j in 1:nz
            cell = (j == 1 || j == nz) ? Δz / 2 : Δz
            lo = j > 1 ? (kmat[i, j - 1] + kmat[i, j]) / 2 /
                         (Δz * cell * rcmat[i, j]) : zero(Tv)
            up = j < nz ? (kmat[i, j] + kmat[i, j + 1]) / 2 /
                          (Δz * cell * rcmat[i, j]) : zero(Tv)
            aa[j] = -h * lo
            cc[j] = -h * up
            bb[j] = 1 + h * (lo + up)
            dd[j] = T[i, j]
        end
        for kz in 1:2
            jz = kz == 1 ? 1 : nz
            βz = h_c / (rcmat[i, jz] * Δz / 2)
            qx = (surface_heat_in(Tpit, T[i, jz], h_c, cfg) -
                  h_c * (Tpit - T[i, jz]) - cfg.λ_vap * jz2[i, kz]) /
                 (rcmat[i, jz] * Δz / 2)
            bb[jz] += h * βz
            dd[jz] += h * (βz * Tpit + qx)
        end
        thomas_col!(T, aa, bb, cc, dd, nz, i)
        for j in 1:nz
            cell = (j == 1 || j == nz) ? Δz / 2 : Δz
            lo = j > 1 ? D_eff / (Δz * cell) : zero(Tv)
            up = j < nz ? D_eff / (Δz * cell) : zero(Tv)
            aa[j] = -h * lo
            cc[j] = -h * up
            bb[j] = 1 + h * (lo + up)
            dd[j] = W[i, j]
        end
        thomas_col!(W, aa, bb, cc, dd, nz, i)
    end
    return nothing
end

function run2d(θ2::AbstractVector, Tpit, nsteps::Int, nr::Int, nz::Int,
               R, Hz, h, cfg)
    D_eff, k_m, h_c, k_mult, k_e, a_myosin = θ2
    Tv = eltype(θ2)
    Δr = R / (nr - 1)
    Δz = Hz / (nz - 1)
    T = fill(Tv(cfg.T0), nr, nz)
    W = fill(Tv(cfg.W0), nr, nz)
    kmat = zeros(Tv, nr, nz)
    rcmat = zeros(Tv, nr, nz)
    nmax = max(nr, nz)
    aa = zeros(Tv, nmax); bb = zeros(Tv, nmax)
    cc = zeros(Tv, nmax); dd = zeros(Tv, nmax)
    rF = [(i - 0.5) * Δr for i in 1:(nr - 1)]
    rV = [i == 1 ? Δr^2 / 8 :
          i == nr ? ((nr - 1)^2 - (nr - 1.5)^2) * Δr^2 / 2 :
          ((i - 0.5)^2 - (i - 1.5)^2) * Δr^2 / 2 for i in 1:nr]
    jr = zeros(Tv, nz)
    jz2 = zeros(Tv, nr, 2)
    for _ in 1:nsteps
        step2d!(T, W, kmat, rcmat, aa, bb, cc, dd, jr, jz2, rF, rV,
                D_eff, k_m, h_c, k_mult, k_e, a_myosin,
                Tpit, h, Δr, Δz, nr, nz, cfg)
    end
    num = zero(Tv)
    den = zero(Tv)
    @inbounds for j in 1:nz, i in 1:nr
        zwj = (j == 1 || j == nz) ? 0.5 : 1.0
        num += rV[i] * zwj * W[i, j]
        den += rV[i] * zwj
    end
    return num / (den * cfg.W0)
end

θ2 = [3.8e-9, 0.0018, 10.0, 1.0, 5.6e-4, 0.20]
const NSTEP, NR, NZ = 720, 16, 14  # 6 h at h = 30 s, the probe grid
loss(v) = run2d(v, 383.15, NSTEP, NR, NZ, 0.05, 0.06, 30.0, CFG)

L0 = loss(θ2)
@printf("w_retained (transport core) = %.6f\n", L0)
t0 = time(); loss(θ2); tp = time() - t0

g_enz = zero(θ2)
Enzyme.autodiff(Enzyme.Reverse, loss, Enzyme.Active, Enzyme.Duplicated(θ2, g_enz))
t0 = time()
let g = zero(θ2)
    Enzyme.autodiff(Enzyme.Reverse, loss, Enzyme.Active, Enzyme.Duplicated(θ2, g))
end
tr = time() - t0

g_fwd = ForwardDiff.gradient(loss, θ2)
t0 = time(); ForwardDiff.gradient(loss, θ2); tf = time() - t0

rel = maximum(abs.(g_enz .- g_fwd) ./ (abs.(g_fwd) .+ 1e-30))
@printf("max rel |Enzyme - ForwardDiff| = %.3e\n", rel)
@printf("timings: primal %.3fs  reverse %.3fs (%.1fx primal)  forward-6 %.3fs\n",
        tp, tr, tr / tp, tf)
