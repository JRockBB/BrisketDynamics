# M1: the discrete adjoint ported to the FULL meat physics.
#
# Fixed-step IMEX-BE over the production RHS: the explicit part is
# meat_rhs_core! itself (Enzyme reverses through it — probe_rhs.jl), and
# the implicit part subtracts and re-solves the three diffusion operators
# (T, W, salt) as tridiagonal systems assembled from the SAME
# finite-volume metric formulas (stretch st, face means, geometry
# factors, bone blend) at the lagged state:
#
#   (I − h·A_X(u_n))·X_{n+1} = X_n + h·(f_X(u_n) − A_X(u_n)·X_n)
#
# Local reactions, surface fluxes and the bark/smoke/ring scalars stay
# explicit (their rates are ~0.01/s at h = 30 s). Any small mismatch
# between A_X and the RHS's discrete operator lands in the explicit
# remainder — the scheme stays consistent, and the FBDF comparison below
# measures what is left.
#
# Outputs: (1) fixed-step accuracy vs the production FBDF solve across
# step sizes, (2) Enzyme-reverse gradient of a trajectory loss wrt θ
# validated against ForwardDiff on the SAME stepper, (3) timings.
#
# Run:  julia --project=adjoint adjoint/full_port.jl

using Pkg
Pkg.activate(@__DIR__)
using Enzyme
using ForwardDiff
using Printf
using BrisketDynamics
using BrisketDynamics: meat_rhs_core!, MeatConfig, _meat_u0, shrink_exponents,
    bone_weights, specific_volume, conductivity_mix, heat_capacity_dry,
    meat_solve

const CFG = MeatConfig()
const NX = CFG.nx
const NS = 5 * NX + 3
const HR = 3600.0
const T_END = 8.0 * HR
const TPIT = 383.15

"""Assemble the lagged tridiagonal diffusion operators for T, W and salt
using the exact stencil formulas of `meat_rhs_core!` (slab, m_g = 0;
bone blend via `wb`). `lo/di/up` are (3, NX) work arrays per field."""
function assemble!(loT, diT, upT, loW, diW, upW, θ, u, cfg, n_s, v0, wb)
    D_eff, _, _, L, k_mult = θ[1], θ[2], θ[3], θ[4], θ[5]
    Δξ = L / (NX - 1)
    W = @view u[(NX + 1):2NX]
    F = @view u[(3NX + 1):4NX]
    T = @view u[1:NX]
    st = similar(diT); kc = similar(diT); ρc = similar(diT)
    for i in 1:NX
        v = specific_volume(W[i], F[i], cfg)
        st[i] = (v / v0)^n_s
        kc[i] = k_mult * ((1 - wb[i]) * conductivity_mix(W[i], F[i], T[i], v, cfg) +
                          wb[i] * cfg.k_bone)
        ρc[i] = (1 - wb[i]) * heat_capacity_dry(W[i], F[i], T[i], cfg) / v +
                wb[i] * cfg.ρc_bone
    end
    # center rows (mirror ghost)
    f_c = (st[1] + st[2]) / 2
    rT = (kc[1] + kc[2]) / (f_c * st[1] * Δξ^2 * ρc[1])
    loT[1] = 0.0; diT[1] = -rT; upT[1] = rT
    rW = 2 * D_eff / (f_c^2 * Δξ^2)
    loW[1] = 0.0; diW[1] = -rW; upW[1] = rW
    # interior rows
    for i in 2:(NX - 1)
        f_m = (st[i - 1] + st[i]) / 2
        f_p = (st[i] + st[i + 1]) / 2
        cTm = (kc[i - 1] + kc[i]) / f_m
        cTp = (kc[i] + kc[i + 1]) / f_p
        den = 2 * st[i] * Δξ^2 * ρc[i]
        loT[i] = cTm / den; upT[i] = cTp / den; diT[i] = -(cTm + cTp) / den
        cWm = D_eff / f_m^2; cWp = D_eff / f_p^2
        loW[i] = cWm / Δξ^2; upW[i] = cWp / Δξ^2; diW[i] = -(cWm + cWp) / Δξ^2
    end
    # surface rows (half cell; the flux terms stay explicit)
    f_s = (st[NX - 1] + st[NX]) / 2
    cT = (kc[NX - 1] + kc[NX]) / 2 / (f_s * Δξ) / (ρc[NX] * st[NX] * Δξ / 2)
    loT[NX] = cT; diT[NX] = -cT; upT[NX] = 0.0
    cW = D_eff / (f_s^2 * Δξ) / (Δξ / 2)
    loW[NX] = cW; diW[NX] = -cW; upW[NX] = 0.0
    return nothing
end

"""Thomas solve of (I − h·A)x = r for a tridiagonal A given by lo/di/up."""
function thomas!(x, lo, di, up, r, h, a, b, d)
    for i in 1:NX
        a[i] = -h * lo[i]
        b[i] = 1 - h * di[i]
        d[i] = r[i]
    end
    for i in 2:NX
        m = a[i] / b[i - 1]
        b[i] -= m * (-h * up[i - 1])
        d[i] -= m * d[i - 1]
    end
    x[NX] = d[NX] / b[NX]
    for i in (NX - 1):-1:1
        x[i] = (d[i] - (-h * up[i]) * x[i + 1]) / b[i]
    end
    return nothing
end

"""Apply A_X·x into out (tridiagonal matvec)."""
function tmul!(out, lo, di, up, x)
    out[1] = di[1] * x[1] + up[1] * x[2]
    for i in 2:(NX - 1)
        out[i] = lo[i] * x[i - 1] + di[i] * x[i] + up[i] * x[i + 1]
    end
    out[NX] = lo[NX] * x[NX - 1] + di[NX] * x[NX]
    return nothing
end

function solve_fixed(θ::Vector{Tv}, h::Float64, nsave::Int) where {Tv}
    n_s, n_a, v0 = shrink_exponents(CFG)
    wb = bone_weights(CFG)
    u = Tv.(_meat_u0(CFG, Float64[], Float64[], Float64[]))
    du = zeros(Tv, NS)
    loT = zeros(Tv, NX); diT = zeros(Tv, NX); upT = zeros(Tv, NX)
    loW = zeros(Tv, NX); diW = zeros(Tv, NX); upW = zeros(Tv, NX)
    ax = zeros(Tv, NX); tmp = zeros(Tv, NX)
    wa = zeros(Tv, NX); wb2 = zeros(Tv, NX); wd = zeros(Tv, NX)
    nt = round(Int, T_END / h)
    every = max(1, nt ÷ nsave)
    core = Tv[]
    θt = (θ[1], θ[2], θ[3], θ[4], θ[5], θ[6], θ[7], θ[8])
    for n in 1:nt
        fill!(du, zero(Tv))
        meat_rhs_core!(
            du, u, (n - 1) * h, θt, TPIT, 0.0, 2000.0, 0.0, CFG.g_ref,
            1.0e9, 1.0e9, 1.0e9, 3600.0, 0.05, CFG, n_s, n_a, v0, wb,
        )
        assemble!(loT, diT, upT, loW, diW, upW, θ, u, CFG, n_s, v0, wb)
        # T field
        Tn = @view u[1:NX]
        tmul!(ax, loT, diT, upT, Tn)
        for i in 1:NX
            tmp[i] = Tn[i] + h * (du[i] - ax[i])
        end
        thomas!(Tn, loT, diT, upT, tmp, h, wa, wb2, wd)
        # W field
        Wn = @view u[(NX + 1):2NX]
        tmul!(ax, loW, diW, upW, Wn)
        for i in 1:NX
            tmp[i] = Wn[i] + h * (du[NX + i] - ax[i])
        end
        thomas!(Wn, loW, diW, upW, tmp, h, wa, wb2, wd)
        # salt shares W's metric structure with D_salt (inert at s = 0):
        # explicit here (D_salt is 4 orders below D_eff — never stiff)
        for i in 1:NX
            u[4NX + i] += h * du[4NX + i]
        end
        # C, F fields and the three scalars: explicit
        for i in (2NX + 1):(4NX)
            u[i] += h * du[i]
        end
        for i in (5NX + 1):NS
            u[i] += h * du[i]
        end
        n % every == 0 && push!(core, u[1])
    end
    return core
end

loss(θ, h, tgt) = sum(abs2, solve_fixed(θ, h, length(tgt)) .- tgt)

function main()
    θ0 = [3.8e-9, 0.0018, 10.0, 0.03, 1.0, 5.6e-4, 0.20, 0.6]
    nsave = 96

    # --- accuracy vs the production FBDF solve ----------------------------
    # meat_solve integrates FROM tsave[1]: include t = 0 so both solvers
    # cook the same horizon, then compare on the shared later grid
    ts = collect(range(0.0, T_END; length=nsave + 1))
    ref = meat_solve(θ0, fill(TPIT, 2), [0.0, T_END], ts).Tcore[2:end]
    println("fixed-step IMEX-BE vs production FBDF (max |ΔTcore| [K]):")
    for h in (60.0, 30.0, 10.0, 5.0)
        c = solve_fixed(θ0, h, nsave)
        m = min(length(c), length(ref))
        @printf("  h = %4.0f s: %.4f K\n", h, maximum(abs.(c[1:m] .- ref[1:m])))
    end

    # --- gradients: Enzyme reverse vs ForwardDiff on the SAME stepper -----
    h = 30.0
    tgt = collect(solve_fixed(θ0 .* 1.02, h, nsave))  # a nearby target
    L(θv) = loss(θv, h, tgt)
    g_enz = zero(θ0)
    Enzyme.autodiff(Enzyme.Reverse, L, Enzyme.Active, Enzyme.Duplicated(θ0, g_enz))
    t_rev = @elapsed begin
        fill!(g_enz, 0.0)
        Enzyme.autodiff(Enzyme.Reverse, L, Enzyme.Active,
                        Enzyme.Duplicated(θ0, g_enz))
    end
    g_fwd = ForwardDiff.gradient(L, θ0)
    t_fwd = @elapsed (g_fwd = ForwardDiff.gradient(L, θ0))
    L(θ0); t_prim = @elapsed L(θ0)
    rel = maximum(abs.(g_enz .- g_fwd) ./ (abs.(g_fwd) .+ 1e-30))
    @printf("gradient (8 θ): Enzyme vs ForwardDiff max rel %.3e\n", rel)
    @printf("primal %.3f s | Enzyme reverse %.3f s (%.1fx) | ForwardDiff %.3f s (%.1fx)\n",
            t_prim, t_rev, t_rev / t_prim, t_fwd, t_fwd / t_prim)
    per_dual = (t_fwd - t_prim) / length(θ0)
    @printf("=> marginal forward cost/param %.4f s; reverse crossover ~%.0f params\n",
            per_dual, (t_rev - t_prim) / per_dual)
    rel < 1e-5 || error("gradient validation FAILED")
    println("FULL-RHS gradient validation PASSED")
end

main()
