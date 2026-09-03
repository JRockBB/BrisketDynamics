# Discrete-adjoint legwork for roadmap item 2 (reverse mode when the
# parameter count demands it).
#
# The claim to test: a discrete adjoint through a FIXED-STEP stiff stepper
# beats forward mode once the parameter count is large (spatial fields,
# dense controls), and the Enzyme-friendly stepper for THIS physics is
# IMEX backward-Euler with a hand-coded tridiagonal (Thomas) solve —
# 1-D diffusion implicit with lagged coefficients, local reactions
# explicit, pure Julia loops end to end (no LAPACK, nothing Enzyme needs
# custom rules for). The same structure scales to ADI sweeps on the 2-D
# axisymmetric slab that triggers this item.
#
# The prototype integrates a REDUCED but structurally faithful
# heat–moisture system (temperature + dry-basis moisture, nonlinear
# moisture-dependent conductivity, evaporative surface BC, purge sink —
# the skeleton of meat_rhs_core!) and differentiates a trajectory loss
# with respect to a PER-NODE diffusivity field (nx parameters) plus the
# two scalar rates: reverse mode gets the whole gradient in one backward
# pass; forward mode pays one dual solve per parameter.
#
# Run:  julia --project=adjoint adjoint/discrete_adjoint.jl
# (first run: ]activate adjoint; add Enzyme ForwardDiff; develop .)

using Enzyme
using ForwardDiff
using Printf

const NX = 40
const NT = 480            # 4 h at 30 s fixed steps
const H = 30.0
const DX = 0.03 / (NX - 1)
const T_PIT = 383.15
const T0 = 281.15
const W0 = 2.33
const RHOC = 3.5e6
const LAM = 2.3e6
const KM = 1.8e-3
const HC = 12.0

# nonlinear conductivity: rises with moisture (Choi–Okos flavor)
k_of(W) = 0.30 + 0.08 * W

"""IMEX-BE step: implicit tridiagonal diffusion with coefficients lagged
at the start-of-step state, explicit local reactions and surface fluxes.
Overwrites `Tn/Wn` with the end-of-step state. `D` is the PER-NODE
moisture diffusivity field (the many-parameter axis); `p_e` (purge rate)
and `p_a` (purge threshold shift) are the scalar tail."""
function imex_step!(Tn, Wn, D, p_e, p_a, a, b, c, d)
    # --- surface fluxes at the lagged state (explicit) --------------------
    Ts, Ws = Tn[NX], Wn[NX]
    j = KM * max(Ws / (1 + Ws), 0.05) * 2.0e-5 * exp(0.06 * (Ts - 300.0))
    q = HC * (T_PIT - Ts) - LAM * j

    # --- implicit diffusion for T (Thomas algorithm) ----------------------
    @inbounds begin
        # assemble: (I - H·A(T_lagged)) T_new = T_old (+ surface source)
        for i in 1:NX
            ki = k_of(Wn[i]) / (RHOC * DX * DX)
            r = H * ki
            a[i] = -r
            b[i] = 1 + 2r
            c[i] = -r
            d[i] = Tn[i]
        end
        b[1] = 1 + 2 * H * k_of(Wn[1]) / (RHOC * DX * DX)  # mirror center
        c[1] = -2 * H * k_of(Wn[1]) / (RHOC * DX * DX)
        a[NX] = -2 * H * k_of(Wn[NX]) / (RHOC * DX * DX)
        b[NX] = 1 + 2 * H * k_of(Wn[NX]) / (RHOC * DX * DX)
        d[NX] += H * q * 2 / (RHOC * DX)
        # forward sweep
        for i in 2:NX
            m = a[i] / b[i - 1]
            b[i] -= m * c[i - 1]
            d[i] -= m * d[i - 1]
        end
        Tn[NX] = d[NX] / b[NX]
        for i in (NX - 1):-1:1
            Tn[i] = (d[i] - c[i] * Tn[i + 1]) / b[i]
        end

        # --- implicit diffusion for W with per-node D ---------------------
        for i in 1:NX
            r = H * D[i] / (DX * DX)
            a[i] = -r
            b[i] = 1 + 2r
            c[i] = -r
            d[i] = Wn[i]
        end
        b[1] = 1 + 2 * H * D[1] / (DX * DX)
        c[1] = -2 * H * D[1] / (DX * DX)
        a[NX] = -2 * H * D[NX] / (DX * DX)
        b[NX] = 1 + 2 * H * D[NX] / (DX * DX)
        d[NX] -= H * j * 2 / (1050.0 * DX)
        for i in 2:NX
            m = a[i] / b[i - 1]
            b[i] -= m * c[i - 1]
            d[i] -= m * d[i - 1]
        end
        Wn[NX] = d[NX] / b[NX]
        for i in (NX - 1):-1:1
            Wn[i] = (d[i] - c[i] * Wn[i + 1]) / b[i]
        end

        # --- explicit purge sink (local, smooth one-sided) ----------------
        for i in 1:NX
            weq = 0.6 + p_a * exp(-0.08 * max(Tn[i] - 315.0, 0.0))
            Wn[i] -= H * p_e * 0.05 * log1p(exp((Wn[i] - weq) / 0.05))
        end
    end
    return nothing
end

"""Trajectory loss: squared distance of the core temperature from a
target curve, summed over saved steps — the calibration-shaped scalar."""
function loss(θ::AbstractVector{Tv}) where {Tv}
    D = @view θ[1:NX]
    p_e, p_a = θ[NX + 1], θ[NX + 2]
    Tn = fill(Tv(T0), NX)
    Wn = fill(Tv(W0), NX)
    a = zeros(Tv, NX); b = zeros(Tv, NX); c = zeros(Tv, NX); d = zeros(Tv, NX)
    L = zero(Tv)
    for n in 1:NT
        imex_step!(Tn, Wn, D, p_e, p_a, a, b, c, d)
        tgt = T0 + (355.0 - T0) * (1 - exp(-n * H / 9000.0))
        L += (Tn[1] - tgt)^2
    end
    return L
end

function main()
    θ0 = vcat(fill(3.8e-9, NX), [5.6e-4, 0.20])
    np = length(θ0)
    println("reduced heat-moisture IMEX-BE: NX=$NX, NT=$NT, params=$np")
    L0 = loss(θ0)
    @printf("loss(θ0) = %.6e\n", L0)

    # --- Enzyme reverse: one backward pass for the FULL gradient ----------
    g_enz = zeros(np)
    t_rev = @elapsed Enzyme.autodiff(
        Enzyme.Reverse, loss, Enzyme.Active, Enzyme.Duplicated(θ0, g_enz)
    )
    t_rev = @elapsed begin
        fill!(g_enz, 0.0)
        Enzyme.autodiff(
            Enzyme.Reverse, loss, Enzyme.Active, Enzyme.Duplicated(θ0, g_enz)
        )
    end

    # --- ForwardDiff: chunked duals over all params (the current mode) ----
    g_fwd = ForwardDiff.gradient(loss, θ0)  # compile
    t_fwd = @elapsed (g_fwd = ForwardDiff.gradient(loss, θ0))

    # --- primal timing + validation ---------------------------------------
    loss(θ0)
    t_prim = @elapsed loss(θ0)
    rel = maximum(abs.(g_enz .- g_fwd) ./ (abs.(g_fwd) .+ 1e-12))
    @printf("max relative gradient mismatch Enzyme vs ForwardDiff: %.3e\n", rel)
    @printf("primal %.4f s | Enzyme reverse %.4f s (%.1fx primal) | ", t_prim,
            t_rev, t_rev / t_prim)
    @printf("ForwardDiff %d-param gradient %.4f s (%.1fx primal)\n", np, t_fwd,
            t_fwd / t_prim)
    per_dual = (t_fwd - t_prim) / np
    crossover = (t_rev - t_prim) / per_dual
    @printf("=> forward-mode marginal cost/param %.5f s; reverse pays off past ~%.0f params\n",
            per_dual, crossover)
    rel < 1e-6 || error("gradient validation FAILED")
    println("gradient validation PASSED")
end

main()
