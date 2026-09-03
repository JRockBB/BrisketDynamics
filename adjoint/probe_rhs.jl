# Probe: does Enzyme reverse through the PRODUCTION meat_rhs_core! ?
# (closures, views, @inbounds loops, Choi-Okos polynomials, Oswin, the
# whole thing). If yes, the M1 full port can call it directly; if not,
# the port reimplements f with an f-agreement check.
using Pkg
Pkg.activate(@__DIR__)
using Enzyme
using BrisketDynamics
using BrisketDynamics: meat_rhs_core!, MeatConfig, _meat_u0, shrink_exponents,
    bone_weights

const CFG = MeatConfig()
const NXS = 5 * CFG.nx + 3
const θ0 = [3.8e-9, 0.0018, 10.0, 0.03, 1.0, 5.6e-4, 0.20, 0.6]

function f_sum(u::Vector{Float64}, θ::Vector{Float64})
    du = zeros(NXS)
    n_s, n_a, v0 = shrink_exponents(CFG)
    wb = bone_weights(CFG)
    meat_rhs_core!(
        du, u, 1000.0, (θ[1], θ[2], θ[3], θ[4], θ[5], θ[6], θ[7], θ[8]),
        383.15, 0.0, 2000.0, 0.0, CFG.g_ref, 1.0e9, 1.0e9, 1.0e9, 3600.0, 0.05,
        CFG, n_s, n_a, v0, wb,
    )
    return sum(abs2, du)
end

u0 = _meat_u0(CFG, Float64[], Float64[], Float64[])
u0[1:CFG.nx] .+= range(0, 40; length=CFG.nx)  # a mid-cook-ish profile
println("primal f_sum = ", f_sum(u0, θ0))

gu = zero(u0)
gθ = zero(θ0)
Enzyme.autodiff(
    Enzyme.Reverse, f_sum, Enzyme.Active,
    Enzyme.Duplicated(u0, gu), Enzyme.Duplicated(θ0, gθ),
)
println("Enzyme reverse through meat_rhs_core!: OK")
println("|grad_u| = ", sum(abs, gu), "   grad_θ = ", gθ)

# spot-check one component against central FD
i = 3 * CFG.nx + 7
h = 1e-6
up = copy(u0); up[i] += h
um = copy(u0); um[i] -= h
fd = (f_sum(up, θ0) - f_sum(um, θ0)) / (2h)
println("FD check du[$i]: enzyme=", gu[i], " fd=", fd,
        " rel=", abs(gu[i] - fd) / (abs(fd) + 1e-12))
