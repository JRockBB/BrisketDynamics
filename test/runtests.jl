using BrisketDynamics
using Test

# shared literature-plausible default parameters used across the test suite
const MEAT_θ_DEFAULT = [3.8e-9, 0.0018, 10.0, 0.03, 1.0, 5.6e-4, 0.20, 0.6] # D_eff, k_m, h_c, L, k_mult, k_e, a_myosin, m_conv
# C_th, UA, r_max, ΔH_eff, u_leak, k_spread, a0 (a0 = 1 → ignition inert)
const PIT_φ_DEFAULT = [15e3, 20.0, 3e-4, 1.2e7, 0.2, 3e-4, 1.0]
# off-kink variant for gradient tests: a0 = 1 sits exactly on the clamp
const PIT_φ_GRAD = [15e3, 20.0, 3e-4, 1.2e7, 0.2, 3e-4, 0.85]

@testset "BrisketDynamics.jl" begin
    include("test_meat.jl")
    include("test_pit.jl")
    include("test_monolith.jl")
end
