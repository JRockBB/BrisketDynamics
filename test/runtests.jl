using BrisketDynamics
using Test

# shared literature-plausible default parameters used across the test suite
const MEAT_θ_DEFAULT = [1.5e-8, 0.0018, 10.0, 0.03, 1.0] # D_eff, k_m, h_c, L, k_mult
const PIT_φ_DEFAULT = [15e3, 20.0, 3e-4, 1.2e7, 0.2]     # C_th, UA, r_max, ΔH_eff, u_leak

@testset "BrisketDynamics.jl" begin
    include("test_meat.jl")
    include("test_pit.jl")
    include("test_gradients.jl")
end
