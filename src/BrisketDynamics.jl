module BrisketDynamics

using SciMLBase: ODEProblem, solve
using SciMLBase: SciMLBase
using OrdinaryDiffEqBDF: FBDF
using OrdinaryDiffEqTsit5: Tsit5
using ForwardDiff

include("thermophysics.jl")
include("interpolation.jl")
include("meat.jl")
include("pit.jl")
include("monolith.jl")

export MeatConfig,
    meat_solve,
    wrap_blend,
    WRAP_NEVER,
    PULL_NEVER,
    pull_blend,
    PitConfig,
    pit_solve,
    Humidity,
    monolith_solve,
    monolith_solve_tf,
    monolith_jacobian,
    monolith_stack,
    monolith_stack_jacobian,
    monolith_stack_jvp,
    monolith_stack_jacobian_tf,
    monolith_stack_jvp_tf,
    FRAC_NEVER,
    picard_cook,
    lininterp,
    hat_matrix,
    p_sat,
    evap_flux

end
