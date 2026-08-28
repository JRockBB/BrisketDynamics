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
    monolith_jacobian,
    monolith_stack,
    monolith_stack_jacobian,
    monolith_stack_jvp,
    picard_cook,
    lininterp,
    hat_matrix,
    p_sat,
    evap_flux

end
