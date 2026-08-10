module BrisketDynamics

using SciMLBase: ODEProblem, solve
using OrdinaryDiffEqBDF: FBDF
using OrdinaryDiffEqTsit5: Tsit5
using SciMLSensitivity: InterpolatingAdjoint, ReverseDiffVJP
using ForwardDiff
using Zygote

include("thermophysics.jl")
include("interpolation.jl")
include("meat.jl")
include("pit.jl")

export MeatConfig, meat_solve, meat_vjp,
       PitConfig, pit_solve, pit_vjp,
       lininterp, hat_matrix, p_sat, evap_flux

end
