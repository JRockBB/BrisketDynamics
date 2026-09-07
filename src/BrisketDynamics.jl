module BrisketDynamics

using SciMLBase: ODEProblem, solve
using SciMLBase: SciMLBase
using OrdinaryDiffEqBDF: FBDF
using OrdinaryDiffEqTsit5: Tsit5
using ForwardDiff

include("thermophysics.jl")
include("interpolation.jl")
include("meats/common.jl")
include("meats/brisket.jl")
include("meats/chicken.jl")
include("meats/ribs.jl")
include("meats/pork_butt.jl")
include("meats/newman.jl")
include("meats/sausage.jl")
include("meats/salmon.jl")
include("cookers/common.jl")
include("cookers/smoker.jl")
include("cookers/sous_vide.jl")
include("cookers/pellet.jl")
include("cookers/offset.jl")
include("cookers/pibil.jl")
include("monolith.jl")
# after the monolith: coupled_solve_2d uses PitConfig, Humidity and the
# pit algebra
include("meats/slab2d.jl")

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
    evap_flux,
    with_config,
    brisket,
    chicken,
    ribs,
    sous_vide,
    pibil_pv,
    electric_pit,
    pork_butt,
    pellet,
    earth_pit,
    newman_center,
    burn_cleanliness,
    offset_smoker,
    gas_grill,
    kamado,
    sausage,
    hot_smoked_salmon,
    meat_solve_2d,
    coupled_solve_2d,
    slab2d_stack,
    slab2d_stack_jacobian,
    slab2d_stack_jvp

end
