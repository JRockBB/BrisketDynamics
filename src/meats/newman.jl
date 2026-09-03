"""
    newman_center(Tcores, Tpit_ts, T0)

Newman product-rule quasi-3D center temperature (the cheap taste of 3D;
docs/roadmap.md fidelity item): given per-axis 1-D center trajectories
`Tcores = (Tcore_x, Tcore_y, ...)` under the SAME forcing `Tpit_ts` and
initial temperature `T0`, the multi-dimensional center follows from the
product of the dimensionless deficits,

    θ_i(t) = (Tpit(t) − Tcore_i(t)) / (Tpit(t) − T0),
    T_center(t) = Tpit(t) − (Tpit(t) − T0)·∏ θ_i(t).

EXACT for linear conduction under constant ambient (Newman 1936);
here it is the documented approximation for the nonlinear
heat–moisture physics under slowly-varying pit forcing. Each axis is
one ordinary [`meat_solve`](@ref) with that axis' half-thickness (and
`geom_m = 0`), so a finite brick costs ~3 solves. θ is clamped to
[0, 1] so late-time noise cannot push the corrected center above the
pit or below the 1-D solution's floor.
"""
function newman_center(Tcores::Tuple, Tpit_ts::AbstractVector, T0::Real)
    drive = max.(Tpit_ts .- T0, 1.0e-6)
    θprod = ones(length(Tpit_ts))
    for Tc in Tcores
        θprod .*= clamp.((Tpit_ts .- Tc) ./ drive, 0.0, 1.0)
    end
    return Tpit_ts .- drive .* θprod
end
