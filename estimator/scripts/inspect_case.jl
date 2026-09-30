# One matched-control cook in physical units at its last arrival: every reduced entry's truth, estimate and sd
# (first-order decoded), and the latents standardized. Run: julia --project=estimator estimator/scripts/inspect_case.jl seed [suite] [nx] [t_a]
include(joinpath(@__DIR__, "consistency_control.jl"))

function inspect_case(seed; suite=:full, nx=4, t_a=1800.0)
    model = consistency_model(; nx, suite, stop=t_a + 7200.0)
    bl = model.bl; saved = Ref{Any}()
    control_cook(model, seed; times=collect(0.0:120.0:t_a), lag=typemax(Int), inspect=(sm, r, truth) -> (saved[] = (r, truth)))
    r, truth = saved[]
    xt = belief_decode(truth.y, bl); xe, Px = belief_physical_moments(r.y, r.y_cov, bl)
    println("seed $seed at t=$t_a  (group: node -> truth | estimate ± sd [coordinate error in sd])")
    for (g, rg) in bl.groups
        for i in rg
            @printf("  %-14s %-3d %12.5g | %12.5g ± %-10.3g [%+.2f]\n", g, i - first(rg) + 1, xt[i], xe[i], sqrt(Px[i, i]),
                    (r.y[i] - truth.y[i]) / sqrt(r.y_cov[i, i]))
        end
    end
    println("latents (estimate − truth)/sd:")
    for (i, nm) in enumerate(model.blk.names)
        @printf("  %-20s %+.2f   (sd %.3g, prior-scale coordinates)\n", nm, (r.z[i] - truth.z[i]) / sqrt(r.z_cov[i, i]), sqrt(r.z_cov[i, i]))
    end
end

inspect_case(parse(Int, ARGS[1]); suite=length(ARGS) > 1 ? Symbol(ARGS[2]) : :full, nx=length(ARGS) > 2 ? parse(Int, ARGS[3]) : 4)
