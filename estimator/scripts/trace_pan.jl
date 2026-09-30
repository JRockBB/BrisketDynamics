# The pan's water and the ignited fraction, truth against belief (first-order physical sd), at every arrival of
# the matched control. Run: julia --project=estimator estimator/scripts/trace_pan.jl [seeds...]
include(joinpath(@__DIR__, "consistency_control.jl"))

function trace_pan(seeds; nx=4, t_a=1800.0)
    model = consistency_model(; nx, suite=:full, stop=t_a)
    bl = model.bl
    ip = first(E.belief_index(bl, :pit, :m_pan)); ia = first(E.belief_index(bl, :pit, :ignited))
    for seed in seeds
        println("seed $seed: t | m_pan truth, estimate ± sd, coordinate estimate ± sd | ignited truth, estimate ± sd")
        control_cook(model, seed; times=collect(0.0:120.0:t_a), lag=typemax(Int), inspect=(sm, r, truth) -> begin
            xt = belief_decode(truth.y, bl); xe, Px = belief_physical_moments(r.y, r.y_cov, bl)
            @printf("  %5.0f | %.4f  %.4g ± %.3g  (y %.2f ± %.2f) | %.4f  %.4f ± %.3g\n", r.t, xt[ip], xe[ip], sqrt(Px[ip, ip]),
                    r.y[ip], sqrt(r.y_cov[ip, ip]), xt[ia], xe[ia], sqrt(Px[ia, ia]))
        end)
        flush(stdout)
    end
end

trace_pan(isempty(ARGS) ? (3001:3008) : parse.(Int, ARGS))
