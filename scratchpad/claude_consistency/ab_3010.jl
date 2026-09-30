# Seed 3010 with the damped truth, the estimator's map switched per ARGS[1] ("on"/"off").
include(joinpath(@__DIR__, "..", "..", "estimator", "scripts", "consistency_control.jl"))
est_on = ARGS[1] == "on"
@eval function truth_process_root(model, y1, t0, t1, events=Symbol[])
    E.BOUNDARY_DAMPING[] = true
    Q = belief_process_cov(model.bl, model.sn, (t1 - t0) / 3600; events)
    L = cholesky(Symmetric(E._stable_cov(belief_cov_to_coord(Q, y1, model.bl)))).L
    E.BOUNDARY_DAMPING[] = $est_on
    return L
end
E.BOUNDARY_DAMPING[] = est_on
model = consistency_model(; nx=4, suite=:full, stop=1200.0)
try
    control_cook(model, 3010; times=collect(0.0:120.0:1200.0), lag=typemax(Int))
    println("estimator damping $(ARGS[1]): every arrival committed")
catch err
    s = sprint(showerror, err); println("estimator damping $(ARGS[1]): ", s[1:min(end, 400)])
end
