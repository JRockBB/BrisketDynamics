# The whitened dynamics residual's sensitivity to k_spread (the column of the factor's latent Jacobian) along a matched
# static cook, at the truth's state and latents; then with the belief's ignited fraction moved below the cap; then at
# the tune round's estimated latents. Physics: k_spread acts only through da/dt = k a(1-a) airflow, nothing once the
# fire is lit. Any sensitivity after that is the model's own artefact and is where the record's spurious information
# on k_spread enters.
const SEED = parse(Int, get(ENV, "SEED", "1"))
empty!(ARGS); append!(ARGS, ["prior_next_cook.json", "1", "1", "sens_unused.json", "14", "0", "default", "0", "0", "0", "matched"])
include(joinpath(@__DIR__, "worker_setup.jl"))
using Statistics
tune = JSON.parsefile(joinpath(REPO, "data", "mock", "coverage_system_tune_1.json"))
iign = first(E.belief_index(bl, :pit, :ignited)); imf = first(E.belief_index(bl, :pit, :m_fuel))
ik = idx["k_spread"]; ia = idx["a0"]; icth = idx["C_th"]
groups = [(g, rg) for (g, rg) in bl.groups]
seed = SEED
rng = MersenneTwister(2000 + seed)
z0 = draw_truth(rng, sites)
dd0 = DriftDraw(rng, idx; sites)
dd = DriftDraw(Dict(k => zero(v) for (k, v) in dd0.inc), Dict(k => 0.0 for (k, v) in dd0.jump))
seg_edges, blocks = drift_segments(t, z0, dd, events.t_add)
xs = [from_z(z, sites) for z in blocks]
A, depth_fraction = xs[1][AREA], xs[1][DEPTH]
depth = probe_depth(xs[1][4], depth_fraction)
o = solve_segments([x[1:8] for x in xs], [x[9:15] for x in xs], u, t, seg_edges; pit, cfg, A_meat=A, events...)
readings, clean, masks, lid = realcook_readings(o, t, events, nc, rng; A=A, L=xs[1][4], depth=depth, cfg)
k_end = searchsortedlast(t, t_end + 1e-9)
recs = MeasurementRecord[]
for k in 1:k_end, id in SENSOR_IDS
    getfield(masks, id)[k] && push!(recs, MeasurementRecord(id, t[k], getfield(readings, id)[k]))
end
clog = CookLog(t, u; events=cook_events, sensors, measurements=recs)
model = SystemModel(sys, clog, blk; sn)
zt = truth_at(z0, dd, 0.0, events.t_add)
ci = findfirst(c -> c["seed"] == seed, tune["cooks"])
z_est = Float64.(tune["cooks"][ci]["batches"][end]["z"])
println("seed ", seed, ": truth k_spread ", round(zt[ik]; digits=3), " (k = ", round(exp(zt[ik]); sigdigits=3), " /s), a0 score ", round(zt[ia]; digits=3),
        "; the estimate at 14 h: k_spread ", round(z_est[ik]; digits=3), " a0 ", round(z_est[ia]; digits=3))
kz = VarKey(:cook, :z, 0)
cases = [("truth", :truth), ("a=0.999 (cap)", 0.999), ("a=0.995", 0.995), ("a=0.99", 0.99), ("a=0.95", 0.95), ("z=estimate", :est)]
for h in [parse(Float64, s) for s in split(get(ENV, "HOURS", "0.25,0.5,1.0,1.5,2.0,3.0,4.0,6.0,8.0,10.0,11.9,12.0,12.1,13.0"), ",")]
    k = searchsortedlast(t, h * 3600 + 1e-9)
    ka, kb = VarKey(:cook, :y, k), VarKey(:cook, :y, k + 1)
    ev_k = sensor_events(clog, t[k], t[k + 1])
    _, y0 = truth_belief(o, k); _, y1 = truth_belief(o, k + 1)
    @printf("\n%.2f h: the plant's ignited %.5f -> %.5f, airflow u %.2f, events %s\n", h, o.ignited[k], o.ignited[k + 1], u[k], isempty(ev_k) ? "none" : join(string.(ev_k), ","))
    @printf("  %-14s %-9s %-9s | %-10s %-10s | per group (whitened column norm): %s\n", "case", "a0 belief", "a1 model", "|dr/dlogk|", "|dr/da0sc|", join([g for (g, _) in groups], " "))
    for (label, c) in cases
        z = c === :est ? z_est : zt
        y0c = copy(y0)
        if c isa Number
            x0 = belief_decode(y0, bl); x0[iign] = c; y0c = belief_encode(x0, bl)
        end
        f = DynamicsFactor(model, kz, ka, kb, t[k], t[k + 1]; first=false, events=ev_k)
        Js = E.jacobians(f, [z, y0c, y1])
        Jz = Js[1]
        a1 = belief_decode(f.cache.f, bl)[iign]
        @printf("  %-14s %.5f   %.5f   | %10.3e %10.3e | %s\n", label, belief_decode(y0c, bl)[iign], a1, norm(Jz[:, ik]), norm(Jz[:, ia]),
                join([@sprintf("%.1e", norm(Jz[rg, ik])) for (_, rg) in groups], " "))
    end
    flush(stdout)
end
