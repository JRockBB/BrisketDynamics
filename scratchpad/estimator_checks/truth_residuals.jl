# The smoother's dynamics factor evaluated along the PLANT's own trajectory at the TRUE latents (a matched, static
# cook of the bench): the whitened residual per belief group and per slice, and the raw physical step difference of
# the pit temperature, the core temperature, the ignited fraction and the fuel mass. The same at the latents the tune
# round's smoother ended with, when the round holds the seed. Zero systematic residual at the truth is what
# "matched" means; a residual that is SMALLER at the estimate than at the truth says the estimator is fitting a
# plant that is not its model.
const SEEDS = [parse(Int, s) for s in split(get(ENV, "SEEDS", "1,10"), ",")]
const HOURS = parse(Float64, get(ENV, "HOURS", "14"))
const STEP = parse(Int, get(ENV, "STEP", "1"))
empty!(ARGS); append!(ARGS, ["prior_next_cook.json", "1", "1", "truth_residuals_unused.json", string(HOURS), "0", "default", "0", "0", "0", "matched"])
include(joinpath(@__DIR__, "worker_setup.jl"))
using Statistics
tune = JSON.parsefile(joinpath(REPO, "data", "mock", "coverage_system_tune_1.json"))
iign = first(E.belief_index(bl, :pit, :ignited)); imf = first(E.belief_index(bl, :pit, :m_fuel))
groups = [(g, rg) for (g, rg) in bl.groups]
hour_bins = [(0.0, 2.0), (2.0, 6.0), (6.0, 12.0), (12.0, HOURS + 1e-9)]
rms(v) = sqrt(mean(abs2, v))
for seed in SEEDS
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
    z_est = ci === nothing ? nothing : Float64.(tune["cooks"][ci]["batches"][end]["z"])
    println("\n=== seed ", seed, ": segments ", length(blocks), "; truth k_spread ", round(zt[idx["k_spread"]]; digits=3), " a0 ", round(zt[idx["a0"]]; digits=3),
            z_est === nothing ? "" : "; the tune round's estimate at 14 h: k_spread $(round(z_est[idx["k_spread"]]; digits=3)) a0 $(round(z_est[idx["a0"]]; digits=3))")
    # the truth's state through the belief's coordinates and back to the system's state
    for k in (1, 61, 301)
        ics, a = handoff(o, k)
        pf = Tuple(f for f in keys(ics) if !(f in PIT_ICS_FIELDS))
        piece_ics = NamedTuple{pf}(Tuple(getfield(ics, f) for f in pf))
        pit_ics = (Tpit0=ics.Tpit0, mfuel0=ics.mfuel0, mpan0=ics.mpan0, Tpan0=ics.Tpan0, Tfb0=ics.Tfb0, ignited=clamp(a, 1e-6, 1 - 1e-6))
        x_sys = initial_state(sys, p_pack, play_pack; ics=[piece_ics], pit_ics)
        _, y = truth_belief(o, k)
        x_back = system_from_belief(belief_decode(y, bl), bl)
        d = abs.(x_back .- x_sys); j = argmax(d)
        @printf("  round trip at k=%d (%.1f h): max |lift(decode(encode)) - x_sys| = %.3g at state %d (value %.4g); encode∘decode on y: %.2g\n",
                k, t[k] / 3600, d[j], j, x_sys[j], maximum(abs.(belief_encode(belief_decode(y, bl), bl) .- y)))
    end
    kz = VarKey(:cook, :z, 0)
    wt = Dict(g => Float64[] for (g, _) in groups); we = Dict(g => Float64[] for (g, _) in groups); th = Float64[]
    raw = NTuple{5,Float64}[]; rawe = NTuple{5,Float64}[]
    t0w = time()
    for k in 1:STEP:(k_end - STEP)
        ka, kb = VarKey(:cook, :y, k), VarKey(:cook, :y, k + STEP)
        ev_k = sensor_events(clog, t[k], t[k + STEP])
        _, y0 = truth_belief(o, k); _, y1 = truth_belief(o, k + STEP)
        f = DynamicsFactor(model, kz, ka, kb, t[k], t[k + STEP]; first=(k == 1), events=ev_k)
        r = E.residual(f, [zt, y0, y1])
        x1 = belief_decode(y1, bl); xf = belief_decode(f.cache.f, bl)
        push!(th, t[k + STEP] / 3600)
        push!(raw, (t[k + STEP] / 3600, x1[ip] - xf[ip], x1[ic] - xf[ic], x1[iign] - xf[iign], x1[imf] - xf[imf]))
        for (g, rg) in groups; push!(wt[g], rms(r[rg])); end
        if z_est !== nothing
            f2 = DynamicsFactor(model, kz, ka, kb, t[k], t[k + STEP]; first=(k == 1), events=ev_k)
            r2 = E.residual(f2, [z_est, y0, y1])
            xf2 = belief_decode(f2.cache.f, bl)
            push!(rawe, (t[k + STEP] / 3600, x1[ip] - xf2[ip], x1[ic] - xf2[ic], x1[iign] - xf2[iign], x1[imf] - xf2[imf]))
            for (g, rg) in groups; push!(we[g], rms(r2[rg])); end
        end
    end
    @printf("  %d slices of %d grid steps in %.0f s\n", length(th), STEP, time() - t0w)
    println("  whitened dynamics residual, RMS per belief group (1 = the assumed state noise), at the TRUE latents | at the tune round's ESTIMATE")
    @printf("  %-12s", "group"); for (a, b) in hour_bins; @printf(" %5.0f-%-4.0fh", a, min(b, HOURS)); end; println("   |", z_est === nothing ? "" : "  the same at the estimate")
    for (g, _) in groups
        @printf("  %-12s", g)
        for (a, b) in hour_bins; sel = [i for i in eachindex(th) if a < th[i] <= b]; @printf(" %10.2f", isempty(sel) ? NaN : rms(wt[g][sel])); end
        if z_est !== nothing
            print("   |"); for (a, b) in hour_bins; sel = [i for i in eachindex(th) if a < th[i] <= b]; @printf(" %10.2f", isempty(sel) ? NaN : rms(we[g][sel])); end
        end
        println()
    end
    println("  raw step difference plant − model over one slice, MEAN per hour bin (a systematic sign is a mismatch): T_pit [K], T_core [K], ignited, m_fuel [kg]")
    for (a, b) in hour_bins
        sel = [i for i in eachindex(raw) if a < raw[i][1] <= b]; isempty(sel) && continue
        m_ = [mean(r[j] for r in raw[sel]) for j in 2:5]; s_ = [std(r[j] for r in raw[sel]) for j in 2:5]
        @printf("    %4.0f-%-4.0fh truth: dT_pit %+9.4f (sd %.4f)  dT_core %+9.5f (sd %.5f)  d_ign %+10.3e (sd %.1e)  dm_fuel %+10.3e (sd %.1e)\n", a, min(b, HOURS), m_[1], s_[1], m_[2], s_[2], m_[3], s_[3], m_[4], s_[4])
        if z_est !== nothing
            m2 = [mean(r[j] for r in rawe[sel]) for j in 2:5]
            @printf("    %9s est.:  dT_pit %+9.4f             dT_core %+9.5f              d_ign %+10.3e            dm_fuel %+10.3e\n", "", m2[1], m2[2], m2[3], m2[4])
        end
    end
    println("  the plant's ignited fraction: ", join([@sprintf("%.3f@%.0fh", o.ignited[searchsortedlast(t, h * 3600 + 1e-9)], h) for h in (0, 1, 2, 4, 8, 12, 12.5, 14)], " "))
    flush(stdout)
end
