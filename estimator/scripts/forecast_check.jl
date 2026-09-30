# Forecast calibration on the matched control: the question planning asks of a belief. Each cook is assimilated
# to `t_a`; its truth then runs on to `t_a + H` under the declared process noise; the belief's forecast is the
# same propagation from draws of its joint latent/state Gaussian, each with its own process noise. The probability
# integral transform of the truth under the forecast (the share of forecast draws below the truth) should be
# uniform across independent cooks, per quantity and horizon. Nothing but the declared model and the belief enters.
# Run: julia --threads=4 --project=estimator estimator/scripts/forecast_check.jl [n=16] [seed0=3001] [suite=full] [nx=4] [t_a=1800] [draws=64]
include(joinpath(@__DIR__, "consistency_control.jl"))

const FORECAST_STEP = 600.0
const HORIZONS = [1800.0, 3600.0, 7200.0]

# Future process noise is the declared physical noise (consistency_control.jl `truth_process_root`), the same for the
# truth's continuation and every forecast draw.
process_root(model, y1, t0, t1) = truth_process_root(model, y1, t0, t1)

"Run a (z, y) pair from `t0` through the horizons in `FORECAST_STEP` slices with process noise; the physical states at each horizon."
function run_forward(model, z, y, t0, rng)
    out = Dict{Float64,Vector{Float64}}()
    t = t0; stops = t0 .+ HORIZONS
    while t < last(stops) - 1e-9
        t1 = t + FORECAST_STEP
        y, _ = E.propagate(model, y, z, t, t1)
        y = y + process_root(model, y, t, t1) * randn(rng, length(y))
        t = t1
        for (h, s) in zip(HORIZONS, stops); abs(t - s) < 1e-6 && (out[h] = belief_decode(y, model.bl)); end
    end
    return out
end

function quantities(model, x)
    bl = model.bl
    T = x[E.belief_index(bl, :brisket, :T)]; W = x[E.belief_index(bl, :brisket, :W)]
    return [x[first(E.belief_index(bl, :pit, :T_pit))], first(T), last(T), sum(T) / length(T), sum(W) / length(W)]
end
const QNAMES = ["T_pit", "T_centre", "T_surface", "T_mean", "W_mean"]

"The probe-depth latents of a model: the bank's axes under `backend=:depth`."
depth_axes(model) = [nm for nm in model.blk.names if endswith(nm, "depth_fraction")]

"A draw of the joint latent/state at the newest node from a committed belief: the Gaussian's, or a mixture component's."
function belief_draw(sm, r, rng)
    hasproperty(r, :posterior) || return (L = cholesky(Symmetric(r.zy_cov + 1e-12I)).L; m = vcat(r.z, r.y); m + L * randn(rng, length(m)))
    p = r.posterior; A = sm.anchor
    j = findfirst(>=(rand(rng)), cumsum(p.weights)); j === nothing && (j = length(p.weights))
    c = p.nodes[j]
    v = c.mean + c.root * randn(rng, size(c.root, 2))
    return vcat(v[p.offsets[A.key_z]], v[p.offsets[last(A.nodes)[2]]])
end

function run_forecast_check(; n=16, seed0=3001, suite=:full, nx=4, t_a=1800.0, draws=64, lag=typemax(Int), backend=:gauss, orders=nothing, bank_tol=Inf)
    model = consistency_model(; nx, suite, stop=t_a + maximum(HORIZONS))
    times = collect(0.0:120.0:t_a)
    nz = length(model.blk)
    pit = Dict(h => zeros(length(QNAMES), 0) for h in HORIZONS)
    zsc = Dict(h => zeros(length(QNAMES), 0) for h in HORIZONS)
    uncommitted = Int[]
    for seed in seed0:seed0+n-1
        t0 = time(); saved = Ref{Any}()
        try
            init = backend === :gauss ? nothing :
                (m -> conditional_init(m; axes=Symbol.(depth_axes(m)), orders=something(orders, length(depth_axes(m)) == 1 ? (5, 9) : (3, 5)),
                                       lag, record_limit=0, mean_tol=bank_tol, covariance_tol=bank_tol))
            control_cook(model, seed; times, lag, init, inspect=(sm, r, truth) -> (saved[] = (sm, r, truth)))
        catch err
            err isa InterruptException && rethrow()
            msg = sprint(showerror, err); push!(uncommitted, seed)
            println("seed $seed skipped: ", msg[1:min(end, 160)]); continue
        end
        smv, r, truth = saved[]
        rng = MersenneTwister(seed + 7_000_000)
        tr = try
            run_forward(model, truth.z, truth.y, t_a, rng)
        catch err
            err isa InterruptException && rethrow(); println("seed $seed skipped: the truth's own continuation failed its solve"); continue
        end
        # A draw whose solve fails is counted and left out (reported: a belief that puts mass where the physics
        # cannot run is itself a finding); it is not redrawn.
        fc = Any[]; failed = 0
        for _ in 1:draws
            v = belief_draw(smv, r, rng)
            try
                push!(fc, run_forward(model, v[1:nz], v[nz+1:end], t_a, rng))
            catch err
                err isa InterruptException && rethrow(); failed += 1
            end
        end
        failed > 0 && println("seed $seed: $failed of $draws forecast draws failed their solve")
        length(fc) >= 8 || (println("seed $seed skipped: too few forecast draws"); continue)
        draws_ok = length(fc)
        for h in HORIZONS
            qt = quantities(model, tr[h]); qd = reduce(hcat, [quantities(model, f[h]) for f in fc])
            u = [(count(<(qt[i]), qd[i, :]) + 0.5) / (draws_ok + 1) for i in eachindex(qt)]
            zs = [(qt[i] - mean(qd[i, :])) / std(qd[i, :]) for i in eachindex(qt)]
            pit[h] = hcat(pit[h], u); zsc[h] = hcat(zsc[h], zs)
        end
        @printf("seed %d (%.0fs) 1h forecast z-scores: %s\n", seed, time() - t0,
                join(["$(QNAMES[i])=$(round(zsc[3600.0][i, end]; digits=2))" for i in eachindex(QNAMES)], " "))
        flush(stdout)
    end
    isempty(uncommitted) || println("\nUpdates that did not commit (cooks left out): seeds ", join(uncommitted, ", "))
    println("\nForecast calibration over $(size(pit[HORIZONS[1]], 2)) of $n cooks from t=$(t_a) s ($(suite), nx=$(nx), $(backend)): rms z (1 is calibrated), mean z, share outside the 90% band (0.10 is calibrated)")
    for h in HORIZONS, i in eachindex(QNAMES)
        u = pit[h][i, :]; z = zsc[h][i, :]
        @printf("  +%4.0f min %-9s rms z %.2f  mean z %+.2f  outside 90%% %.2f\n", h / 60, QNAMES[i], sqrt(mean(z .^ 2)), mean(z), mean((u .< 0.05) .| (u .> 0.95)))
    end
    return pit, zsc
end

if abspath(PROGRAM_FILE) == @__FILE__
    a = ARGS
    run_forecast_check(n=length(a) > 0 ? parse(Int, a[1]) : 16, seed0=length(a) > 1 ? parse(Int, a[2]) : 3001,
                       suite=length(a) > 2 ? Symbol(a[3]) : :full, nx=length(a) > 3 ? parse(Int, a[4]) : 4,
                       t_a=length(a) > 4 ? parse(Float64, a[5]) : 1800.0, draws=length(a) > 5 ? parse(Int, a[6]) : 64,
                       backend=length(a) > 6 ? Symbol(a[7]) : :gauss, bank_tol=length(a) > 7 ? parse(Float64, a[8]) : Inf)
end
