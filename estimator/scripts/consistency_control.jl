# Independent Monte Carlo control for the graph's declared stochastic model.
# Run: julia --threads=4 --project=estimator estimator/scripts/consistency_control.jl [n=32] [seed=10001] [output] [nx=6]
if !isdefined(Main, :BrisketEstimator)
    include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
end
using .BrisketEstimator, BrisketDynamics, LinearAlgebra, Random, Statistics, Printf
using SpecialFunctions: gamma_inc, erfcinv, loggamma
isdefined(Main, :E) || (const E = BrisketEstimator)

function consistency_control_fingerprint()
    ctx=E.SHA.SHA1_CTX()
    for path in (joinpath(@__DIR__,"consistency_control.jl"),joinpath(@__DIR__,"consistency_posterior.jl"),
                 joinpath(@__DIR__,"../../data/calibration/prior_next_cook.json"))
        E.SHA.update!(ctx,codeunits(basename(path)))
        E.SHA.update!(ctx,filter(!=(0x0d),read(path)))
    end
    return bytes2hex(E.SHA.digest!(ctx))[1:12]
end

function consistency_model(; nx=6, stop=3600.0, suite=:full, noise_scale=1.0)
    suite in (:full,:core,:pit,:pit_core,:two_core) || throw(ArgumentError("unknown sensor suite $suite"))
    noise_scale>0 || throw(ArgumentError("noise_scale must be positive"))
    isfinite(stop) && stop>0 || throw(ArgumentError("the cook horizon must be finite and positive"))
    cfg = MeatConfig(nx=nx)
    sys = CookSystem(Slab1D(:brisket; cfg); pit=PitConfig())
    prior = latent_block(joinpath(@__DIR__, "..", "..", "data", "calibration", "prior_next_cook.json"), cfg)
    blk = system_latents(sys, prior.prior_phys, prior.phys_names; probes=[:brisket])
    noise(w,a,t) = NoiseModel(white_sd=noise_scale*w, ar_sd=noise_scale*a, tau=t)
    specs = Dict(:pit=>SensorSpec(:pit, PitTemperature(), noise(0.1,1.5,3000.0)),
                 :core=>SensorSpec(:core, TemperatureAt(:brisket; depth_latent=:core_depth_fraction), noise(0.1,1.2,12000.0)),
                 :humidity=>SensorSpec(:humidity, ChamberVapor(), noise(10.0,100.0,2700.0)),
                 :mass=>SensorSpec(:mass, TotalMass(), noise(0.005,0.02,2600.0)))
    ids = suite===:full ? [:pit,:core,:humidity,:mass] : suite===:pit_core ? [:pit,:core] : suite===:pit ? [:pit] : [:core]
    sensors = [specs[id] for id in ids]
    if suite===:two_core
        # Each uncertain probe position gets its own latent; two probes must not share one depth.
        sites=vcat(blk.sites,[probe_depth_site(mean=0.4,concentration=10.0)])
        names=vcat(blk.names,["second_depth_fraction"])
        semantic=vcat(blk.keys,[LatentKey(:sensor,:second,:depth_fraction)])
        blk=LatentBlock(sites,names,semantic,copy(blk.bindings),Dict(n=>i for (i,n) in enumerate(names)))
        push!(sensors,SensorSpec(:second,TemperatureAt(:brisket;depth_latent=:second_depth_fraction),noise(0.1,1.2,12000.0)))
    end
    # The control grid must cover arbitrary observation times, including horizons between knots.
    ts = 120.0 .* collect(0:ceil(Int,stop/120.0))
    return SystemModel(sys, CookLog(ts, fill(0.5, length(ts)); sensors), blk)
end

"The lower root of a slice's physical process covariance in the belief's coordinates at `y1`: damped at the boundaries (`belief_cov_to_coord` `damp`), no integrator floor."
function truth_process_root(model, y1, t0, t1, events=Symbol[])
    Q = belief_process_cov(model.bl, model.sn, (t1 - t0) / 3600; events)
    return cholesky(Symmetric(E._stable_cov(belief_cov_to_coord(Q, y1, model.bl; damp=true)))).L
end

function control_cook(model, seed; times=[0.0, 120.0, 600.0, 1800.0, 3600.0], lag=0, transition=nothing, inspect=nothing, init=nothing)
    rng = MersenneTwister(seed)
    # `init(model)` supplies another estimator over the same model (a conditional bank); its anchor smoother then
    # carries the graph the truth reads its first prior and slice structure from
    sm = init === nothing ? online_init(model; lag, record_limit=0) : init(model)
    A = hasproperty(sm, :anchor) ? sm.anchor : sm
    # Score-coordinate priors are exactly normal. Do not approximate them by drawing a beta
    # on the legacy bench's finite inverse-CDF grid and transforming that draw back to a score.
    zt = map(model.blk.sites) do site
        if isscore(site)
            randn(rng)
        elseif E._betalike(site)
            x, xc = E.beta_inc_inv(site.p1,site.p2,rand(rng))
            log(x/xc)
        else
            site.p1+site.p2*randn(rng)
        end
    end
    ky = A.nodes[1][2]
    prior = only(f for f in factors_of(A.graph, ky) if f isa GaussianPrior)
    yt = prior.m + prior.R \ randn(rng, length(prior.m))
    disc = zeros(length(model.log.sensors))
    hs = [compile_observable(s.observable, model.sys, model.bl, model.blk, model.log) for s in model.log.sensors]
    ti = [first(E.belief_index(model.bl, :pit, :T_pit)), first(E.belief_index(model.bl, :brisket, :T)), last(E.belief_index(model.bl, :brisket, :T))]
    rows = NamedTuple[]
    for (j, t) in enumerate(times)
        if j > 1
            kp = ky
            Cprev = transition === nothing ? nothing : joint_cov(A.graph; keys=[A.key_z,kp])[1]
            ky = E._node_at!(A, t)
            f = A.dyn[(kp, ky)]
            yt, _ = E.propagate(model, yt, zt, times[j-1], t; first_slice=f.first)
            # The truth draws the declared PHYSICAL process noise of the slice at its own state. The factor's
            # whitener also carries the integrator's floor, a numerical allowance that is one coordinate unit per
            # slice for a state at its floor; drawn as a disturbance it inflates near-zero states over a long cook.
            # The estimator keeps the floor, which makes it slightly conservative there, never overconfident.
            yt += truth_process_root(model, yt, times[j-1], t, f.events) * randn(rng, length(yt))
            transition === nothing || transition(sm, kp, ky, Cprev)
        end
        p, _, tk = E._point_params(model, zt, t)
        xt = belief_decode(yt, model.bl)
        xs = system_from_belief(xt, model.bl)
        readings = MeasurementRecord[]
        for (i, s) in enumerate(model.log.sensors)
            rho, sd = j == 1 ? (0.0, s.noise.ar_sd) : E.ou_step(s.noise, t-times[j-1])
            disc[i] = rho * disc[i] + sd * randn(rng)
            v = hs[i](xs, zt, p, tk, t) + disc[i] + s.noise.white_sd * randn(rng)
            push!(readings, MeasurementRecord(s.id, t, min(v, s.noise.cap)))
        end
        r = assimilate!(sm, readings)
        r.committed || error("uncommitted arrival at $t for seed $seed: " *
            join(["$k=$(getfield(r, k))" for k in propertynames(r) if !(getfield(r, k) isa AbstractArray) || length(getfield(r, k)) <= 8], ", "))
        inspect === nothing || inspect(sm, r, (z=copy(zt), y=copy(yt), x=copy(xt)))
        ez = r.z - zt; ey = r.y - yt; et = r.x[ti] - xt[ti]
        gn = Dict(g => dot(ey[rg], r.y_cov[rg,rg] \ ey[rg])/length(rg) for (g,rg) in model.bl.groups)
        push!(rows, (t=t, nis=hasproperty(r, :nis) ? r.nis : NaN, dof=hasproperty(r, :dof) ? r.dof : 0, nees_z=dot(ez, r.z_cov \ ez),
                     nees_T=dot(et, r.x_cov[ti,ti] \ et), nees_y=dot(ey, r.y_cov \ ey),
                     group_nees=gn,
                     zerr=ez ./ sqrt.(diag(r.z_cov)), Terr=et ./ sqrt.(diag(r.x_cov)[ti]),
                     innov=hasproperty(r, :innov_std) ? [r.innov_std[s.id] for s in model.log.sensors] : fill(NaN, length(model.log.sensors))))
    end
    return rows
end

function chi_square_quantile(p, d)
    lo, hi = 0.0, Float64(max(d, 1))
    while gamma_inc(d/2, hi/2)[1] < p; hi *= 2; end
    for _ in 1:80
        mid = (lo+hi)/2
        if gamma_inc(d/2, mid/2)[1] < p; lo = mid; else; hi = mid; end
    end
    return (lo+hi)/2
end

function binomial_quantile(p, n, probability)
    total = 0.0
    for k in 0:n
        total += exp(loggamma(n+1)-loggamma(k+1)-loggamma(n-k+1)+k*log(probability)+(n-k)*log1p(-probability))
        total >= p && return k
    end
    return n
end

function control_summary(cooks, failures, model, times)
    n = length(cooks)
    n > 0 || return (n=0, failures=failures, names=model.blk.names, points=NamedTuple[], gates=NamedTuple[], passed=false)
    bands(d) = [chi_square_quantile(0.025, n*d)/(n*d), chi_square_quantile(0.975, n*d)/(n*d)]
    results = map(eachindex(times)) do j
        rs = [c[j] for c in cooks]
        nz = length(model.blk); nt = 3; ny = model.bl.n
        z = mean(r.nees_z/nz for r in rs); t = mean(r.nees_T/nt for r in rs)
        y = mean(r.nees_y/ny for r in rs); ni = mean(r.nis/r.dof for r in rs)
        coverage_z = [mean(abs(r.zerr[i]) <= 1.959963984540054 for r in rs) for i in 1:nz]
        coverage_T = [mean(abs(r.Terr[i]) <= 1.959963984540054 for r in rs) for i in 1:nt]
        (t=times[j], nis=ni, nis_band=bands(rs[1].dof), nees_z=z, z_band=bands(nz),
         nees_T=t, T_band=bands(nt), nees_y=y, y_band=bands(ny),
         coverage_z=coverage_z, coverage_T=coverage_T,
         group_nees=Dict(g => mean(r.group_nees[g] for r in rs) for (g,_) in model.bl.groups),
         z_bias=[mean(r.zerr[i] for r in rs) for i in 1:nz])
    end
    # Cooks are independent; arrivals within a cook are not. Every statistic is formed at a fixed
    # time across cooks. Bonferroni bands control the joint checks, marginal mean errors and 95%
    # interval coverage, without treating repeated arrivals as independent replicates.
    alpha = 0.05
    nsites = length(model.blk)+3
    nch = length(model.log.sensors)
    tail = alpha / (2 * (4+2nsites+2nch) * length(times))
    gates = NamedTuple[]
    for r in results
        for (metric, value, d) in (("NIS",r.nis,length(model.log.sensors)), ("latent NEES",r.nees_z,length(model.blk)),
                                  ("temperature NEES",r.nees_T,3), ("state NEES",r.nees_y,model.bl.n))
            lo = chi_square_quantile(tail,n*d)/(n*d)
            hi = chi_square_quantile(1-tail,n*d)/(n*d)
            push!(gates,(t=r.t,metric=metric,value=value,lower=lo,upper=hi,passed=lo<=value<=hi))
        end
        rs = [c[findfirst(==(r.t),times)] for c in cooks]
        errors = [vcat(row.zerr,row.Terr) for row in rs]
        names = vcat(model.blk.names,["T_pit","T_centre","T_surface"])
        mean_bound = sqrt(2)*erfcinv(2tail)/sqrt(n)
        count_lo = binomial_quantile(tail,n,0.95)
        count_hi = binomial_quantile(1-tail,n,0.95)
        for i in 1:nsites
            bias = mean(e[i] for e in errors)
            count = sum(abs(e[i])<=1.959963984540054 for e in errors)
            push!(gates,(t=r.t,metric="$(names[i]) standardized bias",value=bias,lower=-mean_bound,upper=mean_bound,passed=abs(bias)<=mean_bound))
            push!(gates,(t=r.t,metric="$(names[i]) 95% coverage",value=count/n,lower=count_lo/n,upper=count_hi/n,passed=count_lo<=count<=count_hi))
        end
        j = findfirst(==(r.t),times)
        for ch in 1:nch
            bias = mean(row.innov[ch] for row in rs)
            name = string(model.log.sensors[ch].id)
            push!(gates,(t=r.t,metric="$name innovation bias",value=bias,lower=-mean_bound,upper=mean_bound,passed=abs(bias)<=mean_bound))
            if j > 1
                products = [c[j].innov[ch]*c[j-1].innov[ch] for c in cooks]
                correlation = mean(products)
                bound = sqrt(2)*erfcinv(2tail)*std(products)/sqrt(n)
                push!(gates,(t=r.t,metric="$name innovation lag-one product",value=correlation,lower=-bound,upper=bound,passed=abs(correlation)<=bound))
            end
        end
    end
    return (n=n, failures=failures, names=model.blk.names, points=results, alpha=alpha,
            gates=gates, passed=isempty(failures) && all(g.passed for g in gates))
end

function run_consistency_control(; n=32, seed0=10001, output="", nx=6, times=[0.0,120.0,600.0,1800.0,3600.0], lag=0, transition=nothing, label="working-tree",suite=:full,noise_scale=1.0)
    n >= 2 || throw(ArgumentError("consistency statistics require at least two independent cooks"))
    length(times) >= 2 && first(times)==0 && all(diff(times).>0) || throw(ArgumentError("times must increase from zero"))
    model = consistency_model(; nx, stop=maximum(times),suite,noise_scale)
    cooks = Vector{NamedTuple}[]; failures = NamedTuple[]
    for seed in seed0:(seed0+n-1)
        start = time()
        try
            rows = control_cook(model, seed; times, lag, transition)
            push!(cooks, rows)
            @printf("seed %d: z %.3f T %.3f y %.3f NIS %.3f (%.1fs)\n", seed, rows[end].nees_z/length(model.blk), rows[end].nees_T/3, rows[end].nees_y/model.bl.n, rows[end].nis/rows[end].dof, time()-start)
        catch err
            err isa InterruptException && rethrow()
            push!(failures, (seed=seed, error=sprint(showerror, err)))
            println("seed ", seed, " FAILED: ", sprint(showerror, err))
        end
        flush(stdout)
        isempty(output) || write(output, E.json_document((backend=backend_fingerprint(), label=label, requested=n, seed0=seed0, nx=nx, lag=lag,suite=suite,noise_scale=noise_scale, summary=control_summary(cooks, failures, model, times), cooks=cooks)))
    end
    result = control_summary(cooks, failures, model, times)
    for p in result.points
        @printf("t %.0f: NIS %.3f z %.3f T %.3f state %.3f\n",p.t,p.nis,p.nees_z,p.nees_T,p.nees_y)
    end
    println("Consistency gates: ",result.passed ? "PASS" : "FAIL"); flush(stdout)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_consistency_control(n=length(ARGS)>0 ? parse(Int,ARGS[1]) : 32, seed0=length(ARGS)>1 ? parse(Int,ARGS[2]) : 10001,
                            output=length(ARGS)>2 ? ARGS[3] : "consistency_control.json", nx=length(ARGS)>3 ? parse(Int,ARGS[4]) : 6)
end
