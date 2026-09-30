# Prespecified implementation comparison, not a real-cook calibration study.
# Seeds 22001:22004 are dedicated to this comparison; 21001:21064 remain reserved for calibration.
include(joinpath(@__DIR__,"consistency_control.jl"))
include(joinpath(@__DIR__,"posterior_comparison.jl"))

const ONLINE_COMPARISON_CASES = [
    (suite=:full,seed=22001,times=[0.,120.,600.,1800.],channels=[[:pit,:core,:humidity,:mass] for _ in 1:4]),
    (suite=:pit_core,seed=22002,times=[0.,90.,420.,1800.],channels=[[:pit,:core],[:pit],[:core],[:pit,:core]]),
    (suite=:core,seed=22003,times=[0.,75.,510.,1800.],channels=[[:core] for _ in 1:4]),
    (suite=:two_core,seed=22004,times=[0.,150.,630.,1800.],channels=[[:core,:second],[:core],[:second],[:core,:second]])
]

function online_comparison_fingerprint()
    ctx=E.SHA.SHA1_CTX()
    for path in (joinpath(@__DIR__,"validate_sampled_online.jl"),joinpath(@__DIR__,"posterior_comparison.jl"),joinpath(@__DIR__,"consistency_control.jl"),
                 joinpath(@__DIR__,"../../data/calibration/prior_next_cook.json"))
        E.SHA.update!(ctx,codeunits(basename(path)));E.SHA.update!(ctx,filter(!=(0x0d),read(path)))
    end
    return bytes2hex(E.SHA.digest!(ctx))[1:12]
end

function comparison_readings(model,case)
    # Independent generative simulation. Only past readings choose each process-covariance anchor;
    # truth is never supplied to either inference method.
    rng=MersenneTwister(case.seed);anchor=online_init(model;lag=typemax(Int),record_limit=0)
    z=[isscore(s) ? randn(rng) : s.p1+s.p2*randn(rng) for s in model.blk.sites]
    key=anchor.nodes[1][2];prior=only(f for f in factors_of(anchor.graph,key) if f isa GaussianPrior)
    y=prior.m+prior.R\randn(rng,length(prior.m));disc=zeros(length(model.log.sensors))
    hs=[compile_observable(s.observable,model.sys,model.bl,model.blk,model.log) for s in model.log.sensors]
    batches=Vector{MeasurementRecord}[];truths=Vector{Float64}[]
    for (j,t) in enumerate(case.times)
        if j>1
            previous=key;key=E._node_at!(anchor,t);factor=anchor.dyn[(previous,key)]
            y=E.propagate(model,y,z,case.times[j-1],t;first_slice=factor.first)[1]+factor.cache.W\randn(rng,length(y))
        end
        params,_,tk=E._point_params(model,z,t);state=system_from_belief(belief_decode(y,model.bl),model.bl)
        batch=MeasurementRecord[]
        for (i,s) in enumerate(model.log.sensors)
            rho,sd=j==1 ? (0.,s.noise.ar_sd) : E.ou_step(s.noise,t-case.times[j-1])
            disc[i]=rho*disc[i]+sd*randn(rng)
            value=hs[i](state,z,params,tk,t)+disc[i]+s.noise.white_sd*randn(rng)
            s.id in case.channels[j] && push!(batch,MeasurementRecord(s.id,t,value))
        end
        for r in batch;add_measurement!(anchor,r);end
        solve_window!(anchor)
        push!(batches,batch);push!(truths,vcat(z,y))
    end
    return batches,truths
end

function validate_sampled_online(;output="sampled_online_comparison.json",nx=4,particles=1024,max_particles=4096,replicates=8,moves=4)
    records=NamedTuple[]
    for case in ONLINE_COMPARISON_CASES
        start=time();arrivals=NamedTuple[]
        println("ONLINE CASE ",case.suite," seed ",case.seed);flush(stdout)
        try
            model=consistency_model(;nx,suite=case.suite,stop=maximum(case.times))
            batches,truths=comparison_readings(model,case)
            sm=sampled_init(model;particles,max_particles,replicates,moves,rng=MersenneTwister(case.seed+1000000))
            for batch in batches
                r=assimilate!(sm,batch)
                d=r.diagnostics
                push!(arrivals,(t=last(batch).t,channels=[r.sensor for r in batch],committed=r.committed,
                    update=get(d,:update,:failed),particles=d.particles,mean_ess=d.ess,min_tail_ess=minimum(d.tail_ess),
                    joint_mean_mc_error=d.joint_mean_mc_error,wall=r.wall))
                println("ARRIVAL ",case.suite," ",last(arrivals));flush(stdout)
                r.committed || error("online posterior failed estimated precision at $(last(batch).t)")
            end
            count=particles;rng=MersenneTwister(case.seed+2000000);offline=nothing
            while true
                offline=sample_posterior_smc(sm.graph_backend;particles=count,replicates,moves,rng)
                (offline.diagnostics.mc_precision_ok || count>=max_particles) && break
                count=min(2count,max_particles)
            end
            offline.diagnostics.mc_precision_ok || error("offline comparison failed estimated precision")
            comparison=compare_sampled_posteriors(sm.posterior,offline)
            target=last(truths);quantiles=posterior_quantile(sm.posterior)
            record=(suite=case.suite,seed=case.seed,completed=true,arrivals=arrivals,comparison=comparison,
                online_refreshes=sm.refreshes,online_increments=sm.increments,offline_particles=count,
                online_mean=sm.posterior.mean,offline_mean=offline.mean,truth=target,
                covered=[quantiles[i,1]<=target[i]<=quantiles[i,3] for i in eachindex(target)],
                ranks=[sum(sm.posterior.weights[sm.posterior.draws[i,:].<=target[i]]) for i in eachindex(target)],wall=time()-start)
            push!(records,record);println("COMPARISON ",case.suite," ",comparison);flush(stdout)
        catch err
            err isa InterruptException && rethrow()
            push!(records,(suite=case.suite,seed=case.seed,completed=false,arrivals=arrivals,error=sprint(showerror,err),wall=time()-start))
            showerror(stdout,err,catch_backtrace());println();flush(stdout)
        end
        report=(backend=backend_fingerprint(),physics=physics_fingerprint(),validation=online_comparison_fingerprint(),purpose="implementation comparison; not calibration",
            cases=ONLINE_COMPARISON_CASES,nx=nx,particles=particles,max_particles=max_particles,replicates=replicates,moves=moves,
            records=records,passed=length(records)==length(ONLINE_COMPARISON_CASES) && all(r.completed && r.comparison.passed for r in records))
        write(output,E.json_document(report))
    end
    return records
end

if abspath(PROGRAM_FILE)==@__FILE__
    records=validate_sampled_online(output=isempty(ARGS) ? "sampled_online_comparison.json" : first(ARGS))
    all(r.completed && r.comparison.passed for r in records) || error("sampled online comparison failed; see the complete output record")
end
