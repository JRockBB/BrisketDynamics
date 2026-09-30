# Development/evaluation harness for non-Gaussian posterior ranks. No truth is passed to inference.
include(joinpath(@__DIR__,"consistency_control.jl"))

# Family-wise checks across independent cooks at one fixed endpoint. Coordinate dependence within a
# cook is allowed by Bonferroni; repeated arrival times are never counted as independent observations.
# DKW checks the whole posterior-rank CDF, without assuming a Gaussian posterior or chi-square radii.
function posterior_control_summary(records,requested;alpha=.05,families=5,min_cooks=64)
    complete=filter(r->r.completed,records)
    n=length(complete)
    n==0 && return (completed=0,requested=requested,precision_failures=length(records),gates=NamedTuple[],
                   statistically_eligible=false,passed=false)
    d=length(first(complete).ranks);tests=families*(2d+3)
    bound=sqrt(log(2tests/alpha)/(2n));tail=alpha/(2tests)
    lo=binomial_quantile(tail,n,.95);hi=binomial_quantile(1-tail,n,.95)
    gates=NamedTuple[]
    for i in 1:d+3
        ranks=sort([i<=d ? r.ranks[i] : r.joint_ranks[i-d] for r in complete])
        statistic=max(maximum((1:n)./n-ranks),maximum(ranks-(0:n-1)./n))
        push!(gates,(metric="rank CDF $i",value=statistic,lower=0.0,upper=bound,passed=statistic<=bound))
    end
    for i in 1:d
        covered_count=sum(r.covered[i] for r in complete)
        push!(gates,(metric="95% quantile coverage $i",value=covered_count/n,lower=lo/n,upper=hi/n,passed=lo<=covered_count<=hi))
    end
    bad_precision=count(r->!r.mc_precision_ok,records)
    eligible=length(records)==requested && n==requested && n>=min_cooks && bad_precision==0
    return (completed=n,requested=requested,precision_failures=bad_precision,alpha=alpha,families=families,
            min_cooks=min_cooks,gates=gates,statistically_eligible=eligible,
            passed=eligible && all(g.passed for g in gates))
end

function run_posterior_control(;n=8,seed0=3011,nx=4,suite=:full,particles=16384,
                               times=[0.0,120.0,600.0,1800.0],output="",method=:importance,replicates=4,moves=4,max_particles=particles)
    method in (:importance,:smc) || throw(ArgumentError("unknown posterior method $method"))
    model=consistency_model(;nx,suite,stop=maximum(times));records=NamedTuple[]
    for seed in seed0:seed0+n-1
        start=time();saved=Ref{Any}()
        try
            control_cook(model,seed;times,lag=typemax(Int),inspect=(sm,r,truth)->(saved[]=(sm,r,truth)))
            sm,gaussian,truth=saved[]
            count=particles;sampling_rng=MersenneTwister(seed+1000000)
            posterior=method===:importance ? sample_posterior(sm;n=count,rng=sampling_rng) :
                sample_posterior_smc(sm;particles=count,replicates,moves,rng=sampling_rng,verbose=true)
            while !posterior.diagnostics.mc_precision_ok && count<max_particles
                count=min(2count,max_particles)
                posterior=method===:importance ? sample_posterior(sm;n=count,rng=sampling_rng) :
                    sample_posterior_smc(sm;particles=count,replicates,moves,rng=sampling_rng,verbose=true)
            end
            nz=length(model.blk);target=vcat(truth.z,truth.y);error=posterior.mean-target
            ranks=[sum(posterior.weights[view(posterior.draws,i,:).<=target[i]]) for i in eachindex(target)]
            q=posterior_quantile(posterior);covered=[q[i,1]<=target[i]<=q[i,3] for i in eachindex(target)]
            zix=1:nz;yix=nz+1:length(target);d=posterior.diagnostics
            joint_ranks=Float64[];joint_nees=Float64[]
            for ix in (zix,yix,eachindex(target))
                values=hcat(reshape(error[ix],:,1),posterior.draws[ix,:].-posterior.mean[ix])
                white=E._posterior_whiten(posterior.draws[ix,:],posterior.weights,posterior.mean[ix],values)
                radii=vec(sum(abs2,white;dims=1))
                push!(joint_ranks,sum(posterior.weights[radii[2:end].<=radii[1]]))
                push!(joint_nees,radii[1]/length(ix))
            end
            record=(seed=seed,completed=true,mc_precision_ok=d.mc_precision_ok,ess=d.ess,min_tail_ess=minimum(d.tail_ess),
                    joint_mean_mc_error=d.joint_mean_mc_error,
                    max_weight=d.max_weight,numerical_failures=d.numerical_failures,
                    z_nees=joint_nees[1],y_nees=joint_nees[2],
                    gaussian_z_nees=dot(gaussian.z-truth.z,gaussian.z_cov\(gaussian.z-truth.z))/nz,
                    gaussian_y_nees=dot(gaussian.y-truth.y,gaussian.y_cov\(gaussian.y-truth.y))/length(yix),
                    ranks=ranks,joint_ranks=joint_ranks,covered=covered,mean=posterior.mean,sd=sqrt.(diag(posterior.covariance)),truth=target,
                    particles_used=count,
                    wall=time()-start)
            push!(records,record)
            @printf("posterior %s seed %d: ESS %.0f tail %.0f quality %s z %.2f state %.2f (Gaussian %.2f) %.1fs\n",
                    suite,seed,d.ess,minimum(d.tail_ess),string(d.mc_precision_ok),record.z_nees,record.y_nees,record.gaussian_y_nees,record.wall)
        catch err
            err isa InterruptException && rethrow()
            push!(records,(seed=seed,completed=false,mc_precision_ok=false,error=sprint(showerror,err),wall=time()-start))
            println("posterior ",suite," seed ",seed," FAILED: ",sprint(showerror,err))
        end
        flush(stdout)
        isempty(output) || write(output,E.json_document((backend=backend_fingerprint(),validation=consistency_control_fingerprint(),suite=suite,nx=nx,times=times,
                    requested=n,seed0=seed0,particles=particles,max_particles=max_particles,method=method,replicates=replicates,moves=moves,
                    names=model.blk.names,summary=posterior_control_summary(records,n),records=records)))
    end
    return records
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_posterior_control(n=length(ARGS)>0 ? parse(Int,ARGS[1]) : 8,seed0=length(ARGS)>1 ? parse(Int,ARGS[2]) : 3011,
        suite=length(ARGS)>2 ? Symbol(ARGS[3]) : :full,output=length(ARGS)>3 ? ARGS[4] : "posterior_control.json",
        method=length(ARGS)>4 ? Symbol(ARGS[5]) : :smc,particles=length(ARGS)>5 ? parse(Int,ARGS[6]) : 2048,
        max_particles=length(ARGS)>6 ? parse(Int,ARGS[7]) : 4096,replicates=length(ARGS)>7 ? parse(Int,ARGS[8]) : 8,
        moves=length(ARGS)>8 ? parse(Int,ARGS[9]) : 4)
end
