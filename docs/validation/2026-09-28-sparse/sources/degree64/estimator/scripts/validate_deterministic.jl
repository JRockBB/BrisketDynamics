# Deterministic inference; random simulation is used only to generate independent
# validation observations. No truth or random generator enters the estimator.
include(joinpath(@__DIR__,"validate_sampled_online.jl"))

"Run a declared suite matrix and retain failures, timings and parameter ranks. This is not a coverage certificate."
function validate_deterministic(;output="deterministic_comparison.json",cases=ONLINE_COMPARISON_CASES,nx=4,parameter_ranks=false,kwargs...)
    records=NamedTuple[]
    for case in cases
        start=time();arrivals=NamedTuple[]
        try
            model=consistency_model(;nx,suite=case.suite,stop=maximum(case.times))
            batches,truths=comparison_readings(model,case)
            sm=deterministic_init(model;record_limit=0,kwargs...)
            for (i,batch) in enumerate(batches)
                result=assimilate!(sm,batch)
                # Probability integral transforms are diagnostics only. Reporting
                # failed arrivals prevents selection on successful inference.
                ranks=result.committed && parameter_ranks ? [conditional_cdf(result.posterior,sm.bank.anchor.key_z,j,truths[i][j])
                    for j in 1:length(model.blk)] : nothing
                push!(arrivals,(t=last(batch).t,committed=result.committed,reason=get(result,:reason,:ok),
                    diagnostics=result.diagnostics,wall=result.wall,parameter_ranks=ranks))
                println("DETERMINISTIC ",case.suite," t=",last(batch).t," committed=",result.committed,
                    " reason=",get(result,:reason,:ok)," seconds=",result.wall);flush(stdout)
                result.committed || break
            end
            push!(records,(suite=case.suite,seed=case.seed,completed=length(arrivals)==length(batches) && all(r.committed for r in arrivals),
                arrivals=arrivals,wall=time()-start))
        catch err
            err isa InterruptException && rethrow()
            push!(records,(suite=case.suite,seed=case.seed,completed=false,arrivals=arrivals,error=sprint(showerror,err),wall=time()-start))
            showerror(stdout,err,catch_backtrace());println();flush(stdout)
        end
        report=(backend=backend_fingerprint(),physics=physics_fingerprint(),simulation=online_comparison_fingerprint(),
            purpose="implementation and numerical-approximation checks; not held-out coverage or real-cook calibration",
            cases=cases,nx=nx,records=records,passed=length(records)==length(cases) && all(r.completed for r in records))
        write(output,E.json_document(report))
    end
    return records
end

if abspath(PROGRAM_FILE)==@__FILE__
    records=validate_deterministic(output=isempty(ARGS) ? "deterministic_comparison.json" : first(ARGS))
    all(r.completed for r in records) || error("deterministic validation unresolved; see the complete report")
end
