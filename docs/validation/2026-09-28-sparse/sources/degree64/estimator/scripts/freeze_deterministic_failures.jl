# Reproduce exposed failures, retaining the pre-rollback banks and original target.
include(joinpath(@__DIR__,"validate_deterministic.jl"))
using Serialization

function freeze_deterministic_failures(output;cases=ONLINE_COMPARISON_CASES,nx=4,inspect=nothing,kwargs...)
    mkpath(output)
    for case in cases
        println("FREEZE ",case.suite," seed ",case.seed);flush(stdout)
        model=consistency_model(;nx,suite=case.suite,stop=maximum(case.times))
        batches,_=comparison_readings(model,case)
        sm=deterministic_init(model;record_limit=0,kwargs...)
        arrivals=NamedTuple[]
        for (i,batch) in enumerate(batches)
            # One timestamp per generated batch. The public wrapper invokes this
            # same advance then restores its old state on failure. Retain both.
            work=deepcopy(sm);start=time()
            raw=E._deterministic_advance!(work,batch)
            result=merge(raw,(committed=raw.ok,wall=time()-start))
            push!(arrivals,(t=last(batch).t,committed=result.committed,
                reason=get(result,:reason,:ok),wall=result.wall,diagnostics=result.diagnostics))
            println("FREEZE ",case.suite," t=",last(batch).t," committed=",result.committed);flush(stdout)
            if !result.committed
                snapshot=(schema=1,backend=backend_fingerprint(),physics=physics_fingerprint(),
                    simulation=online_comparison_fingerprint(),case=case,nx=nx,model=model,
                    batches=batches,arrival=i,result=result,accepted=sm,failed=work,arrivals=arrivals)
                stem=joinpath(output,string(case.suite)*"_"*string(case.seed))
                serialize(stem*".jls",snapshot)
                write(stem*".json",E.json_document((backend=snapshot.backend,physics=snapshot.physics,case=case,arrivals=arrivals)))
                inspect===nothing || inspect(snapshot,stem)
                break
            end
            sm=work
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE)==@__FILE__
    freeze_deterministic_failures(isempty(ARGS) ? "deterministic_failures" : first(ARGS))
end
