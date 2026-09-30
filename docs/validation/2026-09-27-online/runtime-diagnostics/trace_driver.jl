# Diagnostic replay, with the same model, draws and settings as the frozen comparison.
# Instrumentation changes timing, not the probability target or random sequence.
for (file,marker,nextmarker,renamed) in (
    ("posterior_sampling.jl","function _transport_draw(ref,u)","\"\"\"\n    sample_posterior", "_timed_transport_draw_original"),
    ("particle_online.jl","function _transport_extend(old,ref,u,draw,predictions)","function _incremental_posterior", "_timed_transport_extend_original"))
    path=joinpath(@__DIR__,"../../../estimator/src",file);source=read(path,String)
    lo=first(findfirst(marker,source));hi=first(findnext(nextmarker,source,lo))
    name=first(split(split(marker,"function ")[2],"("))
    Base.include_string(E,replace(source[lo:prevind(source,hi)],name=>renamed),path)
end
@eval E begin
    const _trace_full_count=Threads.Atomic{Int}(0)
    const _trace_extend_count=Threads.Atomic{Int}(0)
    const _trace_lock=ReentrantLock()
    function _trace_report(kind,n,elapsed)
        if n%512==0 || elapsed>2
            lock(_trace_lock) do
                println("RUNTIME TRACE ",kind," completed=",n," evaluation_seconds=",elapsed," wall_clock=",time());flush(stdout)
            end
        end
    end
    function _transport_draw(ref,u)
        start=time();result=_timed_transport_draw_original(ref,u)
        n=Threads.atomic_add!(_trace_full_count,1)+1
        _trace_report(:full,n,time()-start)
        return result
    end
    function _transport_extend(old,ref,u,draw,predictions)
        start=time();result=_timed_transport_extend_original(old,ref,u,draw,predictions)
        n=Threads.atomic_add!(_trace_extend_count,1)+1
        _trace_report(:increment,n,time()-start)
        return result
    end
end
trace_case=ONLINE_COMPARISON_CASES[2]
trace_model=consistency_model(nx=4,suite=trace_case.suite,stop=maximum(trace_case.times))
trace_batches,trace_truths=comparison_readings(trace_model,trace_case)
trace_sm=sampled_init(trace_model;rng=MersenneTwister(trace_case.seed+1000000))
trace_arrivals=NamedTuple[]
for batch in trace_batches
    println("TRACE ARRIVAL START t=",last(batch).t," wall_clock=",time());flush(stdout)
    result=assimilate!(trace_sm,batch;verbose=true)
    push!(trace_arrivals,(t=last(batch).t,committed=result.committed,wall=result.wall,
        update=result.diagnostics.update,particles=result.diagnostics.particles,
        mean_ess=result.diagnostics.ess,min_tail_ess=minimum(result.diagnostics.tail_ess)))
    println("TRACE ARRIVAL RESULT ",last(trace_arrivals));flush(stdout)
    write(joinpath(@__DIR__,"pit_core_trace.json"),E.json_document((backend=BASE_BACKEND,
        purpose="instrumented diagnostic replay, not an independent comparison",arrivals=trace_arrivals)))
    result.committed || error("diagnostic replay failed sampling precision")
end
