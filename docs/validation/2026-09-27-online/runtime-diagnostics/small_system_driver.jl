function diagnose_pit_core()
    case=ONLINE_COMPARISON_CASES[2]
    model=consistency_model(nx=4,suite=case.suite,stop=maximum(case.times))
    batches,_=comparison_readings(model,case)
    inner=online_init(model;lag=typemax(Int))
    for batch in batches
        start=time();proposal=E._append_sampled_measurements!(inner,batch;verbose=true)
        println("PIT CORE GRAPH t=",last(batch).t," converged=",proposal.converged," wall=",time()-start);flush(stdout)
    end
    ref=E._posterior_transport(inner);center=E._transport_draw(ref,ref.center)
    println("PIT CORE PROPOSAL center_potential=",center.potential," center_norm=",norm(ref.center)," dimension=",ref.n);flush(stdout)
    start=time()
    # One small diagnostic system only. This is not a precision-qualified posterior or a cook comparison.
    result=E._posterior_particle_run(ref;n=128,moves=4,rng=MersenneTwister(5501),maxstages=100,verbose=true)
    record=(backend=BASE_BACKEND,development_seed=case.seed,particles=128,replicates=1,
        purpose="proposal/runtime diagnosis only; no posterior precision claim",center_potential=center.potential,
        stages=result.stages,wall=time()-start)
    write(joinpath(@__DIR__,"pit_core_diagnostic.json"),E.json_document(record))
    println("PIT CORE DIAGNOSTIC FINISHED wall=",record.wall);flush(stdout)
end
diagnose_pit_core()
