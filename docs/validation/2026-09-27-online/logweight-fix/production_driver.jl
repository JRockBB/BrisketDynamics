production_source=joinpath(@__DIR__,"../../../estimator/src/particle_online.jl")
sha256(read(production_source))==sha256(read(joinpath(@__DIR__,"particle_online.jl"))) ||
    error("loaded log-weight candidate and production source differ")
final_backend=backend_fingerprint();final_physics=physics_fingerprint()
println("FINAL PRODUCTION REGRESSION backend=",final_backend," physics=",final_physics);flush(stdout)
final_files=("test_posterior_sampling.jl","test_posterior_smc.jl","test_sampled_online.jl",
    "test_particle_online.jl","test_sampled_offline.jl","test_sampled_planning.jl",
    "test_particle_logweights.jl","test_sampled_history_snapshot.jl")
final_failed=String[]
for file in final_files
    println("PRODUCTION TEST FILE ",file);flush(stdout)
    scope=Module(gensym(:ProductionValidation));Core.eval(scope,:(const BrisketEstimator=Main.E))
    Core.eval(scope,:(include(path)=Base.include(@__MODULE__,path)))
    path=joinpath(@__DIR__,"../../../estimator/test",file)
    final_source=replace(read(path,String),"include(joinpath(@__DIR__,\"..\",\"src\",\"BrisketEstimator.jl\"))"=>"",
        "include(joinpath(@__DIR__, \"..\", \"src\", \"BrisketEstimator.jl\"))"=>"")
    try
        Base.include_string(scope,final_source,path);println("FILE PASS ",file)
    catch err
        push!(final_failed,file);showerror(stdout,err,catch_backtrace());println();println("FILE FAIL ",file)
    end
    flush(stdout)
end
final_backend==backend_fingerprint() && final_physics==physics_fingerprint() || error("production source changed during regression")
final_hashes=Dict("estimator/src/"*file=>bytes2hex(sha256(read(joinpath(@__DIR__,"../../../estimator/src",file))))
    for file in (E.BACKEND_FILES...,"BrisketEstimator.jl"))
for file in final_files
    final_hashes["estimator/test/"*file]=bytes2hex(sha256(read(joinpath(@__DIR__,"../../../estimator/test",file))))
end
write(joinpath(@__DIR__,"production_validation.json"),E.json_document((backend=final_backend,physics=final_physics,
    passed=isempty(final_failed),failed=final_failed,files=final_files,hashes=final_hashes,
    process="warm four-thread worker; candidate particle module at startup, original forecast/transport methods restored before final regression")))
isempty(final_failed) || error("production regression failures: $final_failed")
