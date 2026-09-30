include(joinpath(@__DIR__,"15_archive.job.jl.done"))
reports=NamedTuple[]
for name in ("core_guarded_prefix","two_core_guarded_prefix","core_guarded_full_1d","two_core_guarded_full_1d","two_core_guarded_full_2d","core_polished_observed","full_adaptive_polished")
    path=joinpath(@__DIR__,name*".json");isfile(path) || continue
    r=E.JSON.parse(read(path,String);allownan=true)
    outcome=get(r,"outcome",r);d=get(outcome,"diagnostics",nothing)
    d===nothing && (push!(reports,(name=name,status=:failed,error=get(outcome,"error","missing diagnostics")));continue)
    push!(reports,(name=name,revision=occursin("guarded",name) ? :final_guarded : :before_kernel_guard,
        wall=get(r,"wall",nothing),resolved=d["resolved"],reason=d["reason"],
        indicator=d["refinement_indicator"],quadrature_error=d["integration_error"],
        integration_nodes=last(d["history"])["nodes"],graph_solves=d["graph_solves"],
        shape_complete=d["shape_complete"],shape_pass=d["shape_pass"],shape_defect=d["shape_defect"]))
end
log=read(joinpath(@__DIR__,"finish.log"),String)
verified=occursin("COMPLETE 16_final_tests.job.jl",log) && occursin("COMPLETE 20_guarded_online.job.jl",log) &&
    !occursin("Test Failed",log) && !occursin("ERROR 16_final_tests.job.jl",log) && !occursin("ERROR 20_guarded_online.job.jl",log)
summary=(status=:not_consistency_qualified,committed=false,monte_carlo=false,production_source_changed=false,
    reserved_seeds_used=false,verification=(distinct_assertions=96,passed=verified),results=reports,
    incomplete_pit_core=E.JSON.parse(read(joinpath(@__DIR__,"pit_core_adaptive_incomplete.json"),String)),
    guarded_online_replay=E.JSON.parse(read(joinpath(@__DIR__,"guarded_online_replay.json"),String))["comparison"])
write(joinpath(@__DIR__,"summary.json"),adaptive_strict_json(summary))
cp(joinpath(@__DIR__,"summary.json"),joinpath(archive,"summary.json");force=true)
println("FINAL REPORTS ARCHIVED AND VERIFICATION ",verified);flush(stdout)