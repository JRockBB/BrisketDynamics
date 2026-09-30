include(joinpath(@__DIR__,"15_archive.job.jl.done"))
reports=NamedTuple[]
for name in ("core_decoder_prefix","two_core_decoder_prefix","core_decoder_full_1d","two_core_decoder_full_1d","core_guarded_prefix","two_core_guarded_prefix","core_guarded_full_1d","two_core_guarded_full_1d","two_core_guarded_full_2d","core_polished_observed","full_adaptive_polished")
    path=joinpath(@__DIR__,name*".json");isfile(path) || continue
    r=E.JSON.parse(read(path,String);allownan=true)
    outcome=get(r,"outcome",r);d=get(outcome,"diagnostics",nothing)
    d===nothing && (push!(reports,(name=name,status=:failed,error=get(outcome,"error","missing diagnostics")));continue)
    push!(reports,(name=name,revision=occursin("decoder",name) ? :final_shared_decoder : occursin("guarded",name) ? :before_shared_decoder : :before_kernel_guard,
        wall=get(r,"wall",nothing),resolved=d["resolved"],reason=d["reason"],
        indicator=d["refinement_indicator"],quadrature_error=d["integration_error"],
        integration_nodes=last(d["history"])["nodes"],graph_solves=d["graph_solves"],
        shape_complete=d["shape_complete"],shape_pass=d["shape_pass"],shape_defect=d["shape_defect"]))
end
runlog=read(joinpath(@__DIR__,"finish.log"),String)
verified=occursin("COMPLETE 26_decoder_map_check.job.jl",runlog) && occursin("COMPLETE 22_decoder_tests.job.jl",runlog) && occursin("COMPLETE 25_decoder_online.job.jl",runlog) &&
    !occursin("Test Failed",runlog) && !occursin("ERROR 22_decoder_tests.job.jl",runlog) && !occursin("ERROR 25_decoder_online.job.jl",runlog)
artifact_summary=(status=:not_consistency_qualified,committed=false,monte_carlo=false,production_source_changed=false,
    reserved_seeds_used=false,verification=(distinct_assertions=102,passed=verified),results=reports,
    incomplete_pit_core=E.JSON.parse(read(joinpath(@__DIR__,"pit_core_adaptive_incomplete.json"),String)),
    decoder_online_replay=E.JSON.parse(read(joinpath(@__DIR__,"decoder_online_replay.json"),String))["comparison"])
write(joinpath(@__DIR__,"summary.json"),adaptive_strict_json(artifact_summary))
cp(joinpath(@__DIR__,"summary.json"),joinpath(archive,"summary.json");force=true)
println("FINAL REPORTS ARCHIVED AND VERIFICATION ",verified);flush(stdout)