archive=joinpath(@__DIR__,"../../docs/validation/2026-09-28-adaptive")
mkpath(archive);mkpath(joinpath(archive,"drivers"))
for name in sort(readdir(@__DIR__))
    artifact_path=joinpath(@__DIR__,name)
    if endswith(name,".json")
        raw=E.JSON.parse(read(artifact_path,String);allownan=true)
        write(joinpath(archive,name),adaptive_strict_json(raw))
        E.JSON.parse(read(joinpath(archive,name),String))
    elseif endswith(name,".job.jl.done") || endswith(name,".job.jl") || endswith(name,".job.jl.interrupted") || endswith(name,"worker.jl")
        cp(artifact_path,joinpath(archive,"drivers",name);force=true)
    end
end
manifests=NamedTuple[]
for name in sort(filter(n->endswith(n,".jls"),readdir(@__DIR__)))
    artifact_path=joinpath(@__DIR__,name)
    push!(manifests,(path="scratchpad/adaptive_work/"*name,bytes=filesize(artifact_path),sha256=bytes2hex(sha256(read(artifact_path)))))
end
write(joinpath(archive,"candidate-manifest.json"),adaptive_strict_json(manifests))
println("ADAPTIVE ARTIFACTS ARCHIVED");flush(stdout)