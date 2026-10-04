# The cut from its photos, from the command line:
#
#   julia -t auto --project=tools/cutvision tools/cutvision/reconstruct.jl --meat brisket --tag-mm 150 \
#         --out data/cut/<name> photos/*.jpg [--cap-mm 6] [--focal-px 3000] [--voxel-mm 2] [--no-bone] [--bone-side up]
#
# Writes <out>/shape.json (the labels the shadow scripts cook: scripts/run_shadow3d.py --shape <name>), surface.obj
# and two views of the reconstruction (whole and cut through its middle). The calibration's rms, the cap's source and
# the volume and mass are printed; see docs/measure_cut.md for the photos.

using CutVision
using CutVision: render_voxels, principal_frame
using FileIO
using Statistics

function parse_args(args)
    o = Dict{String,Any}("voxel-mm" => 2.0, "bone" => nothing, "bone-side" => :down, "photos" => String[])
    i = 1
    while i <= length(args)
        a = args[i]
        if a in ("--meat", "--out", "--bone-side")
            o[a[3:end]] = a == "--meat" || a == "--bone-side" ? Symbol(args[i + 1]) : args[i + 1]; i += 2
        elseif a in ("--tag-mm", "--cap-mm", "--focal-px", "--voxel-mm", "--weight-kg")
            o[a[3:end]] = parse(Float64, args[i + 1]); i += 2
        elseif a == "--no-bone"
            o["bone"] = false; i += 1
        elseif startswith(a, "--")
            error("unknown option $a")
        else
            push!(o["photos"], a); i += 1
        end
    end
    for k in ("meat", "tag-mm", "out")
        haskey(o, k) || error("--$k is required")
    end
    length(o["photos"]) >= 6 || error("give at least six photos (see docs/measure_cut.md); got $(length(o["photos"]))")
    return o
end

function main(args)
    o = parse_args(args)
    mkpath(o["out"])
    r = reconstruct(o["photos"]; tag_mm=o["tag-mm"], meat=o["meat"], f0=get(o, "focal-px", nothing), h=o["voxel-mm"],
                    bone=o["bone"], bone_side=o["bone-side"], cap_mm=get(o, "cap-mm", nothing))
    write_shape(joinpath(o["out"], "shape.json"), r; photos=basename.(o["photos"]), weight_kg=get(o, "weight-kg", nothing))
    write_obj(joinpath(o["out"], "surface.obj"), r)
    occ = r.labels .!= 0
    c, U = principal_frame(r.grid, occ)
    span = maximum(r.grid.dims) * r.grid.h
    C = c .+ span .* [0.9, -1.1, 0.9]
    whole = render_voxels(r.grid, r.labels; C, target=c, f=1100.0, width=800, height=600)
    cut = render_voxels(r.grid, r.labels; C, target=c, f=1100.0, width=800, height=600, keep=(x, y, z) -> y > c[2])
    save(joinpath(o["out"], "reconstruction.png"), hcat(whole, cut))
    println("wrote $(o["out"]): $(round(volume_l(r); digits=3)) L, $(round(mass_kg(r); digits=3)) kg",
            haskey(o, "weight-kg") ? " (scale: $(o["weight-kg"]) kg)" : "", "; cap $(round(r.cap_mm; digits=2)) mm, $(r.cap_source)")
end

main(ARGS)
