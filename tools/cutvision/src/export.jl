# The reconstruction on disk: shape.json (the grid, its labels run-length encoded in column-major order, the meat,
# the cap, the calibration's quality, volume and mass) and surface.obj (the hull's surface for viewing: one quad per
# exposed voxel face, fat and lean as two groups).

const RHO = Dict(LEAN => 1060.0, FAT => 920.0, BONE => 1900.0)     # kg/m³: raw lean beef, fat tissue, bone

# runs of (value, count) over the array in column-major order
function rle(A)
    out = Int[]
    v = A[1]; n = 0
    for x in A
        if x == v
            n += 1
        else
            push!(out, v, n); v = x; n = 1
        end
    end
    push!(out, v, n)
    return out
end

function unrle(runs, dims)
    A = Vector{Int8}(undef, prod(dims)); i = 1
    for k in 1:2:length(runs)
        A[i:(i + runs[k + 1] - 1)] .= runs[k]; i += runs[k + 1]
    end
    return reshape(A, dims)
end

"""
    Reconstruction

The cut: its grid, labels (0 empty, 1 lean, 2 fat, 3 bone), the meat profile, the cap depth used [mm] and how it
was found, the calibration and the label votes' diagnostics.
"""
struct Reconstruction
    grid::VoxelGrid
    labels::Array{Int8,3}
    profile::MeatProfile
    cap_mm::Float64
    cap_source::String
    rig::Rig
    dropped::Vector{Tuple{Int,Int}}
    seen_fraction::Float64
end

volume_l(r::Reconstruction) = count(!=(EMPTY), r.labels) * r.grid.h^3 * 1e-6
mass_kg(r::Reconstruction) = sum(RHO[l] for l in r.labels if l != EMPTY; init=0.0) * r.grid.h^3 * 1e-9

function write_shape(path, r::Reconstruction; photos=String[], weight_kg=nothing)
    g = r.grid
    out = Dict(
        "format" => "cutvision-voxels-1", "dims" => collect(g.dims), "voxel_mm" => g.h, "origin_mm" => collect(g.origin),
        "order" => "column-major (x fastest), z up from the board, mm in the reference tag's frame",
        "labels_rle" => rle(r.labels), "label_codes" => Dict("0" => "empty", "1" => "lean", "2" => "fat", "3" => "bone"),
        "meat" => string(r.profile.name), "bone_template" => string(r.profile.bone),
        "cap_mm" => r.cap_mm, "cap_source" => r.cap_source,
        "volume_l" => volume_l(r), "mass_kg" => mass_kg(r), "weight_kg" => weight_kg,
        "counts" => Dict(n => count(==(c), r.labels) for (n, c) in (("lean", LEAN), ("fat", FAT), ("bone", BONE))),
        "calibration" => Dict("rms_px" => r.rig.rms, "view_rms_px" => r.rig.view_rms, "focal_px" => r.rig.f,
                              "tags" => Dict(string(k) => collect(v) for (k, v) in r.rig.map.pose),
                              "tag_mm" => r.rig.map.tag_mm, "dropped_photo_tag" => [collect(d) for d in r.dropped]),
        "surface_seen_fraction" => r.seen_fraction, "densities_kg_m3" => Dict(string(k) => v for (k, v) in RHO),
        "photos" => photos)
    open(io -> JSON3.pretty(io, out), path, "w")
    return path
end

function read_shape(path)
    d = JSON3.read(read(path, String))
    dims = Tuple(Int.(d.dims))
    return (; labels=unrle(Int.(d.labels_rle), dims), h=Float64(d.voxel_mm), origin=Tuple(Float64.(d.origin_mm)), meta=d)
end

"""
    write_obj(path, r)

The surface as quads, one per exposed voxel face (the board's faces left out), in the groups `fat`, `lean`, `bone`.
"""
function write_obj(path, r::Reconstruction)
    g = r.grid; L = r.labels
    verts = Dict{NTuple{3,Int},Int}(); vlist = NTuple{3,Int}[]
    vid(p) = get!(verts, p) do
        push!(vlist, p); length(vlist)
    end
    faces = Dict(LEAN => NTuple{4,Int}[], FAT => NTuple{4,Int}[], BONE => NTuple{4,Int}[])
    # the four corners of the face of cell (i, j, k) toward d, counter-clockwise seen from outside
    quad(i, j, k, d) = d == (1, 0, 0) ? ((i, j - 1, k - 1), (i, j, k - 1), (i, j, k), (i, j - 1, k)) :
                       d == (-1, 0, 0) ? ((i - 1, j - 1, k - 1), (i - 1, j - 1, k), (i - 1, j, k), (i - 1, j, k - 1)) :
                       d == (0, 1, 0) ? ((i - 1, j, k - 1), (i - 1, j, k), (i, j, k), (i, j, k - 1)) :
                       d == (0, -1, 0) ? ((i - 1, j - 1, k - 1), (i, j - 1, k - 1), (i, j - 1, k), (i - 1, j - 1, k)) :
                       d == (0, 0, 1) ? ((i - 1, j - 1, k), (i, j - 1, k), (i, j, k), (i - 1, j, k)) :
                       ((i - 1, j - 1, k - 1), (i - 1, j, k - 1), (i, j, k - 1), (i, j - 1, k - 1))
    for I in CartesianIndices(L)
        l = L[I]; l == EMPTY && continue
        i, j, k = Tuple(I)
        for d in NB6
            (d == (0, 0, -1) && k == 1) && continue
            checkbounds(Bool, L, i + d[1], j + d[2], k + d[3]) && L[i + d[1], j + d[2], k + d[3]] != EMPTY && continue
            push!(faces[l], Tuple(vid(p) for p in quad(i, j, k, d)))
        end
    end
    open(path, "w") do io
        println(io, "# cutvision surface: mm in the reference tag's frame")
        for p in vlist
            println(io, "v ", g.origin[1] + p[1] * g.h, " ", g.origin[2] + p[2] * g.h, " ", g.origin[3] + p[3] * g.h)
        end
        for (l, name) in ((FAT, "fat"), (LEAN, "lean"), (BONE, "bone"))
            isempty(faces[l]) && continue
            println(io, "g ", name)
            for q in faces[l]
                println(io, "f ", join(q, " "))
            end
        end
    end
    return path
end

"""
    reconstruct(paths; tag_mm, meat, f0=nothing, h=2.0, bone=nothing, bone_side=:down, verbose=true)

The cut from its photos: calibrate (the tag map, cameras, focal length; `f0` from the first photo's EXIF when not
given), segment each photo, carve the hull at `h` mm, lower its top to where the photos agree (`stereo`), label the
surface, measure the cap (`cap_mm` overrides it with a measurement on the cut; the profile's default when too few
steep fat faces are seen), fill the interior, place the profile's bone (`bone=false` to leave it out).
"""
function reconstruct(paths; tag_mm, meat::Symbol, f0=nothing, h=2.0, bone=nothing, bone_side=:down, stereo=true,
                     cap_mm=nothing, verbose=true)
    prof = PROFILES[meat]
    imgs = load.(paths)
    if f0 === nothing
        f0 = exif_focal_px(paths[1], size(imgs[1], 2), size(imgs[1], 1))
        f0 === nothing && error("$(paths[1]) carries no EXIF focal length: pass f0 (pixels)")
    end
    t = @elapsed rig, dets, dropped, used = calibrate(imgs; tag_mm, f0)
    imgs = imgs[used]; paths = paths[used]
    verbose && println("calibrated $(length(imgs)) photos in $(round(t; digits=1)) s: rms $(round(rig.rms; digits=3)) px, ",
                       "f $(round(rig.f; digits=1)) px, $(length(rig.map.pose)) tags, dropped $(dropped)")
    rig.rms < 1.0 || error("the calibration's rms is $(round(rig.rms; digits=2)) px (> 1): check the tags are flat and the photos sharp")
    t = @elapsed masks = [meat_mask(img, ds, tag_mm) for (img, ds) in zip(imgs, dets)]
    verbose && println("segmented in $(round(t; digits=1)) s")
    isfat = fat_classifier(masks)
    t = @elapsed g, occ = carve(rig, masks; h)
    verbose && println("carved $(g.dims) at $(h) mm in $(round(t; digits=1)) s: $(round(count(occ) * h^3 * 1e-6; digits=3)) L")
    top = nothing
    if stereo
        t = @elapsed frac_cols, lowered, top = refine_top!(occ, g, rig, masks)
        occ = largest_component(occ)
        verbose && println("refined the top by stereo in $(round(t; digits=1)) s: $(round(100frac_cols; digits=1)) % of columns, ",
                           "lowered $(round(lowered; digits=2)) mm on average: $(round(count(occ) * h^3 * 1e-6; digits=3)) L")
    end
    t = @elapsed S, N, frac, B = surface_labels(g, occ, rig, masks, isfat)
    seen = count(!isnan, frac) / length(frac)
    cap, ncol, bands = cap_depth(g, occ, B, S, N, rig, masks, isfat)
    verbose && length(bands) >= 10 && println("  cap bands: quartiles ", round.(quantile(bands, [0.1, 0.25, 0.5, 0.75, 0.9]); digits=2), " mm")
    photo = isnan(cap) ? "none read" : "$(round(cap; digits=2)) mm read in the photos on $(ncol) steep fat columns"
    if cap_mm !== nothing                                         # measured on the cut: it overrides the photos
        src = "given ($(cap_mm) mm; photos: $(photo))"; cap = Float64(cap_mm)
    elseif isnan(cap)
        src = "the $(meat) profile's default ($(ncol) steep fat columns seen)"; cap = prof.cap_mm
    else
        src = "read in the photos on $(ncol) steep fat columns"
    end
    verbose && println("labelled the surface in $(round(t; digits=1)) s: $(round(100seen; digits=1)) % seen; cap $(round(cap; digits=2)) mm, $(src)")
    L = label_voxels(g, occ, S, frac, cap)
    (bone === nothing ? prof.bone !== :none : bone) && place_bone!(L, g, occ, prof; bone_side)
    return Reconstruction(g, L, prof, cap, src, rig, dropped, seen)
end
