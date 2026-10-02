# The tag map and the cameras from all the photos at once. The tags lie on the board, each at an unknown (x, y, θ) in
# the reference tag's frame (the lowest id, fixed at the origin; X along its right edge, Y up it, Z out of the
# board); every photo has an unknown pose; one focal length is shared. Levenberg–Marquardt on the reprojection
# error of every detected corner, started from single-tag poses chained through the tags two photos share.

"""
    Detection(id, corners)

One tag in one photo: its id and its four corners [px] in the detector's order.
"""
struct Detection
    id::Int
    corners::Matrix{Float64}
end

function detect_tags(img; ids=nothing)
    det = AprilTagDetector()
    tags = try
        det(Gray{N0f8}.(img))
    finally
        freeDetector!(det)
    end
    out = [Detection(t.id, reduce(vcat, (p' for p in t.p))) for t in tags if ids === nothing || t.id in ids]
    length(unique(d.id for d in out)) == length(out) || error("a tag id appears twice in one photo")
    return out
end

rot2(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)]

# Rodrigues: the rotation of the rotation vector ω (ForwardDiff-safe at ω = 0)
function rodrigues(ω)
    θ2 = sum(abs2, ω)
    K = [0 -ω[3] ω[2]; ω[3] 0 -ω[1]; -ω[2] ω[1] 0]
    if θ2 < 1e-12
        return one(eltype(ω)) * I(3) + K + K * K / 2
    end
    θ = sqrt(θ2)
    return one(eltype(ω)) * I(3) + sin(θ) / θ * K + (1 - cos(θ)) / θ2 * K * K
end

# a tag's corners on the board for its (x, y, θ)
board_corners(x, y, θ, tag_mm) = (Q = tag_corners(tag_mm); [x y] .+ Q * rot2(θ)')

"""
    TagMap(ids, xyθ, tag_mm)

The tags on the board: each id's (x, y, θ) [mm, rad] in the reference tag's frame.
"""
struct TagMap
    pose::Dict{Int,NTuple{3,Float64}}
    tag_mm::Float64
end

"""
    Rig(map, cams, f, rms, view_rms)

The calibrated photos: the tag map, each photo's camera on the board, the focal length [px] and the reprojection
error's rms [px], overall and per photo.
"""
struct Rig
    map::TagMap
    cams::Vector{Camera}
    f::Float64
    rms::Float64
    view_rms::Vector{Float64}
end

# ---- initialization -----------------------------------------------------------------------------------------------

# the planar pose (x, y, θ) of tag b in tag a's frame from one photo seeing both
function relative_planar(Pa, Pb, tag_mm, K)
    ca = pose(homography(tag_corners(tag_mm), Pa), K)
    cb = pose(homography(tag_corners(tag_mm), Pb), K)
    # tag b's centre and X axis in camera coordinates, then in tag a's frame
    Rab = ca.R' * cb.R
    tb = cb.R * (-cb.C); ta = ca.R * (-ca.C)
    o = ca.R' * (tb - ta)
    return (o[1], o[2], atan(Rab[2, 1], Rab[1, 1]))
end

function init_map(dets::Vector{Vector{Detection}}, tag_mm, K)
    allids = sort(unique(d.id for ds in dets for d in ds))
    ref = allids[1]
    known = Dict{Int,NTuple{3,Float64}}(ref => (0.0, 0.0, 0.0))
    for _ in 1:length(allids)
        progress = false
        for id in allids
            haskey(known, id) && continue
            ests = NTuple{3,Float64}[]
            for ds in dets
                byid = Dict(d.id => d for d in ds)
                haskey(byid, id) || continue
                for (kid, kp) in known
                    haskey(byid, kid) || continue
                    x, y, θ = relative_planar(byid[kid].corners, byid[id].corners, tag_mm, K)
                    p = [kp[1], kp[2]] + rot2(kp[3]) * [x, y]
                    push!(ests, (p[1], p[2], kp[3] + θ))
                end
            end
            isempty(ests) && continue
            known[id] = (median(first.(ests)), median(getindex.(ests, 2)),
                         atan(median(sin.(last.(ests))), median(cos.(last.(ests)))))
            progress = true
        end
        progress || break
    end
    length(known) == length(allids) ||
        error("tags $(setdiff(allids, keys(known))) never appear with a placed tag: every tag must share a photo with another")
    return TagMap(known, tag_mm)
end

# a photo's camera from all its tags at their (initial) board positions
function init_camera(ds::Vector{Detection}, m::TagMap, K)
    src = reduce(vcat, (board_corners(m.pose[d.id]..., m.tag_mm) for d in ds if haskey(m.pose, d.id)))
    dst = reduce(vcat, (d.corners for d in ds if haskey(m.pose, d.id)))
    return pose(homography(src, dst), K)
end

# ---- the joint refinement -----------------------------------------------------------------------------------------

"""
    calibrate(imgs; tag_mm, f0, iters=60, ids=nothing, outlier_px=2.0)

The rig from the photos: detections (dropping any tag touching the frame's edge), the initial map and cameras, then
Levenberg–Marquardt over every tag's (x, y, θ) (the reference fixed), every photo's pose (a rotation increment and
its centre) and the focal length; the worst detection, when its rms exceeds `outlier_px` and three times the
median (a tag partly hidden by the meat), is dropped and the problem solved again, one at a time.
"""
function calibrate(imgs; tag_mm, f0, iters=60, ids=nothing, verbose=false, outlier_px=2.0, edge_px=4.0)
    sizes = [size(img) for img in imgs]
    dets = map(eachindex(imgs)) do k
        h, w = sizes[k]
        filter(d -> all(edge_px .<= d.corners[:, 1] .<= w - edge_px) && all(edge_px .<= d.corners[:, 2] .<= h - edge_px),
               detect_tags(imgs[k]; ids))
    end
    dropped = Tuple{Int,Int}[]
    used = findall(!isempty, dets)
    length(used) < length(dets) && @warn "photos $(setdiff(eachindex(dets), used)) show no usable tag and are left out"
    dets = dets[used]; sizes = sizes[used]
    while true
        rig, det_rms = _solve_rig(dets, sizes; tag_mm, f0, iters, verbose)
        all_rms = reduce(vcat, det_rms)
        worst, kw, jw = 0.0, 0, 0
        for k in eachindex(dets), j in eachindex(dets[k])
            det_rms[k][j] > worst && ((worst, kw, jw) = (det_rms[k][j], k, j))
        end
        worst > max(outlier_px, 3median(all_rms)) || return rig, dets, dropped, used
        push!(dropped, (kw, dets[kw][jw].id)); deleteat!(dets[kw], jw)   # the worst only: it bends its photo's pose
        verbose && println("  dropped tag $(last(dropped)[2]) in photo $kw ($(round(worst; digits=2)) px)")
        f0 = rig.f
    end
end

function _solve_rig(dets, sizes; tag_mm, f0, iters, verbose)
    for (k, ds) in enumerate(dets)
        isempty(ds) && error("photo $k sees no usable tag: every photo needs one (two or more place the map)")
    end
    K0(k, f) = intrinsics(f, sizes[k][2], sizes[k][1])
    m = init_map(dets, tag_mm, K0(1, f0))
    cams = [init_camera(dets[k], m, K0(k, f0)) for k in eachindex(dets)]
    ref = minimum(keys(m.pose))
    free = sort([id for id in keys(m.pose) if id != ref])
    nt, nv = length(free), length(dets)
    # parameters: [tags (x, y, θ)…, views (δω, C)…, log f]
    p = zeros(3nt + 6nv + 1)
    for (i, id) in enumerate(free)
        p[3i-2:3i] .= m.pose[id]
    end
    for k in 1:nv
        p[3nt + 6k - 2:3nt + 6k] .= cams[k].C
    end
    p[end] = log(f0)
    R0 = [c.R for c in cams]
    function residuals(p)
        local T, f, r, o, R, C, cx, cy, xyθ, B, X
        T = eltype(p)
        f = exp(p[end])
        r = T[]
        for k in 1:nv
            o = 3nt + 6(k - 1)
            R = rodrigues(p[o+1:o+3]) * R0[k]
            C = p[o+4:o+6]
            cx, cy = sizes[k][2] / 2, sizes[k][1] / 2
            for d in dets[k]
                xyθ = d.id == ref ? (zero(T), zero(T), zero(T)) :
                      (i = findfirst(==(d.id), free); (p[3i-2], p[3i-1], p[3i]))
                B = board_corners(xyθ..., tag_mm)
                for j in 1:4
                    X = R * ([B[j, 1], B[j, 2], zero(T)] - C)
                    push!(r, f * X[1] / X[3] + cx - d.corners[j, 1], f * X[2] / X[3] + cy - d.corners[j, 2])
                end
            end
        end
        return r
    end
    λ = 1e-3
    r = residuals(p); cost = sum(abs2, r)
    for it in 1:iters
        J = ForwardDiff.jacobian(residuals, p)
        g = J' * r; A = J' * J
        improved = false
        for _ in 1:10
            δ = -(A + λ * Diagonal(diag(A) .+ 1e-9)) \ g
            pn = p + δ
            rn = residuals(pn); cn = sum(abs2, rn)
            if cn < cost
                p, r = pn, rn
                # fold the rotation increments into the base rotations
                for k in 1:nv
                    o = 3nt + 6(k - 1)
                    R0[k] = rodrigues(p[o+1:o+3]) * R0[k]; p[o+1:o+3] .= 0
                end
                rel = (cost - cn) / cost; cost = cn; λ = max(λ / 3, 1e-9); improved = true
                verbose && println("  LM $it: rms $(round(sqrt(cost / length(r)); digits=3)) px, f $(round(exp(p[end]); digits=1))")
                rel < 1e-10 && @goto done
                break
            end
            λ *= 4
        end
        improved || break
    end
    @label done
    f = exp(p[end])
    pm = Dict{Int,NTuple{3,Float64}}(ref => (0.0, 0.0, 0.0))
    for (i, id) in enumerate(free)
        pm[id] = (p[3i-2], p[3i-1], p[3i])
    end
    cams = [Camera(K0(k, f), R0[k], p[3nt + 6k - 2:3nt + 6k]) for k in 1:nv]
    # the rms per photo and per detection
    vr = Float64[]; dr = Vector{Float64}[]; o = 0
    for k in 1:nv
        n = 8 * length(dets[k]); push!(vr, sqrt(sum(abs2, r[o+1:o+n]) / n))
        push!(dr, [sqrt(sum(abs2, r[o+8(j-1)+1:o+8j]) / 8) for j in eachindex(dets[k])]); o += n
    end
    return Rig(TagMap(pm, tag_mm), cams, f, sqrt(cost / length(r)), vr), dr
end
