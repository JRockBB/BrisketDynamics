# The visual hull on a voxel grid in the board's frame (the reference tag's; z up from the board). A voxel is kept
# when its centre falls on the meat in every photo that sees it: a photo it falls outside of, or onto an unknown
# pixel (a tag's card), says nothing. Carved coarse over the tag ring first, then fine inside the coarse hull's box;
# the largest connected piece is the meat.

"""
    VoxelGrid(origin, h, dims)

Cells of side `h` [mm]; cell (i, j, k) spans origin + h·([i − 1, i], [j − 1, j], [k − 1, k]).
"""
struct VoxelGrid
    origin::NTuple{3,Float64}
    h::Float64
    dims::NTuple{3,Int}
end

cell_centre(g::VoxelGrid, i, j, k) = (g.origin[1] + (i - 0.5) * g.h, g.origin[2] + (j - 0.5) * g.h, g.origin[3] + (k - 0.5) * g.h)

# a camera as plain numbers for the inner loop: K's focal length and principal point, R, C
struct Proj
    f::Float64; cx::Float64; cy::Float64
    R::NTuple{9,Float64}
    C::NTuple{3,Float64}
    h::Int; w::Int
end
Proj(cam::Camera, h, w) = Proj(cam.K[1, 1], cam.K[1, 3], cam.K[2, 3], Tuple(vec(cam.R')), Tuple(cam.C), h, w)

# the pixel (row, col) of a board point, or (0, 0) behind the camera or off the photo
@inline function pixel(p::Proj, x, y, z)
    dx, dy, dz = x - p.C[1], y - p.C[2], z - p.C[3]
    X = p.R[1] * dx + p.R[2] * dy + p.R[3] * dz
    Y = p.R[4] * dx + p.R[5] * dy + p.R[6] * dz
    Z = p.R[7] * dx + p.R[8] * dy + p.R[9] * dz
    Z <= 0 && return (0, 0)
    c = floor(Int, p.f * X / Z + p.cx) + 1; r = floor(Int, p.f * Y / Z + p.cy) + 1
    return (1 <= r <= p.h && 1 <= c <= p.w) ? (r, c) : (0, 0)
end

function _carve(g::VoxelGrid, projs::Vector{Proj}, meat::Vector{BitMatrix}, unknown::Vector{BitMatrix})
    occ = falses(g.dims)
    Threads.@threads for k in 1:g.dims[3]
        for j in 1:g.dims[2], i in 1:g.dims[1]
            x, y, z = cell_centre(g, i, j, k)
            keep = true; seen = 0
            for v in eachindex(projs)
                r, c = pixel(projs[v], x, y, z)
                r == 0 && continue
                unknown[v][r, c] && continue
                seen += 1
                meat[v][r, c] || (keep = false; break)
            end
            occ[i, j, k] = keep && seen >= 2
        end
    end
    return occ
end

# the largest 6-connected component
function largest_component(occ::BitArray{3})
    lab = label_components(occ)                      # 6-connectivity by default
    n = maximum(lab); n == 0 && return occ
    sizes = zeros(Int, n)
    for l in lab
        l > 0 && (sizes[l] += 1)
    end
    return lab .== argmax(sizes)
end

"""
    carve(rig, masks; h=2.0, coarse=8.0, zmax=300.0, dilate_px=1)

The meat's visual hull: the photos' meat masks (`PhotoMask`, dilated by `dilate_px` for the calibration's sub-pixel
error) carve a coarse grid over the tag ring's box, then a fine one of side `h` [mm] over the coarse hull's box
grown by a coarse cell. Returns the grid and the occupancy (the largest connected piece).
"""
function carve(rig::Rig, masks::Vector{PhotoMask}; h=2.0, coarse=8.0, zmax=300.0, dilate_px=1)
    projs = [Proj(cam, size(m.meat)...) for (cam, m) in zip(rig.cams, masks)]
    meat = [dilate_px > 0 ? BitMatrix(dilate(m.meat, strel_diamond((2dilate_px + 1, 2dilate_px + 1)))) : m.meat for m in masks]
    unknown = [m.unknown for m in masks]
    xy = reduce(hcat, ([p[1], p[2]] for p in values(rig.map.pose)))
    lo = vec(minimum(xy; dims=2)); hi = vec(maximum(xy; dims=2))
    gc = VoxelGrid((lo[1], lo[2], 0.0), coarse, (ceil(Int, (hi[1] - lo[1]) / coarse), ceil(Int, (hi[2] - lo[2]) / coarse),
                   ceil(Int, zmax / coarse)))
    oc = largest_component(_carve(gc, projs, meat, unknown))
    any(oc) || error("the carving left nothing: check the masks")
    idx = findall(oc)
    imin = max.(Tuple(minimum(idx)) .- 1, 1) .- 1; imax = Tuple(maximum(idx)) .+ 1
    o = (gc.origin[1] + imin[1] * coarse, gc.origin[2] + imin[2] * coarse, 0.0)
    n = ntuple(d -> ceil(Int, (imax[d] - imin[d]) * coarse / h), 3)
    gf = VoxelGrid(o, h, n)
    return gf, largest_component(_carve(gf, projs, meat, unknown))
end
