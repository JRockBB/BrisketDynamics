# The reconstruction against synthetic photos of known cuts (test/synth.jl: ray-cast scenes, tags in a ring on the
# board, three rings of views and one overhead), gated:
#   calibration  the tag map within 0.5 mm and 0.2°, the focal length within 1 %, the reprojection rms under 0.5 px
#   shape        volume within 3 %, IoU above 0.95, mean surface error under 1.5 mm (a flat; a pork butt)
#   labels       interior fat/lean right above 94 %; the cap read within 1 mm on the butt and fed through when given
#   tools        the run-length encoding round-trips; the EXIF reader finds a 35 mm focal length

using CutVision
using FileIO
using LinearAlgebra
using Statistics
using Test
const CV = CutVision
include(joinpath(@__DIR__, "synth.jl"))

const RINGS = ((8, deg2rad(8)), (8, deg2rad(18)), (6, deg2rad(50)))
const TAG = 150.0

function score(s::Shape, r)
    c = V3(s.centre[1], s.centre[2], 0)
    c0 = tag_ring(c, footprint_radius(s) + 5TAG / 8 + 40, 6, TAG)[1]
    world(x, y, z) = c0.Q + x * c0.ux + y * c0.uy + V3(0, 0, z)
    g = r.grid; L = r.labels
    inter = 0; uni = 0; nt = 0; ok = 0; n = 0
    for I in CartesianIndices(L)
        X = world(CV.cell_centre(g, Tuple(I)...)...)
        t = isinside(s, X); o = L[I] != 0
        inter += t & o; uni += t | o; nt += t
        if t & o
            n += 1; ok += (L[I] == CV.FAT) == isfat(s, X)
        end
    end
    S = CV.surface_voxels(L .!= 0)
    err = mean(abs(s.sdf(world(CV.cell_centre(g, Tuple(I)...)...))) for I in S)
    return (; dvol=CV.volume_l(r) / (nt * g.h^3 * 1e-6) - 1, iou=inter / uni, err, labels=ok / n, c0)
end

@testset "cutvision: the encoding and EXIF" begin
    A = rand(Int8[0, 0, 0, 1, 2, 3], 7, 5, 4)
    @test CV.unrle(CV.rle(A), size(A)) == A
    # a minimal JPEG: SOI, an APP1 Exif segment with FocalLengthIn35mmFilm = 26, then SOS
    tiff = UInt8[0x49, 0x49, 0x2a, 0x00, 0x08, 0x00, 0x00, 0x00,
                 0x01, 0x00, 0x69, 0x87, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                 0x01, 0x00, 0x05, 0xa4, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    seg = vcat(UInt8['E', 'x', 'i', 'f', 0, 0], tiff)
    jpg = vcat(UInt8[0xff, 0xd8, 0xff, 0xe1], UInt8[(length(seg) + 2) >> 8, (length(seg) + 2) & 0xff], seg, UInt8[0xff, 0xda])
    p = tempname() * ".jpg"; write(p, jpg)
    @test CV.exif_focal35(p) == 26.0
    @test exif_focal_px(p, 4000, 3000) ≈ 26 * 5000 / hypot(36, 24)
end

@testset "cutvision: a flat and a pork butt from synthetic photos" begin
    for (s, meat, cap_given) in ((flat_shape(), :brisket, 6.0), (butt_shape(), :pork_butt, nothing))
        dir = mktempdir()
        sc = scene_photos(s; dir, rings=RINGS, tag_mm=TAG)
        r = reconstruct(sc.paths; tag_mm=TAG, meat, f0=1300.0, bone=false, cap_mm=cap_given, verbose=false)
        # the calibration against the cards' truth, in tag 0's frame
        m = score(s, r); c0 = m.c0
        errs = map(sc.cards) do card
            q = card.Q - c0.Q
            e = r.rig.map.pose[card.id]
            (hypot(e[1] - dot(q, c0.ux), e[2] - dot(q, c0.uy)),
             abs(rad2deg(rem2pi(e[3] - atan(dot(card.ux, c0.uy), dot(card.ux, c0.ux)), RoundNearest))))
        end
        @test maximum(first.(errs)) < 0.5 && maximum(last.(errs)) < 0.2
        @test abs(r.rig.f / 1400 - 1) < 0.01 && r.rig.rms < 0.5
        @test abs(m.dvol) < 0.03 && m.iou > 0.95 && m.err < 1.5
        @test m.labels > 0.94
        if cap_given === nothing
            @test abs(r.cap_mm - s.cap) < 1.0                     # the butt's cap read from the photos
        else
            @test r.cap_mm == cap_given && startswith(r.cap_source, "given")
        end
        println("  $(meat): volume $(round(100m.dvol; digits=2)) %, IoU $(round(m.iou; digits=4)), surface ",
                "$(round(m.err; digits=2)) mm, labels $(round(100m.labels; digits=1)) %, cap $(round(r.cap_mm; digits=2)) mm")
    end
end
