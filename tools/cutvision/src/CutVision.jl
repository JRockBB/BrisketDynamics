"""
    CutVision

The cut's shape from photos: printed AprilTags on the board calibrate every photo (the tag map, the cameras and the
focal length in one least-squares problem), the meat's silhouettes carve its visual hull on a voxel grid, the photos'
colours label its surface fat or lean, and per-meat rules place the cap's depth and the bone (docs/measure_cut.md).
"""
module CutVision

using AprilTags
using Clustering: kmeans
using Colors
using FileIO
using ForwardDiff
using ImageCore: channelview, N0f8
using ImageMorphology: closing, dilate, strel_diamond, label_components
using JSON3
using LinearAlgebra
using Random
using Statistics

export calibrate, reconstruct, render_voxels, write_shape, read_shape, write_obj, exif_focal_px, volume_l, mass_kg, PROFILES

include("camera.jl")
include("bundle.jl")
include("segment.jl")
include("carve.jl")
include("stereo.jl")
include("labels.jl")
include("export.jl")
include("view.jl")

end
