# The camera: a pinhole with square pixels and its principal point at the image centre (a phone's main camera,
# its distortion corrected in the phone), the focal length from the photo's EXIF or given. The tag fixes the
# camera's pose in the tag's frame: X along the tag's right edge, Y up the tag, Z out of the tag toward the camera,
# millimetres, the origin at the tag's centre. Image coordinates are continuous with pixel (row r, col c) spanning
# [c − 1, c] × [r − 1, r] (the AprilTag detector's convention), so a pixel's centre is (c − ½, r − ½).

"""
    Camera(K, R, C)

The intrinsics `K` (3×3), the rotation `R` taking the tag frame to the camera frame and the camera's centre `C`
in the tag frame [mm].
"""
struct Camera
    K::Matrix{Float64}
    R::Matrix{Float64}
    C::Vector{Float64}
end

intrinsics(f, width, height) = [f 0.0 width / 2; 0.0 f height / 2; 0.0 0.0 1.0]

# the direct linear transform with Hartley normalization: H maps src (plane) to dst (image), both n × 2, n ≥ 4
function homography(src::AbstractMatrix, dst::AbstractMatrix)
    function normalizer(P)
        c = vec(mean(P; dims=1)); s = sqrt(2) / mean(norm.(eachrow(P .- c')))
        return [s 0 -s * c[1]; 0 s -s * c[2]; 0 0 1]
    end
    Ts, Td = normalizer(src), normalizer(dst)
    n = size(src, 1)
    A = zeros(2n, 9)
    for i in 1:n
        x, y, _ = Ts * [src[i, 1], src[i, 2], 1.0]
        u, v, _ = Td * [dst[i, 1], dst[i, 2], 1.0]
        A[2i - 1, :] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        A[2i, :] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
    end
    h = svd(A; full=true).V[:, end]                                 # the null vector (A is 8 × 9 for four points)
    H = Td \ (reshape(h, 3, 3)' * Ts)
    return H / H[3, 3]
end

"""
    pose(H, K)

The camera from the homography `H` mapping the tag plane (X, Y) [mm] to the image: the rotation's columns are the
normalized columns of K⁻¹H, made orthonormal, the translation the third; the camera in front of the tag.
"""
function pose(H, K)
    M = K \ H
    λ = 2 / (norm(M[:, 1]) + norm(M[:, 2]))
    M[3, 3] * λ < 0 && (λ = -λ)                                     # the tag in front of the camera
    r1, r2, t = λ * M[:, 1], λ * M[:, 2], λ * M[:, 3]
    U, _, V = svd(hcat(r1, r2, cross(r1, r2)))
    R = U * V'
    return Camera(K, R, -R' * t)
end

# the ray through image point (x, y) in the tag frame: the camera's centre and a direction
ray(cam::Camera, x, y) = cam.R' * (cam.K \ [x, y, 1.0])

# where the ray through (x, y) meets the plane n·X = c (tag frame)
function backproject(cam::Camera, x, y, n, c)
    d = ray(cam, x, y)
    μ = (c - dot(n, cam.C)) / dot(n, d)
    return cam.C + μ * d
end

# the tag's corners in its own frame, in the detector's order (counter-clockwise from the bottom left)
tag_corners(a) = [-a/2 -a/2; a/2 -a/2; a/2 a/2; -a/2 a/2]

"""
    exif_focal_px(path, width, height)

The focal length in pixels from a JPEG's EXIF: the 35 mm-equivalent focal length scaled by the image's diagonal
(the 35 mm frame's is 43.27 mm), or `nothing` when the file carries none.
"""
function exif_focal_px(path, width, height)
    f35 = exif_focal35(path)
    f35 === nothing && return nothing
    return f35 * hypot(width, height) / hypot(36.0, 24.0)
end

function exif_focal35(path)
    bytes = read(path)
    length(bytes) > 4 && bytes[1] == 0xFF && bytes[2] == 0xD8 || return nothing
    i = 3
    while i + 3 <= length(bytes)
        bytes[i] == 0xFF || return nothing
        marker = bytes[i + 1]
        len = Int(bytes[i + 2]) << 8 | bytes[i + 3]
        if marker == 0xE1 && String(bytes[(i + 4):(i + 7)]) == "Exif"
            return _tiff_focal35(view(bytes, (i + 10):(i + 1 + len)))
        end
        marker == 0xDA && return nothing                             # the image data: no EXIF before it
        i += 2 + len
    end
    return nothing
end

function _tiff_focal35(t)
    le = t[1] == UInt8('I')
    u16(o) = le ? Int(t[o + 1]) | Int(t[o + 2]) << 8 : Int(t[o + 1]) << 8 | Int(t[o + 2])
    u32(o) = le ? Int(t[o + 1]) | Int(t[o + 2]) << 8 | Int(t[o + 3]) << 16 | Int(t[o + 4]) << 24 :
             Int(t[o + 1]) << 24 | Int(t[o + 2]) << 16 | Int(t[o + 3]) << 8 | Int(t[o + 4])
    function entries(ifd)
        n = u16(ifd)
        return [(u16(ifd + 2 + 12k), ifd + 2 + 12k) for k in 0:(n - 1)]
    end
    for (tag, o) in entries(u32(4))
        tag == 0x8769 || continue                                    # the Exif sub-IFD
        for (tg, oo) in entries(u32(o + 8))
            tg == 0xA405 && return Float64(u16(oo + 8))              # FocalLengthIn35mmFilm
        end
    end
    return nothing
end
