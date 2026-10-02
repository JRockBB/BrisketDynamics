# Synthetic photos for the tests: a pinhole camera ray-casting a known shape (a signed distance function; fat its top
# `cap` millimetres measured vertically, lean below) on a blue board with printed tags lying around it, from views
# walking around the meat. World frame: the board z = 0, z up, millimetres.

using AprilTags: getAprilTagImage, tag36h11
using Colors: RGB
using ImageCore: N0f8
using LinearAlgebra
using Random
using StaticArrays

const V3 = SVector{3,Float64}

tag_bits(id) = Float64.(collect(getAprilTagImage(id, tag36h11)))       # 10 × 10, the black square the inner 8

"""
    Shape(sdf, cap, centre, radius)

A meat on the board: `sdf(p)` negative inside (a bound on the distance outside), the fat cap `cap` [mm] a layer under
every upward-facing surface (a point is fat when an up-facing surface, its normal more than 20° above horizontal,
lies within `cap` of it), a bounding sphere.
"""
struct Shape{F}
    sdf::F
    cap::Float64
    centre::V3
    radius::Float64
end

isinside(s::Shape, p) = s.sdf(p) <= 0

# upward-leaning directions (z ≥ 0.35) to search for an up-facing surface within the cap
const UPDIRS = [normalize(V3(cos(φ) * sin(θ), sin(φ) * sin(θ), cos(θ))) for θ in (0.0, 0.45, 0.9, 1.2) for φ in range(0, 2π; length=(θ == 0 ? 2 : 13))[1:end-1]]

function isfat(s::Shape, p)
    (isinside(s, p) && s.cap > 0) || return false
    for u in UPDIRS
        t = 0.0
        while t <= s.cap
            q = p + t * u
            if s.sdf(q) > 0
                normal(s, q)[3] > 0.35 && return true
                break
            end
            t += 0.25
        end
    end
    return false
end

# ---- the shapes ---------------------------------------------------------------------------------------------------

# a trimmed flat: tapering from its thick end at `O` along `e` (horizontal), its convex polyhedron's plane distances
function flat_shape(; O=V3(0, 0, 0), θ=0.26, L=330.0, w0=230.0, w1=180.0, h0=55.0, h1=25.0, cap=6.0)
    e = V3(cos(θ), sin(θ), 0); ey = V3(-sin(θ), cos(θ), 0)
    kh = (h1 - h0) / L; kb = (w1 - w0) / (2L); b0 = w0 / 2
    function sdf(p)
        q = p - O
        s, y, z = dot(q, e), dot(q, ey), q[3]
        return max(-z, (z - h0 - kh * s) / sqrt(1 + kh^2), -s, s - L, (y - b0 - kb * s) / sqrt(1 + kb^2),
                   (-y - b0 - kb * s) / sqrt(1 + kb^2))
    end
    c = O + e * L / 2 + V3(0, 0, h0 / 2)
    return Shape(sdf, cap, c, hypot(L / 2, b0, h0))
end

# an ellipsoid's distance (the usual bound, exact on the axes)
ellipsoid(p, r) = (k0 = norm(p ./ r); k1 = norm(p ./ (r .* r)); k0 * (k0 - 1) / k1)
smin(a, b, k) = (h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0); b * (1 - h) + a * h - k * h * (1 - h))

# a pork butt: a lumpy block of three blended ellipsoids, cut flat on the board
function butt_shape(; O=V3(0, 0, 0), cap=8.0)
    function sdf(p)
        q = p - O
        d = ellipsoid(q - V3(0, 0, 40), V3(105, 80, 75))
        d = smin(d, ellipsoid(q - V3(55, 25, 35), V3(60, 55, 60)), 25.0)
        d = smin(d, ellipsoid(q - V3(-60, -20, 30), V3(55, 60, 50)), 25.0)
        return max(d, -q[3])
    end
    return Shape(sdf, cap, O + V3(0, 0, 50), 170.0)
end

# a rack of ribs: a rounded slab on the board, its top crowned along its length (`arch` lifts the whole slab instead,
# leaving a hollow under its middle no photo can see: the visual hull's concavity limit)
function ribs_shape(; O=V3(0, 0, 0), L=300.0, W=150.0, T=32.0, cap=3.0, crown=6.0, arch=0.0)
    function sdf(p)
        q = p - O
        bump = 1 - (2q[1] / L)^2
        lift = arch * max(bump, 0.0)
        Tq = T + crown * max(bump, 0.0)                              # thicker in the middle, flat underneath
        r = 8.0
        zc = q[3] - lift
        d3 = abs.(V3(q[1], q[2], zc - Tq / 2)) - V3(L / 2 - r, W / 2 - r, Tq / 2 - r)
        d = norm(max.(d3, 0.0)) + min(maximum(d3), 0.0) - r
        return max(d, -q[3])
    end
    return Shape(sdf, cap, O + V3(0, 0, T / 2), hypot(L / 2, W / 2, T))
end

# a sphere resting on the board (the checks' exact shape)
sphere_shape(; O=V3(0, 0, 0), R=60.0, cap=0.0) = Shape(p -> max(norm(p - (O + V3(0, 0, R))) - R, -p[3]), cap, O + V3(0, 0, R), R)

# ---- the tags -----------------------------------------------------------------------------------------------------

struct Card
    Q::V3; ux::V3; uy::V3       # the tag's centre, right and up (lying on the board)
    a::Float64                  # the black square's side
    p::Float64                  # the paper's half size
    bits::Matrix{Float64}
    id::Int
end
lying_card(x, y, θ, a, id; margin=12.0) =
    Card(V3(x, y, 0.05), V3(cos(θ), sin(θ), 0), V3(-sin(θ), cos(θ), 0), a, 5a / 8 + margin, tag_bits(id), id)

function tag_colour(c::Card, u, v)
    cell = c.a / 8
    i = floor(Int, (u + 5cell) / cell); j = floor(Int, (5cell - v) / cell)
    (0 <= i <= 9 && 0 <= j <= 9) || return 0.97
    return c.bits[j + 1, i + 1] > 0.5 ? 0.97 : 0.04
end

# a ring of `n` tags around the meat (ids 0 … n − 1), each turned a little
tag_ring(centre, r, n, a) = [lying_card(centre[1] + r * cos(2π * k / n + 0.3), centre[2] + r * sin(2π * k / n + 0.3),
                                        2π * k / n + 0.7 * k, a, k) for k in 0:(n - 1)]

# ---- the camera ---------------------------------------------------------------------------------------------------

struct View
    C::V3
    R::SMatrix{3,3,Float64,9}   # world → camera
    f::Float64
    width::Int
    height::Int
end

function look_at(C, target; f=1400.0, width=1600, height=1200, up=V3(0, 0, 1))
    zc = normalize(target - C)
    xc = normalize(cross(zc, up)); yc = cross(zc, xc)
    return View(C, SMatrix{3,3}(hcat(xc, yc, zc)'), f, width, height)
end

# views walking around `centre` at distance `d`, one ring per (count, elevation [rad]), staggered, plus one overhead
function views_around(centre, rings; d=700.0, f=1400.0, width=1600, height=1200)
    vs = View[]
    for (m, (n, el)) in enumerate(rings), k in 0:(n - 1)
        φ = 2π * (k + 0.5(m - 1)) / n
        C = centre + d * V3(cos(el) * cos(φ), cos(el) * sin(φ), sin(el))
        push!(vs, look_at(C, centre; f, width, height))
    end
    push!(vs, look_at(centre + V3(5, -10, d), centre; f, width, height, up=V3(1, 0, 0)))
    return vs
end

# the world point's pixel in the view (continuous, the detector's convention)
function project(v::View, X)
    p = v.R * (X - v.C)
    return v.f * p[1] / p[3] + v.width / 2, v.f * p[2] / p[3] + v.height / 2
end

# ---- the renderer -------------------------------------------------------------------------------------------------

const BLUE = V3(0.16, 0.34, 0.66)
const FAT = V3(0.93, 0.89, 0.80)
const LEAN = V3(0.62, 0.17, 0.19)
const LIGHT = normalize(V3(0.3, -0.4, 1.0))

# the meat's texture (grain and marbling): a sum of plane waves of 2–12 mm wavelength, fixed in the world, so the
# photos agree where they see the same point
const WAVES = let rng = MersenneTwister(11)
    [(normalize(V3(randn(rng), randn(rng), randn(rng))) * (2π / (2 + 10rand(rng))), 2π * rand(rng)) for _ in 1:12]
end
texture(P) = 1 + 0.18 * sum(sin(dot(w, P) + φ) for (w, φ) in WAVES) / sqrt(length(WAVES) / 2)

# sphere tracing inside the bounding sphere; the hit distance or Inf
function trace(s::Shape, o, d)
    oc = o - s.centre
    b = dot(oc, d); c = dot(oc, oc) - s.radius^2
    disc = b^2 - c
    disc < 0 && return Inf
    t = max(0.0, -b - sqrt(disc)); t1 = -b + sqrt(disc)
    for _ in 1:400
        t > t1 && return Inf
        dist = s.sdf(o + t * d)
        dist < 0.02 && return t
        t += max(0.8dist, 0.01)
    end
    return Inf
end

function normal(s::Shape, p; ε=0.2)
    g = V3(s.sdf(p + V3(ε, 0, 0)) - s.sdf(p - V3(ε, 0, 0)), s.sdf(p + V3(0, ε, 0)) - s.sdf(p - V3(0, ε, 0)),
           s.sdf(p + V3(0, 0, ε)) - s.sdf(p - V3(0, 0, ε)))
    return normalize(g)
end

"""
    render(v, shape, cards; noise=0.012, rng)

The photo: the shape (fat or lean at the hit, lit), the cards in front of the board, the board.
"""
function render(v::View, s::Shape, cards::Vector{Card}; noise=0.012, rng=MersenneTwister(3))
    img = Matrix{RGB{N0f8}}(undef, v.height, v.width)
    Rt = v.R'
    seeds = rand(rng, UInt, v.height)
    Threads.@threads for r in 1:v.height
        lr = MersenneTwister(seeds[r])
        for c in 1:v.width
            d = normalize(Rt * V3((c - 0.5 - v.width / 2) / v.f, (r - 0.5 - v.height / 2) / v.f, 1.0))
            best = Inf; col = BLUE; n = V3(0, 0, 1)
            t = trace(s, v.C, d)
            if t < best
                P = v.C + t * d; n = normal(s, P); best = t
                col = (isfat(s, P - 0.3n) ? FAT : LEAN) * texture(P)
            end
            if d[3] < 0
                tb = -v.C[3] / d[3]
                if tb < best
                    P = v.C + tb * d; col = BLUE; n = V3(0, 0, 1); best = tb
                    for card in cards
                        u = dot(P - card.Q, card.ux); w = dot(P - card.Q, card.uy)
                        if abs(u) <= card.p && abs(w) <= card.p
                            g = tag_colour(card, u, w); col = V3(g, g, g)
                        end
                    end
                end
            end
            shade = 0.72 + 0.28 * abs(dot(n, LIGHT))
            px = col * shade + noise * V3(randn(lr), randn(lr), randn(lr))
            img[r, c] = RGB{N0f8}(clamp(px[1], 0, 1), clamp(px[2], 0, 1), clamp(px[3], 0, 1))
        end
    end
    return img
end

# the meat's footprint radius about its centre on the board (sampled)
function footprint_radius(s::Shape)
    r = 0.0
    for φ in range(0, 2π; length=73), z in (0.5, 10.0, 30.0)
        ρ = s.radius
        while ρ > 0 && !isinside(s, V3(s.centre[1] + ρ * cos(φ), s.centre[2] + ρ * sin(φ), z))
            ρ -= 1.0
        end
        r = max(r, ρ)
    end
    return r
end

"""
    scene_photos(shape; rings, ntags=6, tag_mm=120.0, ring, dir, d, kwargs...)

Photos of the shape from rings of views around it (low and high, `(count, elevation)`) plus one overhead, with `ntags` tags in a ring on the board clear of
the meat (`ring` by default its footprint plus a card), the cameras `d` out (2.5 rings); the views, the cards, the paths.
"""
function scene_photos(s::Shape; rings=((8, deg2rad(18)), (6, deg2rad(50))), ntags=6, tag_mm=120.0, ring=nothing,
                      dir=mktempdir(), d=nothing, f=1400.0, width=1600, height=1200)
    c = V3(s.centre[1], s.centre[2], 0)
    ring = something(ring, footprint_radius(s) + 5tag_mm / 8 + 40)    # the cards clear of the meat
    d = something(d, 2.5ring)
    cards = tag_ring(c, ring, ntags, tag_mm)
    vs = views_around(c + V3(0, 0, 20), rings; d, f, width, height)
    paths = String[]
    for (k, v) in enumerate(vs)
        p = joinpath(dir, "view$(lpad(k, 2, '0')).png")
        save(p, render(v, s, cards))
        push!(paths, p)
    end
    return (; views=vs, cards, paths)
end
