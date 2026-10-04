# The meat in a photo: everything unlike the board (its colours clustered from the photo's border strip, the tag
# cards left out), white-balanced on the tags' white borders. Pixels on a tag's card are unknown unless clearly meat
# (the meat may hide part of a card, and pale fat reads like paper), so the carving neither keeps nor removes by them.
# Fat against lean is clustered over all the photos at once.

# the median RGB of the tags' white border rings (between each black square and the tag's outer edge)
function white_reference(rgb, cards, tag_mm)
    a = tag_mm / 2; cell = tag_mm / 8
    samples = Vector{Float64}[]
    for P in cards
        Hp = homography(tag_corners(tag_mm), P)
        for u in range(-a - 0.9cell, a + 0.9cell; length=40), v in (-a - 0.5cell, a + 0.5cell), (x, y) in ((u, v), (v, u))
            p = Hp * [x, y, 1.0]
            c, r = floor(Int, p[1] / p[3]) + 1, floor(Int, p[2] / p[3]) + 1
            1 <= r <= size(rgb, 2) && 1 <= c <= size(rgb, 3) && push!(samples, rgb[:, r, c])
        end
    end
    isempty(samples) && return ones(3)
    return [median(getindex.(samples, k)) for k in 1:3]
end

"""
    lab_image(img, cards, tag_mm)

The photo as an h × w × 3 CIELAB array, white-balanced so the tags' white borders read neutral (`cards` the detected
corner sets).
"""
function lab_image(img, cards, tag_mm)
    rgb = Float64.(channelview(RGB.(img)))                          # 3 × h × w
    wb = white_reference(rgb, cards, tag_mm)
    rgb ./= reshape(wb ./ maximum(wb), 3, 1, 1)
    clamp!(rgb, 0.0, 1.0)
    h, w = size(rgb, 2), size(rgb, 3)
    lab = Array{Float64}(undef, h, w, 3)
    Threads.@threads for c in 1:w
        @inbounds for r in 1:h
            L = convert(Lab, RGB(rgb[1, r, c], rgb[2, r, c], rgb[3, r, c]))
            lab[r, c, 1] = L.l; lab[r, c, 2] = L.a; lab[r, c, 3] = L.b
        end
    end
    return lab
end

# a tag's card: the polygon of the tag (its corners `P`) grown by `grow` (the card's white margin) in the image
function tag_polygon(P, tag_mm; grow=1.6)
    Hp = homography(tag_corners(tag_mm), P)
    Q = grow .* tag_corners(tag_mm)
    return reduce(vcat, (begin p = Hp * [Q[i, 1], Q[i, 2], 1.0]; [p[1] / p[3] p[2] / p[3]] end for i in 1:4))
end

function inpolygon(x, y, P)
    inside = false; n = size(P, 1); j = n
    for i in 1:n
        if (P[i, 2] > y) != (P[j, 2] > y) && x < (P[j, 1] - P[i, 1]) * (y - P[i, 2]) / (P[j, 2] - P[i, 2]) + P[i, 1]
            inside = !inside
        end
        j = i
    end
    return inside
end

# the pixels inside any of the card polygons
function card_mask(h, w, polys)
    m = falses(h, w)
    for P in polys
        c0, c1 = clamp.(extrema(P[:, 1]), 0, w); r0, r1 = clamp.(extrema(P[:, 2]), 0, h)
        for c in (floor(Int, c0) + 1):ceil(Int, c1), r in (floor(Int, r0) + 1):ceil(Int, r1)
            1 <= r <= h && 1 <= c <= w && inpolygon(c - 0.5, r - 0.5, P) && (m[r, c] = true)
        end
    end
    return m
end

const CARD_BITS = Dict{Int,Matrix{Bool}}()
card_bits(id) = get!(() -> collect(getAprilTagImage(id, AprilTags.tag36h11)) .> 0.5, CARD_BITS, id)   # true white

"""
    card_expected(det, tag_mm)

What a detected tag's card should look like at image point (x, y): `:black` (ink), `:white` (paper or a white cell),
`:edge` (within 0.2 of a cell of a black–white boundary, where blur mixes them).
"""
function card_expected(d::Detection, tag_mm)
    Hi = inv(homography(tag_corners(tag_mm), d.corners))
    bits = card_bits(d.id); cell = tag_mm / 8
    bit(u, v) = (i = floor(Int, (u + 5cell) / cell); j = floor(Int, (5cell - v) / cell);
                 (0 <= i <= 9 && 0 <= j <= 9) ? bits[j + 1, i + 1] : true)
    return function (x, y)
        p = Hi * [x, y, 1.0]; u, v = p[1] / p[3], p[2] / p[3]
        δ = 0.2cell
        b = bit(u, v)
        all(==(b), (bit(u + δ, v), bit(u - δ, v), bit(u, v + δ), bit(u, v - δ))) || return :edge
        return b ? :white : :black
    end
end

# Otsu's threshold on a vector
function otsu(v; nbins=256)
    lo, hi = extrema(v)
    edges = range(lo, hi; length=nbins + 1)
    counts = zeros(nbins)
    for x in v
        counts[clamp(floor(Int, (x - lo) / (hi - lo) * nbins) + 1, 1, nbins)] += 1
    end
    p = counts / sum(counts); mids = (edges[1:end-1] .+ edges[2:end]) ./ 2
    best, thr = -Inf, mids[1]; ω = 0.0; μ = 0.0; μT = sum(p .* mids)
    for k in 1:(nbins - 1)
        ω += p[k]; μ += p[k] * mids[k]
        (ω <= 0 || ω >= 1) && continue
        σb = (μT * ω - μ)^2 / (ω * (1 - ω))
        σb > best && (best = σb; thr = edges[k + 1])
    end
    return thr
end

# the pixels' Lab as an n × 3 matrix
labrows(lab, idx) = hcat((lab[:, :, k][idx] for k in 1:3)...)

"""
    PhotoMask(lab, meat, unknown)

One photo segmented: its CIELAB image, the meat and the pixels that say nothing (on a tag's card, not clearly meat).
"""
struct PhotoMask
    lab::Array{Float64,3}
    meat::BitMatrix
    unknown::BitMatrix
end

"""
    meat_mask(img, dets, tag_mm; border=0.03, nbg=2, grey=4.0, colour=8.0)

The meat in the photo: the board's colours clustered (k-means, `nbg` clusters) from the border strip off the cards,
each pixel's distance to the nearest cluster (Mahalanobis) thresholded by Otsu. On a card each pixel is checked
against the tag's own pattern (chroma from a 3 × 3 mean of a* and b*): a grey pixel (chroma below `grey`) where ink or paper is expected is the card, a
coloured one (chroma above `colour`) is meat in front, a bright one over a black cell is pale fat in front; a
faintly tinted one over paper is unknown (fat and paper look alike). Closed; the largest component is kept with any
other above a fifth of it, holes filled.
"""
function meat_mask(img, dets::Vector{Detection}, tag_mm; border=0.03, nbg=2, grey=4.0, colour=8.0, rng=MersenneTwister(1))
    cards = [d.corners for d in dets]
    lab = lab_image(img, cards, tag_mm)
    h, w = size(lab, 1), size(lab, 2)
    polys = [tag_polygon(P, tag_mm) for P in cards]
    oncard = card_mask(h, w, polys)
    bw = max(2, round(Int, border * min(h, w)))
    edge = falses(h, w); edge[1:bw, :] .= true; edge[end-bw+1:end, :] .= true; edge[:, 1:bw] .= true; edge[:, end-bw+1:end] .= true
    X = labrows(lab, findall(edge .& .!oncard))
    X = X[randperm(rng, size(X, 1))[1:min(20000, size(X, 1))], :]
    km = kmeans(Matrix(X'), nbg; rng)
    ctr = NTuple{3,Float64}[]; invs = Matrix{Float64}[]
    for k in 1:nbg
        Xk = X[km.assignments .== k, :]
        size(Xk, 1) < 10 && continue
        push!(ctr, Tuple(vec(mean(Xk; dims=1)))); push!(invs, inv(cov(Xk) + 4.0I))   # a floor of 2 units per axis
    end
    d = Matrix{Float64}(undef, h, w)
    Threads.@threads for c in 1:w
        @inbounds for r in 1:h
            best = Inf
            for k in eachindex(ctr)
                δ1 = lab[r, c, 1] - ctr[k][1]; δ2 = lab[r, c, 2] - ctr[k][2]; δ3 = lab[r, c, 3] - ctr[k][3]
                M = invs[k]
                q = M[1, 1] * δ1^2 + M[2, 2] * δ2^2 + M[3, 3] * δ3^2 + 2(M[1, 2] * δ1 * δ2 + M[1, 3] * δ1 * δ3 + M[2, 3] * δ2 * δ3)
                best = min(best, q)
            end
            d[r, c] = log1p(sqrt(max(best, 0.0)))
        end
    end
    fg = d .> otsu(vec(d[.!oncard]))
    # the cards: each pixel against its tag's pattern
    expect = [card_expected(dd, tag_mm) for dd in dets]
    cardlike = falses(h, w); ambiguous = falses(h, w)
    # the chroma of a 3 × 3 mean of a* and b* (a pixel's own noise reads as a tint on paper)
    box(A, r, c) = (s = 0.0; for dr in -1:1, dc in -1:1; s += A[clamp(r + dr, 1, h), clamp(c + dc, 1, w)]; end; s / 9)
    for I in findall(oncard .& fg)
        r, c = Tuple(I)
        k = findfirst(P -> inpolygon(c - 0.5, r - 0.5, P), polys)
        e = expect[k](c - 0.5, r - 0.5)
        L = lab[r, c, 1]; ch = hypot(box(view(lab, :, :, 2), r, c), box(view(lab, :, :, 3), r, c))
        ch >= colour && continue                                    # coloured: meat in front
        if e === :black
            L < 50 && (cardlike[I] = true)                          # ink (bright over ink: pale fat in front)
        elseif e === :edge
            cardlike[I] = true                                      # grey at a cell edge: the blur of ink and paper
        else
            ch < grey ? (cardlike[I] = true) : (ambiguous[I] = true)
        end
    end
    meat = fg .& .!cardlike .& .!ambiguous
    meat = closing(meat, strel_diamond((7, 7)))
    labels = label_components(meat)
    sizes = zeros(Int, maximum(labels))
    @inbounds for l in labels
        l > 0 && (sizes[l] += 1)
    end
    isempty(sizes) && error("no meat found in the photo")
    keep = Set(findall(>=(0.2maximum(sizes)), sizes))
    meat = BitMatrix(map(l -> l in keep, labels))
    meat = fill_holes(meat)
    return PhotoMask(lab, meat, ambiguous .& .!meat)
end

function fill_holes(mask)
    bg = label_components(.!mask)
    touching = Set{Int}()
    h, w = size(mask)
    for r in 1:h, c in (1, w); bg[r, c] > 0 && push!(touching, bg[r, c]); end
    for c in 1:w, r in (1, h); bg[r, c] > 0 && push!(touching, bg[r, c]); end
    out = copy(mask)
    @inbounds for i in eachindex(bg)
        bg[i] > 0 && !(bg[i] in touching) && (out[i] = true)
    end
    return out
end

"""
    fat_classifier(masks; rng)

Fat against lean over all the photos: two k-means clusters of the pooled meat colours, the fat the one with the
higher L* − a* (pale, not red). Returns a function of a pixel's (L*, a*, b*) giving true for fat.
"""
function fat_classifier(masks::Vector{PhotoMask}; rng=MersenneTwister(2), per_photo=20000)
    X = reduce(vcat, begin
        idx = findall(m.meat)
        labrows(m.lab, idx[randperm(rng, length(idx))[1:min(per_photo, length(idx))]])
    end for m in masks)
    km = kmeans(Matrix(X'), 2; rng)
    C = km.centers
    kf = argmax([C[1, k] - C[2, k] for k in 1:2])
    cf, cl = C[:, kf], C[:, 3 - kf]
    abs(cf[1] - cl[1]) + abs(cf[2] - cl[2]) < 15 && @warn "the meat's colours barely separate: no fat, or no lean, in view"
    return (L, a, b) -> (L - cf[1])^2 + (a - cf[2])^2 + (b - cf[3])^2 < (L - cl[1])^2 + (a - cl[2])^2 + (b - cl[3])^2
end
