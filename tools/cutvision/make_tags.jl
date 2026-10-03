# Printable tags for the photos: one tag36h11 per US-letter page, its black square `--tag-mm` across (150 mm by
# default), with its id and a 100 mm bar to check the printer's scale. SVGs in exact millimetres: print at 100 %
# ("actual size", no fit-to-page), then measure the black square with a ruler and pass that to reconstruct.jl.
#
#   julia --project=tools/cutvision tools/cutvision/make_tags.jl [--tag-mm 150] [--ids 0:5] [--out tags]

using AprilTags: getAprilTagImage, tag36h11

function tag_svg(id, a)
    bits = collect(getAprilTagImage(id, tag36h11)) .> 0.5          # 10 × 10, true white
    W, H = 215.9, 279.4
    cell = a / 8; full = 10cell
    x0 = (W - full) / 2; y0 = 30.0
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$(W)mm" height="$(H)mm" viewBox="0 0 $W $H">""")
    println(io, """<rect x="0" y="0" width="$W" height="$H" fill="white"/>""")
    for j in 1:10, i in 1:10
        bits[j, i] && continue
        println(io, """<rect x="$(x0 + (i - 1) * cell)" y="$(y0 + (j - 1) * cell)" width="$(cell + 0.01)" height="$(cell + 0.01)" fill="black"/>""")
    end
    yb = y0 + full + 18
    println(io, """<rect x="$((W - 100) / 2)" y="$yb" width="100" height="2" fill="black"/>""")
    println(io, """<text x="$(W / 2)" y="$(yb + 9)" font-family="sans-serif" font-size="5" text-anchor="middle">100 mm: print at actual size and check this bar</text>""")
    println(io, """<text x="$(W / 2)" y="$(yb + 20)" font-family="sans-serif" font-size="7" text-anchor="middle">tag36h11 id $id · black square $(a) mm (measure it)</text>""")
    println(io, """<text x="$(W / 2)" y="$(yb + 29)" font-family="sans-serif" font-size="5" text-anchor="middle">tag 0 marks the reference: lay it flat on the board</text>""")
    println(io, "</svg>")
    return String(take!(io))
end

function main(args)
    a = 150.0; ids = 0:5; out = "tags"
    i = 1
    while i <= length(args)
        args[i] == "--tag-mm" && (a = parse(Float64, args[i + 1]))
        args[i] == "--ids" && (r = split(args[i + 1], ":"); ids = parse(Int, r[1]):parse(Int, r[end]))
        args[i] == "--out" && (out = args[i + 1])
        i += 2
    end
    10a / 8 <= 215.9 - 20 || error("a $(a) mm tag does not fit a letter page with its white border")
    mkpath(out)
    for id in ids
        p = joinpath(out, "tag36h11_$(lpad(id, 2, '0')).svg")
        write(p, tag_svg(id, a)); println("wrote ", p)
    end
end

main(ARGS)
