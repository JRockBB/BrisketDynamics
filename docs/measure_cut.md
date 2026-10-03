# Measuring a cut from photos

`tools/cutvision` reconstructs a raw cut's shape from about two dozen phone photos. Its output is the cut's 3-D
labels: lean, fat and, by the meat's template, bone. The 3-D shadow then cooks that shape
(`scripts/run_shadow3d.py --shape <name>`). It handles brisket (flat or packer), pork butt, ribs and chuck.

## What you need

- **Six printed tags.** Run `julia --project=tools/cutvision tools/cutvision/make_tags.jl --out tags` to write six
  US-letter SVGs, ids 0–5, each with a 150 mm black square.
  - Print at **actual size**, not fit-to-page, and check the 100 mm bar on each page with a ruler.
  - Then measure one black square: that number is `--tag-mm`.
  - The tags must lie flat. Tape a curled page down.
- **A plain board** in one colour unlike meat and paper. Blue or green is best. It must be large enough to hold the
  cut and the six tags around it, about 70 × 70 cm for a flat.
- **The raw cut, before the rub.** The colours separate fat from lean; a rub hides them.
- **Optionally, a kitchen scale**, to check the reconstruction's mass.

## Laying it out

Put the cut in the middle of the board, fat side up. Lay the six tags around it in a ring, every tag clear of the
meat by a few centimetres. Tag 0 is the reference: the reconstruction's frame is its own, so lay it flat like the
rest.

## The photos (about 23)

The photos walk around the cut in three rings, plus one from above. Every photo must show the whole cut and at least
one tag (two or more is better). Keep the phone's zoom fixed for the whole set.

1. **Board level, 8 photos.** Hold the phone low, about 10 cm above the board, a step back from the tags, and turn a
   little around the cut between shots. These photos fix the thickness. Without them, a flat cut comes out 30 % too
   thick.
2. **Low, 8 photos at about 20°.** As before, the phone higher, at chest height of someone kneeling.
3. **High, 6 photos at about 50°.** Standing, looking down at the cut.
4. **One from straight above.**

Shoot in even light without hard shadows, and turn the flash off. The photos must be JPEGs (`.jpg`) so their EXIF
carries the focal length; most phones convert HEIC on export. If yours does not, give `--focal-px`.

## Running it

```
julia -t auto --project=tools/cutvision tools/cutvision/reconstruct.jl --meat brisket --tag-mm 150 \
      --out data/cut/<name> photos/*.jpg [--cap-mm 6] [--weight-kg 2.9]
```

This writes:
- `data/cut/<name>/shape.json`, the labels
- `surface.obj`
- `reconstruction.png`: the cut whole, and cut through its middle

It prints the calibration's error, the volume, the mass, and where the cap's depth came from.

**The cap's depth.** It is read from the photos where a trimmed side shows the fat band under the top. On synthetic
cuts that is within a millimetre on rounded edges, but about 2 mm thin on a sharp trimmed edge. If you can measure
it on the cut, with a probe or a ruler at a trimmed edge, give `--cap-mm`: the measurement overrides the photos,
which are still recorded.

**Bone.** Bone isn't visible, so it comes from the meat's template:
- ribs: a row of rods across the rack, on the side given by `--bone-side`, down by default
- pork butt: a blade plate at the middle; `--no-bone` for a boneless butt

## Then cook it

```
cd app
uv run python ../scripts/run_shadow3d.py --shape <name>      # the report: data/control/shadow3d_<name>.json
uv run python ../scripts/plot_shape3d.py --shape <name>      # the figures: figures/shadow3d/<name>/
```

## What it can't see

- **Hollows the camera can't look into.** A rack arched off the board is filled in underneath.
- **The fat inside the meat.** Seams in a chuck and marbling are the meat preset's composition, not geometry.
- **Bone.** It is the template.

On synthetic photos (`tools/cutvision/test/runtests.jl`):

| Measure | Result |
|---|---|
| Tags | within 0.2 mm |
| Focal length | within 0.1 % |
| Volume | within 2.4 % |
| Shape (IoU) | 0.97–0.99 |
| Mean surface error | under 0.9 mm |
| Interior fat/lean correct | 95–99 % |
