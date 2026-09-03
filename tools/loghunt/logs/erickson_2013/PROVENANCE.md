# Provenance: erickson_2013.csv

## Source
- **Post:** user **Alan Erickson**, The Virtual Weber Bulletin Board,
  "Post your HeaterMeter graphs" thread, Jul 8 2013.
  https://tvwbb.com/threads/post-your-heatermeter-graphs.42779/post-446870
- **Image:** HeaterMeter chart (photobucket-hosted, served through a
  semi-transparent watermark overlay; no unwatermarked archive copy
  exists), retrieved 2026-09-04 (cached locally as `cache/add2ebab.jpg`,
  not committed). Forum content (c) its author; small factual data
  extraction with attribution.

## Cook metadata (from post text, paraphrased)
- ~**12 lb brisket**, overnight on a **Char-Griller Kamado** (Akorn
  class), Royal Oak lump (15 h on one load), stock Auber fan capped at
  30 %, top vent at 0.75.
- Started ~**250 F**, ramped to ~**275 F** mid-cook "to push through the
  stall" (the setpoint ramp is visible in the red trace and in the pit).
- **WRAPPED IN BUTCHER PAPER "after around 8 hours"** — the author: "you
  can see the spike on the graph when I took it off to wrap it."
- This is the repo's **wrap log** (roadmap item 6): a documented wrap
  moment makes the wrap a known-input step-response experiment for
  eta_evap / eps_wrap (`scripts/run_wrap_fit.py`).

## Chart facts
- 713x360 JPEG; clock labels 22:42 -> 12:42 (2-h steps, 82.9 px); left
  axis 0-550 F (30.64 px / 50 F). The late 275 F setpoint segment reads
  278 F under this calibration (cross-check); no ambient trace.
- Lid-open bands (light columns; hours after 22:42): 0.6 (early check),
  5.0 and 5.7 (night checks), **8.75 and 9.15 — the wrap pair** (off to
  wrap, back on wrapped; "around 8 hours" per the post), 15.24 (pull;
  "around 15 hours" per the post). t_wrap is set between the pair at
  **535 min after meat-on**.
- The watermark lightens everything under it; the `hm_flot_jpeg` palette
  uses loose hue masks that survive it. Inside lid bands the meat mask
  can catch band pixels (visible plunges) — those samples carry the
  lid_event flag and are masked in every fit.
- Extraction stops before the pull spike (`cook_end_min = 895`).

## Digitization
- `tools/loghunt/digitize.py` with this directory's `log.toml`;
  `hm_flot_jpeg` palette (watermark/JPEG-tolerant). Gaps > 6 min NaN.
- Verification: `overlay_check.png` re-projects the CSV onto the source.

## Estimated uncertainty
- Temperature: 1.63 F/px — line width + JPEG blur ≈ **+/-3 F**; keep an
  observation-noise floor >= 1.5 K.
- Time: +/-1 px ≈ **+/-1.5 min**.
- Fan: envelope of the duty; smoothed proxy.

## Calibration caveats (carried into any fit)
- t_amb (291.5 K) is nominal — no ambient trace, location unknown; the
  wrap fit drives the meat model with the MEASURED pit trace, so t_amb
  is not load-bearing there.
- Meat weight ~12 lb but flat thickness unknown: L fixed at 0.033 m in
  the wrap fit (traded off explicitly against D_eff).
- AR(1) noise scales not fitted; the wrap fit uses iid sigma_core =
  1.5 K (documented in `run_wrap_fit.py`).

## Columns
`t_min` (min since meat-on) - `pit_F` - `meat_F` - `amb_F` (empty) -
`fan_pct` - `setpoint_F` - `lid_event`
