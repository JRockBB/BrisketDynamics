# Provenance: keg_2013.csv

## Source
- **Post:** user **Dale Ward**, The Virtual Weber Bulletin Board,
  "Post your HeaterMeter graphs" thread, 2013.
  https://tvwbb.com/threads/post-your-heatermeter-graphs.42779/post-453436
- **Image:** HeaterMeter chart screenshot embedded in the post, retrieved
  2026-09-03 (cached locally as `cache/6e3272c5.png`, not committed).
  Forum content (c) its author; used as a small factual data extraction
  with attribution.

## Cook metadata (from post text, paraphrased)
- **Big Steel Keg** (kamado-style), HeaterMeter controller.
- Fire relit and set to **225 F** with a **13 lb brisket** on at 10:48 pm;
  ~12 h unwrapped overnight; pulled at **185-190 F**, then a 4.5 h cooler
  rest (rest not on the chart).
- The same chart's afternoon segment is an earlier ribs cook at 275 F on
  the same rig (used below to validate the axis calibration).
- **Same vessel as `keg_2014`** (the 2014 chart header reads "BSK HM1",
  the author's Big Steel Keg rig) — this pair is the same-smoker pooling
  set. The author's posts show the airflow hardware evolving across the
  years (blower-only and RotoDamper+fan eras; the exact sequence per
  post is unknown), so the actuator map is treated per-cook even where
  the vessel is shared.

## Chart facts
- Two-day live view; the committed CSV is the overnight brisket segment
  only (22:48 -> ~10:38, t re-zeroed at meat-on).
- The pit rides AT the 225 F setpoint all night, so the red setpoint line
  is occluded end to end and cannot anchor the y-axis. The anchor is the
  label-derived frame top, which predicts the visible 275 F ribs-segment
  setpoint line to ~1 px.
- A second food probe (darker trace) runs alongside; the extracted
  `meat_F` is the probe that starts fridge-cold (~40 F) and ends at the
  posted 185-190 F pull.
- The gray trace is a **dome-level probe riding 10-15 F above pit** — not
  outdoor ambient; ambient extraction is disabled. Outdoor ambient is
  read from the 3 h fire-out lull before the relight, where the pit and
  meat probes sit flat at **~70 F** (294.3 K).
- No HeaterMeter lid-detect bands in the overnight window (scanned; the
  author slept). The frame-edge live-view cursor column is excluded
  (`col_hi`), and the last ~5 min are trimmed (`cook_end_min = 710`)
  to stop before the edge blend.

## Digitization
- `tools/loghunt/digitize.py` with this directory's `log.toml`;
  `hm_classic` palette. Scale from axis labels: 46.14 px / 50 F,
  29.27 px/hr. Gaps > 6 min NaN, never bridged.
- Verification: `overlay_check.png` re-projects the CSV onto the source.

## Estimated uncertainty
- Temperature: ~1.08 F/px at this resolution; line width + anchor ≈
  **+/-2 F**; keep the >= 2 F observation-noise floor.
- Time: +/-1 px ≈ **+/-2 min**.
- Fan: envelope of the PWM-like duty; smoothed proxy.

## Calibration caveats (carried into any fit)
- Early core rises steeply (probe settling / placement); the first grid
  points are settle-masked and the interior is assumed uniform at the
  first unmasked core sample (~88 F).
- AR(1) noise scales borrowed from the Mache MAP (same digitization
  process) rather than fitted per-log.
- Ambient is a single derived value (294.3 K), not a measured trace.

## Columns
`t_min` (min since meat-on) - `pit_F` - `meat_F` - `amb_F` (empty) -
`fan_pct` - `setpoint_F` (225) - `lid_event`
