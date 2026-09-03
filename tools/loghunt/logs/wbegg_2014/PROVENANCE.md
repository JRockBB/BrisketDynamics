# Provenance: wbegg_2014.csv

## Source
- **Post:** user **WBegg**, The Virtual Weber Bulletin Board,
  "Post your HeaterMeter graphs" thread, Aug 4 2014.
  https://tvwbb.com/threads/post-your-heatermeter-graphs.42779/post-592564
- **Image:** full HeaterMeter web-UI screenshot ("WSM Temp" header),
  retrieved 2026-09-04 (cached locally as `cache/5c2b5648.jpg`, not
  committed). Forum content (c) its author; small factual data
  extraction with attribution.

## Cook metadata (from post text, paraphrased)
- **Stock WSM 18.5** (no modifications) — the same smoker MODEL as the
  repo's reference log (`mache_2013`, an 18.5" WSM) but a different
  physical unit: the same-model pooling pair (`run_pool_real.py
  --set wsm`, partial vessel pooling between units).
- **8 lb pork butt**; Truflame hardwood lump (Australian); fan only; a
  foiled terra-cotta dish instead of the water bowl; PID B2/P3/I0.005/D5.
- **32 F at the 4:30 am start** (Australian winter) — the coldest
  documented ambient in the registry; no ambient trace (probes 2/3 off),
  so t_amb is a day-mean estimate (278 K) with the start value in prose.
- Fully documented timeline: 4:30 charcoal; **5:30 shoulder on** (the
  05:30 lid band); 10:47 unexplained oscillations; 17:04 stoked coals
  (no lid band registered); **19:30 added fuel** (band pair
  19:42/19:57); 21:07 setpoint to 230; 22:47 and 24:00 Thermapen checks
  (bands 22:51 / 00:01 — "spot on"); 1:45 setpoint to 250; **2:30 off
  the grill, wrapped, resting** (live capture 2:33:46 — the chart ends
  4 min after the pull, so no usable rest horizon).

## Chart facts
- Right axis 275-25 F, 25 F gridlines at 41.25 px; 2-h labels
  04:27-02:27 at 73.2 px. Vertical anchor: the label-derived frame top,
  cross-checked by the 225 F setpoint line sitting ON the 225 gridline
  (0.5 px agreement); the setpoint steps 225 -> 230 -> 250 late in the
  cook, exactly as the prose documents.
- `meat_F` = the "8# Shoulder" probe; the Thermapen-check dips sit
  inside lid-flagged windows.

## Digitization
- `tools/loghunt/digitize.py` with this directory's `log.toml`;
  `hm_classic` palette. Gaps > 6 min NaN, never bridged.
- Verification: `overlay_check.png` re-projects the CSV onto the source.

## Estimated uncertainty
- Temperature: ~0.61 F/px; line width + anchor ≈ **+/-1.5 F**; keep the
  >= 2 F observation-noise floor.
- Time: +/-1 px ≈ **+/-1.6 min**.
- Fan: envelope of the PWM-like duty; smoothed proxy.

## Calibration caveats (carried into any fit)
- Scalar t_amb (278 K) over a 21 h cook spanning a full winter
  day/night cycle — the coarsest ambient assumption in the registry;
  the documented 32 F start is the only measured point.
- Pork butt on brisket priors is a stated placeholder (meat per-cook in
  the pool); the 17:04 stoke and the 19:30 fuel add are unmodelled fuel
  events (the fan trace carries their airflow signature).
- AR(1) noise scales borrowed from the Mache MAP (same digitization
  process).

## Columns
`t_min` (min since meat-on) - `pit_F` - `meat_F` - `amb_F` (empty) -
`fan_pct` - `setpoint_F` - `lid_event`
