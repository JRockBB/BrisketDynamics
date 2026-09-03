# Provenance: keg_2014.csv

## Source
- **Post:** user **Dale Ward**, The Virtual Weber Bulletin Board,
  "Post your HeaterMeter graphs" thread, 2014 (image-only post).
  https://tvwbb.com/threads/post-your-heatermeter-graphs.42779/post-605881
- **Image:** full HeaterMeter web-UI screenshot ("BSK HM1" header),
  retrieved 2026-09-03 (cached locally as `cache/85b0ad43.png`, not
  committed). Forum content (c) its author; used as a small factual data
  extraction with attribution.

## Cook metadata (from the chart itself; the post has no text)
- Rig header **"BSK HM1"** = the author's **Big Steel Keg** — the same
  vessel as `keg_2013`, a year later (same-smoker pooling pair). The
  thread shows the airflow hardware evolving between the posts, so the
  actuator map is treated per-cook.
- Probe labels name a **pork butt**; setpoint **325 F** — a hot-and-fast
  butt cook. Meat identity differs from the 2013 brisket: in pooling,
  meat parameters are per-cook (and a butt on brisket priors is a stated
  placeholder).
- Live view spans ~09:00 -> 16:00; meat probe comes alive ~11:05 at the
  first HM lid-detect band; ~203 F at capture (cook effectively done).

## Chart facts
- Vertical anchor: the 325 F setpoint line (median over 430 columns; the
  early low-setpoint warm-up segment and pit-trace antialias widen the
  reported spread, but the median matches the label-derived frame top to
  ~0.3 px).
- **Lid events from the HM UI's own lid-detect bands** (measured in the
  chart): meat insertion at 9-17 min and a mid-cook peek at 186-196 min
  after meat-on. Two further bands during the pre-meat-on fire-up are
  irrelevant (that span is sliced off).
- Ambient probe plotted: flat at **~100 F** (median 100.2 F) — a hot
  afternoon, and the probe may be plume/sun-elevated near the keg; used
  as the loss-driving temperature (311.0 K) with that caveat.
- `meat_F` = the "point"-colored probe (single meat probe in use).

## Digitization
- `tools/loghunt/digitize.py` with this directory's `log.toml`;
  `hm_dark` palette. Scale from axis labels: 27.08 px / 25 F,
  91.36 px/hr. Gaps > 6 min NaN, never bridged.
- Verification: `overlay_check.png` re-projects the CSV onto the source.

## Estimated uncertainty
- Temperature: ~0.92 F/px; line width + anchor ≈ **+/-1.5 F**; keep the
  >= 2 F observation-noise floor.
- Time: +/-1 px ≈ **+/-0.7 min**.
- Fan: envelope of the PWM-like duty; smoothed proxy.

## Calibration caveats (carried into any fit)
- Short horizon (~4.8 h, 26 obs at 10-min grid) at a hot-and-fast
  setpoint: informs the vessel's thermal identity at a different
  operating point than the 2013 overnight, which is exactly what the
  same-smoker pool wants — but per-cook meat physics is weakly
  identified.
- The probe-settle descent after insertion is lid/settle-masked; the
  interior is assumed uniform at the first unmasked core sample (~76 F).
- AR(1) noise scales borrowed from the Mache MAP (same digitization
  process) rather than fitted per-log.

## Columns
`t_min` (min since meat-on) - `pit_F` - `meat_F` - `amb_F` (~100) -
`fan_pct` - `flat_F` (unused second probe color) - `setpoint_F` (325) -
`lid_event`
