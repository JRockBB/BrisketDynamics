# Provenance: keg_2015.csv

## Source
- **Post:** user **Dale Ward**, The Virtual Weber Bulletin Board,
  "Help with HM and Keg? Dale/Andy" thread, May 25 2015.
  https://tvwbb.com/threads/help-with-hm-and-keg-dale-andy.59443/post-660823
- **Image:** full HeaterMeter web-UI screenshot ("BSK HM2 TC" header),
  retrieved 2026-09-03 (cached locally as `cache/3cfd8e37.png`, not
  committed). Forum content (c) its author; used as a small factual data
  extraction with attribution.

## Cook metadata (from post text and the surrounding thread, paraphrased)
- **Big Steel Keg** — the same vessel as `keg_2013`/`keg_2014` (third
  cook of the same-smoker pooling set). By this post the airflow
  hardware is a **RotoDamper + stock fan** ("mine is air tight...only
  air comes from the rotodamper"); actuator map treated per-cook.
- Overnight **pork butt** cook (three "Butt" probes in the header):
  blower startup 8 %, **setpoint 225 F**, top vent 1.5-2; meat in after
  the overshoot settled; **~1 am the top vent was closed to ~3/4** (a
  mid-cook airflow change — the per-cook actuator/leak parameters are
  an average over it); PID P12 I0.004 D6, left for the night.
- Live capture 10:35 am: probes at 171.8/174.5/173.6 F, forecast
  "12h5m to 199" — **partial log** (no pull on the chart).
- Thread context from the same author: pit probe hangs at the back of
  the lid thermometer (little difference vs grate temp); ~3 lb lump
  loads last 26+ h at 225 F (the keg is extremely well insulated).

## Chart facts
- Classic steel-blue theme at full resolution. Right axis 0-350 F,
  50 F gridlines at 46.5 px; 2-h labels 22:04-10:04 at 96.5 px.
- Vertical anchor: the 225 F setpoint line (long visible stretch before
  the pit reaches temp), median row 420 vs 419.75 predicted from the
  axis labels — the two calibrations agree to ~0.3 px.
- Three meat probes: `meat_F` = the light-green Butt probe; the dark
  teal and gray Butt probes are left unextracted (the classic-palette
  gray would collide with the ambient mask; ambient extraction is
  disabled — there is no ambient trace).
- Ambient: the pit probe sits flat at **~69.5 F** all afternoon before
  the fire (main chart left edge + mini-map), used as t_amb (294.0 K).
- No HeaterMeter lid-detect bands in the cook window (scanned). The
  23:30-00:40 pit wiggles + 100 % fan spikes are the author's vent
  adjustments; the fan trace drives the model through them.

## Digitization
- `tools/loghunt/digitize.py` with this directory's `log.toml`;
  `hm_classic` palette. Gaps > 6 min NaN, never bridged.
- Verification: `overlay_check.png` re-projects the CSV onto the source.

## Estimated uncertainty
- Temperature: ~1.08 F/px; line width + anchor ≈ **+/-2 F**; keep the
  >= 2 F observation-noise floor.
- Time: +/-1 px ≈ **+/-1.2 min**.
- Fan: envelope of the PWM-like duty; smoothed proxy.

## Calibration caveats (carried into any fit)
- Partial horizon (stall entry, no pull); butt on brisket priors is a
  stated placeholder; interior assumed uniform at the first unmasked
  core sample.
- The ~1 am top-vent change makes u_leak a per-cook average.
- AR(1) noise scales borrowed from the Mache MAP (same digitization
  process).

## Columns
`t_min` (min since meat-on) - `pit_F` - `meat_F` (light-green Butt) -
`amb_F` (empty) - `fan_pct` - `setpoint_F` (225) - `lid_event`
