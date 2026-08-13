# Provenance: mache_brisket_cook.csv

## Source
- **Post:** "First brisket smoke on HeaterMeter w/ WSM 18.5" by user **Mache**, The Virtual Weber Bulletin Board, Sep 4 2013 (edited Sep 6 2013).
  https://tvwbb.com/threads/post-your-heatermeter-graphs.42779/post-474182
- **Image:** HeaterMeter live web-UI screenshot embedded in the post (i.imgur.com/9aS34Zn.png), saved 2026-08-11. Forum content © its author; used here as a small factual data extraction with attribution. A courtesy note to the author is planned; raw HeaterMeter CSV archive requested.

## Cook metadata (from post text)
- 18.5" Weber Smokey Mountain, known leaky door. Full ring of Kingsford Original briquettes, Minion start with 15–20 lit coals (single fuel load, no refuel; ~20 h burn).
- Vents: two bottom vents closed + taped; third taped except one hole feeding a BBQ Guru Pit Viper 10 CFM blower; top vent ~1/8 open.
- Controller: HeaterMeter, stock PIDs, fan min 10% / max 30%, setpoint 225 °F throughout.
- Meat: 10.5 lb choice packer brisket, **unwrapped for the entire cook** (confirmed visually: no mid-cook lid event with post-event slope change; fork tests near the end only). Fridge-cold at load (~45–60 °F probe reading after settling).
- Weather: outdoor peak ~85 °F. Ambient probe sat in a black case in the sun — **ambient channel over-reads by up to ~35 °F midday; treat as qualitative**.

## Timeline (t = 0 at meat-on lid opening)
- t < 0 (not in CSV): ~90 min empty warm-up; pit drifted to ~250 °F (leaky door); meat probe in a water-filled tin can (~168 °F) — that segment is *not* meat data and was discarded.
- t = 0–12 min: lid open, brisket loaded, probe settling → `meat_F` NaN/flagged.
- t ≈ 1088–1098 min: fork-test lid opening (pit dip + recovery overshoot) → `lid_event` flag.
- t ≈ 1141 min: meat pulled; record ends (subsequent probe readings are pit air).
- Stall (rate < 2.5 °F/h between 140–175 °F): ~2.0 h → ~14.4 h. Plateau at ~194 °F before pull.

## Digitization method
- Automated per-pixel-column color-mask extraction (Python/PIL/NumPy), not hand clicking.
- Axis calibration from detected gridlines: 29.618 px per 20 °F; 30.1 px per hour; fan axis 0–100% spans the same frame.
- **Calibration check:** the known 225 °F setpoint line extracts as 225.16 ± 0.0 °F.
- Curves: pit (orange), meat (green), ambient (gray, continuity-tracked to avoid frame/text pixels), fan (top edge of blue fill). Resampled to a uniform 2-min grid; gaps > 6 min left NaN, never bridged.
- `setpoint_F` is filled as the constant 225.0 (occluded by the pit trace on the chart; value known from post + header).
- Verification: `overlay_check.png` re-projects the CSV onto the source image.

## Estimated uncertainty
- Temperature: ±0.7 °F/px line width + calibration ≈ **±1–1.5 °F** → use ≥2 °F observation-noise floor.
- Time: ±1 px ≈ **±2 min**.
- Fan: envelope of a spiky PWM-like signal at ~2 min/px; treat as a smoothed duty-cycle proxy (±3–5 pp).

## Columns
`t_min` (min since meat-on) · `pit_F` · `meat_F` (center of flat, per post) · `amb_F` (qualitative) · `fan_pct` (blower duty, envelope) · `setpoint_F` (constant 225) · `lid_event` (bool; exclude or inflate noise in likelihood)
