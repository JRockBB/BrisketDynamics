# Provenance: phillip_2018.csv

## Source
- **Post:** user **Phillip P**, The Virtual Weber Bulletin Board,
  "Post your HeaterMeter graphs" thread, Jan 7 2018.
  https://tvwbb.com/threads/post-your-heatermeter-graphs.42779/post-800697
- **Image:** HeaterMeter live web-UI screenshot embedded in the post,
  retrieved 2026-09-03 (cached locally as `cache/b3995e16.png`, not
  committed). Forum content © its author; used as a small factual data
  extraction with attribution.

## Cook metadata (from post text, paraphrased)
- 15 lb packer brisket over lump charcoal with hickory and sassafras wood.
- HeaterMeter controller, 225 °F setpoint (header and red line).
- Extremely cold night — about **1 °F ambient** per the author; the post
  was written roughly ten hours into the cook.
- Smoker model not stated in this post. The same author's 2013 posts used
  a 22.5" WSM; continuity across five years is **unconfirmed** — treat the
  pit as its own smoker (nothing pooled with other logs' pits).

## Chart facts
- Live view captured at 10:24 AM; chart spans ~23:36 → 10:24 (~10.8 h).
- **Partial log**: rise and stall entry only — no stall exit, no pull.
- Dual meat probes ("Point" 172.5 °F / "Flat" 162.2 °F at capture);
  ambient probe off (ambient comes from the prose, not a trace).
- Initial core reads ~28 °F — partially frozen meat. The model carries no
  thaw enthalpy, so the calibration slice starts only once the core is
  well above the phase-change regime (loader `start_at_core_K`; the
  masked/resampled fit actually begins with core ≈ 79 °F, pit already at
  setpoint), with the interior assumed uniform at that temperature.

## Digitization
- `tools/loghunt/digitize.py` with this directory's `log.toml`;
  `hm_dark` palette (black plot, gray gridlines, two green probes).
- Vertical anchor: the red 225 °F setpoint line (row 123 in-crop);
  scale 59.09 px per 25 °F; horizontal 120.2 px/hr anchored at the frame's
  left edge. Gaps > 6 min NaN, never bridged.
- `meat_F` = the **Flat** probe (thicker section, matching the Mache
  convention); the Point probe is kept as `point_F`.
- Verification: `overlay_check.png` re-projects the CSV onto the source.

## Estimated uncertainty
- Temperature: ~±0.5 °F/px line width at this resolution + anchor ≈
  **±1–1.5 °F**; keep the ≥ 2 °F observation-noise floor.
- Time: ±1 px ≈ **±0.5 min**.
- Fan: envelope of the PWM-like duty; smoothed proxy.

## Calibration caveats (carried into any fit)
- Partial horizon → weak identification of stall-exit physics
  (D_eff / k_e see only the entry); no pull/rest data.
- AR(1) noise scales borrowed from the Mache MAP (same digitization
  process) rather than fitted per-log.
- Ambient is a single prose value (255.9 K), not a measured trace.

## Columns
`t_min` (min since chart clock 0) · `pit_F` · `meat_F` (Flat) · `amb_F`
(empty) · `fan_pct` · `point_F` (Point probe) · `setpoint_F` (225) ·
`lid_event`
