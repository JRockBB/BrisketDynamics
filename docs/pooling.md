# Multi-cook pooling: what a parameter is *of*

Every calibration parameter in this model is a property of *something* —
the smoker, the cut of meat, the individual animal, or the one cook where
the fire happened to catch a certain way. Pooling makes that ownership
explicit and testable: parameters owned by hardware should agree when the
hardware repeats; parameters owned by meat chemistry should agree across
cuts; parameters owned by a single cook should not be pooled at all. This
document is the write-up of that program on real, found data (method,
three pooling tiers, what transferred, what did not). Results are also
indexed in `docs/results.md`; equations in `docs/model.md`.

## Method: two-stage Gaussian pooling

Joint hierarchical NUTS over N cooks is neither affordable nor needed.
The supported chain is:

1. **Per-log fit.** Gauss-Newton MAP from the prior medians
   (Levenberg-Marquardt on the whitened AR(1) residuals through the
   monolith Tesseract), then the Laplace covariance from the
   Gauss-Newton Hessian. Per log this is minutes of LM; the Laplace step
   is seconds. Against the 22 h NUTS reference the Laplace is within
   10-20 % of the posterior widths on 13 of 15 sites (the two ignition
   sites are per-cook anyway; `docs/results.md`, "Which posterior per
   log").
2. **Gaussian pool** (`brisket_app.approx.hierarchical_pool`) over the
   per-log Laplace Gaussians in the unconstrained space, with each site
   assigned a ROLE:
   - **shared** — one true value across cooks (precision-weighted
     product, with the K−1 duplicated prior copies subtracted so the
     prior enters once);
   - **partial** — a hierarchy with between-cook variance τ²
     (per-cook shrinkage toward a hypermean);
   - **independent** — reported per cook, never pooled.

The role assignment is the scientific claim, and each tier below flips
it to match what the data actually repeats.

## Data

Six committed cook logs (see `brisket logs` and each directory's
PROVENANCE.md under `tools/loghunt/logs/`), all but the reference log
found, digitized and provenance'd by the `tools/loghunt/` pipeline —
public forum posts, small factual data extractions with attribution,
axis anchors cross-checked (setpoint lines vs label-derived scales,
lid-detect bands vs posted timelines), and every extraction verified by
an overlay re-projection. Noise scales are borrowed from the reference
MAP for all found logs (same digitization process); non-brisket cuts run
on brisket priors as a stated placeholder.

| log | rig | cook | horizon | ambient |
|---|---|---|---|---|
| mache_2013 | 18.5" WSM (reference) | 10.5 lb packer, 225 °F | 19.0 h | +27 °C |
| phillip_2018 | unknown smoker | 15 lb packer, 225 °F, partial log | 9.7 h | −17 °C |
| keg_2013 | Big Steel Keg | 13 lb brisket, 225 °F overnight | 11.7 h | +21 °C |
| keg_2014 | same keg | pork butt, 325 °F hot-and-fast | 4.8 h | +38 °C |
| keg_2015 | same keg, new actuator | pork butt, 225 °F overnight | 11.7 h | +21 °C |
| wbegg_2014 | stock WSM 18.5 (2nd unit) | 8 lb pork butt, 32 °F winter start | 20.8 h | +5 °C |

## Tier 1 — different smokers (`run_pool_real.py`)

mache_2013 × phillip_2018: nothing about the hardware repeats, so
**meat partial, everything pit per-cook**. The "briskets I buy" block
behaves: D_eff within 7 %, k_e within 7 % (0.51 vs 0.54 ×10⁻³),
a_myosin/L/m_conv close — across two smokers and a 44 K ambient spread.
The spread sits where the known confounds live: h_c/k_m on the draft
ridge between different rigs, k_mult absorbing phillip's mid-cook slice
start. (`figures/calibration/pool_real.png`)

## Tier 2 — one smoker, three cooks (`--set keg`)

Dale Ward's Big Steel Keg, 2013/2014/2015: the vessel repeats, so
**vessel identity (C_th, UA, dH_eff) fully shared**; the actuator map
(r_max, u_leak) per-cook because the airflow hardware changed across the
posts; meat and ignition per-cook. The two overnight-225 cooks, two
years apart on different actuator hardware, agree on the vessel — UA
within 11 %, dH_eff within 5 %, C_th within 27 % — isolating the 2014
hot-and-fast log's disagreement as operating-point/short-horizon rather
than vessel drift. u_leak spans 0.29-0.88 per-cook (hardware + vent
settings, including a documented mid-cook vent change). Caveat: the
pooled point estimates are ridge-coupled (C_th/UA trade off; a joint
precision-weighted mean with prior-dedup can land outside the
per-coordinate hull) — the cross-cook AGREEMENT is the meaningful
validation, not the pooled point. (`figures/calibration/pool_keg.png`)

## Tier 3 — one model, two units (`--set wsm`)

mache_2013 × wbegg_2014, both 18.5" WSMs but different physical units:
the *design* repeats, the unit does not, so **vessel partial** (τ²
hierarchy — the new tier), everything else per-cook.

| vessel site | mache | wbegg | shrunk | hypermean | verdict |
|---|---|---|---|---|---|
| dH_eff | 1.47e7 | 1.58e7 | 1.55e7 / 1.50e7 | 1.53e7 | **pools hard (8 %)** — fuel energy density is design+charcoal |
| UA | 16.7 | 11.0 | 18.0 / 10.2 | 13.5 | 52 % spread — unit + config (dry terra-cotta dish vs water pan) + a scalar winter day-mean ambient |
| C_th | 4.1e4 | 5.9e5 | 8.4e4 / 2.1e5 | 1.3e5 | ridge — closed-loop logs cannot pin it |

The UA spread is exactly what partial pooling is for, and it sharpens
the case for the water-pan designed experiment (roadmap item 4).
(`figures/calibration/pool_wsm.png`)

## What transfers, and what does not

- **k_e (denaturation-expulsion rate): 0.51-0.57 ×10⁻³ across all FIVE
  found-plus-reference cooks** — brisket and pork butt, three rigs, a
  60 °F ambient range. The most portable meat parameter in the model.
- **a_myosin**: within ~8 % across cuts. Meat *chemistry* transfers;
  meat *geometry* (D_eff, L) differs between cuts exactly as it should.
- **dH_eff**: repeats across cooks of one vessel (5 %) and across units
  of one model (8 %).
- **What does not pool**: C_th (sloppy ridge in every closed-loop log —
  the open-loop characterization burn is the fix, not more logs);
  u_leak/r_max (actuator + vent settings, honestly per-cook); h_c/k_m
  across rigs (the draft ridge); ignition (per-fire by construction).

Two related single-log calibrations complete the transfer picture: the
wrap step-response (butcher paper η_evap = 0.17 [0.13, 0.23] from a
documented wrap moment) and two zero-fit external validations (Baldwin
sous vide; the CET 43:151 oven roast with measured weight loss) — see
`docs/results.md`.

## Standing caveats

Borrowed AR(1) noise on all found logs; scalar per-log ambients (worst
for wbegg's 21 h winter day); pork butts on brisket priors; the
prior-dedup hull effect on shared-block point estimates; digitized data
at 1-2 px temperature resolution (bounded per log in each
PROVENANCE.md). Still wanted: any smoker log with before/after weights,
and the water-pan pair.
