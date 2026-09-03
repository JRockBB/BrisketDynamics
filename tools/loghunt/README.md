# loghunt — find, retrieve, and digitize public cook logs

The calibration pipeline is data-starved by design (one 2013 log). This
directory turns "we need another log" into a repeatable workflow:

```
1. discover   — find candidate posts (agent-assisted web search, or
                scrape_tvwbb.py over a known graph thread)
2. retrieve   — scrape_tvwbb.py --fetch: post text + chart images -> cache/
3. triage     — triage.py: HeaterMeter-UI fingerprint scores + contact
                sheet; a human or the agent picks the good ones
4. describe   — write logs/<slug>/log.toml: the per-log JUDGMENT layer
                (axis anchors read off the chart, meat-on / cook-end /
                lid events / setpoint from the post prose). This part is
                deliberately not automated: it is the provenance record.
5. digitize   — digitize.py logs/<slug>/log.toml: auto-crop from gridline
                detection, axis calibration from the anchors, the proven
                color-mask extraction -> CSV + overlay + provenance stub
6. adopt      — review the overlay, finish the PROVENANCE prose, then the
                CSV moves to data/ and enters the calibration chain
```

## Ethics / provenance rules (non-negotiable)

- Forum content is © its authors. We take **small factual data
  extractions with attribution** (the temperature traces), exactly as
  `data/cook_log/PROVENANCE.md` did for the 2013 log.
- Source images are cached under `cache/` which is **git-ignored** —
  never commit someone else's screenshot. Committed artifacts are the
  digitized CSV, the overlay check, and the provenance file naming the
  author, post URL, and retrieval date.
- Scraping is polite: public thread pages only, ~1 request / 2 s,
  identified user agent. A courtesy note to authors whose cooks we use
  is the repo's standing practice.

## What each log needs from a human or agent (the log.toml)

Axis anchors (read off the chart): the temperature value of the top and
bottom gridlines, gridline spacings, and the clock value of the leftmost
time gridline. Timeline (from the post prose + chart): meat-on minute,
cook end, probe-settle window, lid/wrap events, setpoint. Everything
mechanical — cropping, px-per-degree, trace extraction, resampling,
gap handling — is derived or fixed.

The frozen original (`digitize_mache_reference.py`) is the documented
method behind the committed 2013 CSV; `digitize.py` is its
generalization. `logs/mache_2013/log.toml` re-expresses that log in the
new format and serves as the regression case when its source image is in
the cache (fetch it via the imgur URL in `data/cook_log/PROVENANCE.md`).
