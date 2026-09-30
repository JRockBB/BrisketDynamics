# Pit/core runtime diagnosis

These experiments used the unchanged `411c33d5521d` backend and case 22002. They do not constitute
additional independent cooks or successful online/offline comparisons. Case 22002 has now been exposed
for development and must not be called held-out data for subsequent algorithm changes.

The small-system diagnostic rebuilt the same chronological graph. Its final Gaussian proposal solve
converged in 1.15 seconds. A single 128-particle SMC system reached the full likelihood in four stages
and 16.84 seconds. This run did not meet the replicated-system precision protocol and provides no
qualified posterior. Its purpose was to check that the target and proposal could be evaluated normally.

The instrumented replay used the original online RNG seed and all original sampling settings. Added
wrappers counted evaluations and reported elapsed time; they did not change draws or the target. The
first three accepted arrivals reproduced the original particle counts and mean/tail precision values
exactly. At the last arrival, incremental propagation completed and a full-history refresh began.

The replay was stopped during that refresh to release memory and CPU for the original comparison and
the separate log-weight regression. All completed arrivals, logs, drivers and the interruption status
are retained. No final posterior was obtained from this replay. Elapsed times include instrumentation,
compilation and concurrent processes and are not isolated performance benchmarks.

The drivers retain their original workspace-relative paths. They were run in a warm worker with
production methods restored before validation, as described in the parent record.
