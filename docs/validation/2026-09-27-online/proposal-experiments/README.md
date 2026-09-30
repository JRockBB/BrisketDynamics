# Proposal experiments: none promoted

These are development experiments on backend `411c33d5521d`, physics `4e4a6800e4a2`.
The probability model, prior, noise settings and numerical precision gates were unchanged.
Seed 22001 was inspected during this work and is **not held out for these proposals**. There is no
claim that these experiments establish calibration or globally rank all proposal methods.

| Candidate | Development result | Decision |
| --- | --- | --- |
| Observation-conditioned state guide, with a 20% prior mixture and exact mixture correction | Initial update required a 4,096-particle refresh; the second update also triggered a refresh. The latter was interrupted. | No demonstrated full-cook improvement; not promoted. |
| Invertible affine transport of the joint parameter/history samples, with exact density and Jacobian correction | Initial importance update failed precision; after a reference refresh, 120 seconds passed in 26.49 seconds, but 600 seconds failed precision (minimum mean ESS 687, tail ESS 154). | Partial benefit, insufficient evidence to replace the incremental proposal; not promoted. |
| Extend weak-complement elliptical moves to parameters as well as states | At 1,024 particles/system, minimum mean ESS changed from 438 to 83, and tail ESS from 352 to 49, on the same initial graph and paired seed. Both failed precision. | Worsened this development case; not promoted. |

Times include concurrent computation and are not isolated benchmarks. `PASS` on a development job
means the experiment completed and recorded its results, **not** that its inference precision passed.
The JSON files retain the actual precision flags and diagnostics.

The corrected state-guide oracles passed 70 assertions: Gaussian conditionals, complete defensive
mixture density, retained OU residual conditioning, and a nonlinear change-of-variables check.
The affine proposal passed 27 inversion, Jacobian and exact-weight assertions. The modified weak
kernel passed the existing 38 analytic/measurement assertions, yet performed worse in the cook
experiment. These are separate candidate checks, not additions to the production regression count.
See [the complete check log](proposal_checks.log).

An initial analytic expectation omitted the decoder's positive physical offset and failed eight
checks by about 1.1e-9. The expectation was corrected without relaxing its 1e-9 tolerance. A subsequent
scratch-file copy introduced an encoding parse error; the UTF-8 source was restored. Both failures
are retained in the logs. Neither required changing the candidate algorithm.

Candidate code and corrected oracle sources are copied here for inspection, not included by the
production module. The oracle files retain their original scratch-worker paths; they are not direct
standalone entry points in this archive. The workspace development drivers are under
`scratchpad/online_work/guided` and `scratchpad/online_work/transport`.

The mathematical motivation is proposal adaptation with exact importance correction, as described
in [Doucet and Johansen](https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf) and
[Heng, Doucet and Pokern](https://arxiv.org/abs/1509.08787). The affine candidate is not a trained
normalizing flow or an implementation of their Gibbs-flow algorithm. These references support the
construction principles, not a performance claim for this estimator.
