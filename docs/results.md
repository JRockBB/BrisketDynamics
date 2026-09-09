# Results record

Completed roadmap work and the decisions behind it, moved out of
`docs/roadmap.md` (which holds only what remains). Equations:
`docs/model.md`. Derivative verification: `docs/verification.md`.

## The pilot in a pitmaster's terms, against the MPC on the same brisket (2026-09-09)

Two questions were asked of the real-cook closed loop below: are the
timings real, and how does it compare with the app's MPC. Answering
the first exposed that the pilot had been asked for the wrong things
since the planning experiment; fixing that took three more findings;
the comparison then had to be built on one plant.

**Are the timings real?** Two questions, asked after the run.

*The instruments.* The mock's cadences are on the slow side of real
hardware and one of them is a placeholder. A pit or meat probe
(Fireboard, ThermoWorks Signals, a HeaterMeter) samples every second
and logs every minute; the two-minute grid loses nothing the physics
can use, since the cook's own time constants are hours, but a real
feed is thirty times denser and the estimator would simply see
thirty-times-longer runs at the same ρ per second. A humidity sensor
also reads every few seconds; the ten-minute spacing stands for a
cheap sensor polled slowly and is not a property of the instrument.
The saturation cap is real (a sensor in a smoker reaches its ceiling
and sits there), the value is a guess. The half-hour scale reading is
what a load cell under the grate can honestly deliver once draft, lid
lifts and thermal drift are averaged out. The lid windows are the
weak point: six minutes is right for a wrap or a pull, three times too
long for a spritz, and wrong for a fuel add on an offset, whose
firebox door does not open the cook chamber at all. The fuel-add
noise bump (three times the base for half an hour) is the right shape
for a chimney of fresh coals. None of this changes what the estimator
does; it changes how much data it gets.

*The cook.* The plant is the calibration's real cook: mache_2013, a
19-hour overnight cook at 225 °F on a WSM whose core reached 70 °C at
7.7 h, 80 °C at 15.7 h and 85 °C at 16.7 h and never saw 90. So the
brisket the pilot flew is a slow one by construction, and what the
pilot did with it was slower still: a 238 °F pit, the core at 70 °C by
6.3 h, then an unwrapped stall from 6 to 17 h in which the core
gained two degrees, the wrap at 16.6 h (at 72 °C), 3 °C/h after it,
the pull at 21.2 h at 86 °C, a 1.5 h rest. Against the reference the
question named, an Aaron Franklin cook (a 275 °F offset, butcher-paper
wrap at about 165 °F internal around six hours in, a pull at about
203 °F when probe-tender near twelve hours, a one-to-three-hour rest),
this is not a realistic brisket day: it is twice as long, wraps ten
hours late and pulls nine degrees under. The reason is not the
estimator or the trigger, both of which did what the plan asked. It is
the goals the plan was given, which were mine from the planning
experiment: a retained-water target of 0.35 of the initial dry-basis
water is 45 % moisture wet basis, drier than any brisket one would
serve (a cooked brisket sits near 55 to 60 %), and the only way to get
there is to leave the meat unwrapped through the stall until it has
dried enough, which is exactly what the plan did, holding the water on
its goal to the third decimal and giving up the pull temperature. The
plan's damper prior (a low, 0.15 damper, held with a narrow width) set
the 238 °F pit.

**The goals, in a pitmaster's terms.** `PlanGoals` now reads as a
pitmaster would state it. The pull is probe-tender, which in this
physics is the collagen conversion at the center node, in the band the
app's quality objective uses (0.90 done, 0.95 pot roast): 0.92 ± 0.02;
the pull temperature is reported, not targeted. The water at serve is
what this plant retains at doneness on a cook of normal length, 0.47
of the initial dry-basis water (52 % wet basis): I first wrote a
served brisket's 57 %, and the plan under it stretched back toward a
cool 22-hour cook by the third hour, because on this plant every
short cook lands at 0.45 to 0.48, where the app's own optimum lands
too; the model's water budget is a roadmap residue, and a goal the
plant cannot reach buys nothing but length. The bark is the mahogany
band, L* 31 ± 3. The floor
at serve stays at 65 °C. The damper prior was the last and quietest
culprit: its mean is a nearly closed damper (0.15) and at one logit-sd
of width a half-open damper cost 1.5 loss units per knot, 4.5 for the
cook, as much as the three goals together, so every plan in this
series bought a cool pit by fiat and paid for it in hours. Doubling
the width was not enough (the run under it drifted to a 25 h pull by
its fourth check); the prior is a regulariser for the graph, not a
fuel preference, and is now three logit-sd wide, a sixth of a unit
per knot at half open. The first run under these goals exposed a
flaw in the trigger: at the first hour the standing plan's loss FELL
(134 to 27) because the posterior had tightened, so the loss-rise
condition held while a far better plan had become available. A
standing plan scored against itself is blind to that. Two cheap
conditions catch it: the plan is bad on its own terms (its expected
loss above 3, one goal 2.4 sd off) and the belief it was made on is
gone (the KL divergence of the current posterior from the one the
plan was made under above 2 nats; one full-sd shift of one latent is
half a nat). The closed loop under these goals and this policy
follows; the app's MPC on the same brisket after it.

**The closed loop under these goals** (`estimator/mock_closed_loop_realcook.jl`,
record `data/mock/closed_loop_realcook.json`, figure
`figures/estimator_closed_loop_realcook.png`; the policy now has six
conditions and the record carries the posterior's move per check):

| check | reason | committed | loss: standing → new | KL [nats] | wrap / pull / serve [h] | spritz / add [h] | collagen at the pull | water | bark L* |
|---|---|---|---|---|---|---|---|---|---|
| 0 h | the first plan |  | 137.94 |  | 5.5 / 19.5 / 19.7 | 6.8 / 17.9 | 0.771 ± 0.199 | 0.575 ± 0.319 | 39.7 ± 10.1 |
| 1 h | loss high |  | 16.25 → 10.19 | 20.4 | 4.7 / 18.7 / 19.8 | 6.3 / 16.4 | 0.970 ± 0.049 | 0.390 ± 0.076 | 33.1 ± 4.1 |
| 2 h | loss high |  | 9.55 → 8.85 | 10.0 | 5.8 / 18.4 / 19.9 | 6.9 / 11.8 | 0.985 ± 0.037 | 0.429 ± 0.048 | 33.2 ± 4.0 |
| 3 h | loss high |  | 10.08 → 5.37 | 50.9 | 13.7 / 18.9 / 20.4 | 9.2 / 14.1 | 0.957 ± 0.046 | 0.478 ± 0.042 | 33.3 ± 3.8 |
| 4 h | loss high |  | 7.88 → 5.24 | 10.5 | 16.6 / 22.1 / 24.0 | 9.2 / 13.6 | 0.961 ± 0.048 | 0.465 ± 0.011 | 32.8 ± 3.5 |
| 5 h | loss high |  | 5.77 → 2.99 | 12.9 | 15.0 / 17.6 / 19.0 | 8.0 / 9.1 | 0.941 ± 0.043 | 0.440 ± 0.008 | 30.7 ± 2.2 |
| 6 h | loss high |  | 6.00 → 0.59 | 19.4 | 16.9 / 22.9 / 24.7 | 11.8 / 14.2 | 0.921 ± 0.007 | 0.459 ± 0.008 | 33.8 ± 0.9 |
| 7 h | loss high |  | 9.60 → 0.72 | 19.1 | 15.5 / 24.9 / 26.6 | 12.6 / 16.8 | 0.920 ± 0.011 | 0.466 ± 0.011 | 33.5 ± 2.1 |
| 8 h | loss rose |  | 2.18 → 0.94 | 15.9 | 15.2 / 25.9 / 27.7 | 13.5 / 15.4 | 0.919 ± 0.010 | 0.462 ± 0.007 | 34.5 ± 1.6 |
| 9 h | loss rose |  | 2.90 → 1.37 | 3.6 | 14.4 / 23.9 / 25.7 | 14.7 / 16.6 | 0.923 ± 0.024 | 0.453 ± 0.005 | 34.4 ± 0.9 |
| 10 h | hold | | 1.80 vs 1.37 | 1.3 | | | | | |
| 11 h | loss rose |  | 2.49 → 1.85 | 3.3 | 13.2 / 23.2 / 24.9 | 14.7 / 15.4 | 0.923 ± 0.031 | 0.471 ± 0.014 | 34.4 ± 1.5 |
| 12 h | posterior moved |  | 2.15 → 0.43 | 13.2 | 13.8 / 19.0 / 20.5 | 14.3 / 16.5 | 0.920 ± 0.017 | 0.454 ± 0.001 | 30.5 ± 0.6 |
| 13 h | event due |  | 15.65 → 0.92 | 41.8 | 14.6 / 20.1 / 21.7 | 15.3 / 15.8 | 0.921 ± 0.022 | 0.443 ± 0.008 | 32.0 ± 1.6 |
| 14 h | event due | wrap | 1.37 → 0.66 | 3.4 | 14.9 / 19.3 / 20.8 | 15.6 / 17.1 | 0.920 ± 0.017 | 0.439 ± 0.003 | 31.0 ± 1.0 |
| 15 h | event due |  | 1.05 → 0.62 | 8.3 | 14.9 / 21.0 / 22.5 | 16.8 / 17.8 | 0.920 ± 0.016 | 0.439 ± 0.003 | 31.7 ± 0.4 |
| 16 h | event due |  | 1.06 → 0.52 | 11.3 | 14.9 / 19.4 / 21.0 | 17.0 / 17.5 | 0.920 ± 0.014 | 0.442 ± 0.003 | 31.9 ± 0.5 |
| 17 h | event due | add, spritz | 1.54 → 0.42 | 12.7 | 14.9 / 20.1 / 21.7 | 17.9 / 17.9 | 0.920 ± 0.009 | 0.443 ± 0.003 | 32.4 ± 0.3 |
| 18 h | loss rose |  | 2.16 → 0.43 | 16.6 | 14.9 / 20.7 / 22.3 | 17.9 / 17.9 | 0.919 ± 0.005 | 0.441 ± 0.003 | 32.6 ± 0.2 |
| 19 h | posterior moved |  | 0.60 → 0.58 | 7.2 | 14.9 / 20.8 / 22.4 | 17.9 / 17.9 | 0.920 ± 0.010 | 0.439 ± 0.002 | 32.7 ± 0.4 |
| 20 h | event due | pull | 0.60 → 0.49 | 1.8 | 14.9 / 21.0 / 22.5 | 17.9 / 17.9 | 0.920 ± 0.007 | 0.436 ± 0.002 | 32.2 ± 0.6 |
| 21 h | hold | | 0.51 vs 0.49 | 0.7 | | | | | |
| 22 h | event due |  | 0.50 → 0.50 | 3.8 | 14.9 / 21.0 / 22.5 | 17.9 / 17.9 | 0.928 ± 0.003 | 0.440 ± 0.001 | 32.6 ± 0.3 |

Executed: wrap 14.89 h, fuel add and spritz train at 17.9 h, pull
20.99 h, serve 22.52 h. The plant: collagen conversion 0.937 at the
pull with the core at 90.0 °C, retained water 0.442 (51 % moisture),
bark L* 33.1, core at serve 82.9 °C: every goal inside one sd, and the
last replan's prediction (0.928, 0.440, 32.6) was the plant. Twenty-one
replans in twenty-three checks, 75 minutes of replans and 47 seconds of
holds; the final latents at a median of one posterior sd from truth.
Reading it:

- **The absolute-loss condition carried the first seven hours.** While
  the posterior was wide the standing plan's expected loss was mostly
  variance (collagen ± 0.05 across the draws is five units on its own),
  and a plan scored against itself would have held; the absolute
  condition replanned every hour until the belief tightened at 6 h,
  when the loss fell to 0.6.
- **The plan hedges while it does not know the brisket.** Every early
  plan was a long cool cook (pull 18 to 26 h): under a wide posterior
  the schedule whose outcomes vary least across the draws wins, and
  that is the slow one. Hours 1 to 5 were executed that way, and the
  cook that followed had to finish from there. A certainty-equivalent
  plan would have run hotter sooner and been wrong more often; this is
  the price of planning on the posterior rather than on its mean, and
  it is a real property, not a bug.
- **The posterior-move condition found the four-hour-shorter plan.** At
  12 h the loss rise was under threshold (2.15 against 1.85) but the
  belief had moved 13 nats; the replan brought the pull from 23.2 h to
  19.0 h with every goal on target. The loss-rise rule alone would have
  missed it.
- **The event-due treadmill is still there** (seven replans from 13 h
  on, each deferring an event by an hour until the posterior wanted
  it); the rule that commits an event a replan leaves where it stood is
  still not built.
- **The goal form decides the cook's length.** With the posterior tight
  the plan still chose a 21-hour pull, because a Gaussian at collagen
  0.92 ± 0.02 asks for a precise landing and in this physics a hot cook
  overshoots the center before the bark darkens: the slow approach is
  the only way to arrive at 0.92 with a mahogany bark. The app's flat
  band (0.90 to 0.95, no cost inside) accepts 0.947 and stops hours
  earlier. That is the comparison below.

**Against the MPC, on one brisket** (`estimator/run_mpc_mock1d.py`,
`export_mpc_trace.py`, `pilot_comparison_trace.jl`, record
`data/mock/pilot_comparison.json`, figure
`figures/estimator_pilot_comparison.png`). The app's MPC
(`brisket_app/mpc.py`: state feedback, the plant state read exactly, an
ambient Kalman filter, a replan every hour, the KCBS quality objective
of collagen conversion at the center in a 0.90 to 0.95 band, a mahogany
L* band, juiciness, smoke, fuel and safety terms) had only ever flown
the calibration's own brisket, a thinner one (half-thickness 1.9 cm
against the mock's 3.9, a different surface coupling), from a cold pit
with fridge meat after a twelve-hour brine. `scripts/run_fullplan.py`,
`scripts/run_mpc.py` and `mpc.run_cook` now take the pit and meat
starting temperatures as arguments (defaults unchanged), so the full
plan was scanned on the mock's brisket under the mock's start (pit
89 °C, meat 27 °C, unsalted) over 12 to 24 h (the app picks 16 h:
wrap 8.6 h, pull 11.2 h) and one MPC cook flown from it (76 minutes).
Both executed cooks were then re-run in the Julia monolith at the same
parameters and scored under both objectives; the Julia core tracks the
Tesseract's own trace within 1.4 °C over the MPC's cook, so it is one
plant.

| | the pilot's goal | RxInfer pilot | MPC |
|---|---|---|---|
| probe-tender at the pull (collagen at the center) | 0.92 ± 0.02 | 0.936 (core 90 °C) | 0.854 (core 89 °C) |
| retained water W/W0 at serve | 0.47 ± 0.04 | 0.442 (51 % wet basis) | 0.483 (53 %) |
| bark L* at serve | 31 ± 3 | 33.1 | 34.4 |
| core at serve | floor 65 °C | 82.9 °C | 70.4 °C |
| collagen at the center at serve (the MPC's band) | 0.90 to 0.95 | 0.966 | 0.932 |
| juiciness (the MPC's term) | higher | 0.526 | 0.563 |
| wrap / pull / serve | | 14.9 / 21.0 / 22.5 h | 8.1 / 11.7 / 16.0 h |
| core at 70 / 80 / 90 °C | | 11.9 / 16.1 / 19.7 h | 6.8 / 7.9 / never |
| pit, mean to the pull / max | | 114 °C (236 °F) / 136 °C | 120 °C (248 °F) / 177 °C |
| what it knows | | a 17-latent posterior, noise per channel | the plant state, an ambient filter |
| replans | | 21 of 23 hourly checks, 75 min | every hour, 76 min |

Reading it:

- **The same brisket, two ways.** At serve both cooks are the same
  brisket: collagen 0.93 to 0.97, water 0.44 to 0.48, bark 33 to 34.
  The MPC gets there in 16 hours with a 350 °F push around the wrap
  and a four-hour rest during which the center finishes (0.85 at the
  pull, 0.93 at serve); the RxInfer pilot takes 22.5 hours at 236 °F
  with a 1.5 h rest and the center done at the pull. Neither is wrong
  about the brisket; they were asked different questions.
- **The question is the goal form.** The MPC's bands cost nothing
  inside them, so a cook that lands anywhere in 0.90 to 0.95 and 28 to
  34 is as good as any other and the shortest one wins on fuel. The
  pilot's Gaussian goals want the center of each band, and in this
  physics the center of the collagen band with a mahogany bark is
  reached only slowly. A band-shaped goal (the hinge already used at
  serve, on both sides) would let the pilot stop where the MPC stops;
  that is a two-line change to `goal_residuals` and the next thing to
  try.
- **Knowing the plant is worth hours.** The MPC reads the plant state
  exactly and commits to a hot push at 5 h; the pilot, working from
  readings through a posterior that was still wide at 5 h, hedged. On a
  real pit only the pilot's position is available, and the hedge is
  what a posterior buys: the MPC's plan would be as sharp and as
  wrong if the brisket were not the one it assumed.
- **Neither is an Aaron Franklin cook.** Franklin wraps at 6 h and
  pulls at 12 on a 275 °F offset; the MPC's 16 hours is the nearer.
  What keeps both long is this model's bark: calibrated on a 19-hour
  225 °F WSM cook, its bark kinetics take ten hours to darken to a
  mahogany 33, where an offset at 275 °F does it in six. Until the
  bark model is calibrated on a hot offset cook (roadmap C6), every
  planner on this physics will find the long cook, whichever objective
  it is given.

**Band-shaped goals and the commit rule** (the same day, later). Two of
the three items above are built; the third, the bark on a hot offset,
needs hourly color-card photos from such a cook and there are none.
`PlanGoals` are bands now: a goal costs nothing inside its band and half
the squared distance outside it in its sd, with a softplus corner a
quarter-sd wide so the graph's affine nodes see a gradient
(`band_residual`, `floor_residual`): collagen at the center 0.90 to
0.95 (sd 0.02), water at serve at least 0.45 (sd 0.04), bark L* 28 to
34 (sd 3), the serve floor as before. And an event a replan leaves
within half an hour of where it stood commits at once, however far
ahead (`stable_events`, `commit!(...; force=)`; `ReplanPolicy.stable`).
The closed loop rerun under them (the same plant, the same six
conditions), the figures overwritten with this run:

| check | reason | committed | loss: standing → new | KL [nats] | wrap / pull / serve [h] | spritz / add [h] | collagen at the pull | water | bark L* |
|---|---|---|---|---|---|---|---|---|---|
| 0 h | the first plan |  | 87.29 |  | 1.3 / 16.9 / 17.0 | 5.1 / 15.3 | 0.782 ± 0.181 | 0.575 ± 0.324 | 42.0 ± 9.5 |
| 1 h | event due |  | 13.50 → 2.46 | 25.2 | 6.1 / 10.7 / 12.2 | 3.7 / 5.2 | 0.959 ± 0.050 | 0.454 ± 0.084 | 33.6 ± 4.0 |
| 2 h | posterior moved |  | 2.73 → 2.32 | 14.6 | 10.7 / 15.3 / 16.9 | 5.9 / 8.2 | 0.976 ± 0.044 | 0.478 ± 0.015 | 32.5 ± 3.6 |
| 3 h | posterior moved |  | 2.21 → 0.70 | 20.9 | 13.8 / 17.9 / 19.6 | 6.8 / 10.9 | 0.942 ± 0.039 | 0.459 ± 0.024 | 33.9 ± 2.8 |
| 4 h | loss rose |  | 2.81 → 0.32 | 14.4 | 13.8 / 25.1 / 26.7 | 9.3 / 13.8 | 0.934 ± 0.031 | 0.499 ± 0.016 | 34.3 ± 2.9 |
| 5 h | loss high |  | 3.82 → 0.14 | 9.3 | 11.6 / 15.2 / 16.8 | 7.9 / 11.5 | 0.932 ± 0.023 | 0.457 ± 0.007 | 34.4 ± 1.1 |
| 6 h | posterior moved |  | 0.07 → 0.02 | 9.0 | 12.7 / 15.9 / 17.4 | 8.9 / 10.9 | 0.932 ± 0.016 | 0.461 ± 0.005 | 33.2 ± 1.3 |
| 7 h | loss high |  | 7.45 → 0.10 | 20.2 | 12.5 / 15.6 / 17.2 | 9.6 / 11.2 | 0.933 ± 0.028 | 0.455 ± 0.007 | 33.6 ± 1.2 |
| 8 h | loss high |  | 3.72 → 0.34 | 24.5 | 14.0 / 18.1 / 19.8 | 11.1 / 12.7 | 0.942 ± 0.032 | 0.453 ± 0.009 | 34.8 ± 0.7 |
| 9 h | loss high |  | 6.14 → 0.25 | 5.5 | 12.3 / 15.8 / 17.4 | 11.0 / 13.0 | 0.935 ± 0.028 | 0.447 ± 0.003 | 34.5 ± 0.8 |
| 10 h | event due | add (3.3 h ahead, stable) | 0.13 → 0.12 | 41.6 | 12.8 / 16.7 / 18.2 | 12.1 / 13.3 | 0.923 ± 0.028 | 0.443 ± 0.003 | 30.1 ± 0.8 |
| 11 h | loss high |  | 4.74 → 0.21 | 3.2 | 13.2 / 18.3 / 19.8 | 13.4 / 13.3 | 0.924 ± 0.031 | 0.447 ± 0.007 | 31.6 ± 1.4 |
| 12 h | posterior moved |  | 0.52 → 0.10 | 4.2 | 13.7 / 17.3 / 18.8 | 13.6 / 13.3 | 0.928 ± 0.024 | 0.436 ± 0.001 | 31.0 ± 0.7 |
| 13 h | event due | wrap (1.2 h ahead, stable) | 15.33 → 0.20 | 47.1 | 14.2 / 17.9 / 19.4 | 14.6 / 13.3 | 0.930 ± 0.026 | 0.430 ± 0.002 | 33.6 ± 0.9 |
| 14 h | event due |  | 0.34 → 0.19 | 19.5 | 14.2 / 18.6 / 20.1 | 15.4 / 13.3 | 0.926 ± 0.026 | 0.433 ± 0.003 | 33.1 ± 0.7 |
| 15 h | event due | pull (3.7 h ahead, stable) | 0.18 → 0.11 | 11.7 | 14.2 / 18.7 / 20.2 | 16.1 / 13.3 | 0.930 ± 0.019 | 0.434 ± 0.002 | 32.1 ± 0.6 |
| 16 h | event due |  | 0.40 → 0.40 | 6.3 | 14.2 / 18.7 / 20.2 | — / 13.3 | 0.962 ± 0.006 | 0.428 ± 0.002 | 31.8 ± 0.4 |
| 17 h | posterior moved |  | 0.18 → 0.18 | 14.5 | 14.2 / 18.7 / 20.2 | — / 13.3 | 0.954 ± 0.005 | 0.432 ± 0.002 | 32.6 ± 0.3 |
| 18 h | posterior moved |  | 0.13 → 0.13 | 5.1 | 14.2 / 18.7 / 20.2 | — / 13.3 | 0.948 ± 0.003 | 0.432 ± 0.003 | 32.6 ± 0.3 |
| 19 h | posterior moved |  | 0.16 → 0.16 | 8.7 | 14.2 / 18.7 / 20.2 | — / 13.3 | 0.946 ± 0.003 | 0.429 ± 0.002 | 32.7 ± 0.8 |
| 20 h | event due |  | 0.18 → 0.18 | 2.1 | 14.2 / 18.7 / 20.2 | — / 13.3 | 0.949 ± 0.003 | 0.428 ± 0.002 | 32.3 ± 0.6 |

Executed: fuel add 13.27 h, wrap 14.20 h, pull 18.68 h, serve
20.15 h, no spritz; a 250 °F pit with the damper half open at the
end. The plant: collagen 0.956 at the pull with the core at 97.5 °C
(0.3 sd past the band's top), water 0.431 (0.5 sd under its floor),
bark L* 32.5 inside its band, core at serve 91 °C. Twenty-one replans
in twenty-one checks, 53 minutes; the latents at a median of 0.9 sd
from truth, none past 1.7.

| | the pilot's band | RxInfer pilot | MPC |
|---|---|---|---|
| collagen at the center at the pull | 0.90 to 0.95 | 0.955 (core 98 °C) | 0.854 (core 89 °C) |
| water W/W0 at serve | at least 0.45 | 0.431 (50 %) | 0.483 (53 %) |
| bark L* at serve | 28 to 34 | 32.5 | 34.4 |
| collagen at serve (the MPC's band) | 0.90 to 0.95 | 0.995 | 0.932 |
| wrap / pull / serve | | 14.2 / 18.7 / 20.2 h | 8.1 / 11.7 / 16.0 h |
| pit, mean / max | | 250 °F / 155 °C | 248 °F / 177 °C |

Reading it:

- **The bands shortened the cook by 2.4 hours** (22.5 to 20.2 h) and
  let the damper open (the pit 250 °F against 236, half open at the
  end against 0.13); the gap to the MPC's 16 h is 4.2 h now.
- **Inside the bands the plan is indifferent, so while the posterior
  is wide its width decides the length.** Hours 2 to 4 went from a
  15 h pull to 25 h and back to 15 h by hour 5 as the bark's spread
  fell from ± 3.6 to ± 1.1 L*: a longer, cooler cook keeps more draws
  inside the mahogany band, and once the draws agree the short cook
  wins. The first five hours were executed while it hedged.
- **The commit rule ended the treadmill for three events.** The fuel
  add committed 3.3 hours ahead, the wrap 1.2 hours, the pull 3.7
  hours, each because the replan moved it by under half an hour. The
  spritz moved 48 minutes at 14 h, stayed open, and left the free set
  when the pull committed, so this cook had none.
- **Collagen belongs at serve, not at the pull.** The band was met at
  the pull (0.956) and the rest took it to 0.995: pot roast by the
  MPC's own band, which it scores at serve. The MPC pulls at 0.85 with
  the core at 89 °C and lets a 4.3-hour rest finish the center to
  0.93; that is why it can pull earlier and cooler and keep more
  water. Two things follow: the pilot's collagen goal should be read at
  serve, and its rest prior (1.5 h ± 0.5 in the log, so a four-hour
  rest costs two units) fixes the rest by fiat as the damper prior
  fixed the pit. Both are next.
- **The posterior-move condition fires nearly every hour** (seven of
  twenty-one, at 2 to 47 nats) because a seventeen-latent posterior
  that sharpens hourly moves several nats routinely; two nats is a
  threshold for one latent, not seventeen. Scaled with the dimension
  (n/4, four nats) it would have kept three of the seven. And one
  replan (16 h) fired on a spritz the pull's commit had already taken
  out of the free set; `replan_due` now counts only free events.

## Reading like a real cook: rates, censoring, the noise level as a process, and the replan as a decision (2026-09-09)

Nothing in a cook is stationary and no two instruments agree on a
clock. The pit probe reports every two minutes and its noise doubles
when fresh fuel goes in; the humidity sensor reports every ten and
saturates; the scale is read every half hour; every lid lift blanks
all of them for six minutes. The estimator had been fed one grid, one
noise level per channel and readings at every point, and replanned on
a two-hour clock. This entry makes the plant read like a real cook,
makes the estimator model what it reads, and lets it decide when a
replan is worth making.

**The plant** (`estimator/realcook.jl`, `mock_realcook.jl`, record
`data/mock/mock_realcook.json`). The matched mock on a two-minute grid
over 30 h with the schedule the plan block found (wrap 11 h, spritz
from 8.5 h every 1.6 h at 46 g, fuel add 12 h, pull 15.7 h). Each
channel is a unit AR(1) times a level: the pit's level is 1 + 2
exp(−(t − t_add)/30 min) after the fuel add, the humidity reading is
min(true, 6500 Pa), and every event including each spritz opens a
six-minute lid window that all four channels drop. Of 901 grid points
the pit and core probes have 877, the humidity 176 (34 at the cap), the
scale 58.

**Channels at their own rates.** A channel's whitening is gap-aware:
samples g grid steps apart are correlated ρ^g, a run of the AR chain
breaks only at a gap over three times the channel's spacing (a
full-rate channel breaks at any gap), and the AR node identifies the
noise at the channel's own spacing with the prior converted to it (ρ^g,
its width by the slope of that map). The calibration's ten-minute
priors convert the same way to the two-minute step.

**Censoring.** `CensoredHumidity` predicts min(model, cap) as a smooth
corner (cap − s·softplus((cap − model)/s), slope logistic((cap −
model)/s): one below the cap, zero above), so a reading at the cap says
"the true value was at least the cap" and penalises a model that dips
below it. The first stream pass kept the censored samples in the AR
chain and in the noise identification; their near-zero residuals
identified the humidity's sd at 27 Pa against the plant's 100, and a
channel weighted fourteen times too heavily bent the physics: the
median latent 1.6 sd from truth, one at 4.2. The registry now has three
hooks: `noise_mask` (the samples whose residuals are the noise
process), `extra_rows` (independent rows a channel appends after its AR
rows; a censored sample is one, weighted like a reading) and
`misfit_rows` (the rows the misfit check scores). With them the
humidity's sd came out at 94 ± 25 Pa.

**The noise level as a process** (`estimator/volatility.jl`). Given the
AR node's ρ, the innovations η_k = e_k − ρ e_{k−1} are white with
variance s_k²(1 − ρ²) when the level s moves slowly, and their
cumulative sum is a random walk whose increments have that variance:
RxInfer's GCV node exactly, x_k ~ N(x_{k−1}, exp(z_k)), with the
log-variance z itself a random walk (the hierarchical Gaussian filter's
two layers, Gauss–Hermite messages, mean-field between the layers,
`limit_stack_depth` for a 900-node chain). The smoother's posterior on
z is the level. A `Levelled` channel wrapper carries it: the whitening
divides each sample by its level and is exact for the scaled process,
and the AR node sees the de-levelled residuals so the base sd stays the
base. On a synthetic unit AR(1) (ρ = 0.9) scaled by 3 for a stretch the
smoother reads 2.89 inside and 0.95 outside; the flat whitening leaves
a residual power of 9.8 inside the stretch, the levelled one 0.91
overall. On the real-cook mock's pit residual at the truth the level
after the fuel add reads 2.0 against 2.4 in the first quarter hour,
1.64 against 1.74 in the next half hour, 1.04 against 1.12 after that,
in one second for 877 samples (`estimator/test_volatility.jl`).

**The stream on the real-cook plant** (`estimator/mock_stream_realcook.jl`,
record `data/mock/mock_realcook_stream.json`, figure
`figures/estimator_realcook_stream.png`): two-hourly refits to the pull
on the graph engine, every channel levelled, the noise carried from
batch to batch, one AR round each.

| channel (spacing) | the plant's sd | identified at 14 h | ρ per step, identified / the plant's | misfit z at 14 h |
|---|---|---|---|---|
| pit (2 min) | 1.5 °C | 1.39 ± 0.19 | 0.963 ± 0.010 / 0.961 | 3.0 |
| core (2 min) | 0.5 °C | 1.2 (the calibration's floor) | 0.992 ± 0.005 / 0.990 | −3.4 |
| humidity (10 min, censored) | 100 Pa | 94 ± 25 | 0.971 ± 0.073 / 0.957 | −0.7 |
| scale (30 min) | 20 g | 10.5 ± 1.5 | 0.881 ± 0.20 / 0.955 | 0.9 |

The final latents sit at a median of 0.97 posterior sd from truth (one
at 3.9, the spread parameter), the total residual power 0.94 of its
expectation. Reading it:

- **The pit's level follows the fire.** The inferred level at the 14 h
  refit tracks the plant's bump at 12 h (the figure's first panel) and
  the pit's identified sd and ρ land on the plant's within their
  widths. Without the level the bump would have been read as a rise in
  the base noise and carried forward, as the first pass did (2.4 °C at
  10 h).
- **The core sits at the calibration's floor, and the check says so.**
  The between-cook floor holds the meat probe's sd at 1.2 °C; this mock
  generates 0.5. The misfit check reads −3 to −5 for the core all cook:
  the model claims more noise than the data show, which is the floor's
  doing, not the identification's. On a real probe the calibration
  says 1.2 is what one gets; the mock is the optimist here.
- **The scale's sd reads half.** Twenty-six samples of a process with
  ρ = 0.5 per sample, and the mass block's own latents absorb part of
  the residual; its ρ is wide (± 0.2). More samples or a tighter
  prior; the misfit stays inside the band.
- **Near the unit root, the marginal is not what is identified.** At a
  two-minute step the core's ρ is 0.99 and the AR node's marginal sd
  and ρ trade off along a ridge (the first pass reported 2.05 ± 3.9 at
  ρ = 0.999) while the innovation sd, the quantity the whitening uses,
  was within 35 %. The prior's width floor keeps the pair in range.
- **The iterator's floor.** Two batches of the first pass stalled with
  every negligible step rejected at the objective's jitter and skipped
  their noise round; a rejected step under the tolerance is now
  convergence, as in the outer engine.

**The replan as a decision** (`estimator/horizon.jl`: `plan_loss`,
`ReplanPolicy`, `replan_due`). The question raised was whether the
reactive graph could make replanning intelligent rather than periodic.
Replanning on every posterior update is the wrong reading of that: a
check is cheap (the standing plan's expected loss under the new draws
is six plant runs, no optimisation) and a replan is not (a graph solve
with a Jacobian per draw per iteration), and the standing plan is
already the best answer to the last posterior. What changes the plan's
value is new information, and what makes a decision urgent is
irreversibility. So the policy: after each hourly refit the standing
plan is re-encoded in the current free set and its expected loss
computed under the new draws; a replan fires when that loss has risen
by more than one half-squared-sd unit since the plan was made, when a
planned event falls due within the next hour (it is about to commit
and become irreversible: decide it on the freshest posterior), when
the misfit check reports excess residual power (the model is wrong
somewhere; the plan's outlook is unreliable; too little power means
the noise model is conservative, which threatens nothing, and on this
plant the core's floor keeps that side firing all cook), or when four
hours have passed without one (`estimator/test_horizon.jl`, seven
testsets).

**The closed loop on the real-cook plant**
(`estimator/mock_closed_loop_realcook.jl`, record
`data/mock/closed_loop_realcook.json`, figure
`figures/estimator_closed_loop_realcook.png`). The plant above, read
by its instruments; every hour the estimator refits on the graph
engine with the channels at their rates, the noise identified and
levelled; the standing plan is scored under six posterior draws; the
policy decides. The plan's physics runs on a ten-minute grid, the
executed damper interpolated onto the plant's two-minute one.

| check | reason | committed | loss: standing → new | wrap / pull / serve [h] | spritz / add [h] | core at the pull [°C] | water | bark L* | core at serve [°C] |
|---|---|---|---|---|---|---|---|---|---|
| 0 h | the first plan |  | 192.41 | 5.6 / 8.8 / 9.2 | 6.3 / 2.8 | 76.7 ± 30.2 | 0.625 ± 0.329 | 40.0 ± 11.2 | 75.1 ± 29.4 |
| 1 h | loss rose |  | 589.02 → 6.63 | 12.8 / 15.0 / 16.4 | 6.0 / 8.5 | 91.7 ± 6.4 | 0.279 ± 0.061 | 28.5 ± 2.7 | 83.4 ± 6.5 |
| 2 h | loss rose |  | 48.45 → 10.07 | 12.9 / 15.8 / 17.9 | 7.0 / 9.6 | 90.9 ± 8.5 | 0.421 ± 0.039 | 34.7 ± 4.1 | 82.3 ± 10.2 |
| 3 h | hold | | 8.75 vs 10.07 | | | | | | |
| 4 h | loss rose |  | 11.88 → 3.34 | 14.5 / 21.2 / 22.9 | 11.2 / 13.6 | 92.5 ± 4.6 | 0.411 ± 0.014 | 32.8 ± 3.2 | 86.6 ± 5.0 |
| 5 h | hold | | 0.85 vs 3.34 | | | | | | |
| 6 h | hold | | 1.14 vs 3.34 | | | | | | |
| 7 h | hold | | 2.75 vs 3.34 | | | | | | |
| 8 h | floor |  | 1.62 → 0.31 | 15.9 / 20.5 / 22.0 | 12.0 / 13.2 | 92.7 ± 0.8 | 0.367 ± 0.011 | 31.8 ± 2.5 | 86.9 ± 1.0 |
| 9 h | hold | | 0.70 vs 0.31 | | | | | | |
| 10 h | loss rose |  | 2.06 → 0.51 | 15.2 / 19.7 / 21.3 | 13.2 / 14.2 | 92.5 ± 0.9 | 0.378 ± 0.007 | 33.6 ± 2.3 | 85.5 ± 1.2 |
| 11 h | loss rose |  | 3.59 → 1.07 | 14.4 / 18.3 / 20.0 | 13.7 / 13.7 | 92.0 ± 2.0 | 0.393 ± 0.005 | 32.9 ± 2.3 | 86.1 ± 2.7 |
| 12 h | hold | | 0.87 vs 1.07 | | | | | | |
| 13 h | event due |  | 2.43 → 0.52 | 15.2 / 19.7 / 21.3 | 15.6 / 15.3 | 92.0 ± 1.3 | 0.375 ± 0.002 | 33.2 ± 1.5 | 85.3 ± 1.9 |
| 14 h | hold | | 0.42 vs 0.52 | | | | | | |
| 15 h | event due |  | 0.39 → 0.37 | 16.2 / 20.5 / 21.9 | 16.8 / 16.5 | 92.0 ± 1.0 | 0.360 ± 0.004 | 30.5 ± 1.1 | 86.7 ± 1.2 |
| 16 h | event due | wrap, add | 1.19 → 0.45 | 16.6 / 20.7 / 22.1 | 17.6 / 16.8 | 91.4 ± 0.8 | 0.350 ± 0.004 | 31.0 ± 0.7 | 85.8 ± 1.2 |
| 17 h | event due | spritz | 2.16 → 2.23 | 16.6 / 21.0 / 22.4 | 17.7 / 16.8 | 89.0 ± 1.2 | 0.351 ± 0.003 | 31.0 ± 0.6 | 83.2 ± 1.5 |
| 18 h | loss rose |  | 3.72 → 2.32 | 16.6 / 21.0 / 22.4 | 17.7 / 16.8 | 88.8 ± 0.7 | 0.351 ± 0.003 | 31.2 ± 0.5 | 82.8 ± 0.9 |
| 19 h | loss rose |  | 6.10 → 5.10 | 16.6 / 21.0 / 22.5 | 17.7 / 16.8 | 86.7 ± 1.0 | 0.350 ± 0.004 | 31.6 ± 0.7 | 80.6 ± 1.4 |
| 20 h | event due |  | 5.82 → 6.14 | 16.6 / 21.0 / 22.5 | 17.7 / 16.8 | 86.0 ± 0.4 | 0.351 ± 0.002 | 32.0 ± 0.6 | 79.7 ± 0.5 |
| 21 h | event due | pull | 5.73 → 5.88 | 16.6 / 21.2 / 22.8 | 17.7 / 16.8 | 86.2 ± 0.2 | 0.351 ± 0.001 | 31.9 ± 0.8 | 79.1 ± 0.3 |
| 22 h | event due |  | 5.58 → 5.58 | 16.6 / 21.2 / 22.7 | 17.7 / 16.8 | 86.3 ± 0.1 | 0.352 ± 0.001 | 32.2 ± 0.5 | 79.6 ± 0.1 |

Executed: wrap 16.59 h, fuel add 16.76 h, spritz from 17.7 h hourly at
50 g, pull 21.23 h, serve 22.73 h. The plant's actual outcome: core at
the pull 86.4 °C (goal 93 ± 2), retained water 0.352 (goal 0.35), bark
L* 31.8 (goal 32), core at serve 79.4 °C (floor 65). Twenty-three
checks, sixteen replans (seven with an event about to commit, seven on
a loss rise, one on the floor, the first), seven holds; 66 minutes in
all, of which the seven holds cost 96 seconds together (9 to 17 s
each: a warm-started refit and six plant runs) and the replans the
rest. The final latents sit at a median of 0.5 posterior sd from
truth, one at 2.8 (the spread parameter); the thickness's posterior sd
ends at 0.025. Reading it:

- **The prediction at the commit was the plant.** At the last replan
  the plan predicted 86.3 ± 0.1 °C at the pull, 0.352 ± 0.001 water,
  32.2 ± 0.5 L*, 79.6 ± 0.1 °C at serve; the plant delivered 86.4,
  0.352, 31.8, 79.4. The estimator knew this brisket.
- **The compromise went the other way from the matched run.** There
  the plan held the water above goal and gave the pull temperature
  1.6 sd; here it holds the water and the bark on goal and gives the
  pull temperature 3.3 sd. Both were right about their own outcome;
  which trade the optimiser picks depends on where the posterior sits
  (the two runs read different plants through different instruments),
  and the loss it settles at (5.6) says the goals conflict on this
  brisket: a hotter pull costs water at one sd per 0.05.
- **The trigger placed the replans, it did not thin them.** Sixteen
  replans against the twelve a two-hour clock made on the matched run
  and the twenty-three an hourly clock would have made here. Seven were
  spent where the information changed the plan's value (the loss rose
  from 192 to 589 after the first hour of readings, from 6.6 to 48
  after the second, by two to four units at 10, 11, 18 and 19 h as the
  posterior tightened on the thickness and the pull moved), seven
  where an irreversible decision was about to be taken, and the seven
  holds were hours in which the standing plan was worth as much or
  more under the new posterior than when it was made (0.85 against
  3.34 at 5 h, 0.42 against 0.52 at 14 h).
- **The event-due rule is a treadmill for a deferring plan.** From
  13 h the plan kept its wrap, fuel add and spritz just beyond the
  one-hour lead and the rule replanned every hour until the posterior
  wanted them (the wrap and fuel add committed at 16 h, the spritz at
  17 h, the pull at 21 h). The two-hour clock did the same in coarser
  steps (the matched run's wrap sat at 15 to 16 h for six hours before
  it committed at 15.8). A rule that commits an event when a replan
  leaves it where it stood would cut those hours; it is not built.
- **The misfit trigger never fired, and rightly.** The total z-score
  ran from −1 to −3.9 all cook: the core probe's floor, as in the
  stream. The two-sided form fired at the first hour and would have
  fired every hour after; the one-sided form is what the policy needs.
- **The pit's base sd and its level share one product.** The base
  sat at the calibration's floor (1.2 °C) with the level carrying the
  rest (the product 1.5 to 2.0 °C against the plant's 1.5 base), and
  the level read 2.6 at the check after the executed fuel add. The
  whitening uses the product, which is identified; the split is the
  prior's.
- **A replan can score above the standing plan by a few tenths** (17 h:
  2.16 → 2.23; 20 h: 5.82 → 6.14): the replan minimises the goals plus
  the plan prior and starts from the standing plan re-encoded in the
  current window (its damper knots move with the window), while the
  check scores the goals alone.


## The receding horizon on the graph: the pilot without hardware (2026-09-08)

`estimator/horizon.jl` closes the plan block into a pilot. The past is
data: every replan solves the whole cook from the start with the
executed damper log and the executed events as inputs, and only the
future carries decisions, in seconds on one long grid, with the
duration as the serve time at which the outcomes are read. Events are
candidates until they fall due: one whose planned time lands inside the
next interval is executed at that time and fixed in every later replan,
so the free set shrinks (the spritz train starts when its first spritz
is due; after the pull only the serve time is free). The Gaussian goal
at serve became a one-sided floor, a softplus hinge per posterior draw
that costs nothing above 65 °C and half the squared shortfall below it,
the expected shortfall over the draws. The estimator's slab physics
takes the executed events as inputs like the damper log. Unit tests
(`estimator/test_horizon.jl`) hold the hinge, the free set, the
encode/decode round trip with the ordering wrap < pull < serve and
spritz, fuel add before the pull, the commit rule, the executed damper
preserved, and the plan physics at the truth with every event on.

`estimator/mock_closed_loop.jl` then runs the pilot on the matched mock
plant with nothing but readings: every two hours the plant has run the
executed schedule and produced pit, core, humidity and scale readings
(its AR(1) noise pre-drawn on the grid), the estimator refits on the
graph engine, the posterior's six draws feed the replan on the graph
(warm-started from the previous plan), the due events commit, the
damper executes (record `data/mock/closed_loop.json`, figure
`figures/estimator_closed_loop.png`):

| replan at | committed | wrap | pull | serve | spritz from | fuel add | predicted core at the pull | water | bark L* | posterior sd of L | misfit z |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 h, the prior | | 5.6 h | 8.8 h | 9.2 h | 6.3 h | 2.8 h | 77 ± 30 °C | 0.63 ± 0.33 | 40 ± 11 | 0.250 | |
| 2 h | | 8.6 | 16.0 | 18.3 | 8.8 | 8.5 | 90 ± 9 | 0.43 ± 0.07 | 32 ± 4 | 0.122 | −0.4 |
| 6 h | | 14.1 | 17.6 | 19.3 | 11.6 | 10.7 | 92.4 ± 2.9 | 0.41 ± 0.02 | 31 ± 5 | 0.105 | 0.1 |
| 12 h | | 15.7 | 19.3 | 21.2 | 16.1 | 14.3 | 92.5 ± 1.2 | 0.39 ± 0.01 | 35.1 ± 0.5 | 0.098 | 0.8 |
| 14 h | wrap at 15.8 h, fuel at 15.8 h | | 20.0 | 21.7 | 16.3 | | 91.8 ± 1.9 | 0.386 | 34.3 | 0.096 | 1.2 |
| 16 h | spritz from 17.3 h, hourly, 50 g | | 20.3 | 21.8 | | | 90.2 ± 1.6 | 0.387 | 30.5 | 0.094 | 1.1 |
| 20 h | pull at 20.5 h | | | 22.1 | | | 90.0 ± 1.3 | 0.388 | 31.7 | 0.075 | 0.7 |
| 22 h | serve at 22.0 h | | | | | | 89.4 ± 1.2 | 0.387 | 31.3 | 0.065 | 0.8 |

The plant's actual outcome under the executed schedule: core at the
pull 89.7 °C (goal 93 ± 2), retained water 0.387 (goal 0.35), bark
L* 32.4 (goal 32), core at serve 83.0 °C (floor 65). Twelve replans,
estimator and plan together, in 33 minutes: under three minutes per
replan on a two-hour cadence.

Reading it:

- **The plan settles within one interval of readings.** From the prior
  alone the pilot would run the last knot nearly open and pull at 8.8 h,
  with a predicted pull temperature spread of 30 K across the draws;
  two hours of readings take the spread to 9 K and the plan to a
  recognizable cook, and by hour 6 to 3 K. The posterior's width on the
  thickness halves in the first interval and reaches a quarter by the
  end.
- **The commits are the plan's own, not surprises.** The wrap sat at
  15 to 16 h from hour 8 on and committed at 15.8; the pull drifted
  later as the cook went (16 h at the first replan with data, 20.5 h at
  the commit) while the predicted pull temperature stayed within a
  degree of its final value; the serve time followed the pull by 1.5
  to 2 h throughout.
- **The pull lands 1.6 sd under its goal, and that is the compromise,
  not an error.** The plan holds the water above its goal and the bark
  on it; the pull temperature is what it trades. The prediction at the
  commit (90.0 ± 1.3) matched the plant (89.7).
- **The floor at serve never binds.** With a 1.5 h rest from a 90 °C
  pull the core is 83 °C at serve, eighteen degrees above the floor, so
  the tail term is zero throughout: a floor, unlike a target, does not
  pull the plan toward itself. That is the behaviour wanted; on a
  plant or a goal where the tail is live the same term carries the
  shortfall over the draws.
- **The estimator's misfit check stays inside ±1.8 for the whole cook**,
  as it should on the matched plant, with the executed events inside
  its physics.

What remains of pilot integration is the hookup: the readings from the
pit instead of the mock plant, the executed damper and events from the
pilot's journal, the plan back to it, through juliacall on the Julia
thread the pilot already holds, and an information term from the
estimator's Jacobian if the pilot is to pay for learning the fire.

## The full decision set on the graph (2026-09-08)

The planning experiment's block had three decisions. `estimator/plan_block.jl`
puts the full set on the graph: three damper knots in fractional time,
the wrap, the pull, the duration, the spritz (start, interval, dose)
and a fuel addition, ten unconstrained latents of one plan block with
the ordering built into the parameterization (the wrap and the spritz
start as fractions of the pull, the pull and the fuel add as fractions
of the duration, so wrap < pull < serve and spritz and fuel before the
pull hold by construction). The physics is the duration-scaled solve
`monolith_solve_tf`, which now takes the runtime exposed area like the
plain solve (one physics patch, the suite 368/368), at the estimator's
eight posterior draws from 6 h of the matched mock cook. The goals are
Gaussian: the core at the pull 93 ± 2 °C, the retained water 0.35 ±
0.05, the bark lightness 32 ± 4 L*, the core at serve 65 ± 3 °C; the
priors prefer a low damper and a 14 h cook. The same expected loss is
minimized by the gradient reference (damped Gauss–Newton, ForwardDiff
through the physics, ten columns per draw) and by the graph (per-draw
affine nodes inside the graph iterator), each writing its own record
(`data/mock/plan_block_reference.json`, `plan_block_graph.json`; the
summary and gate in `plan_block.json`).

| decision | prior | reference | graph |
|---|---|---|---|
| damper, three knots | 0.15, 0.15, 0.15 | 0.230, 0.395, 0.270 | 0.234, 0.396, 0.268 |
| wrap | 6.7 h | 10.96 h | 10.92 h |
| pull | 11.2 h | 15.66 h | 15.63 h |
| duration | 14.0 h | 22.28 h | 22.27 h |
| spritz start, interval, dose | 3.4 h, 0.98 h, 50 g | 8.48 h, 1.56 h, 46 g | 8.47 h, 1.57 h, 46 g |
| fuel add | 5.6 h | 12.08 h | 12.12 h |
| expected loss | 337.5 | 36.485 (423 s) | 36.476 (26 iterations, 492 s) |

| outcome over the draws | prior plan | the optimum | goal |
|---|---|---|---|
| core at the pull | 79.8 ± 1.7 °C | 92.1 ± 4.2 °C | 93 ± 2 |
| retained water | 0.61 ± 0.01 | 0.41 ± 0.03 | 0.35 ± 0.05 |
| bark L* | 42.3 ± 0.4 | 31.6 ± 3.3 | 32 ± 4 |
| core at serve | 72.1 ± 1.6 °C | 64.9 ± 2.6 °C | 65 ± 3 |

Reading it:

- **The graph and the reference find the same plan.** The objectives
  are 36.485 and 36.476, the graph's the lower; the plans sit 0.07 sd
  apart on a valley that flat, and every ordering holds. The first gate
  I wrote (0.05 sd) was tighter than two independently converged
  optimizers can be held to on such a valley; it is 0.2 now, with the
  objective within half a percent and the ordering as the other
  conditions.
- **The plan reads like a pitmaster's.** From a low damper everywhere
  and no events, the optimum runs the pit harder in the middle of the
  cook, wraps at 11 h (seventy percent of the way to the pull, the
  crutch's usual place), pulls at 15.7 h, spritzes from 8.5 h every
  hour and a half, adds fuel at 12 h, and lengthens the cook to 22 h so
  a 92 °C pull rests down to 65 °C at serve. The duration prior (14 h,
  30 % in log) is overridden by 1.5 of its widths: the serve-temperature
  goal decides the cook's length, as the time planner's framing says it
  should.
- **What the plan does not yet have** is the receding-horizon layer:
  the candidate-commit-irreversible logic for the wrap and pull, the
  replan from the current state with executed events fixed, and a tail
  objective in place of the Gaussian goal at serve. Those are the
  pilot-integration item, and the block now exists for them to act on.

## The iterator on the graph (2026-09-08)

The planning experiment's diagnosis was that RxInfer's delta node
expands at the inbound messages' means and offers no expansion point,
so iterated linearization has to be supplied from outside. It can be
supplied from inside `infer`, and now is (`estimator/iterated.jl`). The
model stays exactly affine (the estimator's Affine object, which the
built-in Linearization rule handles exactly wherever it expands), and
two facts of the batch loop carry the iteration: the data are re-fed
from the passed arrays every iteration, and an `after_iteration`
callback can read the current marginal (through `on_marginal_update`)
and stop the loop. The callback evaluates the true objective at the
marginal's mean, halves the move until the objective drops (the
estimator's line search), moves the center, rebuilds the affine model
there (one physics Jacobian) and re-centers the prior's local
quadratic, whose mean and precision are data arrays. Damping is a
Levenberg–Marquardt node on the graph: an identity delta node anchored
at the current center with a data precision the callback scales, raised
on a rejected move, lowered on an accepted full one. One `infer` call is
the whole solve; a final pass with the damping released gives the
posterior at the converged center.

Two gates (`estimator/test_iterated.jl`) and a regression
(`estimator/test_engine.jl`):

| problem | reference | on the graph |
|---|---|---|
| the planner, posterior width | expected loss 101.2894 | 101.2897, the plan within 0.001 sd, 10 iterations, 56 s |
| the planner, width shrunk tenfold | 9.6263 | 9.6263, the plan within 0.000 sd, 11 iterations |
| the real cook (the estimator's gate) | MAP within 0.33 sd, F 133.1206 (the outer loop's fixed point) | MAP within 0.324 sd (u_leak), median 0.046, widths within 2.8 % of the Fisher, F 133.1118, 17 iterations, 180 s |
| three hourly batches of the matched mock, humidity + scale | the outer loop's record | 1e-4 sd, identical objectives and iteration counts |

`solve_map_channels` and `stream_fit` take `engine=:graph` (the default)
or `:outer` (the loop around `infer`, kept as the reference). The
graph's own cost is 40 ms per iteration against a 3.5 s physics
Jacobian, so the streaming refits, the noise alternation and the misfit
check run unchanged on the graph engine. What the graph does is still
the linear-Gaussian solve at the current center; what it now also does
is own the loop, which is the difference between a solver that uses a
factor graph and a factor graph that solves. The upstream item stands:
an expansion-point hook on the delta node would let the nonlinear
function itself sit in the graph, and the affine rebuild in the callback
is what that hook would do.

## Planning as inference in RxInfer: the experiment, and the decision (2026-09-08)

The question was whether the pilot's uncertainty-aware planner should be
a factor graph. The bounded experiment (`estimator/plan_experiment.jl`
and `plan_experiment_outer.jl`, record `data/mock/plan_experiment.json`):
a plan ζ of two damper knots (before and after 6 h) and the pull time,
the monolith solved from the start of the matched mock cook per
parameter draw, Gaussian goals on the core temperature at the pull
(93 °C ± 2) and the retained water (0.35 ± 0.05), a prior on ζ that
prefers a low damper and a 12 h pull, and the estimator's posterior at
6 h of that cook (pit + core + humidity + scale, its full covariance
recomputed) as the parameter uncertainty. Three solvers of the same
problem: **R**, the expected loss over eight posterior draws minimized
by damped Gauss–Newton with ForwardDiff through the physics (the
reference optimum); **B**, the graph with the posterior held fixed as
the same eight draws inside per-draw Unscented delta nodes (two outputs,
three inputs each), so message passing minimizes the same expected
loss; **A**, the naive graph with the physics latents in one block with
the controls and one Unscented node, where the goal is free to update
the belief about the world. Then the certainty-equivalence test: R and
B again with the posterior's width shrunk tenfold (the same standard
normals, scaled). A second pass (`plan_experiment_damped.jl`) put the
graph inside the estimator's damped outer loop, and a third
(`plan_experiment_affine.jl`) built the graph the estimator's way, with
per-draw affine nodes at the current plan from a three-column Jacobian,
inside the same loop. The sample-based node
was not run: CVIProjection cannot be installed here
(ExponentialFamilyProjection pins RecursiveArrayTools at or below 3.54,
the physics' BDF solver needs 4.2 or later), and the older CVI is
deprecated in this RxInfer (5.5.2, ReactiveMP 6.5.0). The Unscented
node ran at
its default spread, alpha = 1e-3, which makes it a numerical
linearization at its evaluation point, not a moment match over the
uncertainty; the uncertainty was carried by the draws.

| solver | plan: damper before 6 h, after, pull | expected loss | pull-time core temperature over the draws | retained water |
|---|---|---|---|---|
| prior plan | 0.15, 0.15, 12.0 h | 645 | 69.2 ± 2.2 °C | 0.566 ± 0.012 |
| R, posterior width | 0.097, 0.572, 11.75 h | 101.3 | 89.6 ± 9.3 °C | 0.402 ± 0.025 |
| B, 12 inner iterations | 0.137, 0.439, 12.85 h | 141.0 | 83.7 ± 7.1 °C | 0.402 ± 0.026 |
| B inside the outer loop, line search only | 0.116, 0.513, 12.94 h | 103.8 | 89.9 ± 9.9 °C | 0.374 ± 0.026 |
| B inside the damped outer loop | 0.143, 0.504, 13.11 h | 107.6 | 90.0 ± 10.2 °C | 0.366 ± 0.026 |
| A, the naive graph | 0.193, 0.189, 20.1 h | | | |
| R, width shrunk tenfold | 0.155, 0.560, 12.50 h | 9.6 | 92.7 ± 1.2 °C | 0.334 ± 0.003 |
| B, width shrunk tenfold | 0.152, 0.528, 12.81 h | 15.5 | 90.3 ± 1.1 °C | 0.335 ± 0.003 |
| B in the damped outer loop, width shrunk tenfold | 0.161, 0.547, 12.75 h | 9.7 | 92.6 ± 1.2 °C | 0.328 ± 0.003 |
| B with affine nodes at the current plan, posterior width | 0.098, 0.573, 11.74 h | 101.289 | 89.6 ± 9.3 °C | 0.402 ± 0.025 |
| B with affine nodes at the current plan, width shrunk tenfold | 0.155, 0.560, 12.50 h | 9.626 | 92.7 ± 1.2 °C | 0.334 ± 0.003 |

Reading it:

- **Certainty equivalence does not hold, and the mechanism is plain.**
  The expected-loss optimum moves with the posterior width: at the
  posterior's width the plan runs the early pit at 0.10 and pulls at
  11.75 h, at a tenth of the width 0.155 and 12.5 h. The pull-time
  temperature's spread across the draws (9.3 K against 1.2 K) enters the
  quadratic loss as a variance term that the plan can reduce, and it
  does, holding the mean temperature 3 K below the target rather than
  widening the spread. That is a cautious plan; a serve-time chance
  constraint would need the tail, i.e. draws, which both R and B
  already use.
- **The naive graph is unusable.** With the physics latents in the
  block, one Unscented node at a 20-dimensional input took one step at
  the prior plan (the evaluation point of a single node never moves,
  the measured behaviour) and pushed the pull to 20 h, past the end of
  the cook, because at the prior plan the local model says a later
  pull is hotter; the belief about the world moved by 0.16 sd.
- **The per-draw graph computes the right objective and is cheap, but
  it is a local-model provider, not an optimizer, and its evaluation
  point is not under the caller's control.** Its inner iterations
  oscillate and stop 3 sd from the optimum (141 against 101). A
  nonlinear delta node linearizes at its inbound message's mean, and
  with a Gaussian prior the local quadratic at the current plan is the
  prior itself, so that mean is the prior mean until the damping is
  large, and by then the step is negligible (the evaluation point does
  move with the initialization once several nodes share the block: a
  step from a point near the optimum is [0.27, −0.04, 0.08] with the
  graph initialized there and [0.95, −0.66, 0.47] with it initialized
  at the prior mean). Inside the estimator's damped outer loop the
  graph converges at the small width (9.72 against 9.63, within 0.6 sd
  of the reference plan) and stops six units short at the posterior's
  width (107.6 against 101.3, within 1.9 sd), in about two thousand
  solves against the reference's few hundred. The estimator never met
  this because its affine node is built at the current point outside
  the graph, which is also why it converges: the graph then solves a
  linear-Gaussian problem. The proof is the third pass: with the
  per-draw nodes made affine at the current plan, the graph reproduces
  the reference to the last digit at both widths (101.289 and 9.626,
  the plan within 0.01 sd), and its iteration trace is the reference's
  own, in 392 solve-equivalents and 50 s. The whole shortfall was the
  expansion point. The reason it cannot be fixed inside the graph is in
  the rules: the delta node's marginal rule (ported from ForneyLab)
  expands at the inbound messages' means, smooths against the output's
  message, and the input rule divides that local marginal by the
  inbound; no rule expands at the current marginal and there is no hook
  to set the point, for Linearization, Unscented or CVI alike. The
  iterated extended Kalman fix of the literature is not implemented,
  and "iterative" in RxInfer 5.5.2 means supplying the linearization
  from outside, which is the estimator's design and why it converges.
  A monolith solve costs 60 ms; it is the Jacobian that costs, and R
  needs only three columns of it. Whether a sample-based message does
  better is untested, and it would expand at the same point.
- **Decision.** The planner stays on the gradient side: the app's
  expected-objective optimizer over posterior draws (`control.py`,
  Adam over draws with the wrap and pull shared) and the time planner's
  chance layer, fed by the estimator's hourly posterior instead of the
  offline calibration draws, with the serve-time and safety
  constraints on a posterior quantile and an information term from the
  Jacobian the estimator already computes. RxInfer stays the estimator
  and its misfit check. To plan with the graph one rebuilds the
  estimator's affine machinery around it, at which point the graph is
  solving a linear-Gaussian problem the reference solves directly; the
  elegance of one graph does not survive the outer loop it needs.

## Noise as graph structure: the AR node, and what it taught about priors (2026-09-08)

`estimator/noise.jl`: a channel's AR(1) discrepancy is RxInfer's AR
node now, not a fixed whitening. The residual process of a channel,
r_k = y_k − pred_k(ẑ), is one AR chain per contiguous run of the mask
(a lid lift breaks the chain) with the coefficient θ and the innovation
precision γ shared across runs, q(x) q(θ) q(γ) as in RxInfer's own AR
tests, and the whitening's parameters read off as the marginal sd
σ = 1/√(γ(1−ρ²)) and ρ. The physics block does not change: it sees the
noise through the whitening, and the two alternate (fit the physics at
the current whitening, the AR node on the residuals at the mode, refit)
until the noise stops moving; the stream carries one batch's estimate
into the next. The prior is the calibration's posterior in data form
(`estimator/export_noise_prior.py` writes the real cook's NUTS moments,
σ_eff = σ + floor, to `data/calibration/noise_prior_real.json`; the
loader turns them into the AR node's Normal on θ and moment-matched
Gamma on γ). `estimator/test_noise.jl`: a synthetic AR(1) at the cook's
length is recovered within its posterior width, a run split by a lid
lift agrees with the unsplit series, a strong wrong prior is overridden
by 115 points, the posterior narrows with the series. The node's
mean-field split between θ and γ costs a bias on the marginal sd that
grows with the correlation (σ/truth 1.00 at ρ 0.6, 0.95 at 0.83 and at
0.95 over twelve draws each, half a posterior width); the conjugate
variant that keeps (θ, γ) joint fails in its marginal rule at order 1
(a scalar added to a 1 × 1 matrix), the third upstream finding.

Two things had to be fixed in the solver before the alternation could
work, both found by it. The graph's local-quadratic prior node goes flat
where a Beta site (the draft exponent, the leak fraction, the fire base
rate: the pit ridge) sits at an extreme logit, and an undamped
Gauss–Newton step then runs to infinity (1e5 in z); the line search
failed at its floor, the fit stopped unconverged, and the AR node read
the unconverged residuals as a huge near-unit-root noise (a pit sd of
90 K, ρ 0.999) that switched the channel off and locked the fit. A
rejected or absurd step now raises a Levenberg–Marquardt damping on the
prior node's precision and retries on the same Jacobian, a solve that
fails is an infinite objective, a rejected negligible step is
convergence at the jitter floor, and the noise is estimated only on a
converged fit. The gate's reference run reproduces the calibration's MAP
unchanged (0.33 sd, F 133.12).

`estimator/gate_noise_real.jl`, the real cook (record
`data/calibration/estimator_gate_noise_result.json`):

| start and prior | pit σ (calibration 1.54) | pit ρ (0.831) | core σ (1.70) | core ρ (0.947) | MAP vs calibration, max / median sd | residual power |
|---|---|---|---|---|---|---|
| wrong start (3.0 K, 0.5), weak prior | 1.20 ± 0.14 (the floor) | 0.729 ± 0.065 | 1.27 ± 0.18 | 0.917 ± 0.021 | 1.10 / 0.31 | 1.11 |
| the calibration's prior, started at its means | 1.26 ± 0.13 | 0.758 ± 0.045 | 1.27 ± 0.16 | 0.934 ± 0.015 | 0.87 / 0.22 | 1.20 |
| fixed calibration noise (reference, no AR node) | | | | | 0.33 / 0.05 | 0.93 |

`estimator/mock_stream_noise.jl` streams `mock_plant.jl --altnoise`,
the slab plant with a noisier pit probe, a quieter and less correlated
core probe, a worse humidity sensor and a better scale, every channel
starting at its prior's means (records
`data/mock/mock_1d_alt_tightprior_noise_stream.json` and
`mock_1d_alt_noise_stream.json`, figures
`figures/estimator_noise_mock_1d_alt_tightprior.png` and
`figures/estimator_noise_mock_1d_alt.png`). At 19 h, the full set:

| channel | truth σ, ρ | prior σ, ρ | the calibration's widths as the prior | with the between-cook floor |
|---|---|---|---|---|
| pit | 3.0 K, 0.60 | 1.42, 0.78 | 3.27 ± 0.28, 0.686 ± 0.047 | 3.36 ± 0.30, 0.689 ± 0.048 |
| core | 1.0 K, 0.85 | 1.65, 0.94 | 1.62 ± 0.11, 0.942 ± 0.005 | 1.20 (the floor) ± 0.18, 0.880 ± 0.038 |
| humidity | 200 Pa, 0.5 | 100, 0.8 | 204 ± 16, 0.454 ± 0.080 | 204 ± 16, 0.456 ± 0.080 |
| scale | 10 g, 0.2 | 20, 0.5 | 9.4 ± 0.6, 0.067 ± 0.090 | 9.4 ± 0.6, 0.066 ± 0.090 |
| physics at 19 h, max / median err/sd | | | 2.09 / 0.59 | 2.05 / 0.57 |
| residual power per row, misfit z | | | 1.04, 1.1 | 1.00, 0.4 |

Reading it:

- **The noise is identified as the cook goes.** The pit's sd climbs from
  the prior's 1.4 K to 3.3 K within four hours, the humidity sensor's
  from 100 Pa to 200 and its correlation from 0.8 to 0.5, the scale's
  from 20 g to 9, each with a width that narrows through the cook; the
  physics lands (every latent within 2.1 sd) and the whitened residual
  power ends at 1.00. What every channel identifies well is the
  innovation variance; the decomposition into a marginal sd and a
  correlation is a ridge (σ²(1−ρ²) constant) on which 115 points say
  little, so it follows the prior's correlation: the pit's ρ stops at
  0.69 between the prior's 0.78 and the truth's 0.60, with the sd 10 %
  high to match.
- **A within-cook posterior is the wrong prior across cooks.** The core
  probe's correlation is 0.943 ± 0.005 on the real cook, a hundred times
  more precise than 115 points of another cook can be, and another
  probe or thermal contact has another ρ; as the prior it pinned the
  mock's core at 0.942 against a truth of 0.85 and its sd at 1.62
  against 1.0, and the misfit check could not object (the innovation
  variance was right). A between-cook floor on the prior widths (0.05
  on ρ, 20 % on σ, `noise_priors_from_file`) lets the core move: ρ 0.88
  ± 0.04, the sd to the 1.2 K floor with the truth inside the width.
  "The pooled posterior as the prior in data form" therefore means the
  single-cook posterior with that floor until C3's pool across cooks
  gives the between-cook spread itself.
- **On the real cook the AR node's sd sits 15 to 20 % below the
  calibration's on both channels**, from either start, with ρ within 1.6
  sd and the physics MAP within 0.9 sd; the whitened core residual then
  reads a power of 1.2 (z 2.5), i.e. the estimate is somewhat tight.
  Part is the node's bias (5 % at these correlations), part the
  calibration's additive-floor parameterization with its own prior on
  the excess and a joint posterior rather than residuals at the mode.
  The NUTS posterior stays the reference; the AR node is what a cook can
  run in an hour.

## The 2-D physics behind the registry: the cylinder read by its own engine (2026-09-08)

The channel registry now takes its physics as an argument
(`estimator/physics2d.jl`): `Physics1D` is the monolith slab with the
exposed area as a latent, `Physics2D` the axisymmetric engine, a
cylinder of radius exp(z_R) and height 2 L on the engine's grid with
its fixed convective coefficient. Both build ONE stacked output, so
every channel works unchanged. The cylinder is mapped onto the slab's
conventions: its axis sampled at uniform depths from the top face to
the mid-plane is the probe line (index 1 the center, the slab's order),
the top face's center is the surface temperature, the volume integral
is the mass, and the bark lightness is not available (the 2-D bark is a
distribution at the end of the cook, not a series). The mass block
belongs to the physics now and the scale reads it, times the shape
factor; the area and thickness columns come through the physics'
Jacobian rather than an explicit factor, which the sensor table
confirms to 4.5e-6. The latents are the fifteen physics with L as the
half-height and m_conv unused (the engine has no draft exponent; its
column is zero and it keeps its prior), plus the radius (LogNormal,
30 %, a 10 cm median) and the probe depth. `estimator/test_physics2d.jl`
holds the stacks to their plants: at the true latents the 1-D stack
reproduces the monolith mock's clean readings through every channel to
the adaptive solver's tolerance (2e-6 to 6e-5) and the 2-D stack the
cylinder mock's to machine precision (1e-15), the radius column of the
2-D Jacobian matches a directional ForwardDiff derivative, and the mass
block's area column its derivative. Cost at the mock's 16 by 14 grid: a
2-D solve 1.1 s, its 15-column Jacobian 1.6 s (cheaper than the
monolith's), an hourly batch 15 to 50 s.

Found on the way: the cylinder engine blows up on an overshooting
Gauss–Newton step from a cold start against a 20 g scale (a 10 cm
prior radius against 13.5 cm true is a 45 % mass error at 180 σ), and a
NaN objective compared as "not worse" and was accepted. The line search
treats a non-finite objective as a worse step now, and the axis
sampling propagates a blown-up solve instead of throwing.

`estimator/mock_stream_2d.jl` streams both plants through the cylinder
engine; with the slab-physics runs of the earlier entries this closes
the square (records `data/mock/mock_2d_phys2d_stream.json` and
`mock_1d_phys2d_stream.json`, figures
`figures/estimator_stream_mock_2d_phys2d.png`,
`figures/estimator_stream_mock_1d_phys2d.png` and the misfit check of
both in `figures/estimator_misfit_phys2d.png`). At 19 h, m_conv aside:

| plant → engine | channel set | max err/sd (latent) | median | radius (truth) | residual power per row | z χ² total (worst channel) |
|---|---|---|---|---|---|---|
| cylinder → cylinder | pit + core | 2.2 (a0) | 0.55 | 0.121 ± 22 % (0.135) | 1.22 | 2.9 (meat probe 2.5) |
| cylinder → cylinder | + humidity + scale | 1.5 (a0) | 0.83 | 0.133 ± 5.5 % (0.135) | 1.03 | 0.9 (meat probe 2.4) |
| cylinder → cylinder | + rulers (thickness, radius) | 2.4 (h_c) | 1.07 | 0.132 ± 1.8 % (0.135) | 1.03 | 0.9 |
| cylinder → slab (earlier entry) | + humidity + scale | 17.2 (a0) | 3.2 | | 1.58 | 9.4 (pit 9.0) |
| slab → cylinder | pit + core | 4.0 (a0) | 0.67 | 0.135 ± 24 % (equivalent 0.135) | 1.06 | 1.1 |
| slab → cylinder | + humidity + scale | 15.9 (a0) | 2.65 | 0.173 ± 5 % (equivalent 0.135) | 1.83 | 13.2 (humidity 18; 75 at 1 h) |

Reading it:

- **The right physics lands, and the check goes quiet.** The cylinder
  read by its own engine with humidity and a scale has every latent
  within 1.5 sd, the radius at 13.3 cm ± 5.5 % against 13.5, residual
  power 1.03 and a total z of 0.9, where the slab physics on the same
  readings was 17 sd wrong at a z of 9.4. Rulers on the thickness and
  the radius take the radius to 1.8 %. The surviving 2 sd on the fire
  base rate is the pit ridge from a cold start (−5 sd at 1 h,
  recovering), as on the matched slab.
- **The meat-probe channel's z of 4 at hours 6 to 12 is the mock's own
  noise draw**, not the fit: at the true latents it scores 4.2 there
  and the fit's objective is below the truth's, as a MAP's should be;
  whitened at the clean values, the cylinder mock's core noise carries
  1.9 times the expected power in its first six hours (every other
  channel of both mocks sits near unit power with no autocorrelation).
  The check reports what is in the data, at the truth as well as at
  the fit.
- **The mismatch the other way is flagged from the first hour, by the
  humidity sensor.** The slab plant read by the cylinder engine looks
  fine on temperatures alone (residual power 1.06, z 1.1, the fire on
  its ridge). With humidity and a scale the humidity channel scores
  z 75 at hour 1 and 18 at 19 h, the radius is pushed 5 sd from the
  equivalent cylinder's and the fire base rate lands 16 sd off. What the
  two engines disagree about most is the vapor they put in the pit,
  which is what the humidity sensor reads, so it is the channel that
  tells a slab from a cylinder, in either direction and immediately.
- **Choosing the physics is the check's job, not the estimator's.** The
  registry does not pick between slab and cylinder; a cook's operator
  does, and the misfit check on the humidity channel says within an
  hour whether the choice was right.

## A shape factor on the scale: what a geometry latent fixes, and what it does not (2026-09-08)

The misfit check's first ask on the cylinder plant was a geometry latent.
The cheapest one is a shape factor on the scale's mass bookkeeping,
m = s ρ_d A L (1 + W̄ + F̄) with s = 1 for a slab and πR²H / (A L) for a
cylinder (`MassScale(...; shape_latent=)`, prior LogNormal(0, 0.3)).
`estimator/mock_stream_shape.jl` streams the two scale-bearing channel
sets on both plants with that eighteenth latent and the misfit check
recorded per batch; the figures are `figures/estimator_stream_mock_1d_shape.png`
and `figures/estimator_stream_mock_2d_shape.png`, the records
`data/mock/mock_1d_shape_stream.json` and `mock_2d_shape_stream.json`.

| plant | channel set | shape factor at 19 h (truth) | max err/sd (latent) | median | residual power per row | rulers' misfit z |
|---|---|---|---|---|---|---|
| matched | + humidity + scale | 1.003 ± 0.017 (1.000) | 1.4 (k_spread) | 0.96 | 1.04 | |
| matched | + rulers | 1.002 ± 0.017 (1.000) | 1.7 (k_m) | 1.20 | 1.04 | −0.5, −0.5 |
| cylinder | + humidity + scale | 0.81 ± 0.18 (0.637) | 17.2 (a0) | 2.25 | 1.58 | |
| cylinder | + rulers | 0.649 ± 0.078 (0.637) | 17.3 (a0) | 1.9 | 1.57 | −0.2, 0.2 |

Reading it:

- **The shape factor is a real latent, and the water's running out is
  what identifies it.** On the matched plant its width sits at 0.19
  (the scale's product with the area and the thickness, split only by
  the humidity sensor's hold on the area) for seventeen hours, then
  collapses to 0.017 in the batch that reaches the meat's dry-out: the
  scale reading flattens at 18.5 h with 41 % of the mass gone, and a
  dry-out time is a volume-per-area quantity, which is what a shape
  factor is. With the rulers pinning the area and the thickness the
  scale gives s directly from the first hour (0.08, then 0.017 at the
  dry-out). On the cylinder the meat does not dry out and s stays at
  the scale's precision (0.18) until the rulers split the product:
  0.649 ± 0.078 against a truth of 0.637.
- **It fixes the mass and only the mass.** With s free the rulers'
  readings agree with the fit (their misfit z from 6.3 and 13.9 to −0.2
  and 0.2): the scale's product constraint no longer contradicts a
  measured footprint and thickness. Nothing else moves. The fire base
  rate stays 17 sd off, k_spread −11, h_c +5 (+8 with the rulers),
  u_leak −5; the residual power per row is 1.58 against 1.58 without
  the latent, the misfit total z 9.4 against 9.4, the pit, humidity and
  meat-probe channels flagged exactly as before. What those channels see
  is the cylinder's heat and vapor exchange history (edges that heat
  and dry faster than any slab at any area), not its mass.
- **So the next geometry latent is the geometry.** The estimator's
  forward model for a cut that is not a slab has to be the 2-D physics
  (the slab2d Tesseract, with its Jacobian), not a factor on the slab;
  the shape factor stays in the registry as the honest mass bookkeeping
  for the scale channel.

## The misfit check: a confident posterior arrives with its own warning (2026-09-08)

`estimator/misfit.jl`: every refit reports, per channel and in total,
two statistics of its whitened residuals r = w − a(ẑ). The residual
power Σr² against its expectation n − tr(H), with H = B V Bᵀ from the
graph's posterior covariance so the fit's own absorption is charged to
each channel, as a χ² z-score; and the lag-1 autocorrelation as a
z-score (sd 1/√n for white residuals). Both are approximate (a fixed
truth under a proper prior leaves a small shrinkage bias, and the
autocorrelation ignores the hat matrix), which is the right precision
for a warning light. `estimator/test_misfit.jl` checks them where the
answer is known: white least-squares residuals over 400 draws sit in
the band with the exact degrees of freedom, and a slow structured error
lights both statistics on its own channel only. The check is attached
to every solve and stream record; `estimator/misfit_stream.jl` scores a
stored run, and `figures/estimator_misfit.png` draws it against the
cook hour for the two mock plants (matched against cylinder, the
humidity-plus-scale set solid, pit + core dashed).

Scored on the streaming runs of the entry below (the total over all
rows, and the channel that stands out):

| plant | channel set | hour | residual power per row | z χ² total | z ρ₁ total | worst channel: z χ² / z ρ₁ |
|---|---|---|---|---|---|---|
| matched | pit + core | 6 | 1.01 | 0.6 | −0.9 | meat probe 0.8 / −1.2 |
| matched | pit + core | 19 | 1.06 | 1.0 | −0.7 | pit 1.0 / −0.5 |
| matched | + humidity + scale | 19 | 1.04 | 1.1 | −1.8 | scale −0.2 / −1.6 |
| matched | + rulers | 19 | 1.04 | 1.1 | −1.8 | rulers −0.5, −0.5 |
| cylinder | pit + core | 6 | 1.39 | 3.0 | −1.4 | meat probe 3.7 / −1.0 |
| cylinder | pit + core | 19 | 1.22 | 2.8 | −0.9 | meat probe 2.4 / −0.9 |
| cylinder | + humidity + scale | 6 | 1.33 | 3.7 | 1.0 | meat probe 4.8 / −0.2 |
| cylinder | + humidity + scale | 19 | 1.58 | 9.4 | 5.6 | pit 9.0 / 3.7 |
| cylinder | + rulers | 1 | 1.28 | 2.9 | 0.9 | rulers 9.0, 9.6 |
| cylinder | + rulers | 19 | 1.63 | 10.2 | 5.9 | footprint ruler 13.9 |

Reading it:

- **The matched plant is inside the band at every hour, every set and
  every channel** (no z beyond 1.8 across the 57 refits). The z-score
  is the statistic, not the raw power: at 1 h, 14 rows against 17
  latents leave a residual power of 0.30 per row that the degrees of
  freedom account for (z −1.3).
- **The cylinder is flagged from hour 6 with temperature probes alone**,
  by the meat probe (z χ² 3.7 at 6 h and 12 h): the posterior looked
  fine there (every latent within 2.5 sd) and the residual said
  otherwise. With humidity and the scale the total climbs to 9.4 by
  19 h with autocorrelated residuals on the pit, the humidity sensor and
  the scale (z ρ₁ 3.3 to 3.7): the confident-and-wrong posterior of the
  entry below carries a warning from hour 6. With the rulers it carries
  one from hour 1: the rulers' readings sit 9 sd from the thickness and
  footprint the fit needs, because a slab's product constraint from the
  scale cannot agree with a cylinder's measured dimensions.
- The entry below quoted the objective per reading as the tell; that
  number carried the prior term (23 nats on this deliberately off-prior
  truth). The residual power per row is the clean version: 1.04 to 1.06
  on the matched plant, 1.22 to 1.63 on the cylinder, and its table is
  corrected accordingly.

## Streaming estimator on mock plants: watch it work, watch it fail (2026-09-08)

`estimator/stream.jl` is the streaming form of the gate's graph: at every
hour of the cook a FULL-HISTORY refit of the 17 latents (15 physics, the
exposed area, the core probe's depth) on every reading so far, with the
original prior, warm-started at the previous hour's posterior mean. Not a
filter: there is no process noise to invent, and the posterior at hour k
is exactly the batch posterior on the readings to hour k. The outer loop
is the gate's (one solve and one forward-mode Jacobian of the stacked
output per accepted step, the graph's mean as the step, halved on the
objective evaluated with solves alone, a step-or-objective-floor stop).
Three to fourteen outer steps and 20 to 110 s per hourly batch on a
10-minute reading grid; a 19-batch cook with three channel sets takes
about 35 minutes.

`estimator/mock_plant.jl` builds the plants and `estimator/mock_stream.jl`
runs the sets. The truth is the calibrated MAP with a deliberate cut
into the water-budget map's retention corner (h_c times 0.6, L times 2),
an exposed area of 0.18 m² against a prior median of 0.12, the core probe
at 0.8 L. Two plants: the monolith itself (same solver, same tolerances:
matched), and the 2-D axisymmetric cylinder (R 0.135 m, height 0.078 m,
16 by 14 cells) whose volume-to-area ratio the slab cannot represent
(structural mismatch). Readings carry AR(1) noise at the calibrated pit
and core σ, ρ; the humidity sensor 100 Pa (ρ 0.8), the scale 20 g
(ρ 0.5), the rulers 3 mm on L and 0.01 m² on the footprint. Three
channel sets, cumulative: pit + core; + humidity + scale; + the two
rulers. The figures (`figures/estimator_stream_mock_1d.png`,
`figures/estimator_stream_mock_2d.png`) show, per latent, the posterior
width against the cook hour and the standardized error
(estimate − truth) / sd with its ±2 band; the records are
`data/mock/mock_1d_stream.json` and `mock_2d_stream.json`.

At the end of the cook (19 h), the largest and the median standardized
error over the 17 latents, the objective, and the whitened residual
power per row from the misfit check (the entry above; 1.0 is a right
model with right noise):

| plant | channel set | max err/sd (latent) | median | F | residual power per row |
|---|---|---|---|---|---|
| monolith (matched) | pit + core | 3.8 (a0) | 0.55 | 144.6 | 1.06 |
| | + humidity + scale | 1.4 (k_m, m_conv) | 1.05 | 267.1 | 1.04 |
| | + rulers | 1.7 (k_m) | 1.25 | 267.2 | 1.04 |
| 2-D cylinder (mismatched) | pit + core | 2.5 (u_leak) | 0.52 | 164.4 | 1.22 |
| | + humidity + scale | 17.2 (a0) | 3.2 | 416.5 | 1.58 |
| | + rulers | 17.1 (a0) | 4.1 | 430.0 | 1.63 |

Reading it:

- **Watch it work.** On the matched plant with humidity and a scale,
  every one of the 17 latents is within 1.4 sd of its truth at 19 h
  (median 1.05): the widths are honest. The widths shrink where the
  sensor table said they would, and when the physics lets them: the
  water-holding latent a_myosin from 0.40 to 0.11 only once the drip
  starts (0.40 at 1 h, 0.18 at 6 h), k_e from 0.50 to 0.35, h_c from
  0.38 to 0.20 (0.12 with the rulers). The exposed area stays at its
  prior width (0.30) under temperature probes, goes to 0.09 with
  humidity and the scale, 0.04 with the footprint ruler. With a scale
  the errors on L and A_meat are mirror images (+0.7 and −0.7 sd; +1.4
  and −1.4 with the rulers, the rulers' own noise draw): the scale pins
  their product, the footprint ruler splits it.
- **The pit ridge, in a fit.** With temperature probes alone the fire's
  base rate a0 sits 4 to 5 sd from its truth for the whole cook, with
  u_leak, r_max and dH_eff co-moving along the ridge, while the reported
  width halves (0.62 to 0.32). The Fisher width is the local curvature
  at a point a closed-loop pit log cannot pin; along the ridge it is not
  the posterior's width (the NUTS chain of the calibration is the
  reference there). The humidity sensor moves the point off the ridge:
  a0 lands at −1.3 sd, wider (0.42) and right. That is the sensor
  table's first row confirmed by a fit rather than a Fisher matrix.
- **Watch it fail.** On the cylinder with temperature probes alone
  nothing looks wrong: all latents within 2.5 sd, median 0.5. The slab
  absorbs the cylinder's temperature history into small moves of h_c,
  L and k_m, and the widths stay prior-wide wherever the probes do not
  reach. Add humidity and the scale and the estimator becomes confident
  and wrong: a0 at −17 sd, k_spread −11, r_max −8, u_leak −5, UA −4.5,
  h_c +5, A_meat −5. The scale's operator is the slab's mass
  bookkeeping and the humidity sensor sees the slab's evaporation; the
  cylinder's mass and vapor history is not a slab's at any area and
  thickness, so the fire model is bent to reproduce it, and the widths
  shrink as if the model were right. The errors grow through the cook
  (a0 −12 sd at 1 h, −17 at 19 h) instead of averaging out: bias, not
  noise. What the scale measures directly stays right (a_myosin −0.02
  sd, the probe depth −0.05); what has to pass through the geometry goes
  wrong.
- **The tell.** The whitened residual is the diagnostic nobody has to
  invent. For the same readings the cylinder's residual power per row
  is 1.58 against the matched plant's 1.04 (the misfit check's total χ²
  z-score 9.4 against 1.1), and the objectives separate from the second
  hour:

  | hour | F, matched | F, cylinder |
  |---|---|---|
  | 1 | 34.1 | 33.6 |
  | 6 | 97.7 | 132.9 |
  | 12 | 175.2 | 218.8 |
  | 19 | 267.1 | 416.5 |

  A posterior this confident has to come with that number next to it.

Decisions: full-history refits rather than `autoupdates` filtering (an
hourly exact refit is cheap against a 10-minute reading interval and has
no process noise to make up); the noise held at its calibrated values
(the AR node is a separate step); the channel sets are the C2 order.
What this buys the pilot: the humidity-plus-scale set is the one to
fly, every refit must report its misfit against expectation, and the
cylinder plant makes the case for a geometry latent before a bigger
sensor suite.

## Sensor value on the calibrated cook: the channel registry (2026-09-08)

`estimator/channels.jl` is the sensor registry the estimator design
asked for: every channel is a measurement operator on ONE stacked
monolith output (pit and surface temperature, the temperature, water and
fat fields over time, the pit vapor pressure) with its own AR(1) or iid
whitening, plus static readings on latents; one solve and one
forward-mode Jacobian of that stack serve any sensor set. Channels so
far: pit probe; meat probe at a fixed or latent depth (linear
interpolation in the material coordinate, the chain rule to the depth
and to the thickness analytic); surface IR; humidity (pit vapor
pressure); a scale reading \(\rho_d A_{\mathrm{meat}} L\,(1 + \bar W + \bar F)\),
the model's own bookkeeping; a static reading of any latent (a ruler on
the thickness or the footprint, a measured insertion depth); and, added
later the same day, a fuel scale under the firebox (the pit's fuel
state), a bark colorimeter on L* (the model's lightness map of its bark
states) and a wet-bulb probe (the psychrometer equation solved on the
pit temperature and vapor pressure, its Jacobian by implicit
differentiation; `estimator/test_channels.jl` checks the map against
ForwardDiff). Ambient and rest-environment readings are inputs of the
physics, not measurement operators, and enter through the configs. The exposed
area became a runtime parameter of the monolith for this (an optional
last entry of its parameter vector, `A_meat=` on `monolith_solve`,
dual-capable; the config's constant when absent, bitwise; FD-checked),
so the area is a latent whose column carries the pit coupling and the
vapor balance as well as the scale's explicit factor. Conditioned on the
area and the probe depth, registry-built rows reproduce the gate's
Fisher widths to 4.5e-5 (the whitened Jacobian rows to 1.3e-4, the
documented last-bit jitter of the adaptive solve under a different
chunk).

`estimator/sensor_value.jl` then answers "what does a sensor buy" from
the Fisher information at the MAP, before the instrument exists: the
exposed area (prior 0.12 m², 30 %) and the core probe's depth (median
0.8 L, 30 %, since nobody hits the exact center) are the 16th and 17th
latents, and each candidate channel is added to the pit + core baseline.
Posterior width ratios against that baseline, and the information gained
(nats, half the log-determinant ratio):

| channel added to pit + core (depth unknown) | nats | D_eff | k_m | h_c | L | A_meat | k_e | a_myosin | m_conv | r_max | u_leak | a0 | core_depth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ruler (3 mm on L) | 0.2 | 0.94 | 0.89 | 0.84 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 |
| insertion depth measured | 0.4 | 1.00 | 0.99 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 |
| second probe, 1 cm, known depth | 2.1 | 0.71 | 0.68 | 0.64 | 0.58 | 0.99 | 1.00 | 0.99 | 0.98 | 0.95 | 0.95 | 0.96 | 0.95 |
| humidity sensor | 6.2 | 0.84 | 0.75 | 0.80 | 0.86 | 0.39 | 0.99 | 0.95 | 0.64 | 0.62 | 0.63 | 0.65 | 0.99 |
| surface IR | 2.0 | 0.81 | 0.84 | 0.81 | 0.90 | 0.99 | 1.00 | 0.94 | 0.96 | 0.97 | 0.97 | 0.97 | 0.98 |
| fuel scale (load cell under the firebox) | 2.0 | 0.99 | 0.93 | 0.97 | 0.99 | 0.95 | 1.00 | 1.00 | 1.00 | 0.66 | 0.68 | 0.85 | 1.00 |
| bark colorimeter, hourly | 0.3 | 0.91 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| wet-bulb probe | 1.8 | 0.96 | 0.90 | 0.93 | 0.93 | 0.50 | 1.00 | 0.99 | 0.94 | 0.92 | 0.93 | 0.87 | 1.00 |
| scale, before/after | 4.0 | 0.85 | 0.93 | 0.96 | 0.93 | 0.39 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 0.91 | 1.00 |
| scale, continuous | 8.9 | 0.72 | 0.85 | 0.91 | 0.92 | 0.38 | 0.73 | 0.40 | 1.00 | 0.97 | 0.97 | 0.91 | 0.99 |
| scale, continuous + footprint ruler | 9.3 | 0.53 | 0.68 | 0.63 | 0.57 | 0.24 | 0.72 | 0.40 | 1.00 | 0.97 | 0.96 | 0.91 | 0.98 |
| rulers + scale + second probe | 10.9 | 0.43 | 0.54 | 0.49 | 0.42 | 0.17 | 0.71 | 0.35 | 0.97 | 0.91 | 0.91 | 0.87 | 0.95 |
| everything | 16.3 | 0.37 | 0.37 | 0.36 | 0.39 | 0.16 | 0.68 | 0.28 | 0.52 | 0.42 | 0.43 | 0.56 | 0.65 |

Baseline widths (sd of the unconstrained latents): D_eff 0.346, k_m 0.156, h_c 0.231, L 0.117, A_meat 0.279, k_e 0.499, a_myosin 0.386, m_conv 0.628, r_max 0.136, u_leak 0.273, a0 0.834, core_depth 0.213. Assumed
noise: humidity 100 Pa, IR 2 K, scale 20 g, rulers 3 mm and 0.01 m²,
second probe as the core probe, fuel load cell 50 g, bark colorimeter
3 L* once an hour, wet-bulb probe 1 K, AR(1) correlation 0.8 on the
hypothetical temperature, humidity and color channels, 0.5 on the
scales.

Reading it:

- **A humidity sensor is the single most informative addition** (6.2
  nats). The pit vapor pressure carries the combustion water and the
  meat's evaporation, so it measures the burn rate — breaking the pit
  ridge a closed-loop temperature log cannot (r_max 0.62, u_leak 0.64,
  a0 0.65, m_conv 0.64) — and the exposed area (0.39), which sets the
  meat's vapor. That is the characterization burn's information,
  obtainable during a cook. It rests on the vapor balance's literature
  constants (AFR, the water yield), which is exactly what the wet-bulb
  probe of C2 validates.
- **A scale is the only channel that moves the water-holding latents**
  (k_e 0.73, a_myosin 0.40 with a continuous scale; nothing else touches
  them). Two weighings pin the product of area and thickness, which
  lands on the area (0.39) with thickness barely moved (0.93); a ruler
  on the footprint then lets the scale reach the thickness (L 0.57,
  A 0.24). The scale cook needs both rulers.
- **A second probe at a known 1 cm depth** buys 2.1 nats, on thickness
  and transport (L 0.58, h_c 0.64, k_m 0.68).
- **A fuel scale reads the burn rate directly** (2.0 nats: r_max 0.66,
  u_leak 0.68, a0 0.85) and nothing on the meat: a third of the
  humidity sensor's effect on the pit ridge for a load cell under the
  firebox. **The wet-bulb probe is a humidity sensor at 600 Pa instead
  of 100** (1 K at a 50 °C wet bulb) and still halves the area's width
  (0.50, 1.8 nats); its worth is what it validates, the vapor balance's
  constants, more than its nats. **A bark colorimeter** at 3 L* once an
  hour buys 0.3 nats: the bark's color rides on the surface temperature
  and water the probes already fix.
- A ruler alone buys 0.2 nats; measuring the insertion depth constrains
  only itself, because a probe near 0.8 L is weakly coupled to the
  physics. Assuming the probe sits exactly at the center, as the
  calibration does, overstates the information by 0.3 nats.
- The value ordering for C2 is therefore humidity, scale, second probe,
  then the fuel scale, IR and the wet-bulb probe at about 2 nats each,
  ruler, colorimeter; rulers plus scale plus the second probe give 11
  nats, everything 16.

Linearized information at the MAP with assumed noise, not a posterior;
the registry's fitting path (the same affine nodes as the gate) is what
turns readings into one; the streaming runs on mock plants (the entry
above) exercise it.

## RxInfer estimator: the calibration as a factor graph, gate passed (2026-09-07)

Review item 8 (the output-feedback estimator), redesigned in discussion:
the sensor suite must be composable (different users, different
instruments), every derivative comes from the AD already in the library,
scale and ruler readings are measurements like any probe, a missing
reading is an absent node rather than a fallback branch, and the
estimator lives in Julia next to the physics. RxInfer (reactive message
passing on a factor graph) was chosen as the host and put through two
steps the same night.

**Step 1, compatibility and the three findings.** RxInfer v5.5.2
resolves next to the package on Julia 1.11.9 with no conflicts;
deterministic nodes take the solves under ForwardDiff, constants and
data pass into them, the AR node exists. Three behaviours of its
deterministic (delta) node under `Linearization`, each measured on
linear cases where the exact posterior is known (`estimator/test_rules.jl`
keeps them measured), fix the graph's structure:

1. A node with more outputs than latent inputs silently returns the
   prior — the joint-marginal rule inverts the forward output covariance,
   singular there. No error, no warning. The graph therefore carries one
   node per whitened residual, scalar output.
2. The rule linearizes at the incoming message mean and `iterations`
   never moves that point (a constant free energy over ten iterations);
   from the calibration-width priors one step overshoots (one batch sent
   log k_m from −6 to +24) or is dropped. The physics therefore enters
   each node affine around a linearization point passed as data,
   f(c) + J(c)(z − c) with J by ForwardDiff through the FBDF solve, and an
   outer loop moves c with a descent safeguard: Gauss–Newton with the
   prior intact, as message passing.
3. Separate latent blocks feeding one node come back with the right
   means and variances narrower than the marginals (a block
   factorization, not the joint; 2–5× too narrow on the real problem).
   The latents are therefore one multivariate block; sensor nuisances
   will live inside it.

With those, on a 3-latent, 37-reading meat solve, the graph reproduced
the direct Gauss–Newton MAP and Laplace widths to four decimals, and one
pass equalled one Gauss–Newton step exactly; twelve graph solves took
1.3 s after compile. Two items to file upstream: the silent prior on a
singular forward output covariance, and iterations not moving the
linearization point.

**Step 2, the gate on the real cook.** `estimator/export_gate_data.py`
writes everything `run_gn_map.py` sees — the resampled log, the
whitening operator of the masked AR(1) likelihood at the MAP's noise
parameters, the priors, the MAP, the Gauss–Newton Hessian and its
inverse — and `estimator/gate_real_cook.jl` rebuilds the objective as a
graph: one 15-dimensional latent block in the unconstrained space, a
prior node that is the local quadratic of −log p(z) at the
linearization point (exact for the LogNormal sites, the logit-Beta
curvature for the Beta sites, so the fixed point and the covariance are
the Python ones), 222 residual nodes, twenty message-passing iterations
per outer step. The objective evaluated in Julia at the Python MAP
agrees to 1.4e-13. From the cold start (F = 5317) the outer loop reached
F = 133.1206 in 17 iterations and 155 s, against the stored
MAP's 133.2059 — a lower value of the same objective; the stored MAP
sits at a residual gradient of 4.8 in whitened units, the known last-bit
Jacobian jitter near the optimum, which is exactly the slack the
pit-ridge sites show below.

  | site | z MAP (GN) | z graph | \|Δ\| / sd | sd GN | sd graph | ratio |
  |---|---|---|---|---|---|---|
  | D_eff | -18.7271 | -18.7099 | 0.051 | 0.3357 | 0.3346 | 0.997 |
  | k_m | -6.2096 | -6.2221 | 0.079 | 0.1586 | 0.1582 | 0.998 |
  | h_c | 2.4856 | 2.4754 | 0.045 | 0.2248 | 0.2234 | 0.994 |
  | L | -3.9379 | -3.9326 | 0.046 | 0.1159 | 0.1159 | 1.000 |
  | k_mult | -0.0882 | -0.0783 | 0.043 | 0.2277 | 0.2278 | 1.000 |
  | k_e | -7.5898 | -7.5920 | 0.004 | 0.4987 | 0.4987 | 1.000 |
  | a_myosin | -1.9717 | -1.9773 | 0.015 | 0.3849 | 0.3851 | 1.001 |
  | m_conv | 0.1119 | 0.1364 | 0.039 | 0.6275 | 0.6291 | 1.002 |
  | C_th | 10.6247 | 10.5773 | 0.162 | 0.2931 | 0.2934 | 1.001 |
  | UA | 2.8175 | 2.8406 | 0.101 | 0.2294 | 0.2289 | 0.998 |
  | r_max | -8.5084 | -8.5517 | 0.318 | 0.1358 | 0.1320 | 0.972 |
  | dH_eff | 16.5022 | 16.5264 | 0.109 | 0.2229 | 0.2225 | 0.999 |
  | u_leak | -0.2932 | -0.2043 | 0.325 | 0.2731 | 0.2738 | 1.003 |
  | k_spread | -8.2234 | -8.1980 | 0.041 | 0.6144 | 0.6175 | 1.005 |
  | a0 | 3.0077 | 3.0354 | 0.036 | 0.7729 | 0.7902 | 1.022 |

Max \|Δ\| / sd 0.325 (r_max, u_leak: the pit ridge), covariance entries
within 4.5 % of the largest, correlations within 0.032. The
per-outer-step cost is the monolith solve plus its 15-column Jacobian;
the 222-node graph solve is a fraction of a second. The estimator
subproject (`estimator/`, own `Project.toml`, RxInfer + ForwardDiff +
JSON over the package) is where the channel registry, the AR node, the
sensor-value table, streaming and the mock-pit runs go next
(`docs/roadmap.md`, online estimator).

## Review fixes: pan closure, 2-D shadow to serve, FSIS come-up time, two-route collagen, water-budget map (2026-09-07)

The independent review (`docs/review.md`) was verified claim by claim
against the tree and its laptop-only items 1–7 acted on the same day
(item 8, the output-feedback estimator, is open).

- **CI.** The app job ran a test file that did not exist
  (`tests/test_local_apply.py`; only a stale `.pyc` remained), so the
  composed-gradient gate, MPC, planner and 2-D tests never ran on CI. It
  now runs the whole app suite (76 tests, 4.5 min locally including the
  Julia init).
- **Water pan: closed humidity balance.** `_pan_algebra` now solves
  \(\dot m_{\mathrm{pan}} = g\tau\,(p_{\mathrm{sat}} - p_{v,0})/(1 + g\tau K)\),
  \(K = p_{\mathrm{atm}}/(0.622\,\dot m_{\mathrm{air}})\) — the same linear
  self-consistent closure the meat's `evap_flux_fb` uses — so the pan sees
  its own vapor and cannot drive the pit past its own saturation. Tests:
  the closed form against the open-loop flux, \(p_{v,\mathrm{base}} <
  p_{\mathrm{sat}}(T_{\mathrm{pan}})\) while evaporating, pan-off inert.
  The 2-D coupled loop shares the function.
- **2-D shadow pulls and rests to serve.** The 1-D pull blend is on every
  face of the 2-D engine (rest convection and radiation, the blocked rest
  evaporation with \(k_m h_r/h_c\), the pit's share of heat and vapor
  gated by \(1 - p\)); `t_pull` is the fifth differentiable input of the
  `slab2d` Tesseract (column FD-checked; never-pulled is bitwise the
  pit-only solve); the plan's rest constants pass through; the radius
  defaults to the 1-D exposed-area equivalent
  \(R = \sqrt{A_{\mathrm{meat}}/2\pi} = 13.8\) cm (it was 10 cm by hand)
  and the report records which was used; `refine` re-runs on a doubled
  grid. On the committed robust plan (11 h; wrap 6.72 h; pull 9.66 h):

  | | before: cooked to the horizon, R 10 cm | after: rested to serve, R 13.8 cm |
  |---|---|---|
  | cold-spot lethality F [60 °C-eq s] | 3.8e11 | 2.5e9 (720 required) |
  | cold-spot come-up 10–54 °C | 1.40 h (as "dwell 4.4–60 °C") | 1.05 h |
  | C_min at serve | 0.99998 | 0.979 |
  | water retained | 0.189 | 0.277 |
  | cold spot at serve | 112 °C (= pit; uniformly done) | 68 °C (peak 94 °C; the rim cools in the cambro) |
  | core at serve | 112 °C | 82 °C |
  | shrunk radius / thickness | 7.7 / 3.1 cm | 11.0 / 3.2 cm |
  | 16×14 → 32×28 grid | — | ΔT_cold(serve) −0.08 K, Δpeak +0.004 K, ΔC_min +0.0005, Δw +0.0003 |

  The old numbers were the surrogate left on the fire 1.3 h past the pull
  (the robust plan pulls at 9.66 h — not the nominal plan's 8.4 h the
  review assumed) on a thinner puck. The shadow now scores in 2-D what the
  plan scored in 1-D (C_min 0.979 against the robust plan's 1-D doneness
  0.975), and the grid check says the 16×14 default is not hiding
  anything. `figures/slab2d/` regenerated (pull marker; "resting" in the
  animation); the math diagram re-rendered.
- **Food safety: the FSIS come-up time replaces the 4 h danger-zone cap.**
  The cap (4.4–60 °C, 4 h, weight 500, on surface and core) was the
  consumer holding rule for cooked food left out, not a roasting limit.
  Replaced by the Appendix A come-up time (10–54.4 °C, 6 h), integrated
  and reported on surface and core; for brisket it is not binding (FSIS
  Scientific Gap 6, `cut_critical=False` default) and `cut_critical=True`
  makes it a penalty and a fullplan rejection for products where Appendix
  A's CUT applies. The report states that the Appendix A humidity
  condition is not modelled; `brisket plan` / `brisket pilot` print the
  disclaimer. On the calibrated slab the old cap never bound (core dwell
  1.40 h), so the published plans are unchanged in substance; the banking
  keys are renamed (`cut_h`, `cut_h_core`). Tests: the Gap 6 semantics
  (reported, not rejected; rejected when critical) and the FSIS window.
- **Two-route collagen.** \(\dot C = [k_C(T) + k_{\mathrm{enz}}(T)](1 - C)\)
  with an enzymatic (collagenase) window 45–60 °C at
  \(2.7\times10^{-5}\) s⁻¹ peak (90 % in 24 h; Baldwin, Tornberg 2005),
  the thermal Arrhenius unchanged, `k_enz0 = 0` bitwise the old rate (the
  salmon preset). Effect on the anchors: the 19 h calibrated cook's core
  \(C\) at the end 0.863 → 0.875; in a bare replay of the robust
  schedule the \(C = 0.90\) crossing moves 7.39 → 7.37 h (2 min earlier)
  and \(C\) at serve and retention do not move; sous vide at 55 °C:
  core \(C\) 0.031 → 0.848 at 24 h and 0.063 → 0.978 at 48 h — the regime
  the single rate could not represent. \(E_a\), \(k_{\mathrm{ref}}\) not
  retuned. The plan chain (fullplan → MPC → robust) has not been re-run
  under this change; the shift it would see is minutes.
- **Water-budget map.** `scripts/run_water_budget.py` re-runs the
  calibrated 19 h cook over \(h_c \times 0.25\)–1, \(D_{\mathrm{eff}}
  \times 0.1\)–3 and \(L \times 1\)–2.5 (196 solves, 57 s) and maps the
  retained water against the stall (hours in the 65–78 °C band; the MAP
  stalls 13.5 h and retains 3.9 %). At the calibrated half-thickness
  (1.95 cm) **no** combination retains more than 33 %. Retaining 50–60 %
  with a ≥ 4 h stall appears only at \(L \times 2\)–2.5 (3.9–4.9 cm
  half-thickness — a packer point, not a flat) with \(h_c \times
  0.4\)–0.63 (4.8–7.6 W/m²K against the MAP 12): 13 such cells inside
  the 2σ prior. Reading: the temperature log pulled \(L\) thin
  (\(z = -1.7\) on its prior) and the water budget wants it thick — the
  thickness / probe-depth ridge, not \(k_m\). The prediction for the
  scale experiment (C2): the ~40 % loss of a real flat is consistent with
  this model only if the cut is about twice the calibrated thickness and
  the meat-side heat input about half the MAP, so the scale cook must
  come with a ruler; if a thick cut still calibrates to a thin \(L\), the
  Fickian slab (review T1) is the wrong model. Figure
  `figures/water_budget_scan.png`, data
  `data/validation/water_budget_scan.json`.
- **Docs.** README, writeup, `model.md`, `monolith.md` and a note on the
  submission carry the \(5n_x + 4 + 6 = 210\) layout; `model.md` gained
  the salt field, the wrap latch, the six-state pit (pan, firebox), the
  closed pan humidity, the two-route rate, the come-up safety term and a
  2-D shadow section; the `slab2d.jl` header and the `timeplan.py`
  docstring match the code; the review's seven inaccuracies are recorded
  at the top of `docs/review.md`.

## One monolith Tesseract; MTK not pursued (2026-09-03)

The monolith landed in shape (a): `src/monolith.jl`, one Tesseract around
the whole coupled physics. The MTK/acausal question was then tested
empirically by Phase A of the generalization: cuts and cookers turned out
to be *configurations of shared physics* (`src/meats/`, `src/cookers/` —
four cookers cost two config fields and two config recipes), so the
composition layer already exists as thin presets over `common.jl` cores,
and the system-level composition boundary is the Tesseract itself. MTK's
strengths map badly here: nothing is a DAE (the humidity loop is
deliberately quasi-steady closed-form), its first-class events are
discrete — the smooth blends are the differentiability story and would
have to be re-derived inside components — and large MOL stencils are its
weak spot. Revisit ONLY on one of these triggers, and then for NEW
components with the monolith as oracle, never as a rewrite: (a)
heterogeneous multi-meat wiring becomes painful (N different cuts sharing
one pit — the hand-rolled form is a vector of meat state-blocks
generalizing `n_meat`); (b) outside contributors need to extend the model
without touching the core; (c) an actual DAE enters (flow networks).
Small fixed vessel graphs (water pan = +2 pit states; two-zone offset =
+2 states and a duct coupling) are component enrichment, hand-written per
archetype. The multi-dimensional slab activates the adjoint/sparsity
item, not a framework change.

## Which posterior per log (scored 2026-09-02)

Against the 22 h NUTS reference (per-site 1-Wasserstein in z;
`data/calibration/approx_summary.md`): Gauss-Newton Laplace is 2.5 s a
log and within 10-20 % of the NUTS widths on 13 of 15 sites; it misses
only k_spread and a0 (ignition — cook-specific, not pooled). The TMB /
INLA nested-Laplace marginal on those two coordinates
(`scripts/run_tmb_ignition.py`: 11x11 grid, converged inner Newton,
half log det correction, ~51 min) closes most of that: W1-to-NUTS 0.095
vs the Laplace's 0.148, tied with an oracle fit to the NUTS draws. Guard
the family with `scripts/run_psis_laplace.py --proposal tmb
--moment-match`: Zhang-Stephens Pareto-khat against the true ODE (0.76
raw; importance-weighted moment matching repairs it to 0.47), with the
sample-size-dependent threshold. Measured and retired (2026-09-02 attic
tarball): profile nested grids, structured t/mixture ignition families,
2-D/5-D surrogate-transition samplers (4+ h, no W1 gain over the grid),
and ADVI/IWAE (5-10x too narrow or divergent). The residual honest gap —
every method still ~0.86-0.89x NUTS width on ignition and weakest on
u_leak — is the 19-D sloppy-ribbon coupling a 2-D-marginal family cannot
carry; more data, not more machinery.

**Not RxInfer / CBFE for batch calibration.** Constrained Bethe Free
Energy is the online-during-cook estimator (a reactive factor graph over
pit state and brisket parameters once the model is pure Julia). Batch
calibration is one stiff ODE likelihood; message passing on that graph is
just Laplace/VI on a custom node. Do not rebuild batch calibration as a
factor graph to match NUTS.

## First real two-cook pool: cross-smoker (2026-09-03)

`scripts/run_pool_real.py`: mache_2013 + phillip_2018 (a 15 lb packer at
~1 °F, found, digitized and provenance'd by `tools/loghunt/`; partial
log, different smoker → meat partial, pit per-cook). Result: D_eff agrees
within 7 % across the two cooks, k_e within 7 %, a_myosin/L/m_conv close
— the "briskets I buy" block behaves — while the spread sits on the known
confounds (h_c/k_m on the draft ridge between different rigs; k_mult
absorbing the uniform-interior slice start). Fits: rms pit 1.33/1.87 K,
core 3.02/1.43 K (`figures/pool_real.png`,
`data/calibration/pool_real.json`). Synthetic two-cook mechanics check:
`scripts/run_hier_synth.py`.

## Same-smoker pool: one keg, three cooks (2026-09-03)

`run_pool_real.py --set keg`: Dale Ward's Big Steel Keg — overnight
225 °F brisket '13 (11.7 h), hot-and-fast 325 °F pork butt '14 (4.8 h),
overnight 225 °F butt '15 (11.7 h, RotoDamper era) — found, digitized
(`hm_dark` palette; lid events read from the HM UI's own lid-detect bands
where present) and provenance'd by `tools/loghunt/`. Roles flipped vs the
cross-smoker set: vessel identity (C_th, UA, dH_eff) fully shared,
actuator map (r_max, u_leak) per-cook (the airflow hardware changed
across the years), meat + ignition per-cook. Fits: rms pit 2.6/4.2/2.6 K,
core 1.1/1.4/1.0 K. Headlines: the two overnight-225 cooks TWO YEARS
apart, on different actuator hardware, agree on the vessel (UA within
11 %, dH_eff within 5 %, C_th within 27 %) — isolating the '14 outlier as
operating-point/short-horizon, not vessel drift; k_e holds within 3.5 %
and a_myosin within 8 % across all three cooks (brisket AND butts — meat
chemistry transfers); u_leak spans 0.29-0.88 per-cook (hardware + vent
settings, incl. a documented mid-cook vent change in '15). Caveat: the
pooled shared block is ridge-coupled (C_th/UA trade off; the joint
precision-weighted mean with K−1 prior-dedup can land outside the
per-coordinate MAP hull) — the cross-cook agreement above is the
meaningful validation, and it was checked against a falsy-default pooling
bug fixed en route (explicit-empty role lists no longer fall back to
defaults; regression-tested). (`figures/pool_keg.png`,
`data/calibration/pool_keg.json`.)

## Same-model pool: two WSM 18.5 units (2026-09-04)

`run_pool_real.py --set wsm`: mache_2013 x wbegg_2014 (a stock WSM 18.5
8 lb pork butt from a 32 F Australian winter start, fully documented
timeline, found and digitized by `tools/loghunt/` — the 225 F setpoint
line sits ON the 225 gridline, a 0.5 px anchor cross-check). A NEW
pooling tier: two different physical units of one smoker model, vessel
identity PARTIALLY pooled (tau2), everything else per-cook. Fits: mache
reproduces its historical MAP exactly (F=133.2); wbegg rms pit 5.25 K
(the pit sits ~5 C low through the cold morning — the scalar 278 K
day-mean ambient and the terra-cotta-dish-for-water-pan swap are the
stated confounds), core 1.42 K over 20.8 h. Results: **dH_eff agrees
within 8 % across units** (fuel energy density is a design+charcoal
property — it pools hard; hypermean 1.53e7); UA spreads 52 %
(16.7 vs 11.0 — unit + configuration + ambient-estimate, which is what
partial pooling is for; shrunk 18.0/10.2, hypermean 13.5); C_th stays on
its known sloppy ridge (14x apart — closed-loop logs cannot pin it; the
characterization burn can). Meat side: **k_e now holds within +/-8 %
across FIVE cooks** (mache, three kegs, wbegg: 0.51-0.57e-3) spanning
brisket and pork butt, two smoker models, and a 60 F ambient range.
(`figures/pool_wsm.png`, `data/calibration/pool_wsm.json`.)

## Wrap constants from a documented butcher-paper wrap (2026-09-04)

`scripts/run_wrap_fit.py` on `erickson_2013`, found by `tools/loghunt/`
(12 lb brisket, Char-Griller Kamado, ~15 h; BUTCHER PAPER wrap documented
in prose AND in the chart's own off/on lid-band pair at 8.75/9.15 h; new
watermark-tolerant `hm_flot_jpeg` palette). The observed core DROPS
~7 °C across the wrap — the brisket spent ~24 min off the pit and the
probe was re-seated — so the forward model is three chained segments (pit
→ room interval → pit-wrapped; full state restarts through T/W/C/F fields
+ bark/smoke/ring) with the probe's material depth as a parameter on each
side of the wrap. Measured pit trace as the boundary forcing; 8-parameter
FD-LM with z-space priors. Primary fit (L = 0.045, point-end probe; every
nuisance lands physical — h_c 11.8, k_m/k_e at priors): **η_evap = 0.17
[0.13, 0.23]** — butcher paper passes ~83 % of the evaporation (paper
breathes; the mechanism behind "paper preserves bark") — and **ε_wrap =
0.31 [0.26, 0.37]** (prior-bound); rms 3.6 K; counterfactuals at the fit:
foil would have finished hottest, unwrapped middle, paper slowest under
this ramped pit. Identifiability caveat (kept in the JSON pair): an
L = 0.033 variant fits identically (rms 3.6) but trades the
radiative/convective split (h_c → 1.3, η_evap 0.80 / ε_wrap 0.16) — ONE
log pins the total post-wrap input reduction, not the full η/ε split; a
FOIL-wrapped log (contrast) or a known-thickness cook breaks the tie.
(`data/calibration/wrap_fit_erickson{,_L33}.json`,
`figures/wrap_fit_erickson{,_L33}.png`.)

## Salt physics: the structure (2026-09-04)

The salting item's model half is in (`src/meats/common.jl`): a fifth PDE
field s(x,t) — dry-basis salt — with the dry brine as the ANALYTIC
semi-infinite profile `s0 = S_surf·erfc(depth/(2√(D_salt·t_salt)))` (no
pre-cook solve; a smooth dependency-free erfc), zero-flux surface during
the cook (salt does not evaporate), and two couplings: water-holding
capacity (`W_eq → W_eq·(1 + β_whc·s)` in the purge sink — salted muscle
purges less) and Raoult-scale water-activity depression of surface
evaporation (`a_w` factor `1/(1 + k_aw·s)`). `t_salt = 0` is EXACTLY
unsalted (both couplings are the identity — tested against absurd
coupling constants), salt is conserved to solver tolerance, a 96 h brine
raises retained water and lowers evaporation directionally, and the
state-restart contract covers the salt field (`Ssalt0f`). D_salt is a
ham-curing literature constant by design; S_surf/β_whc/k_aw are
placeholder scales awaiting the scale + designed experiment. The
monolith carries the field (206 states; FD gradient gates unchanged);
`salt_field` is internal to the Tesseract like geom_m/k_skin until a
consumer needs it, and t_salt lives in MeatConfig until the planner
promotes it to a differentiable decision variable.

## Generalization: geometry, modes, and two external validations

Phase A (2026-09-03): the GEOMETRY axis is in and validated —
`MeatConfig.geom_m` puts the slab/cylinder/sphere curvature into the
material-coordinate operator (exact (1+m) center limit; bit-identical
slab at m = 0), and sous vide needed no new BC code at all (the bath is
the rest environment from t = 0 with η_rest = 1). The doneness-kinetics
axis is configuration, not code: collagen constants (Ea, k_ref, T_ref)
are `MeatConfig` fields, and pasteurization is the existing lethality
integral. The remaining modes followed: skin barrier
(`MeatConfig.k_skin`, a series conductance — heat passes, evaporation
throttled; the poultry mechanism, constants uncalibrated), saturated
steam (pure configuration: `pv_base_knots = p_sat(T_env)` — the
two-sided flux condenses and the stall never forms; the pibil mechanism),
and an electric source (`PitConfig.p_heater`: heat = p_heater·u, nothing
burns — no fuel loss, no combustion water, no smoke, bark goes
Maillard-only). Each has a physics-consistency test, and exposing
geom_m/k_skin/p_heater through the Tesseract schema is one optional field
each (like t_f) when a consumer needs them.

**Sous vide vs Baldwin (zero fitted parameters):**
`scripts/run_sous_vide.py` reproduces the exact conduction series at
Baldwin's α/h to ≤ 8 % on all 21 table cases (his published
cylinder/sphere columns are safety-padded ~+15 % and ~+35 % above that
exact solution; the slab column, us, and theory agree to ~1 %;
`figures/sous_vide_baldwin.png`).

**Oven roast vs Papasidero et al. 2015, CET 43:151 (zero fitted
parameters; the first validation with measured WEIGHT LOSS):**
`scripts/run_oven_roast.py` — a 1.4 kg beef cylinder (18 x 10 cm),
domestic oven 22 → 180 °C cold start, stopped at core 65 °C, with core /
5-mm-depth / intermediate thermocouples and dry-basis water content
2.38 → 1.75 kg/kg (digitized by `scripts/digitize_cet2015.py`,
overlay-verified, into `data/validation/cet2015_roast.json`). With
repo-default θ, the paper's own h = 10 W/m²K and ε = 0.91, and
geom_m = 1.2 as the eigenvalue-matched finite-cylinder equivalent:
time-to-65 °C within x1.04-1.16 of measurement across the
geometry/preheat sensitivity band, core rms 2.6-6.9 °C, and predicted
water loss x1.06-1.27 of measured — where the paper's own fitted COMSOL
model overshot the loss ~2x. The purge + surface-evaporation moisture
physics transfers from smoker calibration to an oven regime it never saw
(`figures/oven_roast_cet2015.png`,
`data/validation/oven_roast_cet2015.json`).

## Smoke quality: the structure (2026-09-04)

The "thin blue smoke" variable is in (`burn_cleanliness`,
`PitConfig.r_smolder/δ_smolder`; monolith outputs `smoke_clean` and
`bitterness`): combustion cleanliness is a logistic in the
draft-to-lit-bed ratio — a choked damper over a large lit bed smolders,
an open draft burns clean — and the DIRTY share of the exact deposition
integrand that drives the bark smoke state integrates to a bitterness
output (a quadrature over saved trajectories; no new ODE states, so the
solve cost is unchanged). Tested: a choked burn is dirtier per unit
deposited than an open one at the same fire, and the electric source is
exactly bitterness-free (nothing burns). Constants are placeholder
scale — bitterness calibration needs a smolder-forced designed cook or
a tasting panel (roadmap item 7 keeps that).

## Composability presets + the quasi-3D fidelity level (2026-09-04)

The notable cases are now one-call compositions: `pork_butt()` (the
near-spherical cut — geom_m = 2, R ≈ 5 cm, fattier; same collagen
kinetics; k_e backed by the five-cook pooling record), `pellet()` (a
controlled-feed source: `PitConfig.m_auger` replaces the Minion bed law
with `m_auger·u·taper` — the damper input becomes the auger duty,
combustion water and smoke stay on, the bed-spread state is inert), and
`earth_pit()` (the pibil/imu as PURE COMPOSITION: the rock reservoir is
the pit's precharged lumped mass — C_th = m_rock·c_rock, Tpit0 = 700 K,
no fuel, r_max = 0 → no smoke — discharging through UA into the meat
under the existing `pibil_pv` saturated-steam environment; "control"
becomes rock mass and seal time; a sealed source exposed a
divide-by-zero in the humidity air-mass flow, now floored). The
fidelity ladder gains its first rung above 1-D: `newman_center` — the
Newman product rule combining per-axis 1-D solves into a quasi-3D
center trajectory (exact for linear conduction under constant ambient;
the documented approximation here; a finite brick costs ~3 solves).
Tested: feed law ignores the bed, the earth pit cooks with zero burn,
Newman bounds and its infinite-axis reduction. Julia 252/252, app
67/67. All constants placeholder scale; the combi-oven case was already
reachable (electric source + humidity forcing).

## Water pan: +2 pit states (2026-09-04)

The vessel component the MTK verdict priced at "+2 pit states" is in:
pan water mass and pan temperature (`PitConfig.m_pan0/T_pan0/UA_pan/
g_pan/C_pan_dry/m_pan_taper`; monolith outputs `m_pan`, `T_pan`,
`pan_evap`). The pan draws pit heat through UA_pan, buffers near boiling
(the latent sink behind a WSM's steadiness — capped below 378 K while
wet in tests), FEEDS THE PIT HUMIDITY with its state-dependent
evaporation (replacing the constant `Humidity.pan_kg_s`; the pan drives
against the pre-pan vapor pressure, self-feedback not closed), and DRIES
OUT mid-cook (tanh taper; a dried pan's metal then legitimately tracks
pit temperature). g_pan defaults to ~0.5 kg/h at boiling (a foiled WSM
pan lasts ~7 h). m_pan0 = 0 is the no-pan default with both couplings
config-gated off. This gives item 3's pan-in-vs-out designed experiment
its model term, and speaks directly to the wbegg UA confound. One
FD-gate threshold moved 5e-3 → 8e-3: the extra states shift adaptive
steps and that check carries solver noise (the documented last-bit
sensitivity). Julia 260/260, app 67/67.

## Two-zone offset + wood splits (2026-09-04)

The last missing source: in two-zone mode (`PitConfig.c_duct > 0`) the
fuel burns in a FIREBOX state (mass C_fb, loss UA_fb) and the cook
chamber receives draft-driven duct advection
`q_duct = c_duct·airflow·(T_fb − T_pit)`; smoke and combustion water
ride the same gas, so the humidity/bark pathways are unchanged.
`offset_smoker()` composes it with HOURLY WOOD SPLITS — a finite
`dt_add` turns the single fuel addition into a pulse train (how a stick
burner is actually fed; the standalone pit inherits it for free since
the machinery is shared). c_duct = 0 is the single-zone bitwise
default; T_firebox rides inert. Tested: firebox runs hotter than the
chamber, the duct lags startup, splits keep fuel in the box, default
inert to 1e-6. Constants placeholder scale — a firebox/chamber probe
pair is the calibrating experiment. Differentiable split timing is a
planner step (dt_add is config).

## Bone region: per-node thermal blend (2026-09-04)

Bone-in cuts get their first fidelity term: a central bone out to
`MeatConfig.bone_frac` of the half-thickness/radius, as a smooth
logistic blend of node conductivity and volumetric heat capacity toward
cortical-bone values (k_bone 0.55 W/mK, ρc_bone 2.2e6 J/m³K —
literature). The weights are static fractions of L, hoisted once in
both RHS builders (no per-call cost, no Dual entanglement); v1 is
thermal-only — moisture/collagen in the bone region still evolve as
meat, a small documented error whose fix is excluding bone from the
water bookkeeping. bone_frac = 0 is the boneless default with weights
exactly zero (tested inert against absurd bone constants). Physics
note the tests encode: cortical bone conducts slightly better than
meat but stores far less heat, so a bone-in center heats FASTER — the
"cold near the bone" folklore is about joints and mass, not material.

## Preset batch: kamado, gas, sausage, salmon (2026-09-04)

Four more one-call compositions. `kamado()` is the first DATA-BACKED
cooker hint — vessel block seeded from the three-cook keg pool,
actuator from keg_2015, with the C_th/UA ridge caveat attached.
`gas_grill()` fills the gap between electric and charcoal: heat like
the element plus the burner's combustion water into the pit humidity
(`PitConfig.y_gas_w`, ~4.5e-8 kg H2O/J for natural gas), smoke exactly
zero — tested: same pit temperature as electric, wetter pit, no smoke.
`sausage()` composes the casing as the existing skin barrier on a small
cylinder (why sausages burst instead of stalling), and
`hot_smoked_salmon()` makes the doneness-kinetics-as-configuration
claim concrete by swapping the Arrhenius triple to a fish-myosin scale
(T_ref 50 °C) on a thin slab. All constants placeholder scale except
where the pooling record backs them.

## Discrete-adjoint legwork: Enzyme through a fixed-step stepper (2026-09-04)

The item-2 question — is a discrete adjoint viable, and where is the
forward/reverse crossover — now has measured answers
(`adjoint/discrete_adjoint.jl`, its own Julia env so Enzyme stays out of
the main deps). The Enzyme-friendly stepper for this physics is IMEX
backward-Euler with a HAND-CODED tridiagonal (Thomas) solve — implicit
1-D diffusion with lagged coefficients, explicit local reactions, pure
Julia loops end to end (no LAPACK, no custom rules) — the same
structure that becomes ADI sweeps on the 2-D axisymmetric slab. On a
structurally faithful reduced heat–moisture system (nonlinear
conductivity, evaporative BC, purge sink; NX = 40, 480 steps, a
per-node diffusivity field + two scalars = 42 parameters):

- Enzyme reverse matches ForwardDiff on the same stepper to **2e-11**
  relative, through all 480 implicit steps — and it works on Windows.
- Reverse costs **3.6× the primal, flat in parameter count**; forward
  mode pays ~0.25 ms per parameter here, putting the crossover at
  **~14 parameters** on this proxy.

Caveats: reduced physics (no shrink metrics / Choi-Okos / bark),
fixed-step vs FBDF's error control, and Enzyme through the full
`meat_rhs_core!` closures untested. The strategic conclusion stands:
when the 2-D slab arrives, the investment is a specialized
IMEX/ADI-tridiagonal stepper differentiated by Enzyme — not Enzyme
through the adaptive FBDF solve.

## Adjoint port to the full physics (2026-09-04, M1 complete)

`adjoint/full_port.jl`: the fixed-step IMEX-BE stepper now runs the FULL
production RHS — the explicit part IS `meat_rhs_core!` (Enzyme reverses
through it unmodified: closures, views, Choi-Okos, Oswin, all of it —
`adjoint/probe_rhs.jl`), and the implicit part subtracts and re-solves
the T/W diffusion operators as tridiagonals assembled from the same
finite-volume metric formulas at the lagged state; reactions, surface
fluxes, salt (4 orders less stiff) and the scalars stay explicit.
Measured, 8 h cook at 211 states:

- **Accuracy vs FBDF: clean first order** — max |ΔTcore| 0.28 K at
  h = 60 s, 0.14 K at 30 s, 0.048 K at 10 s, 0.024 K at 5 s. At 30 s the
  fixed-step trade is an order below observation noise. (Debug note: the
  first comparison showed a 2.8 K h-independent offset — `meat_solve`
  integrates from `tsave[1]`, so a reference grid starting at 300 s
  cooked five minutes less. Match the horizons.)
- **Gradients: 7e-14** Enzyme-reverse vs ForwardDiff on the same
  stepper, through 960 implicit steps of the full physics.
- **Cost: reverse 5.3× the primal, flat in parameter count; crossover
  ~10 parameters.** Forward mode's chunked duals remain fine at the
  current 8-19; the adjoint is ready the moment parameter counts grow
  (per-node fields, dense controls, the 2-D slab's ADI sweeps).

Forward mode stays the primary derivative in production; MTK remains
settled.

## Event sensitivities: the state-triggered wrap latch (2026-09-04)

"Wrap at 165 °F internal" is now differentiable without jump conditions:
a smooth IRREVERSIBLE latch state l with dl/dt = k_latch·σ((T_core −
T_trig)/δ_trig)·(1 − l) drives the wrap blend as the smooth OR
s = 1 − (1 − s_time)(1 − l). Because the latch is an ordinary smooth
state, d(anything)/d(T_trig) exists through plain AD — verified:
ForwardDiff through the full solve matches central FD within 5 %, and
the latch commits over the last ~10 minutes of the approach then holds.
Two lessons the tests encode: (1) δ_trig sets the pre-trigger
accumulation, not just softness — a 2 K width was 97 % committed hours
early because the latch integrates the σ tail (0.5 K keeps it
negligible); (2) the SIGN of d(w_retained)/d(T_trig) is
model-dependent — wrapping earlier saves evaporation but runs the meat
hotter and purges more, and at the test fixture the purge effect wins.
`T_wrap_trigger` is a meat_solve argument (Inf = exactly inert); the
monolith carries the latch state inert until the planner consumes the
trigger through the Tesseract (one optional field, like t_f).

## t_salt as a differentiable input (2026-09-04)

`t_salt` is now a solve argument on `meat_solve`, `monolith_solve` and
`monolith_solve_tf` (NaN defers to the config): a Dual brine duration
flows into the state through the analytic erfc profile, kept alive by
the initial-condition promotion, so d(juiciness)/d(t_salt) exists
through the full solve. Verified: the override matches the config path
to 1e-12, and ForwardDiff through an 8 h salted cook matches central FD
within 5 % with the physically right sign (longer brine → more water
held). What remains for the planner join is exposure only: one optional
Tesseract input plus a z-parameter in optimize_schedule — the same
pattern as t_f — shared with the wrap trigger T_trig and the wood-split
interval dt_add.

## Planner exposure of t_salt / T_trig / dt_add (2026-09-04, M1 complete)

The three decision variables the physics already carried are now
first-class through the whole chain, the t_f pattern end to end:

- **Julia**: the monolith p-tail grew to 8 entries (`T_wrap_trigger`,
  `dt_add`); `fuel_add_rate` takes the split interval as a parameter (so
  it can carry Duals); `monolith_solve`/`monolith_solve_tf` take
  `t_salt` / `T_wrap_trigger` / `dt_add` (`frac_dt_add` in duration
  mode — a fraction of the cook like `dt_spritz`; `t_salt` stays seconds
  in both modes, salting happens before the cook) and the stack
  jacobian/jvp carry three more columns. The wrap latch now actually
  fires inside the monolith (it existed as a state; the trigger never
  reached the RHS before).
- **Tesseract**: `t_salt` / `t_trig` / `dt_add` optional differentiable
  scalar inputs, 0 = off; disabled sentinels map to the exactly-inert
  values (Inf trigger/interval) whose derivative columns are exactly
  zero through the smooth blends — no special-casing needed beyond the
  seconds-mode t_f zero column.
- **Planner**: `opt_salt` / `opt_trig` / `opt_splits` flags with bounded
  sigmoid maps (48 h brine cap; 335–372 K trigger; 0.5–4 h splits) join
  `optimize_schedule` and `optimize_schedule_expected`; the objective
  prices a wood-split pulse train correctly (one m_chimney per split
  counted by its own before-the-pull gate). All three default OFF: the
  salt decision has no saltiness counterweight in the objective yet, so
  enabling `opt_salt` pins the brine at its bound — the spritz lesson;
  it gets a taste band when the salt constants are calibrated (C5).

Verified: defaults bit-identical to the old monolith; trigger-wrap
retains +0.12 water on the 8 h fixture; the split train adds 3+
chimneys of fuel mass; the three Jacobian columns match finite
differences (each on the output it actually drives — t_salt's
temperature derivative is ~1e-15 K/s, below FD noise, its signal is
`w_retained`; `dt_add` moves `m_fuel`, not the 2 h fixture's core) and
jvp matches jacobian in both time modes. Julia 305/305, app 73/73.

One Dual lesson worth keeping: ForwardDiff compares eps-aware —
`Dual(0,1) <= 0` is FALSE (0+eps > 0) — so a derivative seeded exactly
at the unsalted sentinel skipped `salt_profile`'s early return, fell
into the sqrt(t) brine singularity, and returned NaN partials that
killed the whole Dual solve (the planner's VJP sweeps every
differentiable input, sentinels included). Sentinel branches guarding a
singular limit must compare the PRIMAL value.

## 2-D axisymmetric core (2026-09-04, M2 solver half)

`src/meats/slab2d.jl`: `meat_solve_2d(θ2, ...)` — temperature and
dry-basis moisture on an r-z finite-cylinder grid with the SAME property
functions as the 1-D model (Choi-Okos mixture properties, Oswin water
activity, smooth purge) and the same convective/radiative/evaporative
surface conditions on rim and end faces. Fixed-step lagged-coefficient
ADI: implicit Thomas sweeps in r then z, exactly the structure the
adjoint work validated as Enzyme-transparent. v1 scope is deliberate: no
shrink, no wrap/spritz/salt/bone, no bark states, prescribed pit forcing
— the quality layers and the monolith coupling stay on the roadmap (M2),
bark calibration gated on thermal-camera fields (C2).

Verified against the 1-D solver in its two limits, plus a Newman sanity
and a ForwardDiff-vs-FD gradient check (7 new tests, suite 293/293):

- **Long-cylinder limit** (tall 2-D cylinder mid-height axis vs 1-D
  `geom_m=1`): max |ΔT| **0.20 K** over a 6 h cook (0.45 K at nr=48).
- **Thick-slab limit** (wide short cylinder axis vs 1-D slab through the
  half-height): max |ΔT| **0.14 K** (0.17 K at nz=120, h=5 s).
- Finite cylinder heats faster than either infinite limit (72.9 vs 70.2
  and 61.1 °C at mid-cook), as Newman predicts.

The limit tests caught two structural bugs worth recording. (1) The
z-sweep treated the boundary half-cells as full cells — flux terms used
Δz/2 capacity, diffusion rows used Δz — so the surface was
under-replenished by exactly 2x, over-drying and overheating it (slab
limit off by 25 K, h-independent, worse under refinement). (2) With the
face evaporation explicit, the face W overshoots the water-activity
equilibrium once the surface dries; the lagged flux then flips sign
every step and the alternating ±λ·j heat kicks rectify into net heating
that DIVERGES under mesh refinement (nr 24→48 pushed the cylinder core
97→108 °C, above any physical equilibrium). Fix: convective heat as an
implicit Robin term inside the sweeps, and face moisture by
backward-Euler on the W-linearized flux, with the SAME applied flux
feeding the heat sink — after which both limits are refinement-stable.

## 2-D productionization (2026-09-04, M1 complete)

The four remaining code pieces of the 2-D line, closing the roadmap's
last modeling goal:

- **Shrink on the grid.** The 1-D material-coordinate metrics mirrored
  per direction: st = (v/v0)^n_shrink stretches both spacings (the
  default 1/3 is the isotropic split, where the 2-D per-direction
  exponent coincides with each 1-D limit's resolved direction), heat
  flux/f_face with st in the capacity, moisture flux/f² dry-basis, and
  the exposed-area factor on the moisture flux only — for heat it
  cancels between current-area intake and current-volume capacity. The
  limit tests now run with shrink ON at the defaults (cylinder and slab
  limits still < 1.5 K); `R_eff`/`Hz_eff` report the shrunk dimensions
  and n_shrink = 0 is exactly inert.
- **Salt + spritz on the faces.** The erfc dry-brine profile from every
  rubbed face as a smooth union (corners saltiest), diffusing with the
  same metrics, driving the WHC purge coupling and the per-face water
  activity; spritz arrives on every face gated by the wrap, f_abs soaks
  in, the rest flashes off with its latent heat (the 1-D q_net terms).
- **Tesseract endpoint + shadow scoring.** `tesseracts/slab2d` wraps
  `coupled_solve_2d` with forward-mode jacobian/jvp/vjp over [θ2; φ;
  u_knots; t_wrap; t_salt; t_spritz; dt_spritz] (t_wrap column vs FD
  2e-4; jvp/vjp consistent). The planners stay 1-D;
  `brisket_app.shadow2d.shadow_report` + `scripts/run_shadow2d.py`
  re-simulate a planned cook in 2-D and report what only the grid can
  see. On the committed robust plan: TRUE-cold-spot lethality passes
  with margin (F_cold 3.8e11 vs 720 required, dwell 1.40 h vs the 4 h
  cap), retention 0.189, c_min 0.99997 — the 11 h plan leaves the
  surrogate uniformly done, the cold gap closed by serve — and the
  shrunk dimensions (R 10 → 7.7 cm) are reported per cook. *(Superseded
  2026-09-07: those numbers cooked the surrogate to the horizon; the
  rested-to-serve shadow is in the review-fixes entry above.)*
- **Reverse-mode performance.** `adjoint/slab2d_port.jl` — the
  transport core as explicit-argument top-level functions (the
  full_port pattern, no closures): gradient exact to 8.7e-15 and
  reverse now **13.7x primal** (blackbox through the closure engine:
  46x), with the primal itself 11x faster than the engine's. Absolute
  reverse cost fell 38x; crossover vs chunked forward sits near ~26
  parameters, so forward mode remains the calibration endpoint at
  today's parameter counts and the port is the template if the 2-D
  parameter space ever grows past that.

What remains of the 2-D line is calibration, not code: bark-distribution
constants against thermal-camera surface fields (folded into C2).

## 2-D slab: layers, pit coupling, reverse gradient (2026-09-04, M2 code complete)

Three commits finish the code half of M2 on top of the limit-verified
core:

- **Layers** (each exactly inert at its default): the 1-D wrap blend on
  every exposed face; `k_grain` / `D_grain` in-plane (along-fiber)
  anisotropy with the z through-thickness direction keeping the
  1-D-calibrated values; point-over-flat as stacked muscles (smooth fat
  step `F_top` above `z_split`·Hz — a uniform cut stays z-symmetric to
  roundoff, which is the asymmetry test). Quality states ride along
  explicitly: collagen per cell (`C_min` = last-done spot), Maillard
  bark per face cell with an area-weighted distribution (burnt ends =
  the tail), and `Tcold`, the true grid-minimum cold spot. Collagen at
  the thick-slab limit matches the 1-D center kinetics to < 0.02.
- **Pit coupling**: the transport refactored into an engine + `step!`
  closure; `coupled_solve_2d` co-integrates the six pit states (the
  monolith's algebra, two-zone included) with the area-integrated heat
  uptake and evaporation of `n_meat` briskets, vapor feeding the
  humidity one step lagged, face evaporation now p_v-aware. The
  consistency test is exact by construction: at `n_meat = 0` with
  pinned humidity, replaying the coupled run's own per-step pit trace
  through the prescribed-forcing solver reproduces the meat fields to
  roundoff. (And a small physical truth the direction test tripped on:
  late in a declining burn the hot meat returns heat to the pit — the
  pull-down only holds while the meat is cold.)
- **Enzyme reverse** (`adjoint/slab2d_reverse.jl`): blackbox
  `Enzyme.autodiff(Reverse, ...)` through the WHOLE solver — engine
  closure, linearized-implicit faces, quality states — matches
  ForwardDiff to **1.3e-14**. But it runs at 46x primal while chunked
  forward over all 6 parameters costs 1.4x primal: for
  calibration-scale parameter counts the forward endpoint stays the
  right choice, and the reverse-mode optimization path is the
  full_port-style explicit-argument stepper (3.6x primal there), not
  blackbox autodiff over the closure engine.

Suite 322/322 after the coupling commit; app gate 73/73. What remains
of the 2-D line is productionization (Tesseract/planner exposure,
shrink on the grid, reverse-mode performance) and the C2-gated bark
calibration — the roadmap's remaining M-goal.

## The cook pilot, validated without hardware (2026-09-04)

`brisket_app/pit_io.py` (one pit contract: `HeaterMeterClient` speaks the
LinkMeter v14 REST API — status GET, setpoint POST + apikey — and
`SimPit` is the calibrated plant behind the same contract with an
HM-style PI turning setpoints into damper duty), `brisket_app/pilot.py`
(the `run_cook` control loop lifted out of simulation: re-plans schedule
+ wrap + pull each interval, pushes the PLANNED PIT TRAJECTORY as
setpoints, advances the belief state under MEASURED fan duty, snaps the
pit state to the sensor, runs the ambient KF on the innovation with an
optional hourly FORECAST as feedforward, and journals everything as
JSONL), `scripts/run_pilot.py` (the daemon; wrap/pull are advisories on
hardware, auto-executed on sim backends), `scripts/mock_heatermeter.py`
(the sim plant behind the real v14 endpoints) and
`scripts/plot_pilot.py`. Validation: the same 11-h fullplan cook flown
twice against an unannounced −15 K cold front at 2 h — in-process, and
over HTTP through the REAL client path against the mock — lands nearly
identically (wrap re-timed 7.06 → 6.83 h, pull 8.39 → 10.07/10.10 h; the
KF recovers ~60 % of the ambient drop and the setpoint plan compensates
with a ~300 °F push; `figures/pilot_http.png`,
`data/control/pilot_{sim,http}*.jsonl/json`).

## The `brisket` CLI (2026-09-04)

`brisket_app/cli.py`: one `brisket` entry point (`uv run brisket ...`
from app/) — `logs`, `digitize`, `calibrate` (any registry log OR a user
CSV via the cook-log column convention; GN-MAP + Laplace; writes a
`--params`-compatible `fit_<name>.json`), and verbatim forwards to the
canonical scripts for `plan`, `pilot`, `pool`, `mock-pit`. The your-log →
your-plan → your-pilot flow is three commands (README "Using it on your
own cook").

## The weights hunt and its lab resolution

No smoker log with BEFORE/AFTER weights was found in the graphs thread or
the keg threads (pre-cook weights are common, yield pairs are not). The
lab route resolved the moisture-physics validation instead (the CET oven
roast above). Blonder's stall experiments (genuineideas.com; core-temp
curves at known smoker temps, coarse loss figures) remain a soft
cross-check option.
