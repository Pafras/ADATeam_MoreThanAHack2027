# More than a Hack 2027, Phase 1: technical document

**Team:** ADA Team

**Repository:** https://github.com/Pafras/ADATeam_MoreThanAHack2027

## Approach

We label every frame H, A or D with a rule-based pipeline and no trained model. It has three steps:

1. **Features.** From the tactical tracking we read, per frame, the ball position, each team's
   nearest player to the ball, and the players' mean speed.
2. **Scores.** Tracking and Smart Tagging give every frame a score for H, A and D.
3. **Viterbi.** Within each half we find the best H/A/D path, with a fixed cost for each change of
   state.

We chose this because the ranking metric, `spellF1`, scores whole possessions. A possession is an
interval, so we treat the labels as a sequence rather than classifying frames one at a time. The
switch cost stops short flickers from splitting a possession. A rule-based model also fits the
rules: no human labels may be used on this match, so there is nothing on it to train against. Every
weight can be traced to a reason stated in the code.

The output contains only H, A and D inside the halves, and X outside them. We do not use X on graded
frames, because the Viterbi path already holds the last team through a gap.

## Possession rules implemented

| Situation | Rule |
|---|---|
| Ball visible and slow | The team with the nearer player gains evidence, more so the closer that player is |
| Ball in flight, above 10 m/s | Proximity evidence is reduced, because the nearest player is not controlling it |
| Touch | A change in ball velocity of at least 1 m/s over 2 frames within 4 m of a player, with one team at least 1 m nearer |
| Change of possession | Takes about three touches in a row, following the "more than two consecutive passes" proxy in `CHALLENGE.md` |
| One or two touches, deflections | Do not pay the switch cost, so possession stays with the same team |
| Loose or missing ball | No evidence for either team, so possession stays with the last team, as `CHALLENGE.md` states |
| Dead ball | Players slowed down **and** the ball is missing for long or standing still; or the ball is out of play |
| Slow players with a moving ball | Not dead: a team walking the ball out from the back looks the same |
| Before each half's kickoff | Dead, from the `Kick Off (Start)` tag |
| From a goal to the next kickoff | Dead, from the `GOALS` and `Kick Off` tags |
| After a set piece | The tagged team gains evidence, except throw-ins, which have no team tag |

## Assets

**Used**

- **Tactical tracking (ATD).** The main input.
- **Metrica Sports public sample data, Sample Games 1 and 2**
  (https://github.com/metrica-sports/sample-data, pinned commit `e706dd5`). External data, used only
  to tune the touch thresholds; see below. The repository asks for responsible use and attribution.
- **Smart Tagging.** Weak evidence: phase tags point at a team, set pieces at the team restarting,
  and kickoffs and goals at stoppages. It is treated as evidence and never copied across as labels.
  We also use it as a reference to compare versions.

**Not used**

- **Broadcast video and tracking.** They are not synchronised to the tactical video, and we did not
  attempt the mapping.
- **Video.** The pipeline does not decode it.
- **Homography.** The tracking is already in pitch coordinates.

**Tried and dropped**

- **Touch thresholds from this match's tracking.** Our first touch detector took its thresholds from
  the ball velocity distributions of this match (`diagnose_touches.py`). The organizers clarified
  that tuned parameters must be tuned on external data, so we replaced them with thresholds tuned on
  Metrica's sample games.
- **Dead-ball evidence from every set piece.** It assumes the restart sits mid-interval, which is
  documented only for kickoffs. It is now used for kickoffs only.
- **Marking down the other team from phase tags.** Two variants were tried so that phases would not
  hold back dead-ball evidence. Both raised the dead share of graded frames to about 54% and were
  reverted.
- **Team spread with the ball hidden.** Measured in `diagnose_hidden.py`: the team with the ball
  tends to be more spread out (0.696 balanced agreement with clear ball frames, 0.5 meaning no
  information). Added to the pipeline, it left our handover measure unchanged, so it was left out.
- **Block movement, block position and line gap.** Did not reach our bar in the same measurement.

## Training data

None. The pipeline trains no model and uses no labelled data from this match or any other.

## Choosing weights without labels

The public five minutes were not used as training data. Weights come from three sources:

- **The challenge's definitions.** `SWITCH_TEAM` and `WEIGHT_TOUCH` are derived from the control
  proxies in `CHALLENGE.md`: holding the ball for a few seconds, and more than two consecutive passes.
- **Tracking distributions, with no labels.** The activity threshold uses Otsu's method, computed
  from the data with no tuned parameter.
- **External data.** `external_touches.py` picks `TOUCH_LAG`, `TOUCH_CHANGE` and `TOUCH_RANGE` by
  grid search against the event touches (passes, receptions, recoveries, shots, ball losses, set
  pieces) of Metrica's Sample Games 1 and 2. The matching rule was fixed before the search: same
  team within 10 frames. The chosen setting scores F1 0.90 on each game, and choosing on one game
  and scoring on the other gives the same setting. Nothing from this match went into the choice.
- **Football reasoning.** The remaining weights are set by judgement and were not fitted.

Whether touches are used at all was decided by `compare_versions.py`, with criteria written before
it ran: touches had to find more tagged changes of possession without breaking the dead-ball share,
restarts or the number of possessions. They passed every criterion.

To compare versions we wrote acceptance criteria before running each one, and measured against
Smart Tagging over the full match (`validate_tags.py`). Smart Tagging is inferred from the same
tracking, so this is a weak reference. On the four published baselines, its measures did not move
in line with the published full-match `spellF1`, so it cannot rank submissions by score.

**Disclosure on the public window.** We looked at the public five-minute score for every version to
check the pipeline. We did not select versions by it, and no weight was fitted to it. Seeing too few
possessions there did influence what we chose to work on next, namely detecting changes of
possession.

## Results

Public five minutes, 17 true possessions. This is a check and not an estimate of the final score:
one possession moves `spellF1` by about six points here.

| Submission | `macroF1` | `spellF1` | Possessions | Mean duration |
|---|---|---|---|---|
| Ours | 0.6527 | 0.4000 | 15 | 19.0 s |
| `baseline_smooth125` | 0.5812 | 0.4242 | 16 | 18.5 s |
| `baseline_smooth50` | 0.6207 | 0.3256 | 26 | 10.8 s |

Over the full match our pipeline reports 218 possessions, and 44.0% of graded frames as dead ball. We have no full-match score, because that ground truth is private.

## Limitations

- **Not validated on the full match.** Apart from the five public minutes, we have no measure of the
  labels' accuracy. Our weights may not be the best ones.
- **Possessions are still merged.** We find 15 of the 17 in the public window, with too long a mean
  duration. Possessions of about a second are likely to be missed by design.
- **Touches can name the wrong team.** The nearest player at a touch is not always the team that
  gains the ball.
- **Touch thresholds come from other games.** Metrica's sample games are tracked differently from
  this match, so the thresholds may not be the best ones here. We did not adjust them to this match.
- **The ball is missing on about 41% of graded frames.** There the label only carries the last team
  forward, or moves to dead ball if the players have also stopped. We lack a signal that reliably
  finds a change of possession during those frames.
- **Restart timing.** After a set piece, team evidence starts mid-interval, an assumption documented
  only for kickoffs.

With more time we would separate a kick from a reception, so that a touch names the team gaining
control, and train on labelled data from other matches.

## Use of LLMs

We used Claude (Anthropic) to help write code and documentation. The pipeline itself makes no LLM
calls. No video, tracking, tagging or homography files were uploaded to it. It saw aggregate output
of our scripts, such as label counts, distribution percentiles and scores.

## Reproduction

The commands are in the **Our submission** section of `README.md`. In short, with Python 3.9 and the
three data files in `data/tactical/`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/python possession.py --out "evaluation/results/ADA Team.csv"
```

The pipeline has no randomness. On our machine, a fresh clone with the pinned dependencies produced
a file byte-identical to our submission. We have not tested it on a different machine.
`python test_possession.py` runs the checks on a synthetic match and needs no data.
