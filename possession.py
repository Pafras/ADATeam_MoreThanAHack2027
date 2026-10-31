"""Frame-level ball possession: per-frame evidence, then a three-state Viterbi over each half.

    python possession.py --out evaluation/results/my_team.csv
    python evaluation/validate_submission.py evaluation/results/my_team.csv

Three steps, no training and no labels:

  1. Read the ATD into per-frame features: how close each team is to the ball, how fast the ball
     moves, how long it has been missing, how fast the players move.
  2. Turn the features and the Smart Tagging into a score for H, A and D on every frame.
  3. Find the best H/A/D path through each half, with a fixed cost for every change of state.

Step 3 is what makes the output possessions rather than frames. A missing ball scores nothing for
either team, so the path simply stays with the last team, which is the challenge's own rule for a
loose ball. A short touch cannot pay the two switches it would take to enter and leave a possession.

The weights below come from football reasoning, not from measuring labels. Do not tune them on the
public five minutes: RULES.md forbids it and EVALUATION.md shows it picks the wrong answer.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from baseline import team_of_each_track, write_submission  # noqa: E402
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from load_smart_tagging import load as load_tags  # noqa: E402

# Params
H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])
FPS = 25

# Ball speed
BALL_SPEED_LAG = 5  # frames between the two positions a speed is measured from
BALL_MAX_SPEED = 40.0  # m/s; anything above this is the tracker jumping, not the ball
BALL_FAST = 10.0  # m/s; above this the ball is in flight and the nearest player does not control it

# Team evidence from the ball
TEAM_MARGIN = 3.0  # metres between the nearest players at which the closer team is clearly closer
CONTROL_DISTANCE = 2.0  # metres; evidence decays with the nearest player's distance on this scale
NO_PLAYER_DISTANCE = 100.0  # metres, standing in for a team with no tracked player on a frame
WEIGHT_BALL = 0.5

# Touches: the ball's velocity changes sharply next to a player
# The ball track is smooth, so a kick is spread over several frames: over 3 frames the velocity
# change only reaches 4.2 m/s at its 99th percentile, over 8 frames 10.8 m/s. These come from that
# distribution in the tracking (diagnose_touches.py), not from any labels, and give about 1400
# touches on the 59% of frames with a ball, in line with a match's passes and receptions.
TOUCH_LAG = 8  # frames either side of the touch that each velocity is measured over
TOUCH_CHANGE = 4.0  # m/s of velocity change; below this it could be tracker jitter
TOUCH_RANGE = 4.0  # metres from the ball to the nearest player, allowing for the smoothing
TOUCH_MARGIN = 1.0  # metres; a closer call than this between the teams names nobody
TOUCH_GAP = 10  # frames; changes closer together than this are one touch
# CHALLENGE.md names "a team completes more than two consecutive passes" as a proxy for control.
# A spell needs 2 * SWITCH_TEAM = 16 of H-A gap and each touch adds 2 * 3 = 6, so three touches in a
# row take possession and one or two do not.
WEIGHT_TOUCH = 3.0

# Dead-ball evidence
ACTIVITY_WINDOW = 50  # half-width in frames of the smoothing over mean player speed
GAP_FULL = 10 * FPS  # a ball missing for this many frames counts as fully suspicious
BALL_STILL = 1.0  # m/s; below this a visible ball is waiting for a restart or at someone's feet
OUT_MARGIN = 0.5  # metres beyond the lines before a ball counts as out
WEIGHT_STOPPED = 0.3  # players stopped AND the ball missing for long or standing still
WEIGHT_LIVE = 0.15  # players moving is evidence against a stoppage
WEIGHT_OUT = 0.5
WEIGHT_FAST_BALL = 0.3  # a ball moving fast is a ball in play

# Smart Tagging evidence
WEIGHT_PHASE = 0.05  # weak on purpose: phases are long and are not possessions
WEIGHT_RESTART = 0.3
WEIGHT_BEFORE_KICKOFF = 5.0  # effectively certain, but still only evidence

# Viterbi switch costs
# CHALLENGE.md names "a player keeps the ball for a few seconds" as a proxy for control. Holding it
# for 3 s at 1.5 m, with the ball visible 60% of the time, adds up to about 8.5 of evidence, and each
# frame moves the H-A gap by twice that. A spell needs two switches, so a cost of 8 lets it through.
SWITCH_TEAM = 8.0
SWITCH_DEAD = 10.0

TAG_TEAMS = {"barcelona": H, "atletic club": A}
ATTACKING_CODES = {"build up", "progression", "creating chances", "attacking transition",
                   "long ball", "ball in final third", "ball in the box", "players in the box"}
DEFENDING_CODES = {"defending in attacking third", "defending in middle third",
                   "defending in defensive third", "defensive transition"}


def read_features(raw_path, team_of_track, n_frames, pitch):
    """Per-frame arrays from the raw ATD: ball, nearest player per team, mean speed, team shape."""
    length, width = pitch
    features = {name: np.full(n_frames, np.nan)
                for name in ("ball_x", "ball_y", "near_home", "near_away", "activity",
                             "home_x", "home_y", "home_spread", "away_x", "away_y", "away_spread")}

    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # frame_id counts from 1, submissions from 0
            ball = parts[2].split(",")
            ball_x, ball_y = float(ball[0]) * length, float(ball[1]) * width

            near = [np.inf, np.inf]
            speed_sum, speed_count = 0.0, 0
            shape = [[0, 0.0, 0.0, 0.0, 0.0], [0, 0.0, 0.0, 0.0, 0.0]]  # n, x, y, x^2, y^2
            for slot, group in enumerate(parts[1].split(";")):
                x, y, speed = (float(value) for value in group.split(","))
                if x != x:
                    continue
                if speed == speed:
                    speed_sum += speed
                    speed_count += 1
                team = team_of_track[slot]
                if ball_x == ball_x:
                    distance = (x * length - ball_x) ** 2 + (y * width - ball_y) ** 2
                    if distance < near[team]:
                        near[team] = distance
                metres_x, metres_y = x * length, y * width
                totals = shape[team]
                totals[0] += 1
                totals[1] += metres_x
                totals[2] += metres_y
                totals[3] += metres_x * metres_x
                totals[4] += metres_y * metres_y

            for side, name in ((H, "home"), (A, "away")):
                count, sum_x, sum_y, sum_xx, sum_yy = shape[side]
                if count:
                    mean_x, mean_y = sum_x / count, sum_y / count
                    features[name + "_x"][frame] = mean_x
                    features[name + "_y"][frame] = mean_y
                    features[name + "_spread"][frame] = max(
                        sum_xx / count - mean_x * mean_x + sum_yy / count - mean_y * mean_y,
                        0.0) ** 0.5

            features["ball_x"][frame] = ball_x
            features["ball_y"][frame] = ball_y
            if ball_x == ball_x:
                features["near_home"][frame] = min(near[H] ** 0.5, NO_PLAYER_DISTANCE)
                features["near_away"][frame] = min(near[A] ** 0.5, NO_PLAYER_DISTANCE)
            if speed_count:
                features["activity"][frame] = speed_sum / speed_count
    return features


def ball_speed(ball_x, ball_y):
    """m/s over BALL_SPEED_LAG frames. NaN where either end is missing or the tracker jumped."""
    speed = np.full(len(ball_x), np.nan)
    moved = np.hypot(ball_x[BALL_SPEED_LAG:] - ball_x[:-BALL_SPEED_LAG],
                     ball_y[BALL_SPEED_LAG:] - ball_y[:-BALL_SPEED_LAG])
    speed[BALL_SPEED_LAG:] = moved * FPS / BALL_SPEED_LAG
    speed[speed > BALL_MAX_SPEED] = np.nan
    return speed


def detect_touches(features):
    """(frames, teams) of ball touches: a sharp change of velocity with one team clearly nearest."""
    ball_x, ball_y = features["ball_x"], features["ball_y"]
    velocity_x = (ball_x[TOUCH_LAG:] - ball_x[:-TOUCH_LAG]) * FPS / TOUCH_LAG
    velocity_y = (ball_y[TOUCH_LAG:] - ball_y[:-TOUCH_LAG]) * FPS / TOUCH_LAG
    with np.errstate(invalid="ignore"):
        jumped = np.hypot(velocity_x, velocity_y) > BALL_MAX_SPEED
        velocity_x[jumped] = np.nan

        change = np.full(len(ball_x), np.nan)
        change[TOUCH_LAG:-TOUCH_LAG] = np.hypot(velocity_x[TOUCH_LAG:] - velocity_x[:-TOUCH_LAG],
                                                velocity_y[TOUCH_LAG:] - velocity_y[:-TOUCH_LAG])
        near = np.minimum(features["near_home"], features["near_away"])
        margin = features["near_away"] - features["near_home"]
        candidates = np.flatnonzero((change > TOUCH_CHANGE) & (near < TOUCH_RANGE)
                                    & (np.abs(margin) >= TOUCH_MARGIN))

    # Keep the sharpest frame of each run of candidates.
    frames = []
    for frame in candidates:
        if frames and frame - frames[-1] <= TOUCH_GAP:
            if change[frame] > change[frames[-1]]:
                frames[-1] = frame
        else:
            frames.append(frame)
    frames = np.array(frames, dtype=np.int64)
    return frames, np.where(margin[frames] > 0, H, A)


def missing_run_lengths(missing):
    """For each frame, the length of the run of missing frames it belongs to. 0 where present."""
    lengths = np.zeros(len(missing), dtype=np.int64)
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        lengths[start:stop] = stop - start
    return lengths


def smooth_ignoring_nan(values, window):
    """Centred mean over +/- window frames, skipping NaN. NaN where the window has no values."""
    known = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(known, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(known)))
    index = np.arange(len(values))
    low, high = np.maximum(0, index - window), np.minimum(len(values), index + window + 1)
    seen = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(seen > 0, (total[high] - total[low]) / seen, np.nan)


def two_levels(values):
    """Means of the two groups that best split the values (Otsu's threshold).

    The split comes from the shape of the distribution, so it assumes nothing about how large
    either group is. A percentile would decide the share of dead ball before looking at the match.
    """
    # ponytail: a fixed 256-bin histogram with no check that two groups exist. Fit a mixture model
    # if the match turns out not to separate into stopped and live play.
    counts, edges = np.histogram(values, bins=256)
    centres = (edges[:-1] + edges[1:]) / 2.0
    below = np.cumsum(counts)[:-1]
    above = counts.sum() - below
    mass_below = np.cumsum(counts * centres)[:-1]
    mass_above = (counts * centres).sum() - mass_below
    with np.errstate(invalid="ignore", divide="ignore"):
        low, high = mass_below / below, mass_above / above
        spread = np.nan_to_num(below * above * (high - low) ** 2)
    best = spread.argmax()
    return low[best], high[best]


def tracking_scores(features, graded, pitch):
    """n_frames x 3 scores for H, A and D from the tracking alone."""
    length, width = pitch
    n_frames = len(features["ball_x"])
    scores = np.zeros((n_frames, 3))
    visible = ~np.isnan(features["ball_x"])
    speed = ball_speed(features["ball_x"], features["ball_y"])

    # Team: the closer team, weighted by how close it is and by how slow the ball is.
    with np.errstate(invalid="ignore"):
        margin = np.clip((features["near_away"] - features["near_home"]) / TEAM_MARGIN, -1, 1)
        control = np.exp(-np.minimum(features["near_home"], features["near_away"])
                         / CONTROL_DISTANCE)
        slow = 1.0 / (1.0 + (np.nan_to_num(speed) / BALL_FAST) ** 2)
        team = np.where(visible, WEIGHT_BALL * margin * control * slow, 0.0)
        out = visible & ((features["ball_x"] < -OUT_MARGIN)
                         | (features["ball_x"] > length + OUT_MARGIN)
                         | (features["ball_y"] < -OUT_MARGIN)
                         | (features["ball_y"] > width + OUT_MARGIN))
    team[out] = 0.0
    scores[:, H] += team
    scores[:, A] -= team

    frames, teams = detect_touches(features)
    touch = WEIGHT_TOUCH * np.where(teams == H, 1.0, -1.0)
    scores[frames, H] += touch
    scores[frames, A] -= touch

    # Dead: players slowing down, relative to this match's own stopped and live levels, so that
    # no absolute speed has to be assumed. Slow players alone are not a stoppage, because a team
    # walking the ball out from the back looks the same, so the ball has to agree: missing for a
    # long time, or visible and not moving.
    activity = smooth_ignoring_nan(features["activity"], ACTIVITY_WINDOW)
    usable = graded & ~np.isnan(activity)
    if usable.any():
        low, live = two_levels(activity[usable])
        middle, scale = (low + live) / 2.0, max((live - low) / 2.0, 1e-3)
        pace = np.clip(np.nan_to_num((middle - activity) / scale), -1, 1)

        gaps = np.minimum(missing_run_lengths(~visible) / float(GAP_FULL), 1.0)
        with np.errstate(invalid="ignore"):
            still = np.where(visible, (speed < BALL_STILL).astype(float), np.nan)
        still = np.nan_to_num(smooth_ignoring_nan(still, ACTIVITY_WINDOW))

        scores[:, D] += WEIGHT_STOPPED * np.maximum(pace, 0.0) * np.maximum(gaps, still)
        scores[:, D] -= WEIGHT_LIVE * np.maximum(-pace, 0.0)
    scores[out, D] += WEIGHT_OUT
    scores[:, D] -= WEIGHT_FAST_BALL * (np.nan_to_num(speed) > BALL_FAST)
    return scores


def tag_scores(events, n_frames, blocks):
    """n_frames x 3 scores from Smart Tagging. Evidence only: every tag can be wrong."""
    scores = np.zeros((n_frames, 3))
    next_kickoffs = []

    for event in events:
        code = (event["code"] or "").strip().lower()
        kind = (event["tags"].get("Type") or "").strip().lower()
        team = TAG_TEAMS.get((event["tags"].get("Team") or "").strip().lower())
        start, stop = max(event["start_frame"], 0), min(event["end_frame"], n_frames)

        if code == "set pieces":
            # ponytail: assumes the restart sits mid-interval, which is only documented for
            # kickoffs, so only kickoffs get dead-ball evidence from it. Measure the timing across
            # the other set pieces if restarts turn out to matter.
            middle = (start + stop) // 2
            if team is not None:
                scores[middle:stop, team] += WEIGHT_RESTART
            if kind.startswith("kick off"):
                scores[start:middle, D] += WEIGHT_RESTART
                next_kickoffs.append(start)
            if kind == "kick off (start)":
                for _, first, last in blocks:
                    if first <= start <= last or first <= stop <= last:
                        scores[first:start, D] += WEIGHT_BEFORE_KICKOFF
        elif team is not None and code in ATTACKING_CODES:
            scores[start:stop, team] += WEIGHT_PHASE
        elif team is not None and code in DEFENDING_CODES:
            scores[start:stop, 1 - team] += WEIGHT_PHASE

    # The ball is dead from a goal until the kickoff that follows it.
    for event in events:
        if (event["code"] or "").strip().lower() == "goals":
            later = [start for start in next_kickoffs if start >= event["end_frame"]]
            if later:
                scores[event["end_frame"]:min(later), D] += WEIGHT_RESTART
    return scores


def viterbi(scores):
    """Best state per frame given the per-frame scores and the switch costs."""
    switch = np.array([[0.0, SWITCH_TEAM, SWITCH_DEAD],
                       [SWITCH_TEAM, 0.0, SWITCH_DEAD],
                       [SWITCH_DEAD, SWITCH_DEAD, 0.0]])
    n_frames = len(scores)
    came_from = np.zeros((n_frames, 3), dtype=np.int8)
    best = scores[0].copy()
    for frame in range(1, n_frames):
        options = best[:, None] - switch
        came_from[frame] = options.argmax(axis=0)
        best = options.max(axis=0) + scores[frame]

    path = np.zeros(n_frames, dtype=np.int8)
    path[-1] = best.argmax()
    for frame in range(n_frames - 1, 0, -1):
        path[frame - 1] = came_from[frame, path[frame]]
    return path


def graded_mask(n_frames, blocks):
    """True on the frames inside the two halves."""
    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    return graded


def label_halves(scores, blocks):
    """The full-length label array. X outside the two halves, H/A/D inside them."""
    labels = np.full(len(scores), "X", dtype="<U1")
    for _, first, last in blocks:
        labels[first:last + 1] = LABELS[viterbi(scores[first:last + 1])]
    return labels


def build_features(game=GAME, data_dir=None):
    """Per-frame features for a whole match, the expensive step."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    return read_features(raw_path, team_of_each_track(meta_path),
                         source_frames("tactical", game), GAMES[game]["pitch"])


def scores_from_features(features, game=GAME):
    """Tracking scores for a whole match, with no Smart Tagging in them."""
    graded = graded_mask(source_frames("tactical", game), half_video_blocks(game))
    return tracking_scores(features, graded, GAMES[game]["pitch"])


def build_tracking_scores(game=GAME, data_dir=None):
    features = build_features(game, data_dir)
    frames, teams = detect_touches(features)
    print("touches: {} ({} H, {} A)".format(len(frames), int((teams == H).sum()),
                                            int((teams == A).sum())))
    return scores_from_features(features, game)


def build(game=GAME, data_dir=None):
    """The full-length label array from tracking and Smart Tagging together."""
    require(["smart_tagging"], game, data_dir)
    n_frames = source_frames("tactical", game)
    blocks = half_video_blocks(game)
    scores = (build_tracking_scores(game, data_dir)
              + tag_scores(load_tags(game, data_dir), n_frames, blocks))
    return label_halves(scores, blocks)


def summary(labels, game=GAME):
    """Aggregate counts only, so the output can be shared without exposing frame-level data."""
    parts = []
    for half, first, last in half_video_blocks(game):
        window = labels[first:last + 1]
        starts = np.flatnonzero(np.concatenate(([True], window[1:] != window[:-1])))
        runs = np.diff(np.concatenate((starts, [len(window)])))
        possession = np.isin(window[starts], ["H", "A"])
        parts.append("half {}: {} possessions, mean {:.1f} s, {} stoppages, H {} A {} D {}".format(
            half, int(possession.sum()),
            runs[possession].mean() / FPS if possession.any() else 0.0,
            int((~possession).sum()),
            int((window == "H").sum()), int((window == "A").sum()), int((window == "D").sum())))
    return "\n".join(parts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="CSV to write")
    args = parser.parse_args()

    result = build()
    print("wrote {}".format(write_submission(args.out, result)))
    print(summary(result))
