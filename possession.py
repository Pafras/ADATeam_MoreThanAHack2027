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

# Dead-ball evidence
ACTIVITY_WINDOW = 50  # half-width in frames of the smoothing over mean player speed
ACTIVITY_LOW, ACTIVITY_LIVE = 10, 50  # percentiles of the match taken as stopped and as live play
GAP_FULL = 10 * FPS  # a ball missing for this many frames counts as fully suspicious
OUT_MARGIN = 0.5  # metres beyond the lines before a ball counts as out
WEIGHT_ACTIVITY = 0.15
WEIGHT_GAP = 0.15
WEIGHT_OUT = 0.5
WEIGHT_FAST_BALL = 0.3  # a ball moving fast is a ball in play

# Smart Tagging evidence
WEIGHT_PHASE = 0.05  # weak on purpose: phases are long and are not possessions
WEIGHT_RESTART = 0.3
WEIGHT_BEFORE_KICKOFF = 5.0  # effectively certain, but still only evidence

# Viterbi switch costs
SWITCH_TEAM = 15.0
SWITCH_DEAD = 10.0

TAG_TEAMS = {"barcelona": H, "atletic club": A}
ATTACKING_CODES = {"build up", "progression", "creating chances", "attacking transition",
                   "long ball", "ball in final third", "ball in the box", "players in the box"}
DEFENDING_CODES = {"defending in attacking third", "defending in middle third",
                   "defending in defensive third", "defensive transition"}


def read_features(raw_path, team_of_track, n_frames, pitch):
    """Per-frame arrays from the raw ATD: ball position, nearest player per team, mean speed."""
    length, width = pitch
    features = {name: np.full(n_frames, np.nan)
                for name in ("ball_x", "ball_y", "near_home", "near_away", "activity")}

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
            for slot, group in enumerate(parts[1].split(";")):
                x, y, speed = (float(value) for value in group.split(","))
                if x != x:
                    continue
                if speed == speed:
                    speed_sum += speed
                    speed_count += 1
                if ball_x == ball_x:
                    distance = (x * length - ball_x) ** 2 + (y * width - ball_y) ** 2
                    team = team_of_track[slot]
                    if distance < near[team]:
                        near[team] = distance

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

    # Dead: players slowing down, relative to this match's own stopped and live levels, so that
    # no absolute speed has to be assumed.
    activity = smooth_ignoring_nan(features["activity"], ACTIVITY_WINDOW)
    usable = graded & ~np.isnan(activity)
    if usable.any():
        low, live = np.percentile(activity[usable], [ACTIVITY_LOW, ACTIVITY_LIVE])
        middle, scale = (low + live) / 2.0, max((live - low) / 2.0, 1e-3)
        scores[:, D] += WEIGHT_ACTIVITY * np.clip(
            np.nan_to_num((middle - activity) / scale), -1, 1)

    gaps = missing_run_lengths(~visible)
    scores[:, D] += WEIGHT_GAP * np.minimum(gaps / float(GAP_FULL), 1.0)
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
            # kickoffs. Measure it across the set pieces if restart timing turns out to matter.
            middle = (start + stop) // 2
            scores[start:middle, D] += WEIGHT_RESTART
            if team is not None:
                scores[middle:stop, team] += WEIGHT_RESTART
            if kind.startswith("kick off"):
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


def build(game=GAME, data_dir=None):
    """The full-length label array. X outside the two halves, H/A/D inside them."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    require(["smart_tagging"], game, data_dir)
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    blocks = half_video_blocks(game)

    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True

    features = read_features(raw_path, team_of_each_track(meta_path), n_frames, pitch)
    scores = (tracking_scores(features, graded, pitch)
              + tag_scores(load_tags(game, data_dir), n_frames, blocks))

    labels = np.full(n_frames, "X", dtype="<U1")
    for _, first, last in blocks:
        labels[first:last + 1] = LABELS[viterbi(scores[first:last + 1])]
    return labels


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
