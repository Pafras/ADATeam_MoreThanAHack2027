"""Can team shape say who has the ball when the ball itself is hidden? Prints aggregates only.

    python diagnose_hidden.py

Nothing here touches the pipeline. Each signal is positive when it points at H, and is scored
against two references, neither of them human labels:

  ball_*    frames where tracking clearly shows one team on the ball. Tells us if the signal works
            at all, but these frames are not hidden ones.
  tags_*    frames with no ball where the phase tags vote for one team. The only view of hidden
            frames, and independent of the ball.

Agreement is balanced: the mean over frames where the reference says H and where it says A. Home
has the ball more often, so plain agreement would reward a signal that just always says H. A signal
with no information scores 0.5.

  *_all     all frames with a reference
  *_top     the third of those frames where the signal is strongest; should beat *_all if the
            signal's size means anything

Result on this match: only spread cleared the bar (ball_all 0.696). Added to the pipeline on hidden
frames it left ho_recall unchanged, so it was left out. tags_* run high for spread and position,
probably because Smart Tagging is inferred from the same tracking, so they are weak evidence.
"""

import numpy as np

from possession import ATTACKING_CODES, DEFENDING_CODES, FPS, GAME, H, TAG_TEAMS, build_features, \
    graded_mask, half_video_blocks, load_tags, smooth_ignoring_nan, two_levels

SMOOTH = FPS  # half-width in frames of the smoothing over each signal
SHIFT_LAG = FPS  # frames either side that the movement of the players is measured over
CONFIDENT_DISTANCE = 2.0  # metres from the ball to the nearest player
CONFIDENT_MARGIN = 2.0  # metres between the nearest players of the two teams


def attack_directions(features, blocks):
    """+1 or -1 per frame: the x direction the home team attacks, read from where each team stands."""
    direction = np.zeros(len(features["home_x"]))
    for _, first, last in blocks:
        gap = np.nanmean(features["away_x"][first:last + 1] - features["home_x"][first:last + 1])
        direction[first:last + 1] = 1.0 if gap > 0 else -1.0
    return direction


def signals(features, blocks):
    """Name -> per-frame signal, positive when it points at the home team."""
    direction = attack_directions(features, blocks)
    block_x = (features["home_x"] + features["away_x"]) / 2.0

    shift = np.full(len(block_x), np.nan)
    shift[SHIFT_LAG:-SHIFT_LAG] = (block_x[2 * SHIFT_LAG:] - block_x[:-2 * SHIFT_LAG]) \
        * FPS / (2.0 * SHIFT_LAG)

    raw = {
        "spread": features["home_spread"] - features["away_spread"],
        "shift": shift * direction,
        "position": (block_x - np.nanmean(block_x)) * direction,
        "line_gap": (features["home_x"] - features["away_x"]) * direction,
    }
    return {name: smooth_ignoring_nan(values, SMOOTH) for name, values in raw.items()}


def balanced_agreement(signal, reference, mask):
    """Mean agreement over frames where the reference is H (+1) and A (-1). NaN without both."""
    usable = mask & ~np.isnan(signal) & (reference != 0)
    home, away = usable & (reference > 0), usable & (reference < 0)
    if not home.any() or not away.any():
        return float("nan")
    return ((signal[home] > 0).mean() + (signal[away] < 0).mean()) / 2.0


def strongest_third(signal, mask):
    usable = mask & ~np.isnan(signal)
    if not usable.any():
        return usable
    with np.errstate(invalid="ignore"):
        return usable & (np.abs(signal) >= np.percentile(np.abs(signal[usable]), 200.0 / 3.0))


def ball_reference(features):
    """+1 / -1 where one team is clearly on the ball, else 0."""
    near = np.minimum(features["near_home"], features["near_away"])
    margin = features["near_away"] - features["near_home"]
    with np.errstate(invalid="ignore"):
        clear = (near < CONFIDENT_DISTANCE) & (np.abs(margin) >= CONFIDENT_MARGIN)
        return np.where(clear, np.sign(margin), 0.0)


def tag_reference(events, n_frames):
    """+1 / -1 where the phase tags agree on a team, else 0."""
    vote = np.zeros(n_frames)
    for event in events:
        team = TAG_TEAMS.get((event["tags"].get("Team") or "").strip().lower())
        code = (event["code"] or "").strip().lower()
        if team is None or code not in ATTACKING_CODES | DEFENDING_CODES:
            continue
        side = 1.0 if team == H else -1.0
        vote[event["start_frame"]:event["end_frame"]] += side if code in ATTACKING_CODES else -side
    return np.sign(vote)


def live_mask(features):
    """Frames where the players are moving like live play, using the same split as the pipeline."""
    activity = smooth_ignoring_nan(features["activity"], 50)
    known = ~np.isnan(activity)
    if not known.any():
        return known
    low, high = two_levels(activity[known])
    with np.errstate(invalid="ignore"):
        return activity > (low + high) / 2.0


def combined(found, ball, mask):
    """All signals scaled and added, each one's sign taken from how it agrees with the ball."""
    total = np.zeros(len(ball))
    for signal in found.values():
        agreement = balanced_agreement(signal, ball, mask)
        scale = np.nanstd(signal[mask])
        if agreement == agreement and scale > 0:
            total += np.nan_to_num(signal) / scale * (1.0 if agreement >= 0.5 else -1.0)
    return total


def report(features, events, blocks):
    n_frames = len(features["home_x"])
    graded = graded_mask(n_frames, blocks)
    live = live_mask(features)
    hidden = graded & live & np.isnan(features["ball_x"])

    ball = ball_reference(features)
    tags = tag_reference(events, n_frames)
    ball_mask = graded & live & (ball != 0)
    tag_mask = hidden & (tags != 0)

    halves = [graded_mask(n_frames, [block]) for block in blocks]
    print("hidden live frames {}, with ball reference {} (H share {:.1%}), with tag reference {} "
          "(H share {:.1%})".format(
              int(hidden.sum()), int(ball_mask.sum()),
              (ball[ball_mask] > 0).mean() if ball_mask.any() else 0.0,
              int(tag_mask.sum()),
              (tags[tag_mask] > 0).mean() if tag_mask.any() else 0.0))
    print("{:<10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "signal", "ball_all", "ball_top", "half_1", "half_2", "tags_all", "tags_top"))

    found = signals(features, blocks)
    found["combined"] = combined(found, ball, ball_mask)
    for name, signal in found.items():
        print("{:<10} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f}".format(
            name,
            balanced_agreement(signal, ball, ball_mask),
            balanced_agreement(signal, ball, strongest_third(signal, ball_mask)),
            balanced_agreement(signal, ball, ball_mask & halves[0]),
            balanced_agreement(signal, ball, ball_mask & halves[1]),
            balanced_agreement(signal, tags, tag_mask),
            balanced_agreement(signal, tags, strongest_third(signal, tag_mask))))


if __name__ == '__main__':
    report(build_features(), load_tags(GAME), half_video_blocks(GAME))
