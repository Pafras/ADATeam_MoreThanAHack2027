"""Why does touch detection find what it finds? Prints aggregates only, no frame-level data.

    python diagnose_touches.py
"""

import numpy as np

import possession
from possession import BALL_MAX_SPEED, FPS, GAME, build_features, detect_touches, graded_mask, \
    half_video_blocks, source_frames

PERCENTILES = (10, 25, 50, 75, 90, 99)
CHANGES = (2.0, 3.0, 4.0, 6.0)
RANGES = (2.5, 4.0, 6.0, 10.0)
LAGS = (3, 5, 8)


def line(name, values):
    values = values[~np.isnan(values)]
    if not len(values):
        print("{:<28} no values".format(name))
        return
    print("{:<28} n {:>7}  ".format(name, len(values)) + "  ".join(
        "p{} {:.2f}".format(p, v) for p, v in zip(PERCENTILES, np.percentile(values, PERCENTILES))))


def velocity_change(ball_x, ball_y, lag):
    velocity_x = (ball_x[lag:] - ball_x[:-lag]) * FPS / lag
    velocity_y = (ball_y[lag:] - ball_y[:-lag]) * FPS / lag
    with np.errstate(invalid="ignore"):
        velocity_x[np.hypot(velocity_x, velocity_y) > BALL_MAX_SPEED] = np.nan
    change = np.full(len(ball_x), np.nan)
    change[lag:-lag] = np.hypot(velocity_x[lag:] - velocity_x[:-lag],
                                velocity_y[lag:] - velocity_y[:-lag])
    return change


def diagnose(features, graded):
    ball_x, ball_y = features["ball_x"], features["ball_y"]
    visible = ~np.isnan(ball_x)
    print("graded frames {}, ball visible {:.1%}".format(int(graded.sum()), visible[graded].mean()))

    edges = np.diff(np.concatenate(([0], (visible & graded).astype(np.int8), [0])))
    runs = (np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).astype(float)
    line("visible run length (frames)", runs)

    near = np.minimum(features["near_home"], features["near_away"])
    margin = np.abs(features["near_away"] - features["near_home"])
    line("nearest player (m)", near[graded])
    line("team margin (m)", margin[graded])

    speed = np.hypot(ball_x[5:] - ball_x[:-5], ball_y[5:] - ball_y[:-5]) * FPS / 5
    line("ball speed, 5 frames (m/s)", speed[graded[5:]])
    with np.errstate(invalid="ignore"):
        print("speed above {:g} m/s: {:.2%} of measured frames".format(
            BALL_MAX_SPEED, (speed[graded[5:]] > BALL_MAX_SPEED).sum()
            / float(max((~np.isnan(speed[graded[5:]])).sum(), 1))))

    for lag in LAGS:
        line("velocity change, lag {} (m/s)".format(lag),
             velocity_change(ball_x, ball_y, lag)[graded])

    change = velocity_change(ball_x, ball_y, possession.TOUCH_LAG)
    with np.errstate(invalid="ignore"):
        sharp = graded & (change > possession.TOUCH_CHANGE)
        close = sharp & (near < possession.TOUCH_RANGE)
        clear = close & (margin >= possession.TOUCH_MARGIN)
    print("frames passing, current settings: measurable {}, sharp {}, close {}, clear {}".format(
        int((graded & ~np.isnan(change)).sum()), int(sharp.sum()), int(close.sum()),
        int(clear.sum())))

    print("\ntouches found, by lag / velocity change / range (margin {:g} m):".format(
        possession.TOUCH_MARGIN))
    print("{:>4} {:>7} ".format("lag", "change") + " ".join(
        "{:>7}".format("{:g} m".format(r)) for r in RANGES))
    defaults = possession.TOUCH_LAG, possession.TOUCH_CHANGE, possession.TOUCH_RANGE
    for lag in LAGS:
        for limit in CHANGES:
            counts = []
            for reach in RANGES:
                possession.TOUCH_LAG, possession.TOUCH_CHANGE, possession.TOUCH_RANGE = \
                    lag, limit, reach
                counts.append(len(detect_touches(features)[0]))
            print("{:>4} {:>7g} ".format(lag, limit) + " ".join(
                "{:>7}".format(count) for count in counts))
    possession.TOUCH_LAG, possession.TOUCH_CHANGE, possession.TOUCH_RANGE = defaults


if __name__ == '__main__':
    diagnose(build_features(),
             graded_mask(source_frames("tactical", GAME), half_video_blocks(GAME)))
