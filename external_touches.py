"""Choose the touch-detection thresholds on external data: Metrica's public sample games.

    python external_touches.py

The competition rules require any tuned parameter to be tuned on external data. This script does
that for TOUCH_LAG, TOUCH_CHANGE and TOUCH_RANGE in possession.py, using Sample Games 1 and 2 from
https://github.com/metrica-sports/sample-data (pinned to one commit below). Those games come with
tracking and events on the same frame clock, so every pass, reception, recovery, shot, ball loss
and set piece marks a real touch by a known team.

It reads nothing from the competition match. Sample Game 3 is not used: its tracking does not say
which team each player slot belongs to.

The matching rule was fixed before looking at any result: a detected touch counts when an event
touch by the same team lies within MATCH_TOLERANCE frames, each event matched once. Dribbling
touches are not events, so precision reads low for every setting alike.
"""

import csv
import os
import urllib.request
import warnings

import numpy as np

import possession
from possession import A, H, detect_touches

# Params
SOURCE = ("https://raw.githubusercontent.com/metrica-sports/sample-data/"
          "e706dd506b360d69d9d123d5b8026e7294b13996/data/Sample_Game_{game}/{name}")
FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external", "metrica")
GAMES = (1, 2)
PITCH = (105.0, 68.0)  # the sample data README gives this size for both games
MATCH_TOLERANCE = 10  # frames, 0.4 s at 25 fps
TOUCH_EVENTS = {"PASS", "SHOT", "RECOVERY", "BALL LOST", "SET PIECE"}
DUPLICATE_GAP = 3  # frames; a set piece and its pass are often logged as two events on one touch
# The first run picked the smallest lag and change on offer, so both ranges were widened downwards.
LAGS = (1, 2, 3, 5, 8, 12)
CHANGES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
RANGES = (2.5, 4.0, 6.0)


def fetch(game):
    """Local paths of one game's files, downloading any that are missing."""
    names = {"events": "Sample_Game_{}_RawEventsData.csv".format(game),
             "home": "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(game),
             "away": "Sample_Game_{}_RawTrackingData_Away_Team.csv".format(game)}
    if not os.path.isdir(FOLDER):
        os.makedirs(FOLDER)
    paths = {}
    for key, name in names.items():
        path = os.path.join(FOLDER, name)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(SOURCE.format(game=game, name=name), path)
        paths[key] = path
    return paths


def read_team(path):
    """(players, ball) as metres: players is frames x players x 2, ball is frames x 2."""
    with open(path) as handle:
        rows = list(csv.reader(handle))[3:]  # three header lines
    values = np.array([[float(v) if v else np.nan for v in row[3:]] for row in rows])
    scale = np.array(PITCH)
    players = values[:, :-2].reshape(len(values), -1, 2) * scale
    return players, values[:, -2:] * scale


def features(paths):
    """The four features detect_touches needs, built the way possession.read_features builds them."""
    home, ball = read_team(paths["home"])
    away, _ = read_team(paths["away"])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with no players tracked give NaN
        near_home = np.nanmin(np.hypot(*(home - ball[:, None, :]).transpose(2, 0, 1)), axis=1)
        near_away = np.nanmin(np.hypot(*(away - ball[:, None, :]).transpose(2, 0, 1)), axis=1)
    return {"ball_x": ball[:, 0], "ball_y": ball[:, 1], "near_home": near_home,
            "near_away": near_away}


def event_touches(path):
    """(frames, teams) of touches named by the events, frames counted from 0 like the tracking."""
    touches = []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            if row["Type"] not in TOUCH_EVENTS:
                continue
            team = H if row["Team"] == "Home" else A
            touches.append((int(row["Start Frame"]) - 1, team))
            if row["Type"] == "PASS" and row["To"]:
                touches.append((int(row["End Frame"]) - 1, team))

    kept = []
    for frame, team in sorted(touches):
        if kept and kept[-1][1] == team and frame - kept[-1][0] <= DUPLICATE_GAP:
            continue
        kept.append((frame, team))
    return np.array([t[0] for t in kept]), np.array([t[1] for t in kept])


def match(found, truth):
    """(true positives, detected, events): each event matched at most once, same team only."""
    truth_frames, truth_teams = truth
    used = np.zeros(len(truth_frames), dtype=bool)
    hits = 0
    for frame, team in zip(*found):
        candidates = np.flatnonzero(~used & (truth_teams == team)
                                    & (np.abs(truth_frames - frame) <= MATCH_TOLERANCE))
        if len(candidates):
            used[candidates[np.argmin(np.abs(truth_frames[candidates] - frame))]] = True
            hits += 1
    return hits, len(found[0]), len(truth_frames)


def f1(hits, detected, events):
    if not hits:
        return 0.0
    precision, recall = hits / float(detected), hits / float(events)
    return 2 * precision * recall / (precision + recall)


def grid(games):
    """(lag, change, reach) -> {game: (hits, detected, events)}."""
    defaults = possession.TOUCH_LAG, possession.TOUCH_CHANGE, possession.TOUCH_RANGE
    results = {}
    for lag in LAGS:
        for change in CHANGES:
            for reach in RANGES:
                possession.TOUCH_LAG, possession.TOUCH_CHANGE, possession.TOUCH_RANGE = \
                    lag, change, reach
                results[(lag, change, reach)] = {
                    game: match(detect_touches(feats), truth)
                    for game, (feats, truth) in games.items()}
    possession.TOUCH_LAG, possession.TOUCH_CHANGE, possession.TOUCH_RANGE = defaults
    return results


def pooled_f1(counts, games):
    return f1(*np.sum([counts[game] for game in games], axis=0))


def report(results, games):
    """Best setting on all games, plus the leave-one-game-out check of how stable that choice is."""
    names = sorted(games)
    print("{:>4} {:>7} {:>6} ".format("lag", "change", "range") + " ".join(
        "{:>12}".format("game {} F1".format(g)) for g in names) + " {:>9}".format("pooled"))
    ranked = sorted(results, key=lambda key: -pooled_f1(results[key], names))
    for key in ranked[:10]:
        print("{:>4} {:>7g} {:>6g} ".format(*key) + " ".join(
            "{:>12.3f}".format(f1(*results[key][g])) for g in names)
            + " {:>9.3f}".format(pooled_f1(results[key], names)))

    print("\nleave one game out: choose on the others, score on the one left out")
    for held in names:
        others = [g for g in names if g != held]
        chosen = max(results, key=lambda key: pooled_f1(results[key], others))
        print("  held out game {}: chose lag {} change {:g} range {:g}, F1 there {:.3f}".format(
            held, chosen[0], chosen[1], chosen[2], f1(*results[chosen][held])))

    best = ranked[0]
    print("\nchosen on all games: TOUCH_LAG = {}, TOUCH_CHANGE = {:g}, TOUCH_RANGE = {:g}".format(
        *best))
    return best


if __name__ == '__main__':
    loaded = {}
    for game in GAMES:
        paths = fetch(game)
        loaded[game] = (features(paths), event_touches(paths["events"]))
        print("game {}: {} event touches".format(game, len(loaded[game][1][0])))
    report(grid(loaded), loaded)
