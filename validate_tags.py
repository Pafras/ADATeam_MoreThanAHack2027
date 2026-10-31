"""Compare pipeline settings over the whole match, using Smart Tagging as an independent reference.

    python validate_tags.py

The public five minutes may not be used to choose between versions (RULES.md). This uses no human
labels instead: the pipeline runs on tracking alone, and its output is compared with the automatic
tags it did not see. Every number printed is an aggregate, so no frame-level data leaves the script.

Smart Tagging has false positives and false negatives, so this is a reference and not an answer key.
The baseline rows are the check on that: their true full-match spellF1 is published, so they show
whether these numbers move with the real score at all.

Columns:
  poss, dur     possessions and their mean length in seconds
  dead          share of graded frames labelled D
  phase         share of tagged phases whose alive frames mostly match the tagged team's possession
  phase_dead    share of frames inside tagged phases labelled D; phases are live play
  rs_recall     share of tagged set pieces with a predicted restart nearby
  rs_prec       share of predicted restarts with a tagged set piece nearby; tags are known to miss
                some restarts, so this reads low even for a perfect system
  rs_team       share of matched restarts handed to the tagged team
"""

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession  # noqa: E402
from possession import ATTACKING_CODES, DEFENDING_CODES, FPS, GAME, H, A, TAG_TEAMS, \
    graded_mask, half_video_blocks, label_halves, load_tags, build_tracking_scores, \
    tag_scores  # noqa: E402
from possession_metrics import read_label_csv  # noqa: E402

# Params
RESTART_TOLERANCE = 5 * FPS  # frames either side of a tagged set piece; the tags are that coarse
SWITCH_TEAM_VALUES = (4.0, 6.0, 8.0, 12.0, 15.0)
SWITCH_DEAD_VALUES = (5.0, 10.0, 20.0)
PUBLISHED_SPELL_F1 = {"baseline_raw": 0.0486, "baseline_smooth25": 0.1996,
                      "baseline_smooth50": 0.2433, "baseline_smooth125": 0.2242}
TEAM_LETTERS = {H: "H", A: "A"}


def tag_team(event):
    return TAG_TEAMS.get((event["tags"].get("Team") or "").strip().lower())


def tag_code(event):
    return (event["code"] or "").strip().lower()


def phase_agreement(labels, events):
    """(share of phases agreeing, share of frames inside phases that are D)."""
    hits, phases, dead, frames = 0, 0, 0, 0
    for event in events:
        team, code = tag_team(event), tag_code(event)
        if team is None or code not in ATTACKING_CODES | DEFENDING_CODES:
            continue
        expected = team if code in ATTACKING_CODES else 1 - team
        window = labels[event["start_frame"]:event["end_frame"]]
        window = window[window != "X"]
        dead += int((window == "D").sum())
        frames += len(window)
        wanted = int((window == TEAM_LETTERS[expected]).sum())
        other = int((window == TEAM_LETTERS[1 - expected]).sum())
        if wanted + other:
            phases += 1
            hits += wanted > other
    return hits / float(max(phases, 1)), dead / float(max(frames, 1))


def restart_agreement(labels, events):
    """(recall, precision, team share) of predicted restarts against tagged set pieces."""
    restarts = 1 + np.flatnonzero((labels[:-1] == "D") & np.isin(labels[1:], ["H", "A"]))
    pieces = [event for event in events if tag_code(event) == "set pieces"
              and labels[min(event["start_frame"], len(labels) - 1)] != "X"]

    matched_restarts = np.zeros(len(restarts), dtype=bool)
    found, team_hits, team_total = 0, 0, 0
    for event in pieces:
        near = (restarts >= event["start_frame"] - RESTART_TOLERANCE) \
            & (restarts <= event["end_frame"] + RESTART_TOLERANCE)
        matched_restarts |= near
        if not near.any():
            continue
        found += 1
        team = tag_team(event)
        if team is not None:
            team_total += 1
            team_hits += labels[restarts[near][0]] == TEAM_LETTERS[team]
    return (found / float(max(len(pieces), 1)),
            matched_restarts.mean() if len(restarts) else 0.0,
            team_hits / float(max(team_total, 1)))


def shape(labels, blocks):
    """(possessions, mean seconds, share of graded frames that are D)."""
    count, frames = 0, 0
    for _, first, last in blocks:
        window = labels[first:last + 1]
        starts = np.flatnonzero(np.concatenate(([True], window[1:] != window[:-1])))
        count += int(np.isin(window[starts], ["H", "A"]).sum())
        frames += int(np.isin(window, ["H", "A"]).sum())
    graded = labels[graded_mask(len(labels), blocks)]
    return count, frames / float(max(count, 1)) / FPS, (graded == "D").mean()


def row(name, labels, events, blocks):
    poss, dur, dead = shape(labels, blocks)
    phase, phase_dead = phase_agreement(labels, events)
    recall, precision, team = restart_agreement(labels, events)
    return "{:<36} {:>5} {:>6.1f} {:>6.3f} {:>6.3f} {:>10.3f} {:>9.3f} {:>7.3f} {:>7.3f}".format(
        name, poss, dur, dead, phase, phase_dead, recall, precision, team)


def report(tracking, events, blocks, baseline_paths=()):
    """Every printable row: baselines, then the settings grid on tracking alone, then with tags."""
    print("{:<36} {:>5} {:>6} {:>6} {:>6} {:>10} {:>9} {:>7} {:>7}".format(
        "version", "poss", "dur", "dead", "phase", "phase_dead", "rs_recall", "rs_prec", "rs_team"))

    graded = graded_mask(len(tracking), blocks)
    for path in baseline_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        labels = read_label_csv(path)["label"].values.astype("<U1")
        labels = np.where(graded, labels, "X")
        print(row("{} (spellF1 {:.4f})".format(name, PUBLISHED_SPELL_F1[name]), labels, events,
                  blocks))

    defaults = possession.SWITCH_TEAM, possession.SWITCH_DEAD
    for team_cost in SWITCH_TEAM_VALUES:
        for dead_cost in SWITCH_DEAD_VALUES:
            possession.SWITCH_TEAM, possession.SWITCH_DEAD = team_cost, dead_cost
            print(row("tracking team={:g} dead={:g}".format(team_cost, dead_cost),
                      label_halves(tracking, blocks), events, blocks))
    possession.SWITCH_TEAM, possession.SWITCH_DEAD = defaults

    # Biased by construction: these tags are also inputs here. Shown only to see how far they pull.
    combined = tracking + tag_scores(events, len(tracking), blocks)
    print(row("current with tags (biased)", label_halves(combined, blocks), events, blocks))


if __name__ == '__main__':
    report(build_tracking_scores(), load_tags(GAME), half_video_blocks(GAME),
           sorted(glob.glob(os.path.join(ROOT, "evaluation", "results", "baseline_*.csv"))))
