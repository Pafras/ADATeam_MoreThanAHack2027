"""End-to-end check of possession.py on a synthetic match. No real data is read or needed.

    python test_possession.py

Every 150 s block is 60 s of H, 60 s of A and 30 s of dead ball. Inside the H stretch an away
player touches the ball for 0.4 s and the ball vanishes for 10 s; neither should change anything.
The second half opens with a stoppage before a tagged kickoff, like the real match does.
"""

import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from games import GAME, GAMES, half_video_blocks, source_frames  # noqa: E402
from possession_metrics import label_codes, read_label_csv, spell_scores, valid_runs, \
    validate_submission  # noqa: E402
from possession import FPS, build, tag_scores, two_levels, viterbi, write_submission  # noqa: E402

N_FRAMES = source_frames("tactical", GAME)
BLOCKS = half_video_blocks(GAME)
SECOND_KICKOFF = 113742

META = """<Root><Teams>
<Team id="h"><Name>Barcelona</Name></Team><Team id="a"><Name>Athletic</Name></Team>
</Teams><Players>
<Player id="1" teamId="h"/><Player id="2" teamId="h"/>
<Player id="3" teamId="a"/><Player id="4" teamId="a"/>
</Players></Root>"""

TAGS = """<file><ALL_INSTANCES><instance><start>{0:.2f}</start><end>{1:.2f}</end>
<code>SET PIECES</code>
<label><group>Type</group><text>Kick Off (Start)</text></label>
<label><group>Half</group><text>2nd Half</text></label>
</instance></ALL_INSTANCES></file>""".format((SECOND_KICKOFF - 60) / 25.0,
                                             (SECOND_KICKOFF + 60) / 25.0)


def truth_and_events():
    """The true label per frame and, per frame, what the tracking should show."""
    truth = np.full(N_FRAMES, "X", dtype="<U1")
    event = np.zeros(N_FRAMES, dtype=np.int8)  # 1 = away touch, 2 = ball missing
    for half, first, last in BLOCKS:
        start = first if half == 1 else SECOND_KICKOFF
        truth[first:last + 1] = "D"
        block = 150 * FPS
        for at in range(start, last + 1, block):
            truth[at:min(at + 60 * FPS, last + 1)] = "H"
            truth[at + 60 * FPS:min(at + 120 * FPS, last + 1)] = "A"
            event[at + 20 * FPS:at + 20 * FPS + 10] = 1
            event[at + 30 * FPS:at + 40 * FPS] = 2
    return truth, event[:N_FRAMES]


def write_match(folder, truth, event):
    tactical = os.path.join(folder, "tactical")
    os.makedirs(tactical)
    stem = os.path.join(tactical, "BAR-ATH_tactical")
    with open(stem + "_atd.xml", "w") as handle:
        handle.write(META)
    with open(stem + "_smart_tagging.xml", "w") as handle:
        handle.write(TAGS)

    with open(stem + "_atd.txt", "w") as handle:
        for frame in range(N_FRAMES):
            label = truth[frame]
            if label == "X":
                handle.write("{}:{}:NaN,NaN,NaN\n".format(frame + 1, ";".join(["NaN,NaN,NaN"] * 4)))
                continue
            # A deterministic wander, so the ball is never exactly on anyone.
            x = 0.5 + 0.2 * np.sin(frame / 200.0)
            carrier = 0.002
            speed = 0.5 if label == "D" else 2.5
            home_gap, away_gap = (carrier, 0.1) if label == "H" else (0.1, carrier)
            if event[frame] == 1:
                home_gap, away_gap = 0.1, carrier
            players = ";".join("{:.4f},{:.4f},{}".format(x + gap, 0.5, speed) for gap in
                               (home_gap, 0.3, away_gap, -0.3))
            hidden = label == "D" or event[frame] == 2
            ball = "NaN,NaN,NaN" if hidden else "{:.4f},0.5,1.0".format(x)
            handle.write("{}:{}:{}\n".format(frame + 1, players, ball))


def test_end_to_end():
    truth, event = truth_and_events()
    with tempfile.TemporaryDirectory() as folder:
        write_match(folder, truth, event)
        labels = build(GAME, folder)
        path = write_submission(os.path.join(folder, "out.csv"), labels)
        table = read_label_csv(path)

    assert validate_submission(table, GAMES[GAME]["n_video_frames"]) is None
    graded = truth != "X"
    assert (labels[~graded] == "X").all()
    accuracy = (labels[graded] == truth[graded]).mean()
    assert accuracy > 0.98, accuracy
    assert (labels[110808:SECOND_KICKOFF - 60] == "D").all()

    gt_codes, pred_codes = label_codes(truth), label_codes(labels)
    spell = spell_scores(gt_codes, pred_codes, valid_runs(gt_codes))
    assert spell["n_pred"] == spell["n_gt"], (spell["n_pred"], spell["n_gt"])
    assert spell["mean_f1"] > 0.95, spell["mean_f1"]


def test_viterbi_ignores_a_short_touch_and_keeps_a_real_change():
    scores = np.zeros((1000, 3))
    scores[:, 0] = 0.3
    scores[500:510, 1] = 0.8
    assert (viterbi(scores) == 0).all()
    scores[500:, 0] = 0.0
    scores[500:, 1] = 0.3
    path = viterbi(scores)
    assert path[0] == 0 and path[-1] == 1


def test_defending_tag_points_at_the_other_team():
    events = [{"code": "DEFENDING IN MIDDLE THIRD", "start_frame": 10, "end_frame": 20,
               "tags": {"Team": "Barcelona"}}]
    scores = tag_scores(events, 30, BLOCKS)
    assert scores[15, 1] > 0 and scores[15, 0] == 0


def test_two_levels_finds_stopped_play_whatever_its_share():
    for stopped_share in (0.05, 0.2, 0.5):
        n_stopped = int(10000 * stopped_share)
        values = np.linspace(-0.1, 0.1, 10000)
        values[:n_stopped] += 1.0
        values[n_stopped:] += 2.5
        low, high = two_levels(values)
        assert abs(low - 1.0) < 0.1 and abs(high - 2.5) < 0.1, (stopped_share, low, high)


def test_a_three_second_hold_changes_possession():
    # The case SWITCH_TEAM is derived from: 8.5 of evidence for A across 3 s, in the middle of H.
    scores = np.zeros((1000, 3))
    scores[:, 0], scores[:, 1] = 0.1, -0.1
    scores[500:575, 0], scores[500:575, 1] = -8.5 / 75, 8.5 / 75
    path = viterbi(scores)
    assert (path[500:575] == 1).all() and path[0] == 0 and path[-1] == 0


if __name__ == '__main__':
    test_two_levels_finds_stopped_play_whatever_its_share()
    test_a_three_second_hold_changes_possession()
    test_viterbi_ignores_a_short_touch_and_keeps_a_real_change()
    test_defending_tag_points_at_the_other_team()
    test_end_to_end()
    print("ok")
