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
from possession import FPS, WEIGHT_TOUCH, build, detect_touches, tag_scores, tracking_scores, \
    two_levels, viterbi, write_submission  # noqa: E402
from validate_tags import handover_recall, phase_agreement, restart_agreement  # noqa: E402

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


def touch_features(near_home, near_away):
    """A ball moving at 10 m/s along x that reverses at frame 100."""
    ball_x = np.concatenate((np.arange(101) * 0.4, 40.0 - np.arange(1, 100) * 0.4))
    n_frames = len(ball_x)
    return {"ball_x": ball_x, "ball_y": np.full(n_frames, 34.0),
            "near_home": np.full(n_frames, near_home), "near_away": np.full(n_frames, near_away)}


def test_a_touch_is_a_sharp_change_next_to_a_clearly_nearer_team():
    frames, teams = detect_touches(touch_features(8.0, 1.0))
    assert list(frames) == [100] and list(teams) == [1], (frames, teams)
    assert len(detect_touches(touch_features(1.0, 8.0))[0]) == 1
    assert list(detect_touches(touch_features(1.0, 8.0))[1]) == [0]
    assert len(detect_touches(touch_features(5.0, 8.0))[0]) == 0  # nobody close enough
    assert len(detect_touches(touch_features(1.0, 1.5))[0]) == 0  # too close to call

    straight = touch_features(8.0, 1.0)
    straight["ball_x"] = np.arange(len(straight["ball_x"])) * 0.4
    assert len(detect_touches(straight)[0]) == 0


def test_three_touches_take_possession_and_two_do_not():
    def path_with_touches(count):
        scores = np.zeros((1000, 3))
        scores[0, 0] = scores[-1, 0] = 50.0  # H at both ends
        for frame in range(400, 400 + 25 * count, 25):
            scores[frame, 0] -= WEIGHT_TOUCH
            scores[frame, 1] += WEIGHT_TOUCH
        return viterbi(scores)

    assert (path_with_touches(2) == 0).all()
    assert 1 in path_with_touches(3)


def test_handover_recall_needs_the_change_to_go_to_the_right_team():
    labels = np.array(["H"] * 500 + ["A"] * 500)
    recovered = [{"code": "ATTACKING TRANSITION", "start_frame": 510, "end_frame": 600,
                  "tags": {"Team": "Atletic Club"}}]
    lost = [{"code": "DEFENSIVE TRANSITION", "start_frame": 510, "end_frame": 600,
             "tags": {"Team": "Barcelona"}}]
    wrong = [{"code": "ATTACKING TRANSITION", "start_frame": 510, "end_frame": 600,
              "tags": {"Team": "Barcelona"}}]
    far = [{"code": "ATTACKING TRANSITION", "start_frame": 900, "end_frame": 950,
            "tags": {"Team": "Atletic Club"}}]
    assert handover_recall(labels, recovered) == 1.0
    assert handover_recall(labels, lost) == 1.0
    assert handover_recall(labels, wrong) == 0.0
    assert handover_recall(labels, far) == 0.0


def test_slow_players_need_the_ball_to_agree_before_it_counts_as_dead():
    n_frames = 3000
    moving_ball = np.arange(n_frames) * 0.2  # 5 m/s, too slow for "in flight", not "still" either
    features = {"ball_x": moving_ball % 100.0, "ball_y": np.full(n_frames, 34.0),
                "near_home": np.full(n_frames, 1.0), "near_away": np.full(n_frames, 8.0),
                "activity": np.full(n_frames, 2.5)}
    features["activity"][1000:2000] = 0.5  # the whole team walks, the ball keeps moving
    graded = np.ones(n_frames, dtype=bool)
    pitch = GAMES[GAME]["pitch"]

    walking = tracking_scores(features, graded, pitch)
    assert walking[1200:1800, 2].max() < 0.01, walking[1200:1800, 2].max()

    features["ball_x"] = features["ball_x"].copy()
    features["ball_x"][1000:2000] = 50.0  # the same walking, with the ball standing still
    waiting = tracking_scores(features, graded, pitch)
    assert waiting[1200:1800, 2].min() > 0.25, waiting[1200:1800, 2].min()
    assert waiting[100:900, 2].max() <= 0.0


def test_only_kickoffs_get_dead_evidence_from_a_set_piece():
    throw_in = {"code": "SET PIECES", "start_frame": 100, "end_frame": 200,
                "tags": {"Type": "Throw-in"}}
    kickoff = {"code": "SET PIECES", "start_frame": 100, "end_frame": 200,
               "tags": {"Type": "Kick Off (After Goal)"}}
    assert tag_scores([throw_in], 300, BLOCKS)[:, 2].max() == 0.0
    assert tag_scores([kickoff], 300, BLOCKS)[120, 2] > 0.0


def test_tag_agreement_counts_the_right_team_and_the_restart():
    labels = np.array(["D"] * 100 + ["H"] * 100 + ["A"] * 100)
    events = [{"code": "BUILD UP", "start_frame": 100, "end_frame": 200,
               "tags": {"Team": "Barcelona"}},
              {"code": "DEFENDING IN MIDDLE THIRD", "start_frame": 200, "end_frame": 300,
               "tags": {"Team": "Barcelona"}},
              {"code": "PROGRESSION", "start_frame": 200, "end_frame": 300,
               "tags": {"Team": "Barcelona"}},
              {"code": "SET PIECES", "start_frame": 95, "end_frame": 105,
               "tags": {"Team": "Barcelona"}}]
    phase, phase_dead = phase_agreement(labels, events)
    assert abs(phase - 2 / 3.0) < 1e-9 and phase_dead == 0.0
    assert restart_agreement(labels, events) == (1.0, 1.0, 1.0)
    events[-1]["tags"]["Team"] = "Atletic Club"
    assert restart_agreement(labels, events)[2] == 0.0


if __name__ == '__main__':
    test_a_touch_is_a_sharp_change_next_to_a_clearly_nearer_team()
    test_three_touches_take_possession_and_two_do_not()
    test_handover_recall_needs_the_change_to_go_to_the_right_team()
    test_slow_players_need_the_ball_to_agree_before_it_counts_as_dead()
    test_only_kickoffs_get_dead_evidence_from_a_set_piece()
    test_tag_agreement_counts_the_right_team_and_the_restart()
    test_two_levels_finds_stopped_play_whatever_its_share()
    test_a_three_second_hold_changes_possession()
    test_viterbi_ignores_a_short_touch_and_keeps_a_real_change()
    test_defending_tag_points_at_the_other_team()
    test_end_to_end()
    print("ok")
