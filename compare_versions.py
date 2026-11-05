"""Does v9 (touches, thresholds from external data) replace v3 (no touches)? Prints a verdict.

    python compare_versions.py

Both versions are the current possession.py: v3 is the same pipeline with touches switched off.
They are compared with automatic checks against Smart Tagging over the full match, the kind of
check the rules allow. The ground truth and the video are not used.

The criteria below were written before either version was run on the match.

Keep the output to yourself: the numbers come from the competition data, which may not be shared
with AI services.
"""

import numpy as np

import possession
from possession import GAME, build_features, detect_touches, half_video_blocks, label_halves, \
    load_tags, scores_from_features, tag_scores
from validate_tags import handover_recall, restart_agreement, shape

# Params
TOUCHES_MIN, TOUCHES_MAX = 600, 2400  # passes and receptions on the ~59% of frames with a ball
HANDOVER_GAIN = 0.02  # about one more tagged transition found
POSSESSIONS_MAX = 300
DEAD_SHIFT = 0.02
RESTART_DROP = 0.03
TEAM_DROP = 0.03


def measure(labels, events, blocks):
    poss, _, dead = shape(labels, blocks)
    recall, _, team = restart_agreement(labels, events)
    return {"poss": poss, "dead": dead, "rs_recall": recall, "rs_team": team,
            "ho_recall": handover_recall(labels, events)}


def verdict(touches, old, new):
    """[(passed, Indonesian description)] for each criterion."""
    return [
        (TOUCHES_MIN <= touches <= TOUCHES_MAX,
         "Jumlah sentuhan wajar ({} sampai {})".format(TOUCHES_MIN, TOUCHES_MAX)),
        (new["ho_recall"] >= old["ho_recall"] + HANDOVER_GAIN,
         "Pergantian possession lebih banyak tertangkap (ho_recall naik minimal {:g})".format(
             HANDOVER_GAIN)),
        (new["poss"] <= POSSESSIONS_MAX,
         "Jumlah possession tidak meledak (maksimal {})".format(POSSESSIONS_MAX)),
        (abs(new["dead"] - old["dead"]) < DEAD_SHIFT,
         "Porsi bola mati bergeser kurang dari {:g} poin".format(DEAD_SHIFT * 100)),
        (new["rs_recall"] >= old["rs_recall"] - RESTART_DROP,
         "Restart tidak banyak hilang (rs_recall turun kurang dari {:g})".format(RESTART_DROP)),
        (new["rs_team"] >= old["rs_team"] - TEAM_DROP,
         "Tim setelah restart tetap benar (rs_team turun kurang dari {:g})".format(TEAM_DROP)),
    ]


def main():
    features = build_features()
    events = load_tags(GAME)
    blocks = half_video_blocks(GAME)
    tags = tag_scores(events, len(features["ball_x"]), blocks)
    touches = len(detect_touches(features)[0])

    new = measure(label_halves(scores_from_features(features) + tags, blocks), events, blocks)
    weight, possession.WEIGHT_TOUCH = possession.WEIGHT_TOUCH, 0.0
    old = measure(label_halves(scores_from_features(features) + tags, blocks), events, blocks)
    possession.WEIGHT_TOUCH = weight

    print("{:<12} {:>8} {:>8}".format("ukuran", "v3", "v9"))
    for key in ("poss", "dead", "rs_recall", "rs_team", "ho_recall"):
        print("{:<12} {:>8.3f} {:>8.3f}".format(key, old[key], new[key]))
    print("{:<12} {:>8} {:>8}\n".format("touches", "-", touches))

    results = verdict(touches, old, new)
    for passed, description in results:
        print("[{}] {}".format("LOLOS" if passed else "GAGAL", description))
    if all(passed for passed, _ in results):
        print("\nKesimpulan: v9 LOLOS semua kriteria dan menggantikan v3.")
    else:
        print("\nKesimpulan: v9 TIDAK lolos, v3 tetap versi yang dikirim.")


if __name__ == '__main__':
    np.seterr(invalid="ignore")
    main()
