"""Parameter-point splits for the Platonic Representation Hypothesis study.

The Hidden Valley grid available in ``fisher_geometry/cache.pt`` is

    m_Phi in {125, 250, 300, 350, 400, 450, 500, 550}   (Lambda = 1, 2, 3, 4)
    m_Phi in {275, 325, 425, 475}                       (Lambda = 2 only)

The four intermediate-mass shells carry only 1600/200/200 events and are therefore
statistically much weaker than the eight complete shells.  They are kept in a separate
``LOWSTAT`` group that is used for nothing but a clearly flagged auxiliary check.

Splits are defined on whole parameter points (never on individual events):

``TRAIN``  17 points, all eight mass shells, Lambda = 1..4 with a balanced count
           (L1 x 6, L2 x 4, L3 x 3, L4 x 4).  Used to fit both encoders.
``VAL``     6 points, unseen by both encoders, Lambda = 1..4.  Hyper-parameter selection.
``TEST``    9 points, unseen by both encoders, Lambda = 1..4, seven distinct masses
           (so the test set is *not* dominated by mediator-mass variation).
           Two masses carry two Lambda values each, giving within-mass Lambda contrasts.
``LOWSTAT`` 4 points, intermediate masses never seen by any encoder.

Every event of a non-training point is unused by every encoder, so evaluation at
VAL/TEST/LOWSTAT points is out-of-sample in the strictest sense.  Evaluation at TRAIN
points uses the ``split == 2`` events, which are disjoint from the ``split == 0``
events used for training.
"""

# --- complete shells -------------------------------------------------------------------------
FULL_MASSES = (125, 250, 300, 350, 400, 450, 500, 550)
LAMBDAS = (1, 2, 3, 4)

# --- intermediate-mass shells (low statistics) ------------------------------------------------
LOW_MASSES = (275, 325, 425, 475)

ALL_POINTS = [(m, l) for m in FULL_MASSES for l in LAMBDAS] + [(m, 2) for m in LOW_MASSES]

TRAIN_POINTS = [
    (125, 1),
    (250, 1), (250, 3),
    (300, 1), (300, 4),
    (350, 1), (350, 2), (350, 3), (350, 4),
    (400, 1), (400, 4),
    (450, 1), (450, 2), (450, 4),
    (500, 2), (500, 4),
    (550, 2),
]

VAL_POINTS = [
    (125, 3),
    (250, 2),
    (300, 2),
    (400, 2),
    (500, 3),
    (550, 1),
]

TEST_POINTS = [
    (125, 2), (125, 4),
    (250, 4),
    (300, 3),
    (400, 3),
    (450, 3),
    (500, 1),
    (550, 3), (550, 4),
]

LOWSTAT_POINTS = [(275, 2), (325, 2), (425, 2), (475, 2)]

GROUPS = {
    "train": TRAIN_POINTS,
    "val": VAL_POINTS,
    "test": TEST_POINTS,
    "lowstat": LOWSTAT_POINTS,
}


def check_splits():
    """Validate the split definition; returns a dict of diagnostics."""
    seen = {}
    for name, pts in GROUPS.items():
        for p in pts:
            assert p not in seen, f"point {p} in both {seen.get(p)} and {name}"
            seen[p] = name
    assert set(seen) == set(ALL_POINTS), (
        f"split does not cover the grid: missing {sorted(set(ALL_POINTS) - set(seen))}"
    )
    out = {}
    for name, pts in GROUPS.items():
        masses = sorted({m for m, _ in pts})
        lams = sorted({l for _, l in pts})
        within_mass = {}
        for m, l in pts:
            within_mass.setdefault(m, []).append(l)
        out[name] = {
            "n": len(pts),
            "masses": masses,
            "lambdas": lams,
            "lambda_counts": {l: sum(1 for _, ll in pts if ll == l) for l in lams},
            "masses_with_multiple_lambda": {m: sorted(v) for m, v in within_mass.items() if len(v) > 1},
        }
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(check_splits(), indent=2))
