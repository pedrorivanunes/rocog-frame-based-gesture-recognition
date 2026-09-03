from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aggregation import (
    accuracy_by_class,
    accuracy_curve,
    aggregate_mean,
    aggregate_vote,
    parse_args,
    probability_cube,
    random_indices,
    segment_indices,
)


def probability_rows(frames_per_video: int = 4) -> pd.DataFrame:
    """Two videos of two classes, with made-up scores.

    Video A is confidently Halt on every frame; video B is confidently Rally.
    """
    rows = []
    for video, label in (("videoA", 0), ("videoB", 1)):
        for frame in range(frames_per_video):
            rows.append(
                {
                    "video_id": video,
                    "frame_number": frame,
                    "position": frame / frames_per_video,
                    "label": label,
                    "p_Halt": 0.9 if label == 0 else 0.1,
                    "p_Rally": 0.1 if label == 0 else 0.9,
                }
            )
    return pd.DataFrame(rows)


def test_probability_cube_stacks_frames_by_video():
    cube, labels, video_ids = probability_cube(probability_rows())

    assert cube.shape == (2, 4, 2)
    assert video_ids == ["videoA", "videoB"]
    assert labels.tolist() == [0, 1]
    assert cube[0, 0, 0] == pytest.approx(0.9)


def test_probability_cube_rejects_uneven_frame_counts():
    """Stacking anyway would silently shift one video's frames into another."""
    uneven = probability_rows().drop(index=0)

    with pytest.raises(ValueError, match="different frame counts"):
        probability_cube(uneven)


def test_probability_cube_rejects_a_video_with_two_labels():
    confused = probability_rows()
    confused.loc[0, "label"] = 1

    with pytest.raises(ValueError, match="more than one label"):
        probability_cube(confused)


def test_probability_cube_rejects_a_table_without_class_columns():
    with pytest.raises(ValueError, match="p_<class>"):
        probability_cube(probability_rows().drop(columns=["p_Halt", "p_Rally"]))


def test_segment_indices_spread_the_picks_over_the_gesture():
    """Centres of four equal segments of 24 frames, not the first four."""
    assert segment_indices(24, 4).tolist() == [3, 9, 15, 21]


def test_segment_indices_of_one_frame_lands_mid_gesture():
    """A single frame should come from the middle, not from either end."""
    assert segment_indices(24, 1).tolist() == [12]


def test_segment_indices_of_every_frame_is_every_frame():
    assert segment_indices(6, 6).tolist() == [0, 1, 2, 3, 4, 5]


def test_segment_indices_rejects_more_picks_than_frames():
    with pytest.raises(ValueError, match="cannot pick"):
        segment_indices(4, 5)


def test_random_indices_are_distinct_and_sorted():
    generator = np.random.default_rng(0)

    for _ in range(20):
        picked = random_indices(24, 8, generator)
        assert len(set(picked.tolist())) == 8
        assert picked.tolist() == sorted(picked.tolist())


def test_random_indices_reproduce_from_a_seed():
    first = random_indices(24, 8, np.random.default_rng(3))
    second = random_indices(24, 8, np.random.default_rng(3))

    assert first.tolist() == second.tolist()


def test_aggregate_mean_holds_when_the_mild_frames_carry_enough_mass():
    """Three frames at 0.7 outweigh one at 0.98: the mean is not a maximum."""
    probabilities = np.array([[[0.7, 0.3], [0.7, 0.3], [0.7, 0.3], [0.02, 0.98]]])

    assert aggregate_mean(probabilities).tolist() == [0]


def test_aggregate_vote_counts_frames_not_confidence():
    """The same split, mild enough that the mean flips and the vote does not.

    This is the pair's whole reason for existing: the mean can be carried by one
    very sure frame, the vote cannot.
    """
    probabilities = np.array([[[0.6, 0.4], [0.6, 0.4], [0.6, 0.4], [0.02, 0.98]]])

    assert aggregate_mean(probabilities).tolist() == [1]
    assert aggregate_vote(probabilities).tolist() == [0]


def test_aggregate_mean_and_vote_can_disagree():
    """One overwhelming frame carries the mean; the vote still says otherwise."""
    probabilities = np.array([[[0.55, 0.45], [0.55, 0.45], [0.0, 1.0]]])

    assert aggregate_mean(probabilities).tolist() == [1]
    assert aggregate_vote(probabilities).tolist() == [0]


def test_aggregate_vote_breaks_ties_by_probability_not_class_index():
    """A tie handed to the lowest class index would bias the collapse analysis.

    One frame each way, but the losing frame was far less sure, so the class
    with the greater probability mass takes it — even though it is not class 0.
    """
    probabilities = np.array([[[0.51, 0.49], [0.0, 1.0]]])

    assert aggregate_vote(probabilities).tolist() == [1]


def test_accuracy_by_class_separates_overall_from_per_class():
    """Unbalanced classes: overall accuracy is not the mean of the recalls."""
    predictions = np.array([0, 0, 0, 1])
    labels = np.array([0, 0, 0, 0])

    scores = accuracy_by_class(predictions, labels, class_count=2)

    assert scores[0] == pytest.approx(0.75)
    assert scores[1] == pytest.approx(0.75)
    assert np.isnan(scores[2])


def test_accuracy_curve_covers_every_frame_count_up_to_all_of_them():
    cube, labels, _ = probability_cube(probability_rows(frames_per_video=24))

    curve = accuracy_curve(cube, labels, ["Halt", "Rally"], repeats=5)

    assert sorted(curve["frames"].unique()) == [1, 2, 4, 8, 16, 24]
    assert set(curve["aggregation"]) == {"mean", "vote"}
    assert set(curve["selection"]) == {"segments", "random"}
    assert set(curve["scope"]) == {"overall", "Halt", "Rally"}


def test_accuracy_curve_is_flat_when_every_frame_already_decides():
    """Frames that all agree with the label leave nothing for more frames to add.

    This is the shape the research question calls pose-separable, and the curve
    has to be able to report it.
    """
    cube, labels, _ = probability_cube(probability_rows(frames_per_video=24))

    curve = accuracy_curve(cube, labels, ["Halt", "Rally"], repeats=5)
    overall = curve[(curve.scope == "overall") & (curve.selection == "random")]

    assert (overall["accuracy"] == 1.0).all()


def test_accuracy_curve_reports_no_spread_when_all_frames_are_taken():
    """There is one way to take every frame, so the draws cannot differ."""
    cube, labels, _ = probability_cube(probability_rows(frames_per_video=8))

    curve = accuracy_curve(cube, labels, ["Halt", "Rally"], repeats=5)
    everything = curve[(curve.frames == 8) & (curve.selection == "random")]

    assert (everything["spread"] == 0.0).all()


def test_accuracy_curve_reproduces_from_a_seed():
    cube, labels, _ = probability_cube(probability_rows(frames_per_video=24))

    first = accuracy_curve(cube, labels, ["Halt", "Rally"], repeats=5, seed=1)
    second = accuracy_curve(cube, labels, ["Halt", "Rally"], repeats=5, seed=1)

    pd.testing.assert_frame_equal(first, second)


def test_parse_args_takes_several_tables():
    args = parse_args(
        ["data/predictions/a.csv", "data/predictions/b.csv", "--output", "curve.csv"]
    )

    assert args.tables == [
        Path("data/predictions/a.csv"),
        Path("data/predictions/b.csv"),
    ]
    assert args.output == Path("curve.csv")
    assert args.repeats == 200
    assert args.seed == 11


def test_parse_args_requires_at_least_one_table():
    with pytest.raises(SystemExit):
        parse_args([])
