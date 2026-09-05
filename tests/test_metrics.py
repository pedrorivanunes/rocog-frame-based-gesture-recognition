import numpy as np
import pandas as pd
import pytest

from metrics import (
    class_report,
    compare,
    confusion_matrix,
    format_class_report,
    format_delta,
    mean_and_spread,
    parse_args,
    prediction_imbalance,
    prediction_trace,
    predictions,
)

CLASSES = ["Halt", "Rally"]


def view(*values: float) -> pd.DataFrame:
    """Build a one-cell-per-class square view, the shape every report has."""
    filled = np.full((len(CLASSES), len(CLASSES)), float(values[0]))
    for offset, value in enumerate(values):
        filled[offset % len(CLASSES), offset % len(CLASSES)] = value
    return pd.DataFrame(filled, index=CLASSES, columns=CLASSES)


def scored(rows: list[tuple[str, int, list[float]]]) -> pd.DataFrame:
    """Build a probability table from ``(video, label, halt_probabilities)``.

    Each video contributes one row per probability given, in frame order.
    """
    records = []
    for video, label, halt_probabilities in rows:
        for frame, halt in enumerate(halt_probabilities):
            records.append(
                {
                    "video_id": video,
                    "frame_number": frame,
                    "position": frame / len(halt_probabilities),
                    "label": label,
                    "p_Halt": halt,
                    "p_Rally": 1.0 - halt,
                }
            )
    return pd.DataFrame(records)


def test_predictions_score_each_frame_on_its_own():
    table = scored([("a", 0, [0.9, 0.9, 0.1, 0.1])])

    truth, guess = predictions(table, "frame")

    assert truth.tolist() == [0, 0, 0, 0]
    assert guess.tolist() == [0, 0, 1, 1]


def test_predictions_average_a_video_before_scoring_it():
    """Frame by frame this video is a three-to-one split; averaged it is Halt.

    The two levels have to be able to disagree, or there would be no reason to
    offer both.
    """
    table = scored([("a", 0, [0.7, 0.7, 0.7, 0.0])])

    assert predictions(table, "frame")[1].tolist() == [0, 0, 0, 1]
    assert predictions(table, "video")[1].tolist() == [0]


def test_predictions_reject_an_unknown_level():
    with pytest.raises(ValueError, match="must be 'frame' or 'video'"):
        predictions(scored([("a", 0, [0.9])]), "clip")


def test_confusion_matrix_normalises_each_true_class():
    """Halt has twice the frames of Rally and must not look twice as wrong."""
    table = scored([("a", 0, [0.9, 0.1]), ("b", 0, [0.9, 0.1]), ("c", 1, [0.1, 0.1])])

    matrix = confusion_matrix(table, CLASSES, "frame")

    assert matrix.loc["Halt", "Halt"] == pytest.approx(50.0)
    assert matrix.loc["Halt", "Rally"] == pytest.approx(50.0)
    assert matrix.loc["Rally", "Rally"] == pytest.approx(100.0)


def test_confusion_matrix_rows_sum_to_a_hundred():
    table = scored([("a", 0, [0.9, 0.1, 0.9]), ("b", 1, [0.2, 0.8, 0.1])])

    matrix = confusion_matrix(table, CLASSES, "frame")

    assert matrix.sum(axis=1).tolist() == pytest.approx([100.0, 100.0])


def test_confusion_matrix_leaves_an_absent_class_undefined():
    """Zero would read as a class that was scored and failed."""
    table = scored([("a", 0, [0.9, 0.9])])

    matrix = confusion_matrix(table, CLASSES, "frame")

    assert np.isnan(matrix.loc["Rally"]).all()


def test_class_report_separates_a_missed_class_from_an_over_claimed_one():
    """Rally is guessed for everything: perfect recall, poor precision."""
    table = scored([("a", 0, [0.1, 0.1]), ("b", 1, [0.1, 0.1])])

    report = class_report(table, CLASSES, "frame")

    assert report.loc["Rally", "recall"] == pytest.approx(100.0)
    assert report.loc["Rally", "precision"] == pytest.approx(50.0)
    assert report.loc["Rally", "share"] == pytest.approx(100.0)
    assert report.loc["Halt", "recall"] == pytest.approx(0.0)


def test_class_report_leaves_precision_undefined_for_a_class_never_guessed():
    table = scored([("a", 0, [0.9, 0.9])])

    report = class_report(table, CLASSES, "frame")

    assert np.isnan(report.loc["Rally", "precision"])


def test_class_report_counts_support_in_items_not_videos():
    table = scored([("a", 0, [0.9, 0.9, 0.9])])

    assert class_report(table, CLASSES, "frame").loc["Halt", "support"] == 3
    assert class_report(table, CLASSES, "video").loc["Halt", "support"] == 1


def test_prediction_imbalance_is_zero_when_every_class_is_named_equally():
    table = scored([("a", 0, [0.9, 0.1])])

    assert prediction_imbalance(table, CLASSES, "frame") == pytest.approx(0.0)


def test_prediction_imbalance_is_greatest_when_one_class_takes_everything():
    table = scored([("a", 0, [0.9, 0.9, 0.9, 0.9])])

    assert prediction_imbalance(table, CLASSES, "frame") == pytest.approx(50.0)


def test_prediction_trace_keeps_the_frames_in_order():
    """A matrix cannot say whether two errors happen at once or in turn."""
    table = scored([("a", 0, [0.9, 0.9, 0.1, 0.1, 0.9])])

    trace = prediction_trace(table, CLASSES)

    assert trace.loc["a", "truth"] == "Halt"
    assert [trace.loc["a", i] for i in range(5)] == [0, 0, 1, 1, 0]


def test_prediction_trace_gives_one_row_per_video():
    table = scored([("a", 0, [0.9, 0.9]), ("b", 1, [0.1, 0.1])])

    trace = prediction_trace(table, CLASSES)

    assert list(trace.index) == ["a", "b"]
    assert trace["truth"].tolist() == ["Halt", "Rally"]


def test_mean_and_spread_averages_matching_views():
    first = pd.DataFrame([[10.0, 20.0]], columns=["a", "b"])
    second = pd.DataFrame([[20.0, 40.0]], columns=["a", "b"])

    mean, spread = mean_and_spread([first, second])

    assert mean.iloc[0].tolist() == pytest.approx([15.0, 30.0])
    assert spread.iloc[0].tolist() == pytest.approx([7.0710678, 14.1421356])


def test_mean_and_spread_of_one_view_has_no_spread():
    """NaN, not zero: one measurement is silent about variation, not proof of none."""
    only = pd.DataFrame([[10.0, 20.0]], columns=["a", "b"])

    mean, spread = mean_and_spread([only])

    assert mean.iloc[0].tolist() == pytest.approx([10.0, 20.0])
    assert np.isnan(spread.to_numpy()).all()


def test_mean_and_spread_rejects_views_of_different_shape():
    with pytest.raises(ValueError, match="disagree in index or columns"):
        mean_and_spread(
            [
                pd.DataFrame([[1.0, 2.0]], columns=["a", "b"]),
                pd.DataFrame([[1.0, 2.0]], columns=["a", "c"]),
            ]
        )


def test_mean_and_spread_rejects_an_empty_list():
    with pytest.raises(ValueError, match="no frames"):
        mean_and_spread([])


def test_parse_args_defaults_to_frames_and_no_traces():
    args = parse_args(["data/predictions/a.csv"])

    assert args.level == "frame"
    assert args.traces == 0


def test_parse_args_takes_several_tables_and_a_level():
    args = parse_args(["a.csv", "b.csv", "--level", "video", "--traces", "3"])

    assert len(args.tables) == 2
    assert args.level == "video"
    assert args.traces == 3


def test_parse_args_rejects_an_unknown_level():
    with pytest.raises(SystemExit):
        parse_args(["a.csv", "--level", "clip"])


def test_compare_reports_the_treatment_minus_the_baseline():
    delta, _ = compare([view(10.0)], [view(30.0)])

    assert delta.iloc[0, 0] == pytest.approx(20.0)


def test_compare_pairs_repetitions_when_both_sides_hold_the_same_number():
    # Both repetitions move by exactly two, so pairing leaves nothing to vary.
    # Differencing the means instead would carry the spread of 10 vs 20 into
    # the answer, and the error would come back far from zero.
    _, error = compare([view(10.0), view(20.0)], [view(12.0), view(22.0)])

    assert error.iloc[0, 0] == pytest.approx(0.0)


def test_compare_falls_back_to_the_difference_of_means_when_the_counts_differ():
    delta, error = compare(
        [view(10.0), view(20.0)], [view(12.0), view(22.0), view(32.0)]
    )

    assert delta.iloc[0, 0] == pytest.approx(22.0 - 15.0)
    assert error.iloc[0, 0] > 0


def test_compare_has_no_error_with_a_single_repetition_on_each_side():
    _, error = compare([view(10.0)], [view(30.0)])

    assert error.isna().all().all()


def test_compare_rejects_a_side_with_nothing_in_it():
    with pytest.raises(ValueError):
        compare([], [view(10.0)])


def test_compare_rejects_views_that_disagree_in_shape():
    other = pd.DataFrame([[1.0]], index=["Halt"], columns=["Halt"])

    with pytest.raises(ValueError):
        compare([view(10.0)], [other])


def test_format_delta_marks_only_the_cells_that_clear_the_threshold():
    delta = pd.DataFrame([[10.0, 1.0], [0.0, 0.0]], index=CLASSES, columns=CLASSES)
    error = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]], index=CLASSES, columns=CLASSES)

    rendered = format_delta(delta, error, threshold=2.0)

    assert "+10*" in rendered
    assert "+1*" not in rendered


def test_format_delta_leaves_a_cell_unmarked_when_its_error_is_unknown():
    delta = pd.DataFrame([[10.0, 0.0], [0.0, 0.0]], index=CLASSES, columns=CLASSES)
    error = pd.DataFrame(np.nan, index=CLASSES, columns=CLASSES)

    assert "*" not in format_delta(delta, error)


def test_format_class_report_shows_an_undefined_figure_as_neither_zero_nor_nan():
    report = pd.DataFrame(
        {
            "recall": [50.0, 50.0],
            "precision": [np.nan, 50.0],
            "support": [4, 4],
            "share": [50.0, 50.0],
        },
        index=CLASSES,
    )

    rendered = format_class_report(report)

    assert "--" in rendered
    assert "nan" not in rendered


def test_parse_args_takes_no_second_group_by_default():
    args = parse_args(["a.csv"])

    assert args.against is None
    assert args.threshold == 2.0


def test_parse_args_takes_a_second_group_to_compare_against():
    args = parse_args(["a.csv", "b.csv", "--against", "c.csv", "--threshold", "1"])

    assert [path.name for path in args.tables] == ["a.csv", "b.csv"]
    assert [path.name for path in args.against] == ["c.csv"]
    assert args.threshold == 1.0
