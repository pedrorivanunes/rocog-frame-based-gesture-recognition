import math

import numpy as np
import pandas as pd
import pytest

from catalog import find
from dataset import clip_batches
from readings import (
    GESTURES,
    accuracy_by_k,
    by_gesture,
    clip_curve,
    clip_partition,
    confidence_auroc,
    frame_accuracy,
    frame_agreement,
    gesture_cube,
    in_the_thesis,
    latency_by_k,
    oracle_and_null,
    prior_marginal,
    ranked_share,
    read,
    reported,
    stored_table,
    summarize,
    video_scores,
)

NAMES = [
    "Advance",
    "Attention",
    "Rally",
    "MoveForward",
    "Halt",
    "FollowMe",
    "MoveInReverse",
    "Idle",
]


def table_of(probabilities: np.ndarray, labels: list[int]) -> pd.DataFrame:
    """A table as scoring writes it, from ``(videos, frames, outputs)`` scores."""
    videos, frames, outputs = probabilities.shape
    rows = {
        "video_id": np.repeat([f"video{index}" for index in range(videos)], frames),
        "frame_number": np.tile(np.arange(frames) * 3, videos),
        "position": np.tile((np.arange(frames) + 0.5) / frames, videos),
        "label": np.repeat(labels, frames),
    }
    for index in range(outputs):
        rows[f"p_{NAMES[index]}"] = probabilities[:, :, index].ravel()
    return pd.DataFrame(rows)


def confident(answers: np.ndarray, outputs: int = GESTURES + 1) -> np.ndarray:
    """Scores that name ``answers[video, frame]`` with probability 0.9."""
    scores = np.full((*answers.shape, outputs), 0.1 / (outputs - 1))
    np.put_along_axis(scores, answers[..., None], 0.9, axis=2)
    return scores


def test_the_idle_output_is_dropped_and_the_rest_left_as_they_were():
    scores = confident(np.zeros((2, 4), dtype=int))
    scores[0, 0] = [0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6]

    cube, labels, ids = gesture_cube(table_of(scores, [0, 3]))

    assert cube.shape == (2, 4, GESTURES)
    assert cube[0, 0].sum() == pytest.approx(0.4)
    assert labels.tolist() == [0, 3]
    assert ids == ["video0", "video1"]


def test_the_clips_read_are_the_clips_scoring_serves():
    """The same partition reached two ways is the check that it is the right one."""
    manifest = pd.DataFrame(
        {"video_id": np.repeat(["a", "b"], 24), "frame": np.tile(np.arange(24), 2)}
    )

    served = clip_batches(manifest, segments=8, frames_per_batch=8)

    expected = [list(clip) for clip in clip_partition(24, 8)]
    assert served[:3] == expected
    assert served[3:] == [[24 + place for place in clip] for clip in expected]


def test_a_clip_length_that_does_not_divide_the_frames_is_refused():
    with pytest.raises(ValueError, match="do not split"):
        clip_partition(24, 16)


def test_the_correction_lifts_a_gesture_the_network_rarely_names():
    cube = np.zeros((1, 2, GESTURES))
    cube[0, :, 0] = 0.5
    cube[0, :, 1] = 0.4

    plain = video_scores(cube).argmax(axis=1)
    marginal = np.full(GESTURES, 0.1)
    marginal[0], marginal[1] = 0.5, 0.2
    corrected = video_scores(cube, marginal).argmax(axis=1)

    assert plain.tolist() == [0]
    assert corrected.tolist() == [1]


def test_every_video_weighs_the_same_in_the_marginal():
    cube = np.zeros((2, 4, GESTURES))
    cube[0, :, 0] = 1.0
    cube[1, :2, 1] = 1.0
    cube[1, 2:, 2] = 1.0

    assert prior_marginal(cube)[:3].tolist() == [0.5, 0.25, 0.25]


def test_frame_accuracy_counts_frames_not_videos():
    answers = np.array([[0, 0, 0, 1], [3, 3, 2, 2]])

    # Three of four frames right in one video, two of four in the other.
    assert frame_accuracy(confident(answers)[..., :GESTURES], np.array([0, 3])) == 5 / 8


def test_the_auroc_is_one_when_the_right_frames_are_the_confident_ones():
    cube = confident(np.array([[0, 0, 1, 1]]))[..., :GESTURES]
    cube[0, 2:, 1] = 0.5
    labels = np.array([0])

    assert confidence_auroc(cube, labels) == 1.0
    cube[0, 2:, 1] = 0.95
    assert confidence_auroc(cube, labels) == 0.0
    cube[0, 2:, 1] = 0.9
    assert confidence_auroc(cube, labels) == 0.5
    assert math.isnan(confidence_auroc(cube, np.array([5])))


def test_the_oracle_needs_one_right_frame_and_its_null_is_reproducible():
    answers = np.array([[1, 1, 0], [2, 2, 2], [3, 3, 3]])
    cube = confident(answers)[..., :GESTURES]
    labels = np.array([0, 4, 3])

    oracle, null = oracle_and_null(cube, labels)

    assert oracle == pytest.approx(2 / 3)
    assert oracle_and_null(cube, labels) == (oracle, null)
    assert 0 <= null <= 1


def test_the_true_gesture_is_always_among_all_seven():
    scores = np.random.default_rng(0).random((5, GESTURES))
    labels = np.array([0, 1, 2, 3, 4])

    assert ranked_share(scores, labels, 1) == (scores.argmax(axis=1) == labels).mean()
    assert ranked_share(scores, labels, GESTURES) == 1.0


def test_taking_every_frame_leaves_the_draw_nothing_to_vary():
    """At the last point there is one way to choose all the frames."""
    generator = np.random.default_rng(1)
    cube = generator.random((6, 24, GESTURES))
    labels = generator.integers(0, GESTURES, 6)

    curve = accuracy_by_k(cube, labels, draws=5)

    spread, drawn, deviation = curve[24]
    assert drawn == spread
    assert deviation == 0.0
    assert sorted(curve) == [1, 2, 4, 8, 16, 24]
    assert sorted(accuracy_by_k(cube[:, :8], labels, draws=5)) == [1, 2, 4, 8]


def test_the_confusion_counts_decisions_by_their_true_gesture():
    recall, precision, confusion = by_gesture(
        np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])
    )

    assert confusion[0, 0] == 1
    assert confusion[1, 0] == 1
    assert confusion[1, 1] == 2
    assert recall[:2].tolist() == pytest.approx([1.0, 2 / 3])
    assert precision[:2].tolist() == pytest.approx([0.5, 1.0])
    assert math.isnan(recall[2])


def test_a_shifted_network_is_decided_per_clip():
    answers = np.zeros((2, 24), dtype=int)
    answers[1] = 2
    answers[1, :8] = 4  # the first block: one frame of every clip of eight
    table = table_of(confident(answers), [0, 2])

    reading = read(table, clip=8)

    assert reading.decisions.shape == (2, 3)
    assert reading.decided.tolist() == [[0, 0, 0], [2, 2, 2]]
    assert reading.clip == 1.0
    assert reading.frame == pytest.approx(40 / 48)
    assert read(table).clip is None


def test_a_corrected_configuration_decides_after_dividing():
    scores = np.zeros((1, 4, GESTURES + 1))
    scores[0, :, 0], scores[0, :, 1] = 0.5, 0.4
    table = table_of(scores, [1])
    marginal = np.full(GESTURES, 0.1)
    marginal[0], marginal[1] = 0.5, 0.2

    assert read(table).video == 0.0
    assert read(table, marginal).video == 1.0


def test_a_shifted_curve_reads_each_length_from_its_own_scoring():
    short = table_of(confident(np.zeros((1, 24), dtype=int)), [0])
    wrong = table_of(confident(np.ones((1, 24), dtype=int)), [0])

    assert clip_curve({8: short, 1: wrong}) == {1: 0.0, 8: 1.0}


def test_the_summary_is_in_percent_with_the_sample_deviation():
    tables = [
        table_of(confident(np.zeros((2, 24), dtype=int)), [0, 1]),
        table_of(confident(np.zeros((2, 24), dtype=int)), [0, 0]),
    ]
    readings = [read(table) for table in tables]

    summary = summarize(readings, raw=readings)

    assert summary.runs == 2
    assert summary.video == pytest.approx((75.0, 35.355339))
    assert summary.raw_video == summary.video
    assert summary.confusion.sum() == 4
    assert summary.curve[24][0] == pytest.approx(75.0)


def test_a_table_is_found_under_the_name_scoring_gives_it(tmp_path):
    assert stored_table("diff_multi", 2, root=tmp_path).name == (
        "diff_multi_s2__real_ground_test.csv"
    )
    assert stored_table("tsm", 0, length=8, trained=8, root=tmp_path).name == (
        "tsm_s0__real_ground_test.csv"
    )
    assert stored_table("tsm", 0, length=4, trained=8, root=tmp_path).name == (
        "tsm_s0__real_ground_test_clips4.csv"
    )


def test_the_thesis_figures_average_the_runs_on_this_computer(tmp_path):
    entry = find("resnet18-frames")
    assert reported(entry, tmp_path) is None
    for seed, labels in ((0, [0, 1]), (2, [0, 0])):
        table = table_of(confident(np.zeros((2, 24), dtype=int)), labels)
        table.to_csv(stored_table(entry.stem, seed, root=tmp_path), index=False)

    summary = reported(entry, tmp_path)

    assert summary.runs == 2
    assert summary.video[0] == pytest.approx(75.0)
    assert in_the_thesis(entry, summary) == (
        "75.00% per frame · 75.00% per video, on the 100 real test videos"
    )


def test_a_corrected_configuration_reads_its_marginal_from_the_training_videos(
    tmp_path,
):
    entry = find("resnet18-prior")
    test_scores = np.zeros((1, 24, GESTURES + 1))
    test_scores[0, :, 0], test_scores[0, :, 1] = 0.5, 0.4
    train_scores = np.zeros((1, 24, GESTURES + 1))
    train_scores[0, :, 0], train_scores[0, :, 1] = 0.6, 0.1
    table_of(test_scores, [1]).to_csv(
        stored_table(entry.stem, 0, root=tmp_path), index=False
    )
    table_of(train_scores, [0]).to_csv(
        stored_table(entry.stem, 0, "real_ground_train", root=tmp_path), index=False
    )

    summary = reported(entry, tmp_path)

    assert summary.video[0] == 100.0
    assert summary.raw_video[0] == 0.0
    assert in_the_thesis(entry, summary).startswith(
        "100.00% per video with the correction (0.00% without)"
    )


def test_the_card_quotes_a_shifted_network_per_decision():
    entry = find("resnet18-shift")
    tables = [table_of(confident(np.zeros((2, 24), dtype=int)), [0, 0])]

    line = in_the_thesis(entry, summarize([read(tables[0], clip=entry.clip)]))

    assert line == (
        "100.00% per frame · 100.00% per decision on 8 frames, on the 100 real test"
        " videos"
    )


def test_frames_are_compared_where_both_tables_have_them():
    stored = table_of(confident(np.zeros((3, 4), dtype=int)), [0, 0, 1])
    now = stored[stored["video_id"] != "video1"].copy()
    now.loc[now.index[0], "p_Advance"] = 0.0

    same, compared, gap = frame_agreement(now, stored)

    assert (same, compared) == (7, 8)
    assert gap == pytest.approx(0.9)


def test_frames_the_stored_table_does_not_have_are_refused():
    stored = table_of(confident(np.zeros((2, 4), dtype=int)), [0, 0])
    now = stored.copy()
    now["video_id"] = now["video_id"].replace("video1", "video9")

    with pytest.raises(ValueError, match="not in the stored table"):
        frame_agreement(now, stored)

    moved = stored.copy()
    moved.loc[0, "frame_number"] = 99
    with pytest.raises(ValueError, match="different frames"):
        frame_agreement(moved, stored)


def write_latency(path, rows):
    pd.DataFrame(
        [
            {"model": model, "level": level, "frames": frames, "median_ms": ms}
            for model, level, frames, ms in rows
        ]
    ).to_csv(path, index=False)


def test_a_decision_costs_its_rows_together_at_the_k_every_row_has(tmp_path):
    write_latency(
        tmp_path / "latency_catalog_cpu_1t_isolated.csv",
        [
            ("pose_detector", "decision", 1, 20.0),
            ("pose_detector", "decision", 8, 160.0),
            ("pose_classifier", "decision", 1, 0.5),
            ("pose_classifier", "decision", 4, 0.6),
            ("pose_classifier", "decision", 8, 0.7),
            ("pose_classifier", "forward", 1, 9.0),
        ],
    )

    table = latency_by_k(find("pose"), root=tmp_path)

    assert table == {"cpu_1t": {1: 20.5, 8: 160.7}}


def test_a_session_without_the_row_or_the_file_reports_nothing(tmp_path):
    write_latency(
        tmp_path / "latency_catalog_cuda_1t_isolated.csv",
        [("resnet18", "decision", 1, 2.0)],
    )

    assert latency_by_k(find("resnet18-shift"), root=tmp_path) == {}
    assert latency_by_k(find("resnet18-frames"), root=tmp_path) == {"cuda_1t": {1: 2.0}}
