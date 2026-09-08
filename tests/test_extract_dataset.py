from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extract_dataset import (
    append_rows,
    completed_videos,
    drop_incomplete_videos,
    parse_args,
    read_annotations,
    sample_stratified,
    video_rng,
)
from manifest import video_metadata


def synthetic_entries(gestures, views, per_stratum):
    """Build annotation entries following the RoCoG-v2 synthetic naming scheme.

    No files are created. video_metadata reads names only, so the paths never
    have to exist on disk, and the scene index doubles as the viewpoint because
    views here are kept below six.
    """
    entries = []
    for label, gesture in enumerate(gestures):
        for view in views:
            for index in range(per_stratum):
                name = f"Scene{view}_{index}_{gesture}_1_1_2026_0_0_0.mp4"
                entries.append((Path("syn/ground") / gesture / name, label))
    return entries


def strata_sizes(entries):
    """Count how many entries fall into each (gesture, viewpoint) stratum."""
    return Counter(
        (path.parent.name, video_metadata(path, "syn").view) for path, _ in entries
    )


def test_every_stratum_contributes_the_requested_number():
    entries = synthetic_entries(["Halt", "Rally"], range(6), 10)

    drawn = sample_stratified(entries, "syn", 4, np.random.default_rng(0))

    sizes = strata_sizes(drawn)
    assert len(sizes) == 12
    assert set(sizes.values()) == {4}
    assert len(drawn) == 48


def test_drawn_videos_are_distinct():
    entries = synthetic_entries(["Halt", "Rally"], range(6), 10)

    drawn = sample_stratified(entries, "syn", 10, np.random.default_rng(0))

    assert len({path for path, _ in drawn}) == len(drawn)


def test_the_same_seed_draws_the_same_videos():
    entries = synthetic_entries(["Halt", "Rally"], range(6), 10)

    first = sample_stratified(entries, "syn", 4, np.random.default_rng(1))
    second = sample_stratified(entries, "syn", 4, np.random.default_rng(1))

    assert first == second


def test_a_different_seed_draws_different_videos():
    entries = synthetic_entries(["Halt", "Rally"], range(6), 10)

    first = sample_stratified(entries, "syn", 4, np.random.default_rng(1))
    second = sample_stratified(entries, "syn", 4, np.random.default_rng(2))

    assert first != second


def test_a_stratum_smaller_than_requested_is_rejected():
    """An unbalanced draw would surface much later as an unexplained result."""
    entries = synthetic_entries(["Halt"], range(6), 10)

    with pytest.raises(RuntimeError):
        sample_stratified(entries, "syn", 11, np.random.default_rng(0))


def test_parse_args_reads_the_annotations_file_as_a_path():
    args = parse_args(["data/annotations/syn_ground_train.txt"])

    assert args.annotations == Path("data/annotations/syn_ground_train.txt")


def test_parse_args_requires_the_annotations_file():
    """The file name is what a run chooses; there is no sensible default."""
    with pytest.raises(SystemExit):
        parse_args([])


def test_read_annotations_pairs_paths_with_integer_labels(tmp_path):
    annotations = tmp_path / "syn_ground_train.txt"
    annotations.write_text(
        "syn/ground/Halt/Scene0_0_Halt_1_1_2026_0_0_0.mp4 4\n"
        "syn/ground/Rally/Scene1_0_Rally_1_1_2026_0_0_0.mp4 2\n"
    )

    entries = read_annotations(annotations)

    assert entries == [
        (Path("syn/ground/Halt/Scene0_0_Halt_1_1_2026_0_0_0.mp4"), 4),
        (Path("syn/ground/Rally/Scene1_0_Rally_1_1_2026_0_0_0.mp4"), 2),
    ]
    assert isinstance(entries[0][1], int)


def manifest_rows(video_id, count, start=0):
    """Build the rows one video contributes, with only the columns resume reads."""
    return [
        {
            "video_id": video_id,
            "frame_number": start + i,
            "path": f"{video_id}_f{i}.jpg",
        }
        for i in range(count)
    ]


def test_the_same_video_is_given_the_same_frames_whatever_its_position():
    """Resuming skips videos, so a draw that moves with the stream would move."""
    first = video_rng("Scene1_0_Halt_1_1_2026_0_0_0").integers(0, 1000, size=24)
    second = video_rng("Scene1_0_Halt_1_1_2026_0_0_0").integers(0, 1000, size=24)

    assert np.array_equal(first, second)


def test_different_videos_are_given_different_frames():
    first = video_rng("Scene1_0_Halt_1_1_2026_0_0_0").integers(0, 1000, size=24)
    second = video_rng("Scene1_1_Halt_1_1_2026_0_0_0").integers(0, 1000, size=24)

    assert not np.array_equal(first, second)


def test_a_different_seed_places_frames_differently():
    first = video_rng("Scene1_0_Halt_1_1_2026_0_0_0", seed=1).integers(0, 1000, size=24)
    second = video_rng("Scene1_0_Halt_1_1_2026_0_0_0", seed=2).integers(
        0, 1000, size=24
    )

    assert not np.array_equal(first, second)


def test_no_manifest_means_nothing_is_done(tmp_path):
    assert completed_videos(tmp_path / "absent.csv", 24) == set()


def test_an_empty_manifest_means_nothing_is_done(tmp_path):
    """A file created but never written to would otherwise fail to parse."""
    manifest = tmp_path / "syn_ground_train.csv"
    manifest.touch()

    assert completed_videos(manifest, 24) == set()


def test_videos_with_every_row_count_as_done(tmp_path):
    manifest = tmp_path / "syn_ground_train.csv"
    append_rows(manifest_rows("first", 4) + manifest_rows("second", 4), manifest)

    assert completed_videos(manifest, 4) == {"first", "second"}


def test_a_video_cut_off_partway_is_extracted_again(tmp_path):
    """Trusting a short group would leave a hole nothing downstream reports."""
    manifest = tmp_path / "syn_ground_train.csv"
    append_rows(manifest_rows("whole", 4) + manifest_rows("cut", 2), manifest)

    assert completed_videos(manifest, 4) == {"whole"}


def test_asking_for_more_frames_per_video_redoes_the_pass(tmp_path):
    """A manifest built under other rules must not be resumed into."""
    manifest = tmp_path / "syn_ground_train.csv"
    append_rows(manifest_rows("first", 4), manifest)

    assert completed_videos(manifest, 8) == set()


def test_appending_writes_one_header_and_keeps_every_row(tmp_path):
    manifest = tmp_path / "syn_ground_train.csv"

    append_rows(manifest_rows("first", 3), manifest)
    append_rows(manifest_rows("second", 3), manifest)

    written = pd.read_csv(manifest)
    assert list(written["video_id"]) == ["first"] * 3 + ["second"] * 3
    assert manifest.read_text().count("video_id") == 1


def test_appending_creates_the_manifest_directory(tmp_path):
    manifest = tmp_path / "manifests" / "syn_ground_train.csv"

    append_rows(manifest_rows("first", 3), manifest)

    assert manifest.exists()


def test_trimming_leaves_whole_videos_untouched(tmp_path):
    manifest = tmp_path / "syn_ground_train.csv"
    append_rows(manifest_rows("first", 4) + manifest_rows("second", 4), manifest)

    assert drop_incomplete_videos(manifest, 4) == 0
    assert len(pd.read_csv(manifest)) == 8


def test_trimming_removes_the_rows_of_a_video_left_half_written(tmp_path):
    """Left in place, they would sit in front of the complete group written next."""
    manifest = tmp_path / "syn_ground_train.csv"
    append_rows(manifest_rows("whole", 4) + manifest_rows("cut", 2), manifest)

    assert drop_incomplete_videos(manifest, 4) == 2

    remaining = pd.read_csv(manifest)
    assert set(remaining["video_id"]) == {"whole"}
    assert len(remaining) == 4


def test_a_trimmed_manifest_still_takes_appended_rows(tmp_path):
    """Trimming everything away leaves a header, which is not an empty file."""
    manifest = tmp_path / "syn_ground_train.csv"
    append_rows(manifest_rows("cut", 2), manifest)
    drop_incomplete_videos(manifest, 4)

    append_rows(manifest_rows("cut", 4), manifest)

    written = pd.read_csv(manifest)
    assert list(written["video_id"]) == ["cut"] * 4
    assert manifest.read_text().count("video_id") == 1


def test_trimming_leaves_the_surviving_rows_byte_for_byte(tmp_path):
    """The CSV reader's fast float parser would edit a position by one ulp."""
    manifest = tmp_path / "syn_ground_train.csv"
    append_rows(
        [{"video_id": "whole", "position": 0.01694915254237288}] * 2
        + [{"video_id": "cut", "position": 0.13559322033898305}],
        manifest,
    )
    before = manifest.read_text().splitlines()

    drop_incomplete_videos(manifest, 2)

    assert manifest.read_text().splitlines() == before[:3]


def test_a_run_samples_across_the_gesture_by_default():
    """The pass that built every manifest so far has to keep its old command."""
    args = parse_args(["data/annotations/syn_ground_train.txt"])

    assert args.idle is False


def test_a_run_can_ask_for_the_stretch_before_the_gesture():
    args = parse_args(["data/annotations/syn_ground_train.txt", "--idle"])

    assert args.idle is True
