import cv2
import numpy as np
import pandas as pd
import pytest

from frame_store import (
    append_rows,
    completed_videos,
    drop_incomplete_videos,
    save_frames,
)


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


def gradient(side):
    """One frame with detail at every scale, so resampling has work to do."""
    row = np.linspace(0, 255, side, dtype=np.uint8)

    return np.dstack([np.repeat(row[None, :], side, axis=0)] * 3)


def test_a_frame_larger_than_the_size_asked_for_is_shrunk_to_it(tmp_path):
    path = tmp_path / "syn" / "Advance" / "Scene1_Advance_f0007.jpg"

    save_frames([gradient(640)], [path], 256)

    assert cv2.imread(str(path)).shape == (256, 256, 3)


def test_a_frame_smaller_than_the_size_asked_for_is_grown_to_it(tmp_path):
    """14% of the real clips are below 640, so growing has to work too."""
    path = tmp_path / "real" / "Halt" / "S00_f0004.jpg"

    save_frames([gradient(408)], [path], 640)

    assert cv2.imread(str(path)).shape == (640, 640, 3)


def test_writing_creates_the_class_directory(tmp_path):
    path = tmp_path / "syn" / "Advance" / "Scene1_Advance_f0007.jpg"

    save_frames([gradient(320)], [path], 256)

    assert path.exists()


def test_a_frame_without_a_path_to_write_to_is_refused(tmp_path):
    """Silently dropping one would leave a manifest row with no file."""
    with pytest.raises(ValueError):
        save_frames([gradient(320), gradient(320)], [tmp_path / "one.jpg"], 256)
