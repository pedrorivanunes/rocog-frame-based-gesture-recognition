import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

import frame_cache
from extract_dataset import NUM_FRAMES, video_rng
from frame_cache import (
    CACHE_DIR,
    EXTERNAL,
    UNLABELLED,
    VideoSource,
    cache_key,
    dataset_videos,
    prepare,
)
from frame_extraction import extract_frames

CLASS_NAMES = {0: "Advance", 3: "Halt"}

# Named the way the dataset names real clips, so the subject reads off it.
REAL_NAME = "S07_10m_ground_label3_start100"


def write_video(path: Path, shades: list[int], size: int = 32) -> Path:
    """Write a video whose frames are solid greys, one per shade.

    Lossless FFV1, so a frame reads back as the value it was written with and a
    test can tell which frame an image is.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (size, size)
    )
    for shade in shades:
        writer.write(np.full((size, size, 3), shade, dtype=np.uint8))
    writer.release()

    return path


def shade_of(root: Path, path: str) -> int:
    """The grey a stored frame was written with, give or take JPEG rounding."""
    return int(cv2.imread(str(root / path))[0, 0, 0])


@pytest.fixture
def real_clip(tmp_path):
    """Forty frames stepping by five, so a frame's shade names its number."""
    return write_video(tmp_path / "videos" / f"{REAL_NAME}.avi", list(range(0, 200, 5)))


def test_a_recorded_video_is_read_at_the_frames_its_manifest_names(tmp_path, real_clip):
    """The record, not the sampler, says which frames the figures came from."""
    source = VideoSource(real_clip, "real", 3, (4, 17, 31))

    rows, _ = prepare(source, CLASS_NAMES, size=32, root=tmp_path)

    assert rows["frame_number"].tolist() == [4, 17, 31]
    shades = [shade_of(tmp_path, path) for path in rows["path"]]
    assert np.allclose(shades, [20, 85, 155], atol=2)


def test_a_video_without_a_record_is_drawn_as_extraction_draws(tmp_path, real_clip):
    source = VideoSource(real_clip, "real", 3)

    rows, _ = prepare(source, CLASS_NAMES, size=32, root=tmp_path)

    expected = extract_frames(real_clip, NUM_FRAMES, video_rng(REAL_NAME))
    assert rows["frame_number"].tolist() == [f.frame_number for f in expected]


def test_rows_carry_what_scoring_reads(tmp_path, real_clip):
    rows, _ = prepare(
        VideoSource(real_clip, "real", 3, (4, 17)), CLASS_NAMES, size=32, root=tmp_path
    )

    assert rows["video_id"].unique().tolist() == [REAL_NAME]
    assert rows["group_id"].unique().tolist() == ["S07"]
    assert rows["label"].unique().tolist() == [3]
    assert rows["class_name"].unique().tolist() == ["Halt"]
    assert rows["position"].tolist() == pytest.approx([4 / 39, 17 / 39])


def test_frames_land_in_the_cache_and_never_among_the_dataset_frames(
    tmp_path, real_clip
):
    """data/frames holds what every figure was computed on; nothing writes there."""
    rows, window = prepare(
        VideoSource(real_clip, "real", 3, (4, 17)),
        CLASS_NAMES,
        dense=True,
        size=32,
        root=tmp_path,
    )

    for path in [*rows["path"], *window["path"]]:
        assert path.startswith(str(CACHE_DIR / "frames"))
        assert (tmp_path / path).exists()
    assert not (tmp_path / "data" / "frames").exists()


def test_the_dense_window_of_a_real_clip_is_every_frame(tmp_path, real_clip):
    """Real clips carry no window of their own, so the gesture is the whole clip."""
    _, window = prepare(
        VideoSource(real_clip, "real", 3, (4,)),
        CLASS_NAMES,
        dense=True,
        size=32,
        root=tmp_path,
    )

    assert window["frame_number"].tolist() == list(range(40))


def test_the_two_kinds_of_frame_never_share_a_folder(tmp_path, real_clip):
    """Neither is overwritten by the other, whichever was asked for first."""
    rows, window = prepare(
        VideoSource(real_clip, "real", 3, (4,)),
        CLASS_NAMES,
        dense=True,
        size=32,
        root=tmp_path,
    )

    assert {Path(path).parent.name for path in rows["path"]} == {"sampled"}
    assert {Path(path).parent.name for path in window["path"]} == {"dense"}


def test_a_video_met_again_is_read_back_without_decoding(
    tmp_path, real_clip, monkeypatch
):
    source = VideoSource(real_clip, "real", 3, (4, 17))
    first, first_window = prepare(
        source, CLASS_NAMES, dense=True, size=32, root=tmp_path
    )

    def refuse(*_, **__):
        raise AssertionError("decoded a video the cache already holds")

    for name in ("extract_frames", "extract_frames_at", "extract_all_frames"):
        monkeypatch.setattr(frame_cache, name, refuse)
    again, again_window = prepare(
        source, CLASS_NAMES, dense=True, size=32, root=tmp_path
    )

    pd.testing.assert_frame_equal(first, again)
    pd.testing.assert_frame_equal(first_window, again_window)


def test_frames_stored_at_another_size_sit_under_a_folder_named_for_it(
    tmp_path, real_clip
):
    """The scoring code reads the size off the folder after frames/."""
    rows, _ = prepare(
        VideoSource(real_clip, "real", 3, (4,)), CLASS_NAMES, size=64, root=tmp_path
    )

    assert Path(rows["path"].iloc[0]).parts[3] == "64"
    assert cv2.imread(str(tmp_path / rows["path"].iloc[0])).shape == (64, 64, 3)


def test_a_dataset_video_drawn_anew_never_shares_a_folder_with_its_record(
    real_clip,
):
    recorded = VideoSource(real_clip, "real", 3, (4, 17))
    drawn = VideoSource(real_clip, "real", 3)

    assert cache_key(recorded) == REAL_NAME
    assert cache_key(drawn) != cache_key(recorded)


def test_an_outside_clip_nobody_labelled_says_so(tmp_path):
    clip = write_video(tmp_path / "mine" / "waving.avi", list(range(0, 200, 5)))

    rows, _ = prepare(VideoSource(clip, EXTERNAL), CLASS_NAMES, size=32, root=tmp_path)

    assert rows["label"].unique().tolist() == [UNLABELLED]
    assert rows["group_id"].unique().tolist() == [EXTERNAL]
    assert rows["video_id"].unique().tolist() == ["waving"]


def test_an_outside_clip_recorded_again_under_the_same_name_is_read_again(tmp_path):
    """Otherwise the new recording would be scored on the old one's frames."""
    clip = tmp_path / "mine" / "waving.avi"
    write_video(clip, [40] * 40)
    first, _ = prepare(VideoSource(clip, EXTERNAL), CLASS_NAMES, size=32, root=tmp_path)

    write_video(clip, [200] * 40)
    os.utime(clip, ns=(1, 1))
    second, _ = prepare(
        VideoSource(clip, EXTERNAL), CLASS_NAMES, size=32, root=tmp_path
    )

    assert shade_of(tmp_path, first["path"].iloc[0]) == pytest.approx(40, abs=2)
    assert shade_of(tmp_path, second["path"].iloc[0]) == pytest.approx(200, abs=2)


def test_dataset_videos_carry_the_frames_their_manifest_records(tmp_path):
    annotations = tmp_path / "real_ground_test.txt"
    annotations.write_text(
        f"real/ground/Halt/{REAL_NAME}.mp4 3\n"
        "real/ground/Advance/S04b_10m_ground_label0_start50.mp4 0\n",
        encoding="utf-8",
    )
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    pd.DataFrame(
        {"video_id": [REAL_NAME, REAL_NAME, "other"], "frame_number": [9, 2, 5]}
    ).to_csv(manifests / "real_ground_test.csv", index=False)

    sources = dataset_videos(annotations, tmp_path / "data", manifests)

    assert [source.domain for source in sources] == ["real", "real"]
    assert [source.label for source in sources] == [3, 0]
    assert sources[0].path == tmp_path / "data" / f"real/ground/Halt/{REAL_NAME}.mp4"
    assert sources[0].frame_numbers == (9, 2)
    assert sources[1].frame_numbers == ()


def test_dataset_videos_without_a_manifest_will_be_drawn(tmp_path):
    annotations = tmp_path / "real_ground_test.txt"
    annotations.write_text(f"real/ground/Halt/{REAL_NAME}.mp4 3\n", encoding="utf-8")

    sources = dataset_videos(annotations, tmp_path / "data", tmp_path / "nowhere")

    assert sources[0].frame_numbers == ()
