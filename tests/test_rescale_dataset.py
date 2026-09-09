from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from manifest import mask_path_for, sized_path_for
from rescale_dataset import (
    parse_args,
    rescaled_manifest_name,
    save_frames,
    source_video,
)


def test_parse_args_reads_the_manifest_and_the_size():
    args = parse_args(["--manifest", "real_ground_test.csv", "--size", "640"])

    assert args.manifest == "real_ground_test.csv"
    assert args.size == 640


def test_parse_args_requires_the_manifest():
    """The manifest is the whole specification of a pass, not a default."""
    with pytest.raises(SystemExit):
        parse_args(["--size", "640"])


def test_parse_args_requires_the_size():
    with pytest.raises(SystemExit):
        parse_args(["--manifest", "real_ground_test.csv"])


def test_a_copy_is_written_beside_the_frames_it_came_from():
    frame = Path("data/frames/syn/Advance/Scene1_Advance_f0007.jpg")

    assert sized_path_for(frame, 640) == Path(
        "data/frames/640/syn/Advance/Scene1_Advance_f0007.jpg"
    )


def test_the_size_every_manifest_points_at_leaves_the_path_alone():
    """A path that already names the default tree is the copy of itself."""
    frame = Path("data/frames/syn/Advance/Scene1_Advance_f0007.jpg")

    assert sized_path_for(frame, 256) == frame


def test_the_silhouette_tree_mirrors_the_frames_at_any_size():
    """mask_path_for has to keep working without being told sizes exist."""
    frame = Path("data/frames/syn/Advance/Scene1_Advance_f0007.jpg")

    assert mask_path_for(sized_path_for(frame, 640)) == Path(
        "data/masks/640/syn/Advance/Scene1_Advance_f0007.png"
    )


def test_the_copy_gets_a_manifest_of_its_own():
    """Sharing one would let a resumed pass read its source's videos as done."""
    assert rescaled_manifest_name("syn_ground_train.csv", 640) == (
        "syn_ground_train_640.csv"
    )
    assert rescaled_manifest_name("syn_ground_train_idle.csv", 640) == (
        "syn_ground_train_idle_640.csv"
    )


def test_the_source_video_is_found_from_any_of_its_rows():
    row = pd.Series(
        {
            "domain": "real",
            "class_name": "Halt",
            "video_id": "S00_10m_ground_label5_start653",
        }
    )

    assert source_video(row) == Path(
        "data/real/ground/Halt/S00_10m_ground_label5_start653.mp4"
    )


def gradient(side):
    """One frame with detail at every scale, so resampling has work to do."""
    row = np.linspace(0, 255, side, dtype=np.uint8)
    image = np.repeat(row[None, :], side, axis=0)

    return np.dstack([image] * 3)


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

    save_frames([gradient(640)], [path], 256)

    assert path.exists()


def test_a_frame_without_a_path_to_write_to_is_refused(tmp_path):
    """Silently dropping one would leave a manifest row with no file."""
    with pytest.raises(ValueError):
        save_frames([gradient(640), gradient(640)], [tmp_path / "one.jpg"], 256)
