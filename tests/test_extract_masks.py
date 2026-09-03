from pathlib import Path

import cv2
import numpy as np
import pytest

from extract_masks import mask_path_for, parse_args, person_silhouette, save_silhouettes


def segmentation_frame(person_colour: tuple[int, int, int]) -> np.ndarray:
    """A tiny mask frame: green terrain, bright sky, and a person of some colour.

    Colours are BGR, as OpenCV reads them.
    """
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[:2, :] = (60, 200, 30)  # terrain: green dominant
    frame[2, :] = (247, 255, 129)  # sky: bright, green still dominant
    frame[3, :] = person_colour

    return frame


def test_person_silhouette_finds_the_red_convention():
    silhouette = person_silhouette(segmentation_frame((3, 0, 232)))

    assert silhouette[3].all()
    assert not silhouette[:3].any()


def test_person_silhouette_finds_the_blue_convention():
    """The person's colour is not constant across the dataset."""
    silhouette = person_silhouette(segmentation_frame((243, 0, 1)))

    assert silhouette[3].all()
    assert not silhouette[:3].any()


def test_person_silhouette_keeps_sky_out():
    """Sky is bright in every channel; a plain not-green test would claim it."""
    frame = segmentation_frame((3, 0, 232))

    assert not person_silhouette(frame)[2].any()


def test_person_silhouette_returns_one_value_per_pixel():
    silhouette = person_silhouette(segmentation_frame((3, 0, 232)))

    assert silhouette.shape == (4, 4)
    assert silhouette.dtype == bool


def test_mask_path_mirrors_the_frame_tree():
    assert mask_path_for(Path("data/frames/syn/Halt/Scene1_x_f0007.jpg")) == Path(
        "data/masks/syn/Halt/Scene1_x_f0007.png"
    )


def test_mask_path_accepts_a_string_as_a_manifest_stores_it():
    assert mask_path_for("data/frames/real/Rally/S02_y_f0012.jpg") == Path(
        "data/masks/real/Rally/S02_y_f0012.png"
    )


def test_save_silhouettes_writes_only_two_values(tmp_path):
    """Resampling that averages would invent greys the source never had."""
    silhouette = np.zeros((8, 8), dtype=bool)
    silhouette[2:6, 2:6] = True
    path = tmp_path / "syn" / "Halt" / "a_f0001.png"

    save_silhouettes([silhouette], [path], size=32)

    written = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert written.shape == (32, 32)
    assert set(np.unique(written).tolist()) == {0, 255}


def test_save_silhouettes_preserves_the_person_share(tmp_path):
    """A quarter of the pixels going in stays a quarter coming out."""
    silhouette = np.zeros((8, 8), dtype=bool)
    silhouette[:4, :4] = True
    path = tmp_path / "a_f0001.png"

    save_silhouettes([silhouette], [path], size=64)

    written = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert (written == 255).mean() == pytest.approx(0.25)


def test_save_silhouettes_creates_missing_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "a_f0001.png"

    save_silhouettes([np.ones((4, 4), dtype=bool)], [path], size=8)

    assert path.exists()


def test_save_silhouettes_rejects_mismatched_lengths(tmp_path):
    with pytest.raises(ValueError):
        save_silhouettes(
            [np.ones((4, 4), dtype=bool), np.ones((4, 4), dtype=bool)],
            [tmp_path / "only_one.png"],
        )


def test_parse_args_requires_the_manifest():
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_defaults_the_size_to_the_frame_size():
    args = parse_args(["--manifest", "syn_ground_train.csv"])

    assert args.manifest == "syn_ground_train.csv"
    assert args.size == 256
