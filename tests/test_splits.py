import pandas as pd
import pytest

from splits import split_by_scene

REAL_SHAPE = [7, 7, 7, 5, 6, 8]


def synthetic_manifest(scenes_by_view, videos_per_scene=2, frames_per_video=3):
    """Build a manifest following the RoCoG-v2 synthetic layout.

    No files are created and no frames exist: split_by_scene reads columns and
    never opens anything. Scene numbers are laid out so that ``Scene % 6``
    recovers the viewpoint, matching how the dataset numbers them.

    Args:
        scenes_by_view: How many scenes each of the six viewpoints holds. The
            training subset is uneven — five scenes for view 3, eight for view
            5 — and the split has to hold up under that.
        videos_per_scene: Videos each scene contributes.
        frames_per_video: Rows each video contributes, as extraction writes one
            per sampled frame.

    Returns:
        One row per frame, carrying the columns the split reads.
    """
    rows = []
    for view, scene_count in enumerate(scenes_by_view):
        for block in range(scene_count):
            scene = f"Scene{block * 6 + view}"
            for video in range(videos_per_scene):
                video_id = f"{scene}_{video}_Halt_1_1_2026_0_0_0"
                rows.extend(
                    {
                        "video_id": video_id,
                        "group_id": scene,
                        "view": view,
                        "frame_number": frame,
                    }
                    for frame in range(frames_per_video)
                )

    return pd.DataFrame(rows)


def test_no_scene_appears_on_both_sides():
    """A scene on both sides leaks silently, inflating validation accuracy."""
    manifest = synthetic_manifest(REAL_SHAPE)

    train, validation = split_by_scene(manifest)

    assert set(train["group_id"]) & set(validation["group_id"]) == set()


def test_every_viewpoint_survives_on_both_sides():
    """A missing viewpoint blinds validation to an angle the model trains on."""
    manifest = synthetic_manifest(REAL_SHAPE)

    train, validation = split_by_scene(manifest)

    assert set(train["view"]) == set(range(6))
    assert set(validation["view"]) == set(range(6))


def test_every_row_lands_on_exactly_one_side():
    """Selecting on the deduplicated frame would drop rows without complaint."""
    manifest = synthetic_manifest(REAL_SHAPE)

    train, validation = split_by_scene(manifest)

    assert len(train) + len(validation) == len(manifest)


def test_each_view_contributes_the_requested_number_of_scenes():
    manifest = synthetic_manifest(REAL_SHAPE)

    _, validation = split_by_scene(manifest, scenes_per_view=2)

    assert validation["group_id"].nunique() == 12


def test_the_same_seed_splits_the_same_way():
    """Two runs meant to be compared have to have trained on the same data."""
    manifest = synthetic_manifest(REAL_SHAPE)

    _, first = split_by_scene(manifest, seed=1)
    _, second = split_by_scene(manifest, seed=1)
    _, other = split_by_scene(manifest, seed=2)

    assert set(first["group_id"]) == set(second["group_id"])
    assert set(first["group_id"]) != set(other["group_id"])


def test_a_view_with_too_few_scenes_is_rejected():
    """Losing the tight viewpoint would surface much later as an odd result."""
    manifest = synthetic_manifest(REAL_SHAPE)

    with pytest.raises(RuntimeError):
        split_by_scene(manifest, scenes_per_view=6)
