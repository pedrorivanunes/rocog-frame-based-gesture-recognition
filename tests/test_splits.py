import numpy as np
import pandas as pd
import pytest

from splits import (
    EDGE_FRAMES,
    add_idle_rows,
    sample_videos,
    select_frames,
    split_by_group,
    split_by_scene,
)

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


def test_split_by_group_holds_out_the_groups_it_is_given():
    manifest = pd.DataFrame(
        {"group_id": ["S01", "S01", "S02", "S03"], "video_id": list("abcd")}
    )

    train, validation = split_by_group(manifest, ["S02"])

    assert sorted(train.group_id.unique()) == ["S01", "S03"]
    assert sorted(validation.group_id.unique()) == ["S02"]


def test_split_by_group_keeps_no_group_on_both_sides():
    manifest = pd.DataFrame(
        {"group_id": ["S01", "S02", "S03"], "video_id": list("abc")}
    )

    train, validation = split_by_group(manifest, ["S02", "S03"])

    assert not set(train.group_id) & set(validation.group_id)


def test_split_by_group_rejects_a_group_the_manifest_does_not_hold():
    # A name that matches nothing would otherwise leave validation empty, and
    # that failure surfaces much later without saying what caused it.
    manifest = pd.DataFrame({"group_id": ["S01", "S02"], "video_id": list("ab")})

    with pytest.raises(ValueError, match="not in the manifest"):
        split_by_group(manifest, ["S99"])


def test_split_by_group_rejects_holding_out_nothing():
    manifest = pd.DataFrame({"group_id": ["S01"], "video_id": ["a"]})

    with pytest.raises(ValueError):
        split_by_group(manifest, [])


def test_split_by_group_rejects_holding_out_every_group():
    manifest = pd.DataFrame({"group_id": ["S01", "S02"], "video_id": list("ab")})

    with pytest.raises(ValueError, match="nothing to train on"):
        split_by_group(manifest, ["S01", "S02"])


def test_the_scene_split_refuses_a_manifest_that_carries_no_viewpoint():
    # The real manifests have no viewpoint, so grouping by one finds nothing and
    # the scene split has no scenes to draw. Failing here is the point: the
    # alternative is an empty validation set, which only fails once a batch of
    # nothing reaches the model and says nothing about the cause.
    manifest = pd.DataFrame(
        {
            "group_id": ["S01", "S02", "S03"],
            "video_id": list("abc"),
            "view": [np.nan] * 3,
        }
    )

    with pytest.raises(ValueError):
        split_by_scene(manifest)

    _, validation = split_by_group(manifest, ["S02"])
    assert len(validation) == 1


def video_manifest(videos=3, frames_per_video=24):
    """Build a manifest holding whole videos, frames consecutive and in order.

    That layout is not decoration: select_frames trims by rank, so it reads the
    order the rows sit in rather than any column. A manifest built any other way
    would test something the extraction never produces.

    Args:
        videos: How many videos the manifest holds.
        frames_per_video: Rows each video contributes.

    Returns:
        One row per frame, carrying ``video_id`` and the frame's position along
        the gesture window, evenly spaced as extraction spaces them.
    """
    return pd.DataFrame(
        [
            {
                "video_id": f"Scene0_{video}_Halt_1_1_2026_0_0_0",
                "frame_number": frame,
                "position": (frame + 1) / frames_per_video,
            }
            for video in range(videos)
            for frame in range(frames_per_video)
        ]
    )


def test_full_window_keeps_every_frame():
    """The default has to leave earlier runs measuring exactly what they measured."""
    manifest = video_manifest()

    assert len(select_frames(manifest, "full")) == len(manifest)


def test_middle_drops_the_same_count_from_both_ends():
    """Trimming one end only would shift the window instead of narrowing it."""
    manifest = video_manifest(frames_per_video=24)

    kept = select_frames(manifest, "middle")

    for _, frames in kept.groupby("video_id"):
        assert frames["frame_number"].min() == EDGE_FRAMES
        assert frames["frame_number"].max() == 24 - EDGE_FRAMES - 1


def test_middle_leaves_every_video_the_same_number_of_frames():
    """The sampler cuts each video into equal blocks and refuses uneven counts."""
    manifest = video_manifest(videos=5)

    kept = select_frames(manifest, "middle")

    assert set(kept.groupby("video_id").size()) == {24 - 2 * EDGE_FRAMES}


def test_scattered_keeps_as_many_frames_as_middle():
    """The control exists to separate which frames from how many."""
    manifest = video_manifest(videos=5)

    middle = select_frames(manifest, "middle")
    scattered = select_frames(manifest, "scattered")

    assert scattered.groupby("video_id").size().tolist() == (
        middle.groupby("video_id").size().tolist()
    )


def test_scattered_reaches_the_ends_middle_removes():
    """A control that avoided the ends too would confound the same thing again."""
    manifest = video_manifest(videos=40)

    kept = select_frames(manifest, "scattered")

    assert kept["frame_number"].min() == 0
    assert kept["frame_number"].max() == 23


def test_scattered_moves_with_the_seed():
    """Three repetitions should meet three subsets, not agree on one lucky draw."""
    manifest = video_manifest(videos=5)

    first = select_frames(manifest, "scattered", seed=0)
    second = select_frames(manifest, "scattered", seed=1)

    assert not first.index.equals(second.index)


def test_a_window_returns_rows_in_manifest_order():
    """The sampler reads consecutive rows as one video's frames, in time order."""
    manifest = video_manifest(videos=4)

    kept = select_frames(manifest, "scattered")

    assert kept.index.tolist() == sorted(kept.index)


def test_an_unknown_window_is_refused():
    """A typo that silently trained on everything would be invisible in the log."""
    with pytest.raises(ValueError, match="not one of"):
        select_frames(video_manifest(), "midle")


def test_a_video_too_short_to_trim_is_refused():
    """Trimming eight from a video of eight would leave a run with no frames."""
    manifest = video_manifest(frames_per_video=2 * EDGE_FRAMES)

    with pytest.raises(ValueError, match="too few"):
        select_frames(manifest, "middle")


def labelled_rows(video, label, class_name, frames, position):
    """Build the manifest columns that adding the idle rows reads."""
    return pd.DataFrame(
        {
            "video_id": [video] * frames,
            "label": [label] * frames,
            "class_name": [class_name] * frames,
            "position": [position] * frames,
        }
    )


def test_idle_rows_lose_the_gesture_they_came_from():
    """What is asked is whether a body gestures, not which gesture follows."""
    gesture = labelled_rows("v0", 4, "Halt", 3, 0.5)
    idle = labelled_rows("v0", 4, "Halt", 2, -0.2)

    combined = add_idle_rows(gesture, idle)

    assert list(combined["label"]) == [4, 4, 4, 7, 7]
    assert list(combined["class_name"]) == ["Halt"] * 3 + ["Idle"] * 2


def test_the_gesture_rows_keep_their_label():
    gesture = labelled_rows("v0", 4, "Halt", 3, 0.5)
    idle = labelled_rows("v0", 4, "Halt", 2, -0.2)

    combined = add_idle_rows(gesture, idle)

    assert list(combined.iloc[:3]["label"]) == [4, 4, 4]


def test_the_combined_rows_are_numbered_from_zero():
    """The dataset serves rows by position, so a sampler names positions in this."""
    gesture = labelled_rows("v0", 4, "Halt", 3, 0.5).iloc[[1, 2]]
    idle = labelled_rows("v0", 4, "Halt", 2, -0.2).iloc[[1]]

    combined = add_idle_rows(gesture, idle)

    assert list(combined.index) == [0, 1, 2]


def test_adding_idle_rows_leaves_the_table_it_was_given():
    idle = labelled_rows("v0", 4, "Halt", 2, -0.2)

    add_idle_rows(labelled_rows("v0", 4, "Halt", 3, 0.5), idle)

    assert list(idle["label"]) == [4, 4]


class TestSampleVideos:
    """Keeping a fraction of the videos, drawn so the fractions nest."""

    @staticmethod
    def manifest(videos_per_class=20, classes=7, frames=3):
        return pd.DataFrame(
            [
                {
                    "video_id": f"c{label}_v{video}",
                    "label": label,
                    "group_id": f"S{video % 4}",
                    "frame_number": frame,
                }
                for label in range(classes)
                for video in range(videos_per_class)
                for frame in range(frames)
            ]
        )

    def test_it_keeps_whole_videos(self):
        """A share of each video's frames would measure something else entirely."""
        rows = self.manifest()
        kept = sample_videos(rows, 0.5, seed=0)

        counts = kept.groupby("video_id").size()
        assert set(counts) == {3}

    def test_every_class_survives_the_smallest_fraction(self):
        """A draw that loses a gesture turns a data curve into a curve about luck."""
        rows = self.manifest()
        kept = sample_videos(rows, 0.01, seed=0)

        assert sorted(kept["label"].unique()) == list(range(7))

    def test_the_fractions_nest(self):
        """Everything a run at a tenth sees, the same run at a fifth sees too."""
        rows = self.manifest()
        small = set(sample_videos(rows, 0.1, seed=3)["video_id"])
        large = set(sample_videos(rows, 0.2, seed=3)["video_id"])

        assert small < large

    def test_a_different_seed_draws_differently(self):
        """The spread across seeds has to include the luck of the draw."""
        rows = self.manifest()
        first = set(sample_videos(rows, 0.25, seed=0)["video_id"])
        second = set(sample_videos(rows, 0.25, seed=1)["video_id"])

        assert first != second

    def test_the_whole_thing_comes_back_untouched(self):
        rows = self.manifest()

        assert sample_videos(rows, 1.0).equals(rows)

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
    def test_a_fraction_outside_the_range_is_refused(self, fraction):
        """Zero would hand back nothing and fail later, further from the cause."""
        with pytest.raises(ValueError, match="fraction must be"):
            sample_videos(self.manifest(), fraction)
