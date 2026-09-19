import numpy as np
import pandas as pd
import pytest

from splits import (
    EDGE_FRAMES,
    add_idle_rows,
    keep_views,
    neighbours_in,
    sample_videos,
    select_frames,
    split_by_group,
    split_by_scene,
    with_neighbour,
    with_neighbours,
    with_previous_anchor,
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


def _dense(videos=2, frames=6, start=10):
    """A dense manifest: contiguous frame numbers, one path each."""
    return pd.DataFrame(
        {
            "video_id": [f"v{v}" for v in range(videos) for _ in range(frames)],
            "frame_number": list(range(start, start + frames)) * videos,
            "path": [
                f"frames/v{v}_f{n:04d}.jpg"
                for v in range(videos)
                for n in range(start, start + frames)
            ],
        }
    )


def test_a_neighbour_sits_the_asked_stride_back():
    annotated = with_neighbour(_dense(), stride=2)

    row = annotated[annotated["frame_number"] == 14].iloc[0]
    assert row["neighbour_path"].endswith("_f0012.jpg")


def test_a_neighbour_before_the_window_falls_back_to_its_first_frame():
    """The first frame then points at itself, and its difference is zero.

    That is the honest reading: no earlier frame exists, so no movement was
    measured — rather than a difference against some other video's frame.
    """
    annotated = with_neighbour(_dense(start=10), stride=3)
    early = annotated[annotated["frame_number"] < 13]

    assert set(early["neighbour_path"]) == {
        path for path in early["path"] if path.endswith("_f0010.jpg")
    }


def test_a_neighbour_never_comes_from_another_video():
    annotated = with_neighbour(_dense(), stride=4)

    for _, row in annotated.iterrows():
        assert row["neighbour_path"].split("_f")[0].endswith(row["video_id"])


def test_a_stride_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        with_neighbour(_dense(), stride=0)


def _served(frames):
    """Rows as a run would be served them: one video, the given frame numbers."""
    return pd.DataFrame(
        {
            "video_id": ["v0"] * len(frames),
            "frame_number": frames,
            "path": [f"frames/v0_f{n:04d}.jpg" for n in frames],
        }
    )


def test_the_previous_anchor_is_the_row_served_before_it():
    """No dense lookup: the neighbour is a frame the run already holds."""
    annotated = with_previous_anchor(_served([4, 11, 19]))

    assert list(annotated["neighbour_path"]) == [
        "frames/v0_f0004.jpg",
        "frames/v0_f0004.jpg",
        "frames/v0_f0011.jpg",
    ]


def test_the_first_anchor_points_at_itself():
    """Same convention as the strided lookup: nothing was served before it."""
    annotated = with_previous_anchor(_served([4, 11]))

    assert annotated.iloc[0]["neighbour_path"] == annotated.iloc[0]["path"]


def test_a_repeated_anchor_is_skipped_rather_than_differenced_against_itself():
    """The row before the repeat is the one that counts.

    Extraction writes some frames twice, and a difference of exactly zero is
    not a weak signal but a distinctive one the network could read a class off.
    """
    annotated = with_previous_anchor(_served([4, 9, 9, 15]))

    assert list(annotated["neighbour_path"])[3] == "frames/v0_f0009.jpg"
    assert list(annotated["neighbour_path"])[2] == "frames/v0_f0009.jpg"


def test_a_previous_anchor_never_comes_from_another_video():
    annotated = with_previous_anchor(_dense())

    for _, row in annotated.iterrows():
        assert row["neighbour_path"].split("_f")[0].endswith(row["video_id"])


def test_the_previous_anchor_keeps_the_row_order_it_was_given():
    """Selection decides the order; naming a neighbour must not re-decide it."""
    rows = _served([19, 4, 11])
    annotated = with_previous_anchor(rows)

    assert list(annotated["frame_number"]) == [19, 4, 11]
    assert list(annotated["neighbour_path"]) == [
        "frames/v0_f0011.jpg",
        "frames/v0_f0004.jpg",
        "frames/v0_f0004.jpg",
    ]


def test_a_repeated_frame_number_is_refused():
    """A dense pass never writes one, and it would make the neighbour ambiguous."""
    doubled = pd.concat([_dense(videos=1), _dense(videos=1)])

    with pytest.raises(ValueError, match="same frame number twice"):
        with_neighbour(doubled, stride=1)


def test_a_neighbour_is_looked_up_somewhere_other_than_it_lands():
    """Their neighbours are not among the rows a run trains on.

    Looking up in the dense pass while attaching to the served rows is what
    lets a run reading differences train on exactly the frames a run without
    them trains on.
    """
    dense = _dense(videos=1, frames=20, start=0)
    chosen = dense[dense["frame_number"].isin([5, 12])]

    annotated = with_neighbour(chosen, stride=2, dense=dense)

    assert list(annotated["path"]) == list(chosen["path"])
    assert [p.rsplit("_f", 1)[1] for p in annotated["neighbour_path"]] == [
        "0003.jpg",
        "0010.jpg",
    ]


def test_rows_whose_neighbour_is_nowhere_are_refused():
    """Silently dropping them would leave a run short of frames it asked for."""
    dense = _dense(videos=1, frames=5, start=0)
    stranger = dense.copy()
    stranger["video_id"] = "elsewhere"

    with pytest.raises(ValueError, match="no neighbour for"):
        with_neighbour(stranger, stride=1, dense=dense)


def views_manifest(scenes_per_view=4):
    """Six camera positions with several scenes each, as the real manifest has.

    Scenes come in consecutive blocks of six and the position within a block
    fixes the viewpoint, which is how ``view`` is derived from the scene index.
    """
    scenes = range(6 * scenes_per_view)
    return pd.DataFrame(
        {
            "video_id": [f"Scene{scene}_x" for scene in scenes for _ in range(2)],
            "group_id": [f"Scene{scene}" for scene in scenes for _ in range(2)],
            "view": [scene % 6 for scene in scenes for _ in range(2)],
            "label": [0, 1] * len(scenes),
        }
    )


def test_keep_views_keeps_only_the_positions_named():
    kept = keep_views(views_manifest(), [3, 4])

    assert sorted(kept["view"].unique()) == [3, 4]
    assert kept["group_id"].nunique() == 8
    assert len(kept) == 16


def test_keep_views_refuses_a_position_the_manifest_does_not_have():
    """An absent name would otherwise hand back fewer rows than asked for.

    On a real manifest, whose viewpoint column is empty, it would hand back
    none at all and fail much later without saying why.
    """
    with pytest.raises(ValueError, match="viewpoints not in the manifest"):
        keep_views(views_manifest(), [4, 9])


def test_keep_views_refuses_an_empty_request():
    with pytest.raises(ValueError, match="no viewpoints named"):
        keep_views(views_manifest(), [])


def test_keep_views_refuses_a_manifest_with_no_viewpoints():
    real = views_manifest().assign(view=None)

    with pytest.raises(ValueError, match="viewpoints not in the manifest"):
        keep_views(real, [3, 4])


def test_keep_views_leaves_the_split_able_to_hold_one_scene_per_view():
    """Filtering runs before the split, so the split sees only what is left.

    Two viewpoints means two held-out scenes rather than six, which is the
    same share within the views the run actually trains on.
    """
    kept = keep_views(views_manifest(), [3, 4])

    train, validation = split_by_scene(kept)

    assert sorted(validation["view"].unique()) == [3, 4]
    assert validation["group_id"].nunique() == 2
    assert train.empty or set(train["view"]).issubset({3, 4})


def dense_run():
    """One video, six consecutive frames, as a dense pass writes them."""
    return pd.DataFrame(
        {
            "video_id": ["v0"] * 6,
            "frame_number": list(range(6)),
            "path": [f"f{index}.jpg" for index in range(6)],
        }
    )


def test_one_spacing_writes_the_column_a_single_spacing_always_wrote():
    """Every measured cell of the difference family reads this column.

    Renaming it, or reordering the table around it, would make those cells
    unreproducible.
    """
    rows = dense_run()

    single = with_neighbour(rows, 2)
    plural = with_neighbours(rows, [2])

    pd.testing.assert_frame_equal(single, plural)


def test_each_spacing_gets_a_column_of_its_own():
    rows = dense_run()

    annotated = with_neighbours(rows, [1, 3])

    assert neighbours_in(annotated) == ["neighbour_path", "neighbour_path_1"]
    assert annotated["neighbour_path"].tolist() == ["f0.jpg"] + [
        f"f{index}.jpg" for index in range(5)
    ]
    assert annotated["neighbour_path_1"].tolist() == ["f0.jpg"] * 4 + [
        "f1.jpg",
        "f2.jpg",
    ]


def test_a_repeated_spacing_is_the_capacity_control_and_is_allowed():
    """Adding spacings widens the stem as well as adding scales.

    Stacking one spacing as many times carries no second signal and holds the
    width fixed, which is what separates the two.
    """
    annotated = with_neighbours(dense_run(), [2, 2, 2])

    columns = neighbours_in(annotated)
    assert len(columns) == 3
    for column in columns[1:]:
        assert annotated[column].tolist() == annotated[columns[0]].tolist()


def test_naming_no_spacing_is_refused():
    with pytest.raises(ValueError, match="no spacings named"):
        with_neighbours(dense_run(), [])


def test_a_manifest_with_no_neighbour_column_names_none():
    assert neighbours_in(dense_run()) == []
