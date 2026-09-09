from pathlib import Path

from manifest import (
    FRAME_SIZE,
    mask_path_for,
    sized_path_for,
    stored_size_for,
    video_metadata,
    with_idle_class,
)


def test_synthetic_scene_number_determines_view():
    meta = video_metadata(
        Path("data/syn/ground/Halt/Scene26_386_Halt_4_2_2022_15_1_38.mp4"), "syn"
    )
    assert meta.video_id == "Scene26_386_Halt_4_2_2022_15_1_38"
    assert meta.group_id == "Scene26"
    assert meta.view == 2
    assert meta.is_frontal is False


def test_real_videos_have_no_scene_viewpoint():
    meta = video_metadata(
        Path("data/real/ground/MoveForward/S04_10m_ground_label4_start4837.mp4"), "real"
    )
    assert meta.video_id == "S04_10m_ground_label4_start4837"
    assert meta.group_id == "S04"
    assert meta.view is None
    assert meta.is_frontal is True


def test_second_session_maps_to_the_same_subject():
    """S04 and S04b are one person recorded twice, in different clothing.

    Treating them as two subjects would let the same person appear on both
    sides of a subject-wise split.
    """
    meta = video_metadata(
        Path("data/real/ground/FollowMe/S04b_10m_ground_label6_start2098.mp4"), "real"
    )
    assert meta.video_id == "S04b_10m_ground_label6_start2098"
    assert meta.group_id == "S04"
    assert meta.view is None
    assert meta.is_frontal is True


def test_mask_path_mirrors_the_frame_tree():
    assert mask_path_for(Path("data/frames/syn/Halt/Scene1_x_f0007.jpg")) == Path(
        "data/masks/syn/Halt/Scene1_x_f0007.png"
    )


def test_mask_path_accepts_a_string_as_a_manifest_stores_it():
    assert mask_path_for("data/frames/real/Rally/S02_y_f0012.jpg") == Path(
        "data/masks/real/Rally/S02_y_f0012.png"
    )


def test_the_idle_class_is_added_past_the_last_gesture():
    """Sitting past them is what leaves a table read without it unchanged."""
    names = with_idle_class({0: "Advance", 1: "Attention"})

    assert names == {0: "Advance", 1: "Attention", 7: "Idle"}


def test_adding_the_idle_class_leaves_the_mapping_it_was_given():
    """A caller still working in seven classes has to keep having seven."""
    seven = {0: "Advance", 1: "Attention"}

    with_idle_class(seven)

    assert seven == {0: "Advance", 1: "Attention"}


def test_a_size_reads_back_from_where_the_frame_sits():
    """Naming a tree is how a run picks a resolution, so the path is the record."""
    frame = Path("data/frames/syn/Advance/Scene1_Advance_f0007.jpg")

    assert stored_size_for(sized_path_for(frame, 640)) == 640


def test_a_path_without_a_size_names_the_tree_extraction_wrote():
    frame = Path("data/frames/syn/Advance/Scene1_Advance_f0007.jpg")

    assert stored_size_for(frame) == FRAME_SIZE
