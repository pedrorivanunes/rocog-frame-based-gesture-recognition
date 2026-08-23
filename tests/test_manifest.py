from pathlib import Path

from manifest import video_metadata


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
