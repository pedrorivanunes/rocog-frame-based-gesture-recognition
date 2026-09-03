from pathlib import Path

import cv2
import numpy as np
import pytest

from frame_extraction import read_frames_at


def write_video(path: Path, shades: list[int], size: int = 32) -> Path:
    """Write a video whose frames are solid greys, one per shade.

    Lossless FFV1 so that a frame reads back as the value it was written with,
    which is what lets a test tell the frames apart by index.
    """
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (size, size)
    )
    for shade in shades:
        writer.write(np.full((size, size, 3), shade, dtype=np.uint8))
    writer.release()

    return path


@pytest.fixture
def video(tmp_path):
    return write_video(tmp_path / "clip.avi", [10, 40, 70, 100, 130, 160, 190, 220])


def test_read_frames_at_returns_the_frames_asked_for(video):
    frames = read_frames_at(video, [0, 3, 7])

    assert [int(frame[0, 0, 0]) for frame in frames] == [10, 100, 220]


def test_read_frames_at_keeps_the_order_requested(video):
    """Out-of-order numbers come back out of order, not sorted."""
    frames = read_frames_at(video, [5, 1, 6])

    assert [int(frame[0, 0, 0]) for frame in frames] == [160, 40, 190]


def test_read_frames_at_returns_a_repeated_number_once_each(video):
    """Short clips round two segments onto one frame, and both rows want it."""
    frames = read_frames_at(video, [2, 2, 4])

    assert [int(frame[0, 0, 0]) for frame in frames] == [70, 70, 130]


def test_read_frames_at_returns_one_frame_per_number(video):
    assert len(read_frames_at(video, [0, 1, 2, 3])) == 4


def test_read_frames_at_asks_for_nothing_and_gets_nothing(video):
    assert read_frames_at(video, []) == []


def test_read_frames_at_rejects_a_missing_video(tmp_path):
    with pytest.raises(RuntimeError, match="could not open"):
        read_frames_at(tmp_path / "absent.avi", [0])


def test_read_frames_at_rejects_a_frame_past_the_end(video):
    """Silently returning fewer frames would misalign a mask from its image."""
    with pytest.raises(RuntimeError, match="failure when reading frame"):
        read_frames_at(video, [99])
