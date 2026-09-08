from pathlib import Path

import cv2
import numpy as np
import pytest

from frame_extraction import (
    extract_frames,
    extract_idle_frames,
    gesture_window,
    read_frames_at,
    read_metadata,
)


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


def write_window(video_path: Path, start_time: float, end_time: float) -> Path:
    """Write the metadata sibling that declares where a video's gesture sits.

    Only the two boundaries are read from it, but the encoding matters: the
    dataset's files declare utf-16 while holding utf-8, and the reader corrects
    that, so a test file has to carry the same mistake to exercise the same path.
    """
    xml_path = video_path.with_suffix(".xml")
    xml_path.write_bytes(
        f'<?xml version="1.0" encoding="utf-16"?>'
        f"<GestureVideo>"
        f"<startTime>{start_time}</startTime>"
        f"<endTime>{end_time}</endTime>"
        f"</GestureVideo>".encode()
    )

    return xml_path


@pytest.fixture
def annotated(tmp_path):
    """A sixty-two frame clip at 10 fps whose gesture runs from frame 5 to 59.

    Shades step by four, so the value a frame reads back with names the frame it
    is. The window divides into six segments on whole frames — 5, 14, 23, 32,
    41, 50, 59 — which lets a test say where a pick belongs without repeating
    the arithmetic that placed it.
    """
    video_path = write_video(tmp_path / "clip.avi", list(range(0, 248, 4)))
    write_window(video_path, 0.5, 5.9)

    return video_path


def test_read_metadata_parses_a_file_that_lies_about_its_encoding(annotated):
    """Following the declaration, the parser refuses the dataset's own files."""
    root = read_metadata(annotated.with_suffix(".xml"))

    assert root.findtext("startTime") == "0.5"


def test_the_window_is_read_from_the_metadata(annotated):
    assert gesture_window(annotated, 62.0, 10.0) == (5, 59)


def test_the_window_scales_with_the_frame_rate(annotated):
    """The metadata gives seconds, and the same instant is a different frame."""
    assert gesture_window(annotated, 124.0, 20.0) == (10, 118)


def test_a_boundary_between_two_frames_lands_on_the_earlier_one(tmp_path):
    video_path = write_video(tmp_path / "between.avi", list(range(0, 248, 4)))
    write_window(video_path, 0.55, 5.97)

    assert gesture_window(video_path, 62.0, 10.0) == (5, 59)


def test_a_video_without_metadata_is_gesture_from_end_to_end(video):
    """Real footage ships none, and is one gesture spanning the whole clip."""
    assert gesture_window(video, 8.0, 10.0) == (0, 7)


def test_an_end_past_the_last_frame_is_clamped_to_it(tmp_path):
    """Around 2% of synthetic annotations, Rally above all, end past the video."""
    video_path = write_video(tmp_path / "over.avi", list(range(0, 248, 4)))
    write_window(video_path, 0.5, 99.0)

    assert gesture_window(video_path, 62.0, 10.0) == (5, 61)


def test_the_number_of_frames_is_honored_exactly(annotated):
    assert len(extract_frames(annotated, 6, np.random.default_rng(0))) == 6


def test_every_frame_comes_from_inside_the_window(annotated):
    frames = extract_frames(annotated, 6, np.random.default_rng(0))

    assert all(5 <= frame.frame_number <= 59 for frame in frames)


def test_frames_arrive_in_increasing_order(annotated):
    frames = extract_frames(annotated, 6, np.random.default_rng(0))

    numbers = [frame.frame_number for frame in frames]
    assert numbers == sorted(numbers)


def test_one_frame_is_drawn_from_each_segment(annotated):
    """Covering the whole window is what separates this from six blind draws."""
    frames = extract_frames(annotated, 6, np.random.default_rng(0))

    boundaries = [5, 14, 23, 32, 41, 50, 59]
    segments = zip(boundaries, boundaries[1:], strict=False)
    for frame, (low, high) in zip(frames, segments, strict=True):
        assert low <= frame.frame_number <= high


def test_a_pick_moves_within_its_segment(annotated):
    """A fixed offset would lock the sample onto one phase of a repeating gesture."""
    first_picks = {
        extract_frames(annotated, 6, np.random.default_rng(seed))[0].frame_number
        for seed in range(20)
    }

    assert len(first_picks) > 1


def test_position_is_where_the_frame_sits_in_the_window(annotated):
    frames = extract_frames(annotated, 6, np.random.default_rng(0))

    for frame in frames:
        assert frame.position == (frame.frame_number - 5) / 54
        assert 0.0 <= frame.position <= 1.0


def test_the_frame_a_number_names_is_the_frame_returned(annotated):
    """A record whose image came from elsewhere is wrong past any later check."""
    frames = extract_frames(annotated, 6, np.random.default_rng(0))

    for frame in frames:
        assert int(frame.frame[0, 0, 0]) == frame.frame_number * 4


def test_the_same_seed_samples_the_same_frames(annotated):
    first = extract_frames(annotated, 6, np.random.default_rng(3))
    second = extract_frames(annotated, 6, np.random.default_rng(3))

    assert [f.frame_number for f in first] == [f.frame_number for f in second]


def test_a_different_seed_samples_different_frames(annotated):
    first = extract_frames(annotated, 6, np.random.default_rng(3))
    second = extract_frames(annotated, 6, np.random.default_rng(4))

    assert [f.frame_number for f in first] != [f.frame_number for f in second]


def test_a_window_shorter_than_the_sample_repeats_frames(tmp_path):
    """Short clips round two segments onto one frame, and the count still holds."""
    video_path = write_video(tmp_path / "short.avi", list(range(0, 248, 4)))
    write_window(video_path, 0.5, 0.7)

    frames = extract_frames(video_path, 8, np.random.default_rng(0))

    numbers = [frame.frame_number for frame in frames]
    assert len(numbers) == 8
    assert len(set(numbers)) < 8


def test_a_window_of_no_length_is_rejected(tmp_path):
    """Dividing by it would make every position infinite rather than fail."""
    video_path = write_video(tmp_path / "empty.avi", list(range(0, 248, 4)))
    write_window(video_path, 0.5, 0.5)

    with pytest.raises(RuntimeError, match="length zero"):
        extract_frames(video_path, 6, np.random.default_rng(0))


def test_extract_frames_rejects_a_missing_video(tmp_path):
    with pytest.raises(RuntimeError, match="could not open"):
        extract_frames(tmp_path / "absent.avi", 6, np.random.default_rng(0))


def test_the_frames_a_seed_draws_are_pinned(annotated):
    """Checking a seed only against itself would let a rewrite pass unnoticed.

    Extraction is reproducible from a seed, and the manifests on disk record
    the frames one particular sampler drew. Any change to how a pick is placed
    inside its segment silently moves every frame a future extraction takes,
    and the images already written stop being the ones the code would produce.
    Pinning the draw is what turns that into a failing test.
    """
    frames = extract_frames(annotated, 6, np.random.default_rng(0))

    assert [frame.frame_number for frame in frames] == [11, 16, 23, 32, 48, 58]


@pytest.fixture
def late_gesture(tmp_path):
    """A clip whose gesture starts at frame 30, leaving thirty frames before it."""
    video_path = write_video(tmp_path / "late.avi", list(range(0, 248, 4)))
    write_window(video_path, 3.0, 5.9)

    return video_path


def test_idle_frames_all_come_from_before_the_gesture(late_gesture):
    frames = extract_idle_frames(late_gesture, 4, np.random.default_rng(0))

    assert all(frame.frame_number < 30 for frame in frames)


def test_the_number_of_idle_frames_is_honored_exactly(late_gesture):
    assert len(extract_idle_frames(late_gesture, 7, np.random.default_rng(0))) == 7


def test_idle_frames_arrive_in_increasing_order(late_gesture):
    frames = extract_idle_frames(late_gesture, 4, np.random.default_rng(0))

    numbers = [frame.frame_number for frame in frames]
    assert numbers == sorted(numbers)


def test_idle_frames_are_spread_across_the_stretch(late_gesture):
    """Four picks clustered at one end would be four copies of one instant."""
    frames = extract_idle_frames(late_gesture, 4, np.random.default_rng(0))

    numbers = [frame.frame_number for frame in frames]
    assert numbers[0] <= 8
    assert numbers[-1] >= 21


def test_idle_position_falls_below_the_start_of_the_gesture(late_gesture):
    """Leaving the zero to one range is what marks the row as idle."""
    frames = extract_idle_frames(late_gesture, 4, np.random.default_rng(0))

    for frame in frames:
        assert frame.position < 0.0
        assert frame.position == (frame.frame_number - 30) / 29


def test_the_frame_an_idle_number_names_is_the_frame_returned(late_gesture):
    frames = extract_idle_frames(late_gesture, 4, np.random.default_rng(0))

    for frame in frames:
        assert int(frame.frame[0, 0, 0]) == frame.frame_number * 4


def test_the_same_seed_samples_the_same_idle_frames(late_gesture):
    first = extract_idle_frames(late_gesture, 4, np.random.default_rng(3))
    second = extract_idle_frames(late_gesture, 4, np.random.default_rng(3))

    assert [f.frame_number for f in first] == [f.frame_number for f in second]


def test_a_video_without_metadata_has_nothing_before_its_gesture(video):
    """Real footage ships none, and is gesture from its very first frame."""
    with pytest.raises(RuntimeError, match="before the gesture"):
        extract_idle_frames(video, 4, np.random.default_rng(0))


def test_asking_for_more_idle_frames_than_exist_is_rejected(annotated):
    """Returning fewer would leave that video weighted below every other."""
    with pytest.raises(RuntimeError, match="before the gesture"):
        extract_idle_frames(annotated, 6, np.random.default_rng(0))


def test_idle_frames_reject_a_missing_video(tmp_path):
    with pytest.raises(RuntimeError, match="could not open"):
        extract_idle_frames(tmp_path / "absent.avi", 4, np.random.default_rng(0))


def test_idle_frames_never_include_the_first_frame_of_the_clip(late_gesture):
    """Its companion carries no person, so nothing would be kept from it."""
    for seed in range(30):
        frames = extract_idle_frames(late_gesture, 4, np.random.default_rng(seed))
        assert all(frame.frame_number >= 1 for frame in frames)


def test_the_dropped_first_frame_is_not_counted_as_available(tmp_path):
    """Four frames sit before this gesture and only three of them can be used."""
    video_path = write_video(tmp_path / "tight.avi", list(range(0, 248, 4)))
    write_window(video_path, 0.4, 5.9)

    with pytest.raises(RuntimeError, match="3 frames before the gesture"):
        extract_idle_frames(video_path, 4, np.random.default_rng(0))
