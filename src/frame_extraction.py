"""Sample frames from RoCoG-v2 videos.

Reads the gesture boundaries a video declares in its metadata, splits that
window into equal segments, and draws one frame from each. Real videos carry
no metadata and are treated as a single gesture spanning the whole clip.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np


class SampledFrame(NamedTuple):
    """One frame drawn from a video's gesture window.

    Attributes:
        frame_number: Index of the frame within the video.
        position: Where the frame sits in the gesture window — 0.0 at the start,
            1.0 at the end. Normalized so that frames from clips of different
            length and frame rate remain comparable across domains.
        frame: The image itself, a BGR array of shape ``(height, width, 3)`` as
            produced by OpenCV — not RGB, which is what most other libraries
            expect.
    """

    frame_number: int
    position: float
    frame: np.ndarray


def read_frames_at(video_path: Path, frame_numbers: list[int]) -> list[np.ndarray]:
    """Read the frames a caller names, in the order given.

    The counterpart to ``extract_frames`` for the case where the frames have
    already been chosen. Two situations need that, and both would break under
    re-sampling:

    A companion video must line up frame for frame with one already extracted.
    ``extract_frames`` places its picks with a random draw inside each segment,
    so running it twice over a pair of videos yields two different sets of
    frames — and a segmentation mask taken from a different instant than the
    image it is meant to describe is wrong in a way nothing downstream can
    detect. Reading the numbers a manifest already recorded removes the draw
    from the question entirely.

    Those companion videos also carry no metadata of their own, so there is no
    gesture window to derive. Naming the frames sidesteps that too.

    Args:
        video_path: Path to the ``.mp4`` to read.
        frame_numbers: Indices to read. Duplicates are allowed and are returned
            once each, in the order requested — short clips do round two
            segments onto the same frame.

    Returns:
        One BGR array per requested number, in the same order.

    Raises:
        RuntimeError: If the video cannot be opened, or a frame cannot be read.
    """
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"could not open video file: {video_path}")

    frames = []
    try:
        for frame_number in frame_numbers:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = video.read()
            if not ok:
                raise RuntimeError(
                    f"{video_path.name}: failure when reading frame {frame_number}"
                )
            frames.append(frame)
    finally:
        video.release()

    return frames


def read_metadata(xml_path: Path) -> ET.Element:
    """Parse a RoCoG-v2 metadata file and return its root element.

    RoCoG-v2 XML files declare encoding="utf-16" but are actually stored as
    utf-8 (verified: one byte per character, no BOM). ElementTree follows the
    declaration and fails to parse, so the declaration is corrected first.

    Args:
        xml_path: Path to the ``.xml`` file accompanying a video.

    Returns:
        The root ``<GestureVideo>`` element.
    """
    data = xml_path.read_bytes()
    data = data.replace(b"utf-16", b"utf-8", 1)

    return ET.fromstring(data)


def gesture_window(
    video_path: Path,
    total_frames: float,
    frames_per_second: float,
) -> tuple[int, int]:
    """Return the frame range spanned by the gesture in a video.

    Synthetic videos ship an XML file with exact gesture boundaries in seconds.
    Real videos do not, and are treated as a single gesture spanning the whole
    clip. About 2% of synthetic annotations — concentrated in the Rally class —
    end past the last frame of the video, so the upper bound is clamped.

    Args:
        video_path: Path to the ``.mp4``. Its ``.xml`` sibling is read when present.
        total_frames: Frame count reported by the decoder.
        frames_per_second: Frame rate reported by the decoder.

    Returns:
        ``(start_frame, end_frame)``, both valid frame indices into the video.
    """
    xml_path = video_path.with_suffix(".xml")

    if xml_path.exists():
        root = read_metadata(xml_path)
        gesture_start_time = root.findtext("startTime")
        gesture_end_time = root.findtext("endTime")
        gesture_start_frame = float(gesture_start_time) * frames_per_second
        gesture_end_frame = float(gesture_end_time) * frames_per_second
        gesture_end_frame = min(total_frames - 1, gesture_end_frame)
    else:
        gesture_start_frame = 0
        gesture_end_frame = total_frames - 1

    return int(gesture_start_frame), int(gesture_end_frame)


def _open_at_gesture(video_path: Path) -> tuple[cv2.VideoCapture, int, int]:
    """Open a video and locate the gesture inside it.

    Both samplers need the same three things before they can choose anything —
    an open reader, the first frame of the gesture and the last — and neither
    can do anything useful without all three.

    The reader is handed over open. The caller releases it, which is why every
    caller wraps its work in ``try``/``finally``: a read that fails partway
    would otherwise leave the file held.

    Args:
        video_path: Path to the ``.mp4`` to open.

    Returns:
        The open reader, the first frame of the gesture and the last.

    Raises:
        RuntimeError: If the video cannot be opened, or its gesture has no
            length, which would make every position infinite rather than fail.
    """
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"could not open video file: {video_path}")

    try:
        gesture_start_frame, gesture_end_frame = gesture_window(
            video_path,
            video.get(cv2.CAP_PROP_FRAME_COUNT),
            video.get(cv2.CAP_PROP_FPS),
        )
        if gesture_end_frame - gesture_start_frame == 0:
            raise RuntimeError(f"{video_path.name}: has length zero")
    except Exception:
        video.release()
        raise

    return video, gesture_start_frame, gesture_end_frame


def _segment_draw(
    first_frame: int,
    last_frame: int,
    num_frames: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Cut a stretch of frames into equal segments and draw one from each.

    This is the sampling scheme of Temporal Segment Networks, and it is the one
    decision both samplers share. Drawing within each segment keeps full
    coverage of the stretch while decoupling the sampled phases from the period
    of whatever repeats inside it — a gesture performed three times over a
    window sampled at fixed offsets is sampled at the same phase every time.

    Args:
        first_frame: First frame of the stretch, included.
        last_frame: Last frame of the stretch, included.
        num_frames: How many to draw. One per segment, so exactly this many.
        rng: Generator that places the pick inside each segment.

    Returns:
        The frame numbers drawn, in increasing order. A stretch shorter than
        the count rounds two segments onto one frame rather than returning
        fewer, which short clips do in about 1.7% of rows.
    """
    edges = np.linspace(first_frame, last_frame, num_frames + 1)

    return rng.uniform(edges[:-1], edges[1:]).round().astype(int)


def _read_at(
    video: cv2.VideoCapture,
    video_path: Path,
    frame_numbers: np.ndarray,
    gesture_start_frame: int,
    gesture_end_frame: int,
) -> list[SampledFrame]:
    """Read the frames chosen and record where each sits in the gesture.

    Position is measured against the gesture rather than the clip, so that
    frames from clips of different length and frame rate stay comparable. The
    formula is not clamped: a frame taken from before the gesture lands below
    0.0 and one from after lands above 1.0, which is what tells the two kinds
    of frame apart once they meet in one table.

    Args:
        video: Open reader, positioned anywhere. Left open for the caller.
        video_path: Only for naming the video in an error.
        frame_numbers: Frames to read, as chosen by the caller.
        gesture_start_frame: Frame the gesture starts on, position 0.0.
        gesture_end_frame: Frame it ends on, position 1.0.

    Returns:
        One ``SampledFrame`` per number, in the order given.

    Raises:
        RuntimeError: If a frame cannot be read. A record whose image came from
            somewhere other than the number it carries is wrong in a way no
            later check would catch.
    """
    frames = []
    for frame_number in frame_numbers:
        video.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ok, frame = video.read()
        if not ok:
            raise RuntimeError(
                f"{video_path.name}: failure when reading frame {frame_number}"
            )
        position = (frame_number - gesture_start_frame) / (
            gesture_end_frame - gesture_start_frame
        )
        frames.append(SampledFrame(frame_number, position, frame))

    return frames


def extract_frames(
    video_path: Path,
    num_frames: int,
    rng: np.random.Generator | None = None,
) -> list[SampledFrame]:
    """Sample frames across the gesture window of a single video.

    The window is split into ``num_frames`` equal segments and one frame is
    drawn at random from within each.

    Sampling is random but reproducible: pass a seeded ``rng`` and the same video
    always yields the same frames.

    Args:
        video_path: Path to the ``.mp4`` to sample.
        num_frames: How many frames to return. Always honored exactly.
        rng: Generator used to place a frame inside each segment. Defaults to an
            unseeded generator; pass a seeded one for reproducible extraction.

    Returns:
        A list of ``num_frames`` ``SampledFrame`` records, in increasing frame
        order. ``position`` is the frame's normalized location in the gesture
        window, 0.0 at the start and 1.0 at the end. ``frame`` is a BGR array
        of shape ``(height, width, 3)``, as produced by OpenCV.

    Raises:
        RuntimeError: If the video cannot be opened, or a frame cannot be read.
    """
    if rng is None:
        rng = np.random.default_rng()

    video, gesture_start_frame, gesture_end_frame = _open_at_gesture(video_path)
    try:
        frame_numbers = _segment_draw(
            gesture_start_frame, gesture_end_frame, num_frames, rng
        )
        frames = _read_at(
            video, video_path, frame_numbers, gesture_start_frame, gesture_end_frame
        )
    finally:
        video.release()

    assert len(frames) == num_frames, (
        f"{video_path.name}: {len(frames)} frames, expected {num_frames}"
    )

    return frames


def extract_idle_frames(
    video_path: Path,
    num_frames: int,
    rng: np.random.Generator | None = None,
) -> list[SampledFrame]:
    """Sample frames from the stretch before a video's gesture begins.

    A synthetic clip declares a start delay, and through it the avatar holds a
    named idle pose. That footage is the only material in the dataset showing a
    body performing none of the seven gestures, and a classifier offered seven
    labels has to call a body at rest one of the seven.

    Only the stretch before the gesture, though the clip holds the same pose
    after it as well. The stretch after runs to the final frame of the clip, and
    the companion files the dataset ships alongside each video do not always
    reach that far — a frame with no companion cannot be composited, and every
    synthetic frame is composited. The stretch before ends where the gesture
    begins, which both files always hold.

    Args:
        video_path: Path to the ``.mp4`` to sample. Its ``.xml`` sibling
            declares the window; a video without one is gesture from its first
            frame and has nothing before it.
        num_frames: How many frames to return. Always honored exactly.
        rng: Generator used to place a frame inside each segment. Defaults to an
            unseeded generator; pass a seeded one for reproducible extraction.

    Returns:
        A list of ``num_frames`` ``SampledFrame`` records, in increasing frame
        order. ``position`` follows the formula it has inside the window and so
        falls below 0.0 for every one of them, which is what tells these frames
        from the gesture's own once the two meet in one table, without a column
        that exists to say so.

    Raises:
        RuntimeError: If the video cannot be opened, if a frame cannot be read,
            if the gesture window has no length, or if fewer than ``num_frames``
            frames sit before the gesture.
    """
    if rng is None:
        rng = np.random.default_rng()

    video, gesture_start_frame, gesture_end_frame = _open_at_gesture(video_path)
    try:
        if gesture_start_frame < num_frames:
            raise RuntimeError(
                f"{video_path.name}: {gesture_start_frame} frames before the "
                f"gesture, need {num_frames}"
            )
        frame_numbers = _segment_draw(0, gesture_start_frame - 1, num_frames, rng)
        frames = _read_at(
            video, video_path, frame_numbers, gesture_start_frame, gesture_end_frame
        )
    finally:
        video.release()

    assert len(frames) == num_frames, (
        f"{video_path.name}: {len(frames)} frames, expected {num_frames}"
    )

    return frames
