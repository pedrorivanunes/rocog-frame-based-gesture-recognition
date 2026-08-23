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


def extract_frames(
    video_path: Path,
    num_frames: int,
    rng: np.random.Generator | None = None,
) -> list[SampledFrame]:
    """Sample frames across the gesture window of a single video.

    The window is split into ``num_frames`` equal segments and one frame is drawn
    at random from within each. Drawing within each segment keeps full coverage
    of the window while decoupling the sampled phases from the gesture period.
    This is the sampling scheme of Temporal Segment Networks.

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

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise RuntimeError(f"could not open video file: {video_path}")
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    frames_per_second = video.get(cv2.CAP_PROP_FPS)

    gesture_start_frame, gesture_end_frame = gesture_window(
        video_path, total_frames, frames_per_second
    )
    if gesture_end_frame - gesture_start_frame == 0:
        raise RuntimeError(f"{video_path.name}: has length zero")
    edges = np.linspace(gesture_start_frame, gesture_end_frame, num_frames + 1)
    frame_indices = rng.uniform(edges[:-1], edges[1:]).round().astype(int)
    frames = []

    for frame_number in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = video.read()
        if not ok:
            raise RuntimeError(
                f"{video_path.name}: failure when reading frame {frame_number}"
            )
        position = (frame_number - gesture_start_frame) / (
            gesture_end_frame - gesture_start_frame
        )
        frames.append(SampledFrame(frame_number, position, frame))

    video.release()
    assert len(frames) == num_frames, (
        f"{video_path.name}: {len(frames)} frames, expected {num_frames}"
    )

    return frames
