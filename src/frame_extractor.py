import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path


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


def gesture_window(video_path: Path, total_frames: float, frames_per_second: float) -> tuple[int, int]:
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


def extract_frames(video_path: Path, num_frames: int) -> list[tuple[int, np.ndarray]]:
    """Sample frames evenly across the gesture window of a single video.

    Args:
        video_path: Path to the ``.mp4`` to sample.
        num_frames: How many frames to return. Always honored exactly.

    Returns:
        A list of ``(frame_number, image)`` pairs, of length ``num_frames``.
        Images are BGR arrays of shape ``(height, width, 3)``, as produced by
        OpenCV.

    Raises:
        RuntimeError: If the video cannot be opened, or a frame cannot be read.
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise RuntimeError(f"could not open video file: {video_path}")
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    gesture_start_frame, gesture_end_frame = gesture_window(video_path, total_frames, frames_per_second)
    frame_indices = np.linspace(gesture_start_frame, gesture_end_frame, num_frames).round().astype(int)
    frames = []
    for frame_number in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = video.read()
        if not ok:
            raise RuntimeError(f"{video_path.name}: failure when reading frame {frame_number}")
        frames.append((frame_number, frame))
    video.release()
    assert len(frames) == num_frames, f"{video_path.name}: {len(frames)} frames, expected {num_frames}"
    return frames


if __name__ == "__main__":
    video_path = Path("data/syn/ground/Rally/Scene77_220_Rally_9_2_2022_3_8_30.mp4")
    extract_frames(video_path, 10)