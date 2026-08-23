"""Build the per-frame manifest that indexes the extracted dataset.

The manifest is a table with one row per extracted frame, carrying everything
a training run needs to select, group and audit its data: which domain and
split a frame belongs to, which video it came from, which camera viewpoint it
was shot from, and where in the gesture it sits. Frames live on disk as plain
images; this table is the single source of truth about them.
"""

import json
from pathlib import Path
from typing import NamedTuple


class VideoMetadata(NamedTuple):
    """Facts about a video that can be read from its file name alone.

    Attributes:
        video_id: File name without extension. Ties every extracted frame back
            to the video it came from.
        group_id: The unit that must never straddle a train/test boundary — the
            recorded subject for real videos, the scene for synthetic ones.
        view: Camera viewpoint, as ``Scene % 6``, for synthetic videos; ``None``
            for real ones, shot from a single fixed camera.
        is_frontal: Whether the subject faces the camera. Always true for real
            videos; true for synthetic views 3 and 4.
    """

    video_id: str
    group_id: str
    view: int | None
    is_frontal: bool


def video_metadata(video_path: Path, domain: str) -> VideoMetadata:
    """Derive a video's metadata from its file name.

    RoCoG-v2 names videos differently in each domain::

        syn:   Scene26_386_Halt_4_2_2022_15_1_38.mp4
        real:  S00_10m_ground_label5_start653.mp4

    For synthetic videos the leading scene index encodes the camera position.
    Scenes come in consecutive blocks of six, and the position within a block
    fixes the viewpoint, so ``Scene % 6`` recovers it. Views 3 and 4 are the
    frontal ones — the only two that match the real domain, which is frontal
    throughout.

    For real videos the leading token identifies the recorded subject. A
    trailing letter marks a second session with the same person: S04 and S04b
    are one subject in different clothing — light shirt in one session, dark
    shirt and a cap in the other — confirmed by inspecting the footage. The
    letter is stripped so that ``group_id`` names the person rather than the
    session; otherwise a split built on this column would count one subject as
    two and leak them across the boundary.

    Args:
        video_path: Path to the video. Only the file name is read — the file
            itself is never opened.
        domain: Either ``"syn"`` or ``"real"``. Asserted by the caller, which
            already knows which annotations file it is walking.

    Returns:
        The metadata derivable from the name.

    Raises:
        ValueError: If ``domain`` is neither ``"syn"`` nor ``"real"``.
    """

    video_id = video_path.stem
    group_id = video_id.split("_")[0]

    if domain == "syn":
        view = int(group_id.removeprefix("Scene"))
        view = view % 6
        is_frontal = view in (3, 4)
    elif domain == "real":
        group_id = group_id.rstrip("abcdefghijklmnopqrstuvwxyz")
        view = None
        is_frontal = True
    else:
        raise ValueError(f"unknown domain: {domain!r}")

    return VideoMetadata(video_id, group_id, view, is_frontal)


def load_class_names(path: Path) -> dict[int, str]:
    """Load the label-to-name mapping ..."""
    text = path.read_text(encoding="utf-8")
    raw_mapping = json.loads(text)
    return {int(label): name for label, name in raw_mapping.items()}