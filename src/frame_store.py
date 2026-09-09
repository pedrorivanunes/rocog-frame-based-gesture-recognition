"""Write extracted frames to disk and keep the manifest that indexes them.

Two passes put frames on disk: one that chooses them, sampling across a video's
gesture window, and one that re-renders a choice already made at another size.
They differ in how a frame and its destination are arrived at and in nothing
else, so what they have in common lives here — the writing itself, and the
manifest that records what has been written.

That manifest is also the ledger a resumed pass reads. A pass over thousands of
videos runs for hours on a machine that has lost power mid-pass more than once,
and the rows already in the manifest are what tells a second attempt where the
first one stopped. Both halves therefore belong together: a file on disk that no
row points at is invisible to everything downstream, and a row pointing at a file
that was never written is worse.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# JPEG quality every pass writes with. A constant rather than an option so that
# two copies of the same frame differ in resolution alone.
QUALITY = 90


def save_frames(
    frames: list[np.ndarray],
    paths: list[Path],
    size: int,
    quality: int = QUALITY,
) -> None:
    """Resize frames to a square of the given side and write them as JPEG.

    How a frame is resampled depends on which way it is going. Averaging over
    the source pixels is right when shrinking and wrong when growing: asked for
    more pixels than it was given, it has nothing to average and returns blocks.
    Extraction only ever shrank, every source video being larger than the size
    it wrote; growing starts to happen at 640, which 14% of the real clips fall
    below.

    Args:
        frames: Images to write, BGR as OpenCV reads them.
        paths: Where to write each, in the same order. Parents are created.
        size: Side length of the square output, in pixels.
        quality: JPEG quality, 0-100.

    Raises:
        RuntimeError: If a file cannot be written.
        ValueError: If the counts disagree. Writing the shorter of the two would
            leave either a frame with no row or a row with no file.
    """
    for frame, path in zip(frames, paths, strict=True):
        shrinking = frame.shape[0] >= size
        resized = cv2.resize(
            frame,
            (size, size),
            interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), resized, [cv2.IMWRITE_JPEG_QUALITY, quality]):
            raise RuntimeError(f"could not write frame to {path}")


def drop_incomplete_videos(manifest_path: Path, frames_per_video: int) -> int:
    """Trim a manifest back to whole videos, before a pass resumes into it.

    A pass cut off while writing leaves a short group of rows behind. That video
    is extracted again, which is right, but appending its rows would leave the
    short group in front of the complete one and the manifest would hold more
    rows for that video than any video should have — enough to weight it above
    the others in training, and to break a frame selection that counts on every
    video carrying the same number.

    Rows that survive have to come back out unchanged, and the default CSV
    reader does not guarantee that: it parses floats with a fast routine that
    can land one unit in the last place away from the value written, so reading
    the manifest and writing it again would quietly edit the position of every
    frame already extracted. ``round_trip`` asks for the parser that returns the
    float the text names.

    Args:
        manifest_path: Manifest to trim in place. Missing or empty is left
            alone, there being nothing to trim.
        frames_per_video: How many rows a finished video contributes.

    Returns:
        How many rows were dropped.
    """
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        return 0

    written = pd.read_csv(manifest_path, float_precision="round_trip")
    rows_per_video = written["video_id"].map(written["video_id"].value_counts())
    whole = written[rows_per_video == frames_per_video]

    if len(whole) < len(written):
        whole.to_csv(manifest_path, index=False)

    return len(written) - len(whole)


def completed_videos(manifest_path: Path, frames_per_video: int) -> set[str]:
    """Read back which videos a previous pass finished.

    A pass over the synthetic subset runs for hours and this machine has lost
    power four times in a week, so it has to be able to pick up where it
    stopped. The manifest is what decides: it is the index the rest of the
    pipeline reads, and frames on disk that no row points at are invisible
    downstream, so a video counts as done only once its rows are written.

    A video is accepted only with its full complement of rows. A pass cut off
    while writing leaves a short group behind, and one video is cheap to extract
    again — while trusting a short group would leave a hole that nothing further
    down reports.

    Counting rows also catches the case where the extraction itself changed: ask
    for a different number of frames per video and no earlier group matches, so
    the pass redoes the work instead of resuming into a manifest built under
    other rules. It does not catch a change that leaves the count alone, such as
    a different output size — deleting the manifest is what forces those.

    Args:
        manifest_path: Manifest a previous pass wrote. Missing or empty means
            nothing is done yet.
        frames_per_video: How many rows a finished video contributes.

    Returns:
        The ids of the videos that need not be extracted again.
    """
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        return set()

    rows_per_video = pd.read_csv(manifest_path, usecols=["video_id"])[
        "video_id"
    ].value_counts()

    return set(rows_per_video[rows_per_video == frames_per_video].index)


def append_rows(rows: list[dict], manifest_path: Path) -> None:
    """Add one video's rows to the manifest, creating the file if needed.

    Holding every row until the end of the pass puts hours of work behind a
    single write. Appending as each video finishes puts at most one video at
    risk, which is what makes the pass resumable at all.

    Rows are appended only after the images they point at are on disk, so a
    manifest row is a promise that its frame exists — the order the resume logic
    depends on.

    Args:
        rows: Manifest rows for a single video.
        manifest_path: Manifest to append to. Its directory is created if
            missing, and the header is written only for a new file.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not manifest_path.exists() or manifest_path.stat().st_size == 0

    pd.DataFrame(rows).to_csv(manifest_path, mode="a", header=is_new, index=False)
