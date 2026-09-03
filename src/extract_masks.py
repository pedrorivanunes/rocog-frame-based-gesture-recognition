"""Extract the person silhouette for frames a manifest already lists.

Every synthetic video ships a companion ``_mask.mp4`` holding a semantic
segmentation of the same footage. The published baselines never used them. They
are what makes background randomisation possible: with the person separated
from the scene, the same gesture can be composited onto anything, which attacks
a model that has learnt the terrain instead of the pose.

This script does not choose frames. It reads the numbers an existing manifest
recorded and pulls exactly those, so that every silhouette lines up with the
image already on disk. Choosing again would draw different frames — the picks
are random within each segment — and a silhouette taken a few frames away from
its image is wrong in a way no later check would catch.

Masks are written as PNG rather than JPEG. The output is one bit per pixel
dressed up as an image, and JPEG would blur its edges into intermediate values
that the source never contained.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from frame_extraction import read_frames_at

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Side length of the square output, matching the extracted frames so that a
# mask and its image can be composited without resampling either.
SIZE = 256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line: which manifest's frames to find silhouettes for.

    Args:
        argv: Arguments to parse. ``None`` reads ``sys.argv``.

    Returns:
        A namespace with ``manifest`` (a file name under data/manifests/) and
        ``size``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--manifest",
        required=True,
        help="frames to find masks for, a file name under data/manifests/",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=SIZE,
        help="side length of the square output, in pixels",
    )
    return parser.parse_args(argv)


def person_silhouette(mask_frame: np.ndarray) -> np.ndarray:
    """Separate the person from the scene in one segmentation frame.

    The person is where green is not the dominant channel. Terrain, vegetation
    and sky all peak in green; the person does not. The rule is indirect on
    purpose: the colour assigned to the person is not constant across the
    dataset — both a red and a blue convention appear — so any fixed threshold
    would silently return an empty silhouette on half the videos.

    Testing dominance rather than "not green" also keeps the sky out. Sky is
    bright in all three channels and a naive non-green test claims it.

    Args:
        mask_frame: One frame of a ``_mask.mp4``, BGR as OpenCV reads it.

    Returns:
        A boolean array of shape ``(height, width)``, true on the person.
    """
    blue, green, red = (mask_frame[:, :, channel].astype(int) for channel in range(3))

    return green < np.maximum(blue, red)


def mask_path_for(frame_path: Path) -> Path:
    """Name the silhouette belonging to an extracted frame.

    The two trees mirror each other — ``data/frames/...`` and ``data/masks/...``
    with the same class folders and file stems — rather than the mask being
    recorded in the manifest. The manifest is a controlled artefact reproduced
    byte for byte across machines, and adding a column to it would mean every
    later run diverged from the ones already measured.

    Args:
        frame_path: The frame's path, as the manifest stores it.

    Returns:
        Where that frame's silhouette belongs, as a relative path.
    """
    parts = list(Path(frame_path).parts)
    parts[parts.index("frames")] = "masks"

    return Path(*parts).with_suffix(".png")


def save_silhouettes(
    silhouettes: list[np.ndarray], paths: list[Path], size: int = SIZE
) -> None:
    """Resize and write silhouettes as PNG.

    Nearest-neighbour resizing, not the area averaging the frames use. Averaging
    a two-valued image invents the values in between, and the compositing this
    feeds would then blend a halo of scene around every person.

    Args:
        silhouettes: Boolean arrays, true on the person.
        paths: Where to write each, in the same order. Parents are created.
        size: Side length of the square output.

    Raises:
        RuntimeError: If a file cannot be written.
    """
    for silhouette, path in zip(silhouettes, paths, strict=True):
        resized = cv2.resize(
            silhouette.astype(np.uint8) * 255,
            (size, size),
            interpolation=cv2.INTER_NEAREST,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), resized):
            raise RuntimeError(f"could not write mask to {path}")


if __name__ == "__main__":
    args = parse_args()
    manifest = pd.read_csv(PROJECT_ROOT / "data/manifests" / args.manifest)
    print(f"manifest: {args.manifest}  —  {len(manifest)} frames")

    start_time = time.perf_counter()
    written = 0
    failures = []
    fractions = []

    for video_id, rows in manifest.groupby("video_id", sort=False):
        domain = rows["domain"].iloc[0]
        gesture_class = rows["class_name"].iloc[0]
        video = PROJECT_ROOT / "data" / domain / "ground" / gesture_class
        video = video / f"{video_id}_mask.mp4"

        try:
            frames = read_frames_at(video, rows["frame_number"].tolist())
            silhouettes = [person_silhouette(frame) for frame in frames]
            paths = [
                PROJECT_ROOT / mask_path_for(path) for path in rows["path"].tolist()
            ]
            save_silhouettes(silhouettes, paths, args.size)
            fractions.extend(silhouette.mean() for silhouette in silhouettes)
            written += len(paths)
        except Exception as error:
            failures.append((video_id, f"{type(error).__name__}: {error}"))

    elapsed = time.perf_counter() - start_time
    print(f"masks written: {written} of {len(manifest)}")
    print(f"videos that failed: {len(failures)}")
    for video_id, message in failures[:5]:
        print(f"  {video_id} -> {message}")

    if fractions:
        share = np.array(fractions) * 100
        # The person covers a narrow and known band of the frame. A run whose
        # median leaves it has segmented something else, and the count of files
        # written would not show that.
        print(
            f"person share of frame: median {np.median(share):.2f}%  "
            f"p1 {np.percentile(share, 1):.2f}%  p99 {np.percentile(share, 99):.2f}%"
        )
        outside = ((share < 1) | (share > 15)).sum()
        print(f"frames outside the 1-15% band: {outside} ({outside / len(share):.2%})")

    print(f"time taken: {elapsed:.0f} s")
