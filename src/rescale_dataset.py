"""Re-render the frames a manifest already lists at a different side length.

The frames on disk were stored at 256 pixels, well below what the sources hold:
every synthetic video is 640 and the real clips run from 408 to 1200. The person
is therefore about 139 pixels tall in an extracted frame against 347 in the
synthetic source, and whether the detail thrown away with those pixels carries
any of the signal is a question that can only be asked by keeping it.

Asking it cleanly means changing the pixel count and nothing else. This script
therefore does not extract anything: it reads the frame numbers an existing
manifest recorded, pulls exactly those, and writes them larger under::

    data/frames/{size}/{domain}/{class}/{video_id}_f{frame_number}.jpg

Its manifest is the source manifest with one column rewritten, so the two index
the same videos, the same frames and the same positions.

Re-extracting instead would look equivalent and is not. Frame placement is drawn
at random inside each segment, and the draw was moved from one generator per
pass to one per video when extraction was made resumable — so a pass run today
would not reproduce the manifests in use even at the same size. Measured on the
real test split: 412 of 2400 frame numbers matched, which is chance.

A pass can be interrupted and run again: the manifest is written as each video
finishes, and a video already listed there in full is not re-rendered.

    python src/rescale_dataset.py --manifest real_ground_test.csv --size 640
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from frame_extraction import read_frames_at
from frame_store import (
    append_rows,
    completed_videos,
    drop_incomplete_videos,
    save_frames,
)
from manifest import sized_path_for

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line: which manifest to re-render, and how large.

    Neither argument has a default. The manifest is the specification of the
    whole pass — which videos, which of their frames, which rows come out — and
    the size is the one thing the pass is for, so leaving either implicit would
    hide the only two decisions there are.

    Args:
        argv: Arguments to parse. ``None`` reads ``sys.argv``.

    Returns:
        A namespace with ``manifest`` (a file name under data/manifests/) and
        ``size`` (side length of the square output).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--manifest",
        required=True,
        help="frames to re-render, a file name under data/manifests/",
    )
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="side length of the square output, in pixels; the synthetic source "
        "holds 640 and asking for more of it stores detail it never had",
    )
    return parser.parse_args(argv)


def rescaled_manifest_name(manifest: str, size: int) -> str:
    """Name the manifest a pass writes its rows to.

    The output manifest is also the ledger a resumed pass reads to decide what
    is already done, so it cannot be the file being read: a pass would find its
    source's videos listed, count them as finished, and write nothing.

    Args:
        manifest: File name of the manifest being re-rendered, with or without
            its suffix.
        size: Side length the frames are stored at.

    Returns:
        The output manifest's file name, with the ``.csv`` suffix.
    """
    return f"{Path(manifest).stem}_{size}.csv"


def source_video(row: pd.Series) -> Path:
    """Locate the video a manifest row's frame came from.

    Args:
        row: Any row belonging to the video.

    Returns:
        Path to the ``.mp4``, relative to the project root.
    """
    return Path(
        "data", row["domain"], "ground", row["class_name"], f"{row['video_id']}.mp4"
    )


if __name__ == "__main__":
    args = parse_args()
    manifest = pd.read_csv(
        PROJECT_ROOT / "data/manifests" / args.manifest, float_precision="round_trip"
    )
    frames_per_video = manifest.groupby("video_id", sort=False).size().max()
    output_path = (
        PROJECT_ROOT
        / "data/manifests"
        / rescaled_manifest_name(args.manifest, args.size)
    )

    print(f"manifest: {args.manifest}  —  {len(manifest)} frames")
    print(f"stored at: {args.size}x{args.size}")
    dropped_rows = drop_incomplete_videos(output_path, frames_per_video)
    already_done = completed_videos(output_path, frames_per_video)
    print(f"videos already in the manifest: {len(already_done)}")
    print(f"rows dropped from videos left half written: {dropped_rows}")

    start_time = time.perf_counter()
    written = 0
    skipped = 0
    failures = []

    for video_id, rows in manifest.groupby("video_id", sort=False):
        if video_id in already_done:
            skipped += 1
            continue

        try:
            video = PROJECT_ROOT / source_video(rows.iloc[0])
            frames = read_frames_at(video, rows["frame_number"].tolist())
            paths = [sized_path_for(path, args.size) for path in rows["path"]]
            save_frames(frames, [PROJECT_ROOT / path for path in paths], args.size)

            append_rows(rows.assign(path=paths).to_dict("records"), output_path)
            written += len(paths)
        except Exception as error:
            failures.append((video_id, f"{type(error).__name__}: {error}"))

    elapsed = time.perf_counter() - start_time
    print(f"frames written: {written} of {len(manifest)}")
    print(f"videos skipped as already done: {skipped}")
    print(f"videos that failed: {len(failures)}")
    for video_id, message in failures[:5]:
        print(f"  {video_id} -> {message}")
    print(f"time taken: {elapsed:.0f} s")
