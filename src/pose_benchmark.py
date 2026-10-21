"""Measure what reading the body costs, per frame and per decision.

The pose branch replaces pixels with the joints a detector finds, and its
accuracy was measured when the branch was trained; what it costs was not. The
network that classifies the joints is tiny, two layers of a hundred and
twenty-eight units over a few dozen numbers, so nearly all of the cost is the
detector, a pair of networks of its own that runs before anything this project
trained. A decision over K frames runs it K times, whatever classifier follows.

Two things differ from timing the image networks, and both are on purpose.

The frames are real. An image network does the same arithmetic whatever the
pixels hold, which is why its rows are timed on random frames. The detector
does not: it first looks for a person and runs its landmark model only where it
finds one, so random noise would time the short path and report a cost no real
frame pays. The frames here are stored test frames, decoded before the clock
starts, as the image rows' frames are.

One thread means one core. The detector has no setting for how many threads it
uses, so the process is pinned to a single core before the detector is built,
which is the closest equivalent to what the image rows ask of their library. A
system that cannot pin a process refuses the request rather than ignoring it.

The detector is built exactly as the extraction built it, in static mode on
every frame. Its own environment, as for the extraction, because MediaPipe and
the ROCm build of torch pin conflicting dependencies; the classifier half needs
torch and runs in the training environment, so the script is run once per part.
The classifier reads the joints and their motion, the input of the arm that the
fusion with the pixel model used; the other inputs differ from it by a few
numbers per row.

Run from the project root, the detector part with the pose environment's
interpreter and the classifier part with the training environment's::

    python src/pose_benchmark.py --part detector --threads 1 --output a.csv
    python src/pose_benchmark.py --part classifier --threads 1 --output b.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pose import make_detector  # noqa: E402
from timing import REPEATS, WARMUP, time_repeatedly  # noqa: E402

MANIFEST = PROJECT_ROOT / "data/manifests/real_ground_test.csv"
# Frames timed, one per video. Enough that no single frame's content decides
# the figure, few enough to hold in memory decoded.
SAMPLE_FRAMES = 30
FRAME_COUNTS = (1, 4, 8, 16, 24)
STORED_SIZE = 256
NUM_CLASSES = 8
# Columns shared with the image networks' table, so one reader takes both.
COLUMNS = (
    "model",
    "source",
    "level",
    "frames",
    "crop",
    "median_ms",
    "iqr_ms",
    "min_ms",
    "samples",
    "parameters_m",
)


def middle_frames(manifest: pd.DataFrame, count: int = SAMPLE_FRAMES) -> list[str]:
    """One stored frame from the middle of each of the first ``count`` videos.

    The middle rather than an end, because the ends of a gesture window are
    where the body is at rest, and a frame of someone standing still is not the
    frame a signal is read from. Fixed rather than drawn, so that every pass
    times the same frames.

    Args:
        manifest: One row per stored frame, in the order frames were stored.
        count: Videos to take a frame from.

    Returns:
        Paths relative to the project root, in manifest order.
    """
    by_video = manifest.groupby("video_id", sort=False)
    position = by_video.cumcount()
    size = by_video["path"].transform("size")

    return manifest.loc[position == size // 2, "path"].head(count).tolist()


def detector_call(detector, frames: Sequence[np.ndarray], count: int) -> Callable:
    """Build the closure that runs the detector on ``count`` frames in turn.

    Each call moves on through the sample rather than repeating a frame, so a
    decision over K frames meets K different people and poses, as a real one
    would. The colour conversion is inside, because the detector expects RGB
    and the frames arrive as OpenCV decodes them.

    Args:
        detector: Anything with the detector's ``process`` method.
        frames: Decoded frames, BGR.
        count: Frames one call processes.

    Returns:
        A callable taking no arguments.
    """
    position = [0]

    def call() -> None:
        for _ in range(count):
            frame = frames[position[0] % len(frames)]
            position[0] += 1
            detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    return call


def classifier_call(count: int) -> tuple[Callable, float]:
    """Build the closure that classifies ``count`` frames' joints and decides.

    Untrained weights, as for every row timed in this project: the arithmetic
    does not depend on their values.

    Args:
        count: Frames the decision aggregates, classified in one batch.

    Returns:
        The callable, and the classifier's parameter count in millions.
    """
    import torch

    from pose_train import JointClassifier, width

    model = JointClassifier(width("motion"), NUM_CLASSES).eval()
    rows = torch.randn(count, width("motion"))

    def call() -> None:
        with torch.inference_mode():
            model(rows).softmax(dim=1).mean(dim=0).argmax()

    parameters = sum(parameter.numel() for parameter in model.parameters())
    return call, parameters / 1e6


def pin_to_one_core() -> None:
    """Confine this process to a single core, before anything starts a thread.

    Raises:
        SystemExit: If the system cannot pin a process. A one-thread row that
            silently ran on every core would be the wrong number with the
            right label.
    """
    if not hasattr(os, "sched_setaffinity"):
        raise SystemExit("this system cannot pin a process to one core")
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})


def measure(part: str, threads: int, warmup: int, repeats: int) -> list[dict]:
    """Time one part at one frame and at every decision size."""
    records = []

    if part == "detector":
        frames = [
            cv2.imread(str(PROJECT_ROOT / path))
            for path in middle_frames(pd.read_csv(MANIFEST))
        ]
        detector = make_detector()
        # The reason for real frames, checked rather than assumed: a frame
        # with nobody found would be timed on the short path.
        found = sum(
            detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks
            is not None
            for frame in frames
        )
        print(f"a person found in {found} of {len(frames)} sample frames")
        calls = {
            count: (detector_call(detector, frames, count), "")
            for count in FRAME_COUNTS
        }
        crop = STORED_SIZE
    else:
        if threads:
            import torch

            torch.set_num_threads(threads)
        calls = {count: classifier_call(count) for count in FRAME_COUNTS}
        crop = 0

    for level, counts in (("forward", (1,)), ("decision", FRAME_COUNTS)):
        for count in counts:
            call, parameters = calls[count]
            summary = time_repeatedly(call, warmup, repeats)
            records.append(
                {
                    "model": f"pose_{part}",
                    "source": "ours",
                    "level": level,
                    "frames": count,
                    "crop": crop,
                    "median_ms": summary.median_ms,
                    "iqr_ms": summary.iqr_ms,
                    "min_ms": summary.min_ms,
                    "samples": summary.samples,
                    "parameters_m": parameters,
                }
            )

    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=("detector", "classifier"), required=True)
    parser.add_argument(
        "--threads",
        type=int,
        choices=(0, 1),
        default=1,
        help="1 confines the detector to one core and the classifier to one "
        "thread; 0 leaves every core to both.",
    )
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Time the requested part and write its rows."""
    args = parse_args(argv)
    if args.part == "detector" and args.threads == 1:
        pin_to_one_core()

    print(
        f"{platform.processor() or platform.machine()}, part {args.part}, "
        f"threads {args.threads or 'all'} | {platform.system()}"
    )
    records = measure(args.part, args.threads, args.warmup, args.repeats)
    for record in records:
        print(
            f"{record['model']:16s} {record['level']:9s} K={record['frames']:<3d}"
            f" {record['median_ms']:9.2f} ms  IQR {record['iqr_ms']:6.2f}"
            f"  min {record['min_ms']:8.2f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
