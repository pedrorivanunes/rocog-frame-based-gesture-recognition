"""Measure what optical flow costs per pair of frames, before any network runs.

The difference between two frames says where something changed; optical flow
says where each pixel went and how fast. It is the stronger form of what the
difference approximates, and the classic motion input of two-stream networks.
What it costs is a tax paid per frame, by a separate algorithm, before the
network sees anything, so it weighs proportionally more the cheaper the network
behind it: a costly preprocessing step in front of a cheap network makes the
choice of network irrelevant. Whether that tax fits the time a decision has is a
question for the clock, and this file asks it before any accuracy is chased.

Three methods, chosen to span what is in use. Farneback's dense flow is the
classical method shipped with OpenCV. TV-L1 is the variational method that
much of the two-stream work computed its flow with, and it needs OpenCV's
contributed modules. RAFT is a recent neural method, in the small version
torchvision ships, timed untrained because its arithmetic does not depend on
its weights.

The pairs are real frames, a stored frame and the one five frames before it in
the video, which is the spacing the frame difference uses. The classical
methods iterate until they settle, so what the frames hold can move their time;
random noise would not be what they meet. Frames are decoded and cropped to the
side the network reads before the clock starts, as for every other row; the
conversion a method needs, to grey or to a normalised tensor, is timed.

One pair per timed call. The methods take pairs one at a time, so a decision
over K frames pays K times this; the network rows are batched and have their
own decision rows, and this file does not pretend to a batching it does not do.

Not every method exists in every environment: TV-L1 is in the pose
environment's OpenCV and RAFT in the training environment's torch. The script
times whichever it is asked for and refuses one it cannot run.

Run from the project root::

    python src/flow_benchmark.py --method farneback --threads 1 --output a.csv
    python src/flow_benchmark.py --method raft_small --device cuda --output b.csv
"""

from __future__ import annotations

import argparse
import csv
import platform
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pose_benchmark import COLUMNS, middle_frames  # noqa: E402
from timing import REPEATS, WARMUP, time_repeatedly  # noqa: E402

MANIFEST = PROJECT_ROOT / "data/manifests/real_ground_test.csv"
DENSE_MANIFEST = PROJECT_ROOT / "data/manifests/real_ground_test_dense.csv"
# The frame difference's spacing, so that the flow is taken over the same gap.
SPACING = 5
CROP = 224
METHODS = ("farneback", "tvl1", "raft_small")


def frame_pairs(
    manifest: pd.DataFrame, dense: pd.DataFrame, spacing: int = SPACING
) -> list[tuple[str, str]]:
    """Pair each sampled frame with the one ``spacing`` frames before it.

    A frame too near the start of its video has nothing that far back, and is
    paired with the one that far ahead instead; the time a pair takes does not
    depend on which of the two comes first.

    Args:
        manifest: The stored frames, one row each.
        dense: Every frame of the same videos, which is where neighbours live.
        spacing: Frames between the two of a pair.

    Returns:
        Path pairs, earlier frame first, relative to the project root.
    """
    rows = manifest.set_index("path").loc[middle_frames(manifest)]
    lookup = dense.set_index(["video_id", "frame_number"])["path"]

    pairs = []
    for path, row in rows.iterrows():
        earlier = (row["video_id"], row["frame_number"] - spacing)
        if earlier in lookup.index:
            pairs.append((lookup.loc[earlier], path))
        else:
            pairs.append(
                (path, lookup.loc[(row["video_id"], row["frame_number"] + spacing)])
            )
    return pairs


def centre_crop(frame: np.ndarray, side: int = CROP) -> np.ndarray:
    """The square of ``side`` pixels at the centre, as the evaluation crops."""
    top = (frame.shape[0] - side) // 2
    left = (frame.shape[1] - side) // 2
    return frame[top : top + side, left : left + side]


def available(method: str) -> bool:
    """Whether this environment can run a method."""
    if method == "farneback":
        return True
    if method == "tvl1":
        return hasattr(cv2, "optflow")
    try:
        import torchvision  # noqa: F401
    except ImportError:
        return False
    return True


def flow_call(
    method: str, pairs: Sequence[tuple], device: str
) -> tuple[Callable, float]:
    """Build the closure that computes the flow of one pair, the next each call.

    Args:
        method: One of ``METHODS``.
        pairs: Decoded and cropped frames, BGR, two per pair.
        device: Where RAFT runs; the classical methods are CPU only.

    Returns:
        The callable, and the method's parameter count in millions, or an empty
        string for a method without parameters.
    """
    position = [0]

    def next_pair() -> tuple:
        pair = pairs[position[0] % len(pairs)]
        position[0] += 1
        return pair

    if method == "farneback":

        def call() -> None:
            first, second = (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in next_pair())
            cv2.calcOpticalFlowFarneback(first, second, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        return call, ""

    if method == "tvl1":
        solver = cv2.optflow.DualTVL1OpticalFlow_create()

        def call() -> None:
            first, second = (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in next_pair())
            solver.calc(first, second, None)

        return call, ""

    import torch
    from torchvision.models.optical_flow import raft_small

    target = torch.device(device)
    model = raft_small(weights=None).eval().to(target)

    def tensor(frame: np.ndarray) -> torch.Tensor:
        rgb = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        scaled = rgb.permute(2, 0, 1).unsqueeze(0).float().div(127.5).sub(1)
        return scaled.to(target)

    def call() -> None:
        with torch.inference_mode():
            first, second = (tensor(f) for f in next_pair())
            model(first, second)
            if target.type == "cuda":
                torch.cuda.synchronize()

    parameters = sum(parameter.numel() for parameter in model.parameters())
    return call, parameters / 1e6


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="cuda only for RAFT; the classical methods run on the CPU.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="CPU threads for OpenCV and torch; 0 leaves their defaults alone.",
    )
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Time one method on real pairs and write its row."""
    args = parse_args(argv)
    if not available(args.method):
        raise SystemExit(f"{args.method} cannot run in this environment")
    if args.device == "cuda" and args.method != "raft_small":
        raise SystemExit(f"{args.method} runs on the CPU only")

    if args.threads:
        cv2.setNumThreads(args.threads)
        if args.method == "raft_small":
            import torch

            torch.set_num_threads(args.threads)

    pairs = [
        tuple(centre_crop(cv2.imread(str(PROJECT_ROOT / path))) for path in pair)
        for pair in frame_pairs(pd.read_csv(MANIFEST), pd.read_csv(DENSE_MANIFEST))
    ]
    call, parameters = flow_call(args.method, pairs, args.device)
    summary = time_repeatedly(call, args.warmup, args.repeats)

    where = "GPU" if args.device == "cuda" else f"threads {args.threads or 'all'}"
    print(
        f"{platform.processor() or platform.machine()}, {where}, OpenCV "
        f"{cv2.__version__} | {platform.system()}"
    )
    print(
        f"flow_{args.method:12s} pair  {summary.median_ms:9.2f} ms"
        f"  IQR {summary.iqr_ms:6.2f}  min {summary.min_ms:8.2f}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "model": f"flow_{args.method}",
                "source": "reference",
                "level": "forward",
                "frames": 1,
                "crop": CROP,
                "median_ms": summary.median_ms,
                "iqr_ms": summary.iqr_ms,
                "min_ms": summary.min_ms,
                "samples": summary.samples,
                "parameters_m": parameters,
            }
        )
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
