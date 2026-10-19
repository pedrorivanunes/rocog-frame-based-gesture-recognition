"""Extract body keypoints for every frame a manifest lists.

The pose branch classifies gestures from where the joints are rather than from
what the frame looks like. Finding the joints is the expensive part, a
detection network run on every frame, and it does not depend on anything the
branch later decides: which joints to keep, how to normalise them, whether to
add motion or confidence. So the detector runs once, here, and everything it
returns is stored. Every later choice is then made on the stored keypoints
without running the detector again.

The frames are the ones the appearance models read, from the same manifests,
so that a difference between the two branches is a difference in what they
take from the same pixels. The detector is the probe's, built by the same
function, so that what the probe measured about this dataset describes what
this extraction produced.

Each manifest becomes one file under ``data/pose/keypoints/``, aligned row for
row with the manifest and holding:

    image       (frames, 33, 4) x, y in image fractions, z relative depth,
                visibility; NaN where no person was found
    world       (frames, 33, 3) metric coordinates centred on the hips
    path        the manifest's frame path per row, to check the alignment
    names       the landmark names in index order
    complexity  the detector's model size

The detector also reports a presence score per joint, but this interface
leaves it at zero, so it is not stored: visibility is the only confidence
that varies.

The work is split into blocks that are written as they finish, so a run that
stops can be started again and picks up where it was; the blocks are merged
once a manifest is complete.

Run it in the pose environment, from the project root::

    .venv-pose/bin/python src/pose_keypoints.py
"""

import argparse
import multiprocessing
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pose import ARM_LANDMARKS, MODEL_COMPLEXITY, VISIBILITY_FLOOR, make_detector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "data/pose/keypoints"

# The four manifests every appearance cell is trained, validated and scored on.
MANIFESTS = (
    "real_ground_test.csv",
    "real_ground_train.csv",
    "syn_ground_train_idle.csv",
    "syn_ground_train.csv",
)

LANDMARKS = 33

# Frames per block. Small enough that an interrupted run loses little, large
# enough that writing a block costs nothing next to detecting it.
BLOCK = 2000

# Worker processes. The detector already spreads one frame over several
# threads, so throughput stops growing near the number of physical cores.
WORKERS = 8

# Built once in each worker by ``start_worker``.
detector = None


def start_worker() -> None:
    """Build this worker's detector.

    Workers are started by spawn, not fork. A forked copy of a process that
    already holds a detector inherits its threads and allocator state, and
    dies in the allocator; a spawned one starts clean and builds its own.
    """
    global detector
    detector = make_detector()


def detect_block(paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Run the detector on a block of frames.

    Args:
        paths: Frame paths relative to the project root.

    Returns:
        The image landmarks and the world landmarks, NaN where no person was
        found.

    Raises:
        FileNotFoundError: If a frame cannot be read. A missing frame would
            otherwise look like a frame without a person, which is a result.
    """
    image = np.full((len(paths), LANDMARKS, 4), np.nan, dtype=np.float32)
    world = np.full((len(paths), LANDMARKS, 3), np.nan, dtype=np.float32)
    for index, path in enumerate(paths):
        frame = cv2.imread(str(PROJECT_ROOT / path))
        if frame is None:
            raise FileNotFoundError(path)
        # The detector expects RGB; OpenCV reads BGR.
        result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks is None:
            continue
        image[index] = [
            (point.x, point.y, point.z, point.visibility)
            for point in result.pose_landmarks.landmark
        ]
        world[index] = [
            (point.x, point.y, point.z)
            for point in result.pose_world_landmarks.landmark
        ]

    return image, world


def landmark_names() -> np.ndarray:
    """The detector's landmark names in index order, for whoever reads the file."""
    import mediapipe as mp

    return np.array([point.name for point in mp.solutions.pose.PoseLandmark])


def extract_manifest(name: str, pool: ProcessPoolExecutor) -> Path:
    """Detect every frame of one manifest, block by block, and merge the blocks.

    Args:
        name: Manifest file name under ``data/manifests/``.
        pool: Workers that each hold a detector.

    Returns:
        The path of the merged file.
    """
    manifest = pd.read_csv(PROJECT_ROOT / "data/manifests" / name)
    output = OUTPUT_DIRECTORY / f"{Path(name).stem}.npz"
    if output.exists():
        with np.load(output) as stored:
            if len(stored["path"]) == len(manifest):
                print(f"{name}: already extracted, {len(manifest)} frames", flush=True)
                return output

    parts = OUTPUT_DIRECTORY / f"{Path(name).stem}.parts"
    parts.mkdir(parents=True, exist_ok=True)
    paths = manifest["path"].tolist()
    blocks = range(0, len(paths), BLOCK)
    pending = {
        start: pool.submit(detect_block, paths[start : start + BLOCK])
        for start in blocks
        if not (parts / f"{start:08d}.npz").exists()
    }
    print(
        f"{name}: {len(manifest)} frames, {len(blocks) - len(pending)} of"
        f" {len(blocks)} blocks already done",
        flush=True,
    )

    start_time = time.perf_counter()
    finished = 0
    starts = {future: start for start, future in pending.items()}
    for future in as_completed(starts):
        image, world = future.result()
        np.savez(parts / f"{starts[future]:08d}.npz", image=image, world=world)
        finished += 1
        if finished % 10 == 0 or finished == len(pending):
            elapsed = time.perf_counter() - start_time
            print(
                f"  {finished}/{len(pending)} blocks"
                f"  {finished * BLOCK / elapsed:6.1f} frames/s"
                f"  {time.strftime('%H:%M:%S')}",
                flush=True,
            )

    loaded = [np.load(parts / f"{start:08d}.npz") for start in blocks]
    image = np.concatenate([part["image"] for part in loaded])
    world = np.concatenate([part["world"] for part in loaded])
    if len(image) != len(manifest):
        raise RuntimeError(f"{name}: {len(image)} rows for {len(manifest)} frames")

    np.savez_compressed(
        output,
        image=image,
        world=world,
        path=np.array(paths),
        names=landmark_names(),
        complexity=np.array(MODEL_COMPLEXITY),
    )
    shutil.rmtree(parts)
    report(name, image)
    return output


def report(name: str, image: np.ndarray) -> None:
    """Print how often a person was found and how many arm joints were located.

    Args:
        name: The manifest's name, for the heading.
        image: The image landmarks as extracted.
    """
    found = ~np.isnan(image[:, 0, 0])
    located = (image[found][:, list(ARM_LANDMARKS), 3] >= VISIBILITY_FLOOR).sum(axis=1)
    print(
        f"{name}: person found in {found.mean():.2%} of {len(image)} frames;"
        f" arm joints located, mean {located.mean():.2f} of {len(ARM_LANDMARKS)},"
        f" all of them in {np.mean(located == len(ARM_LANDMARKS)):.1%}",
        flush=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line: which manifests, and how many workers.

    Args:
        argv: Arguments to parse, or ``None`` for the process's own.

    Returns:
        A namespace with ``manifest`` and ``workers``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--manifest",
        nargs="+",
        default=list(MANIFESTS),
        help="manifests to extract, file names under data/manifests/",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help="detector processes to run at once",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    with ProcessPoolExecutor(
        args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=start_worker,
    ) as pool:
        for name in args.manifest:
            extract_manifest(name, pool)
    print(f"\ntime taken: {(time.perf_counter() - start_time) / 60:.1f} min")
    print(f"keypoints written to {OUTPUT_DIRECTORY}")
