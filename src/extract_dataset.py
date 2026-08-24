"""Extract frames for every video in a RoCoG-v2 annotations file and index them.

For each video listed, samples frames across the gesture window, resizes them to
a fixed square size, and writes them as JPEG under::

    data/frames/{domain}/{class}/{video_id}_f{frame_number}.jpg

Then writes one manifest row per frame to::

    data/manifests/{annotations file name}.csv

The manifest is the index the training pipeline reads: it carries domain, split,
label, source video, camera viewpoint and position within the gesture, so any
subset can be selected without touching the images themselves.

Domain, perspective and split are read from the annotations file name, so the
file to process is the only thing that has to be chosen — everything else
follows from it.

Reports how many videos succeeded, how many failed and why, and how long the
pass took.

Paths are resolved from the location of this file, so it can be run from any
working directory:

    python src/extract_dataset.py
"""

import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from frame_extraction import SampledFrame, extract_frames, read_metadata
from manifest import load_class_names, video_metadata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILE_PATH = PROJECT_ROOT / "data/annotations/real_ground_train.txt"
NUM_FRAMES = 24
NUM_PER_STRATUM = 250
MAX_REPETITIONS = 4
SEED = 42


def save_frames(
    frames: list[SampledFrame],
    output_dir: Path,
    video_id: str,
    size: int = 256,
    quality: int = 90,
) -> list[Path]:
    """Resize and write a video's sampled frames as JPEG files.

    Args:
        frames: Frames returned by ``extract_frames``.
        output_dir: Directory to write into. Created if missing.
        video_id: Video the frames came from; used as the file name prefix.
        size: Side length of the square output, in pixels.
        quality: JPEG quality, 0-100.

    Returns:
        The paths written, in the same order as ``frames``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for sampled in frames:
        image = cv2.resize(sampled.frame, (size, size), interpolation=cv2.INTER_AREA)
        path = output_dir / f"{video_id}_f{sampled.frame_number:04d}.jpg"
        written_ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not written_ok:
            raise RuntimeError(f"could not write frame to {path}")
        written.append(path)

    return written


def read_annotations(file_path: Path) -> list[tuple[Path, int]]:
    """Read a RoCoG-v2 annotations file into video paths and labels.

    Each line pairs a video path, relative to the data directory, with its
    numeric class label.

    Args:
        file_path: Path to the annotations file.

    Returns:
        One ``(video_path, label)`` pair per line, in file order.
    """
    entries = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            video_path, label = line.split(" ")
            video_path = Path(video_path)
            label = int(label)
            entries.append((video_path, label))

    return entries


def filter_by_repetitions(
    entries: list[tuple[Path, int]],
    data_root: Path,
    max_repetitions: int,
) -> list[tuple[Path, int]]:
    """Drop synthetic videos whose gesture repeats more than a given number of times.

    Rally is the only class affected: 1,440 of its 8,160 training videos repeat the
    gesture more than four times, one of them 98 times, while the other six classes
    never exceed four. Real clips run 3.4 to 4.9 seconds and hold at most three
    repetitions, so keeping the long synthetic outliers would train on a temporal
    structure that does not occur in the target domain.

    Videos without a metadata file are dropped as well: their repetition count
    cannot be checked, and without that file the gesture window falls back to the
    whole clip — a different sampling regime from every other synthetic video.
    Exactly one video in the training split is affected.

    Args:
        entries: Video paths and labels, as returned by ``read_annotations``.
        data_root: Directory the entry paths are relative to.
        max_repetitions: Highest repetition count to keep.

    Returns:
        The entries that passed, in the order they arrived.
    """
    accepted_videos = []
    for entry in entries:
        video_path, label = entry
        xml_path = (data_root / video_path).with_suffix(".xml")
        if xml_path.exists():
            repetitions_text = read_metadata(xml_path).findtext("numberOfRepetitions")
            number_of_repetitions = int(repetitions_text)

            if number_of_repetitions <= max_repetitions:
                accepted_videos.append(entry)

    return accepted_videos


def sample_stratified(
    entries: list[tuple[Path, int]],
    domain: str,
    num_per_stratum: int,
    rng: np.random.Generator,
) -> list[tuple[Path, int]]:
    """Draw an equal number of videos from every (class, viewpoint) stratum.

    The synthetic split is uneven on both axes that matter: gesture classes range
    from 2,400 to 8,160 videos, and camera viewpoints from 5,580 to 8,798. Drawing
    at random would carry both imbalances into the subset. Taking the same count
    from every stratum yields a subset balanced on class and viewpoint at once —
    which is what the viewpoint experiment needs, and what removes the need for
    class weights during training.

    Args:
        entries: Video paths and labels, as returned by ``read_annotations``.
        domain: Either ``"syn"`` or ``"real"``; used to read each video's
            viewpoint from its name.
        num_per_stratum: How many videos to draw from each stratum.
        rng: Generator used for the draw. Pass the same one used elsewhere in the
            run, so the whole extraction stays reproducible.

    Returns:
        The drawn videos, grouped by stratum rather than in annotations order.

    Raises:
        RuntimeError: If any stratum holds fewer videos than requested, which
            would silently unbalance the result.
    """
    strata = defaultdict(list)
    for entry in entries:
        video_path, label = entry
        gesture_class = video_path.parent.name
        view = video_metadata(video_path, domain).view
        strata[gesture_class, view].append(entry)

    sampled = []
    for stratum, group in strata.items():
        if len(group) < num_per_stratum:
            raise RuntimeError(
                f"{stratum} has {len(group)} videos, need {num_per_stratum}"
            )

        indices = rng.choice(len(group), size=num_per_stratum, replace=False)
        sampled.extend([group[i] for i in indices])

    return sampled


if __name__ == "__main__":
    start_time = time.perf_counter()
    video_counter = 0
    success_counter = 0
    failures = []
    manifest_rows = []
    rng = np.random.default_rng(SEED)

    entries = read_annotations(FILE_PATH)
    num_read = len(entries)
    class_names = load_class_names(PROJECT_ROOT / "data" / "class_dict.json")
    domain, perspective, split = FILE_PATH.stem.split("_")
    num_after_repetitions = 0
    num_sampled = 0

    if domain == "syn":
        entries = filter_by_repetitions(entries, PROJECT_ROOT / "data", MAX_REPETITIONS)
        num_after_reps = len(entries)
        entries = sample_stratified(entries, domain, NUM_PER_STRATUM, rng)
        num_sampled = len(entries)

    for video_path, label in entries:
        video_counter += 1
        gesture_class = class_names[label]

        if gesture_class != video_path.parent.name:
            raise RuntimeError(
                f"label {label} maps to {gesture_class} "
                f"but the video is in folder {video_path.parent.name}"
            )

        output_dir = PROJECT_ROOT / "data" / "frames" / domain / video_path.parent.name

        try:
            frames = extract_frames(PROJECT_ROOT / "data" / video_path, NUM_FRAMES, rng)
            metadata = video_metadata(video_path, domain)
            frame_paths = save_frames(frames, output_dir, metadata.video_id)

            for sampled, frame_path in zip(frames, frame_paths, strict=True):
                row = {
                    "domain": domain,
                    "perspective": perspective,
                    "split": split,
                    "label": label,
                    "class_name": gesture_class,
                    "video_id": metadata.video_id,
                    "group_id": metadata.group_id,
                    "view": metadata.view,
                    "is_frontal": metadata.is_frontal,
                    "frame_number": sampled.frame_number,
                    "position": sampled.position,
                    "path": frame_path.relative_to(PROJECT_ROOT),
                }
                manifest_rows.append(row)

            success_counter += 1
        except Exception as e:
            failures.append((video_path, f"{type(e).__name__}: {e}"))

    manifests_folder = PROJECT_ROOT / "data" / "manifests"
    manifests_folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(
        manifests_folder / f"{FILE_PATH.stem}.csv", index=False
    )

    end_time = time.perf_counter()
    time_taken = end_time - start_time
    time_per_video = time_taken / video_counter

    print(f"Total number of video files read: {num_read}")

    if domain == "syn":
        print(f"Total number of entries after repetition filter: {num_after_reps}")
        print(f"Total number of entries sampled: {num_sampled}")

    print(f"Total number of videos: {video_counter}")
    print(f"Total number of successfully processed videos: {success_counter}")
    print(f"Total number of videos that failed to be processed: {len(failures)}")

    if failures:
        print(f"\nFirst failures ({len(failures)} total):")
        for path, message in failures[:5]:
            print(f"  {path} -> {message}")

    print(f"Total amount of rows created in manifest: {len(manifest_rows)}")
    print(f"Total amount of time taken: {time_taken} seconds")
    print(f"Total amount of time taken per video: {time_per_video} seconds")
