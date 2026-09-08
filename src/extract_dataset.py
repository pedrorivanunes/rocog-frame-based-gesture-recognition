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

A pass can be interrupted and run again: the manifest is written as each
video finishes, and a video already listed there in full is not extracted a
second time. Deleting the manifest is what forces a pass to start over.

Reports how many videos succeeded, how many were already done, how many failed
and why, and how long the pass took.

Output paths are anchored to this file's location, so it runs from any working
directory:

    python src/extract_dataset.py data/annotations/syn_ground_train.txt
"""

import argparse
import time
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from frame_extraction import SampledFrame, extract_frames, read_metadata
from manifest import load_class_names, video_metadata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NUM_FRAMES = 24
NUM_PER_STRATUM = 250
MAX_REPETITIONS = 4
SEED = 42


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line: which annotations file to turn into frames.

    Args:
        argv: Arguments to parse. ``None`` reads ``sys.argv``.

    Returns:
        A namespace with ``annotations``, the path to the RoCoG-v2 annotations
        file whose videos should be extracted. Domain, perspective and split
        follow from its name, so it is the only thing a run has to choose.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "annotations",
        type=Path,
        help="RoCoG-v2 annotations file to extract, "
        "e.g. data/annotations/syn_ground_train.txt",
    )
    return parser.parse_args(argv)


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


def video_rng(video_id: str, seed: int = SEED) -> np.random.Generator:
    """Build the generator that places one video's frames.

    Where a video's frames land must not depend on how many videos came before
    it. One generator threaded through the whole pass makes it depend on exactly
    that: a video is given whatever the stream had reached by the time its turn
    arrived, so a pass that resumes — skipping everything already done, and so
    consuming none of those draws — hands every remaining video a different set
    of frames than an uninterrupted pass would. The resumed extraction would
    still be valid, but it would no longer be the one the seed names.

    Seeding from the video's own name takes ordering out of the question. A
    video receives the same frames whether it is extracted first or last, and
    whether or not the pass was interrupted.

    ``zlib.crc32`` rather than the built-in ``hash``: Python salts string
    hashing per process, so ``hash`` would place the frames somewhere new on
    every run.

    Args:
        video_id: Identifier of the video, as ``video_metadata`` builds it.
        seed: Fixes the extraction as a whole. The same seed and the same video
            always yield the same frames.

    Returns:
        A generator seeded for this video and no other.
    """
    return np.random.default_rng([seed, zlib.crc32(video_id.encode())])


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


if __name__ == "__main__":
    args = parse_args()
    print(f"annotations: {args.annotations}")

    start_time = time.perf_counter()
    video_counter = 0
    success_counter = 0
    skipped_counter = 0
    rows_written = 0
    failures = []
    selection_rng = np.random.default_rng(SEED)

    entries = read_annotations(args.annotations)
    num_read = len(entries)
    class_names = load_class_names(PROJECT_ROOT / "data" / "class_dict.json")
    domain, perspective, split = args.annotations.stem.split("_")
    num_after_repetitions = 0
    num_sampled = 0

    if domain == "syn":
        entries = filter_by_repetitions(entries, PROJECT_ROOT / "data", MAX_REPETITIONS)
        num_after_reps = len(entries)
        entries = sample_stratified(entries, domain, NUM_PER_STRATUM, selection_rng)
        num_sampled = len(entries)

    manifest_path = PROJECT_ROOT / "data" / "manifests" / f"{args.annotations.stem}.csv"
    dropped_rows = drop_incomplete_videos(manifest_path, NUM_FRAMES)
    already_extracted = completed_videos(manifest_path, NUM_FRAMES)
    print(f"videos already in the manifest: {len(already_extracted)}")
    print(f"rows dropped from videos left half written: {dropped_rows}")

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
            metadata = video_metadata(video_path, domain)
            if metadata.video_id in already_extracted:
                skipped_counter += 1
                continue

            frames = extract_frames(
                PROJECT_ROOT / "data" / video_path,
                NUM_FRAMES,
                video_rng(metadata.video_id),
            )
            frame_paths = save_frames(frames, output_dir, metadata.video_id)

            rows = []
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
                rows.append(row)

            append_rows(rows, manifest_path)
            rows_written += len(rows)
            success_counter += 1
        except Exception as e:
            failures.append((video_path, f"{type(e).__name__}: {e}"))

    end_time = time.perf_counter()
    time_taken = end_time - start_time
    time_per_video = time_taken / video_counter

    print(f"Total number of video files read: {num_read}")

    if domain == "syn":
        print(f"Total number of entries after repetition filter: {num_after_reps}")
        print(f"Total number of entries sampled: {num_sampled}")

    print(f"Total number of videos: {video_counter}")
    print(f"Total number of videos skipped as already extracted: {skipped_counter}")
    print(f"Total number of successfully processed videos: {success_counter}")
    print(f"Total number of videos that failed to be processed: {len(failures)}")

    if failures:
        print(f"\nFirst failures ({len(failures)} total):")
        for path, message in failures[:5]:
            print(f"  {path} -> {message}")

    print(f"Total amount of rows appended to the manifest: {rows_written}")
    print(f"Total amount of time taken: {time_taken} seconds")
    print(f"Total amount of time taken per video: {time_per_video} seconds")
