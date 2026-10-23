"""Turn any video into the frames a trained model is scored on, and keep them.

The models here never read a video. They read frames that an extraction pass
chose, resized and stored as JPEG, listed in a manifest, and every figure the
project reports was computed on those files. A program that takes videos as its
input therefore has to repeat that pass exactly, or it would score a model on
images that differ from the ones its figures came from: slightly, but enough to
move a frame that sat near a tie between two gestures. So nothing here is
reimplemented. The frames are decoded, resized and written by the same code at
the same size and quality, and listed with the columns the scoring code reads.

Which frames, though, is not the sampler's to say for a video the dataset
ships. Its manifests record the frames every figure was computed on, and a draw
repeated today lands elsewhere, because the generator was seeded differently
when the first manifests were written. A dataset video is therefore read at the
frame numbers its manifest records; only a video from outside the dataset, which
has no record, is sampled the way extraction samples now.

The frames go into a cache of their own, never into the dataset's tree.
Extracting a dataset video again would write the very file names the measured
manifests point at, and a copy made on another machine, with another build of
the video decoder, is not guaranteed to match them byte for byte. The frames
that were measured have to stay the ones that were measured.

A video is extracted once. Its rows are written after its frames, so a cached
row file is proof that every frame it lists is on disk, and asking for the same
video again reads the rows back instead of decoding it a second time.
"""

import zlib
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from extract_dataset import NUM_FRAMES, frame_paths, read_annotations, video_rng
from frame_extraction import (
    SampledFrame,
    extract_all_frames,
    extract_frames,
    extract_frames_at,
)
from frame_store import save_frames
from manifest import FRAME_SIZE, video_metadata

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where prepared frames and their rows live, relative to the project root. Under
# data/, which no commit carries, and apart from data/frames/, which holds the
# frames every reported figure was computed on.
CACHE_DIR = Path("data") / "cache"

# The label of a video nobody said the gesture of. Outside the range of every
# class index, so it can never be read as one.
UNLABELLED = -1

DATASET_DOMAINS = ("real", "syn")
EXTERNAL = "external"


class VideoSource(NamedTuple):
    """A video to score, and what is known about it before it is opened.

    Attributes:
        path: The video file.
        domain: ``"real"`` or ``"syn"`` for a video from the dataset, which
            fixes how its subject or scene is read off its name, and
            ``"external"`` for any other file.
        label: Its gesture as a class index, or ``UNLABELLED``.
        frame_numbers: The frames a manifest records for it, in the manifest's
            order. Empty when there is no record, and the frames are then drawn
            the way extraction draws them.
    """

    path: Path
    domain: str
    label: int = UNLABELLED
    frame_numbers: tuple[int, ...] = ()


def dataset_videos(
    annotations: Path, data_dir: Path, manifests_dir: Path | None = None
) -> list[VideoSource]:
    """List the videos an annotations file names, with their labels and frames.

    Args:
        annotations: A RoCoG-v2 annotations file. Its name starts with the
            domain, as every file the dataset ships does.
        data_dir: Directory the file's video paths are relative to.
        manifests_dir: Where the manifest extracted from that file lives, under
            the same name. When it is there, each video carries the frames the
            manifest records; when it is not, its frames will be drawn.

    Returns:
        One source per line, in file order.
    """
    domain = annotations.stem.split("_")[0]
    recorded: dict[str, tuple[int, ...]] = {}
    if manifests_dir is not None:
        manifest_path = manifests_dir / f"{annotations.stem}.csv"
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path, usecols=["video_id", "frame_number"])
            recorded = {
                video_id: tuple(int(number) for number in numbers)
                for video_id, numbers in manifest.groupby("video_id", sort=False)[
                    "frame_number"
                ]
            }

    return [
        VideoSource(data_dir / path, domain, label, recorded.get(Path(path).stem, ()))
        for path, label in read_annotations(annotations)
    ]


def cache_key(source: VideoSource) -> str:
    """Name a video's place in the cache.

    A dataset video read at its recorded frames is named by its identifier,
    which is stable: the dataset ships each file once. One whose frames had to
    be drawn is marked as such, so that the two sets never share a folder. Any
    other file is named by its stem and a fingerprint of where it is and when it
    last changed, because two recordings can share a name, and a clip recorded
    again under the same name would otherwise be scored on the frames of the one
    it replaced.

    Args:
        source: The video.

    Returns:
        A name unique to this file and to how its frames are chosen.
    """
    stem = source.path.stem
    if source.domain in DATASET_DOMAINS:
        return stem if source.frame_numbers else f"{stem}-drawn"

    status = source.path.stat()
    signature = f"{source.path.resolve()}|{status.st_size}|{status.st_mtime_ns}"
    return f"{stem}-{zlib.crc32(signature.encode()):08x}"


def identify(source: VideoSource) -> tuple[str, str]:
    """Name a video and the group it must not be separated from.

    Args:
        source: The video.

    Returns:
        Its id, the file name without extension as in every manifest, and its
        group: the subject of a real video, the scene of a synthetic one, and
        ``EXTERNAL`` for a file from outside the dataset.
    """
    if source.domain in DATASET_DOMAINS:
        metadata = video_metadata(source.path, source.domain)
        return metadata.video_id, metadata.group_id

    return source.path.stem, EXTERNAL


def prepare(
    source: VideoSource,
    class_names: dict[int, str],
    dense: bool = False,
    size: int = FRAME_SIZE,
    root: Path = PROJECT_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """The frames a video is scored on, extracted on first request and kept.

    Args:
        source: The video.
        class_names: Label to gesture name, to fill the rows' class column.
        dense: Also keep every frame of the gesture window, which is where the
            neighbours of a network reading differences are looked up.
        size: Side the frames are stored at. The dataset's frames are 256; a
            run trained at a larger resolution read a copy of the same frames
            at its own size.
        root: Directory the cache lives under, and the one every path in the
            returned rows is relative to.

    Returns:
        The sampled rows, two dozen per video as the dataset's manifests list
        them, and every frame of the window when ``dense`` is asked for.
    """
    sampled = _cached_rows(source, class_names, "sampled", size, root)
    window = _cached_rows(source, class_names, "dense", size, root) if dense else None

    return sampled, window


def _cached_rows(
    source: VideoSource,
    class_names: dict[int, str],
    kind: str,
    size: int,
    root: Path,
) -> pd.DataFrame:
    """Read one kind of row for a video back from the cache, extracting if absent.

    The two kinds keep separate folders. They name some of the same frame
    numbers, and a frame read by seeking and the same frame read in a run are
    one image only as long as the decoder seeks exactly; keeping them apart
    means neither kind is ever overwritten by the other, whatever order they
    were asked for in. A frame stored at another size lands under a folder named
    for that size, which is how the scoring code tells the two apart.
    """
    key = cache_key(source)
    sized = Path() if size == FRAME_SIZE else Path(str(size))
    rows_path = (
        root / CACHE_DIR / "manifests" / sized / source.domain / f"{key}_{kind}.csv"
    )
    if rows_path.exists():
        return pd.read_csv(rows_path, keep_default_na=False)

    video_id, group_id = identify(source)
    if kind == "dense":
        frames = extract_all_frames(source.path)
    elif source.frame_numbers:
        frames = extract_frames_at(source.path, list(source.frame_numbers))
    else:
        frames = extract_frames(source.path, NUM_FRAMES, video_rng(video_id))
    output_dir = root / CACHE_DIR / "frames" / sized / source.domain / key / kind
    paths = frame_paths(output_dir, video_id, frames)
    save_frames([sampled.frame for sampled in frames], paths, size)

    rows = _rows(source, class_names, video_id, group_id, frames, paths, root)
    # Written aside and moved into place, so that a row file is either whole or
    # absent: one cut short would be read back as a video with fewer frames.
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    partial = rows_path.with_suffix(".partial")
    rows.to_csv(partial, index=False)
    partial.replace(rows_path)

    # Read back rather than returned as built, so that a video met for the
    # first time and one met again come back with exactly the same types.
    return pd.read_csv(rows_path, keep_default_na=False)


def _rows(
    source: VideoSource,
    class_names: dict[int, str],
    video_id: str,
    group_id: str,
    frames: list[SampledFrame],
    paths: list[Path],
    root: Path,
) -> pd.DataFrame:
    """One row per stored frame, with the columns the scoring code reads."""
    return pd.DataFrame(
        {
            "domain": source.domain,
            "label": source.label,
            "class_name": class_names.get(source.label, ""),
            "video_id": video_id,
            "group_id": group_id,
            "frame_number": [sampled.frame_number for sampled in frames],
            "position": [sampled.position for sampled in frames],
            "path": [str(path.relative_to(root)) for path in paths],
        }
    )
