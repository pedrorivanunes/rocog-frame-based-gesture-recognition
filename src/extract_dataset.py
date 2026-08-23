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

from frame_extraction import SampledFrame
from frame_extraction import extract_frames
from manifest import video_metadata
from manifest import load_class_names
from pathlib import Path
import pandas as pd
import time
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILE_PATH = PROJECT_ROOT / "data/annotations/real_ground_test.txt"
NUM_FRAMES = 10

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

if __name__ == "__main__":
    start_time = time.perf_counter()
    video_counter = 0
    success_counter = 0
    failures = []
    manifest_rows = []

    with open(FILE_PATH, "r", encoding="utf-8") as file:
        class_names = load_class_names(PROJECT_ROOT / "data" / "class_dict.json")
        domain, perspective, split = FILE_PATH.stem.split("_")

        for line in file:
            video_counter += 1
            video_path, label = line.split(" ")
            video_path = Path(video_path)
            label = int(label)
            gesture_class = class_names[label]

            if gesture_class != video_path.parent.name:
                raise RuntimeError(f"label {label} maps to {gesture_class} but the video is in folder {video_path.parent.name}")
            
            output_dir = PROJECT_ROOT / "data" / "frames" / domain / video_path.parent.name

            try:
                frames = extract_frames(PROJECT_ROOT / "data" / video_path, NUM_FRAMES)
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
                        "path": frame_path.relative_to(PROJECT_ROOT)
                    }
                    manifest_rows.append(row)

                success_counter += 1
            except Exception as e:
                failures.append((video_path, f"{type(e).__name__}: {e}"))
                
    manifests_folder = PROJECT_ROOT / "data" / "manifests"
    manifests_folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(manifests_folder / f"{FILE_PATH.stem}.csv", index=False)            
    
    end_time = time.perf_counter()
    time_taken = end_time - start_time
    time_per_video = time_taken / video_counter

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
