"""Extract frames from every video listed in a RoCoG-v2 annotations file.

Reads an annotations file, samples frames from each video it lists, and
reports how many videos succeeded, how many failed and why, and how long the
pass took.

Nothing is written to disk yet. This pass exists to validate the traversal
and to measure per-video cost before committing to a full extraction run.

Paths are resolved from the location of this file, so it can be run from any
working directory:

    python src/extract_dataset.py
"""

from frame_extraction import extract_frames
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NUM_FRAMES = 10

if __name__ == "__main__":
    start_time = time.perf_counter()
    video_counter = 0
    success_counter = 0
    failures = []
    file_path = PROJECT_ROOT / "data/annotations/real_ground_test.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            video_counter += 1
            video_path, label = line.split(" ")
            label = label.rstrip()
            try:
                extract_frames(PROJECT_ROOT / "data" / video_path, NUM_FRAMES)
                success_counter += 1
            except Exception as e:
                failures.append((video_path, f"{type(e).__name__}: {e}"))
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
    print(f"Total amount of time taken: {time_taken} seconds")
    print(f"Total amount of time taken per video: {time_per_video} seconds")

