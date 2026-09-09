from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from extract_dataset import (
    frame_paths,
    parse_args,
    read_annotations,
    sample_stratified,
    video_rng,
)
from frame_extraction import SampledFrame
from manifest import video_metadata


def synthetic_entries(gestures, views, per_stratum):
    """Build annotation entries following the RoCoG-v2 synthetic naming scheme.

    No files are created. video_metadata reads names only, so the paths never
    have to exist on disk, and the scene index doubles as the viewpoint because
    views here are kept below six.
    """
    entries = []
    for label, gesture in enumerate(gestures):
        for view in views:
            for index in range(per_stratum):
                name = f"Scene{view}_{index}_{gesture}_1_1_2026_0_0_0.mp4"
                entries.append((Path("syn/ground") / gesture / name, label))
    return entries


def strata_sizes(entries):
    """Count how many entries fall into each (gesture, viewpoint) stratum."""
    return Counter(
        (path.parent.name, video_metadata(path, "syn").view) for path, _ in entries
    )


def test_every_stratum_contributes_the_requested_number():
    entries = synthetic_entries(["Halt", "Rally"], range(6), 10)

    drawn = sample_stratified(entries, "syn", 4, np.random.default_rng(0))

    sizes = strata_sizes(drawn)
    assert len(sizes) == 12
    assert set(sizes.values()) == {4}
    assert len(drawn) == 48


def test_drawn_videos_are_distinct():
    entries = synthetic_entries(["Halt", "Rally"], range(6), 10)

    drawn = sample_stratified(entries, "syn", 10, np.random.default_rng(0))

    assert len({path for path, _ in drawn}) == len(drawn)


def test_the_same_seed_draws_the_same_videos():
    entries = synthetic_entries(["Halt", "Rally"], range(6), 10)

    first = sample_stratified(entries, "syn", 4, np.random.default_rng(1))
    second = sample_stratified(entries, "syn", 4, np.random.default_rng(1))

    assert first == second


def test_a_different_seed_draws_different_videos():
    entries = synthetic_entries(["Halt", "Rally"], range(6), 10)

    first = sample_stratified(entries, "syn", 4, np.random.default_rng(1))
    second = sample_stratified(entries, "syn", 4, np.random.default_rng(2))

    assert first != second


def test_a_stratum_smaller_than_requested_is_rejected():
    """An unbalanced draw would surface much later as an unexplained result."""
    entries = synthetic_entries(["Halt"], range(6), 10)

    with pytest.raises(RuntimeError):
        sample_stratified(entries, "syn", 11, np.random.default_rng(0))


def test_parse_args_reads_the_annotations_file_as_a_path():
    args = parse_args(["data/annotations/syn_ground_train.txt"])

    assert args.annotations == Path("data/annotations/syn_ground_train.txt")


def test_parse_args_requires_the_annotations_file():
    """The file name is what a run chooses; there is no sensible default."""
    with pytest.raises(SystemExit):
        parse_args([])


def test_read_annotations_pairs_paths_with_integer_labels(tmp_path):
    annotations = tmp_path / "syn_ground_train.txt"
    annotations.write_text(
        "syn/ground/Halt/Scene0_0_Halt_1_1_2026_0_0_0.mp4 4\n"
        "syn/ground/Rally/Scene1_0_Rally_1_1_2026_0_0_0.mp4 2\n"
    )

    entries = read_annotations(annotations)

    assert entries == [
        (Path("syn/ground/Halt/Scene0_0_Halt_1_1_2026_0_0_0.mp4"), 4),
        (Path("syn/ground/Rally/Scene1_0_Rally_1_1_2026_0_0_0.mp4"), 2),
    ]
    assert isinstance(entries[0][1], int)


def test_the_same_video_is_given_the_same_frames_whatever_its_position():
    """Resuming skips videos, so a draw that moves with the stream would move."""
    first = video_rng("Scene1_0_Halt_1_1_2026_0_0_0").integers(0, 1000, size=24)
    second = video_rng("Scene1_0_Halt_1_1_2026_0_0_0").integers(0, 1000, size=24)

    assert np.array_equal(first, second)


def test_different_videos_are_given_different_frames():
    first = video_rng("Scene1_0_Halt_1_1_2026_0_0_0").integers(0, 1000, size=24)
    second = video_rng("Scene1_1_Halt_1_1_2026_0_0_0").integers(0, 1000, size=24)

    assert not np.array_equal(first, second)


def test_a_different_seed_places_frames_differently():
    first = video_rng("Scene1_0_Halt_1_1_2026_0_0_0", seed=1).integers(0, 1000, size=24)
    second = video_rng("Scene1_0_Halt_1_1_2026_0_0_0", seed=2).integers(
        0, 1000, size=24
    )

    assert not np.array_equal(first, second)


def test_a_run_samples_across_the_gesture_by_default():
    """The pass that built every manifest so far has to keep its old command."""
    args = parse_args(["data/annotations/syn_ground_train.txt"])

    assert args.idle is False


def test_a_run_can_ask_for_the_stretch_before_the_gesture():
    args = parse_args(["data/annotations/syn_ground_train.txt", "--idle"])

    assert args.idle is True


def test_a_frame_is_named_for_the_video_and_the_number_it_carries():
    """The number is what leads back to the video.

    The mask pass reads it, and so does any later pass re-rendering these frames
    at another size.
    """
    frames = [SampledFrame(7, 0.5, None), SampledFrame(140, 0.9, None)]

    assert frame_paths(Path("data/frames/syn/Halt"), "Scene1_Halt", frames) == [
        Path("data/frames/syn/Halt/Scene1_Halt_f0007.jpg"),
        Path("data/frames/syn/Halt/Scene1_Halt_f0140.jpg"),
    ]
