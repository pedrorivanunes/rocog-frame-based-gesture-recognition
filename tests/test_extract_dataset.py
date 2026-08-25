from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from extract_dataset import read_annotations, sample_stratified
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
