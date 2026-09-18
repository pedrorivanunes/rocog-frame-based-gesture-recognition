import numpy as np
import pandas as pd
import pytest

from pose import (
    ARM_LANDMARKS,
    lighting_groups,
    luminance,
    spread_indices,
    visibility_summary,
)


def solid_frame(blue: int, green: int, red: int) -> np.ndarray:
    return np.full((8, 8, 3), (blue, green, red), dtype=np.uint8)


def test_luminance_weights_the_channels_by_perceived_brightness():
    """A flat channel mean would call a green field as bright as the sky.

    This dataset is mostly field, so the distinction decides which clips the
    probe reports as dark.
    """
    green = luminance(solid_frame(0, 255, 0)).level
    red = luminance(solid_frame(0, 0, 255)).level
    blue = luminance(solid_frame(255, 0, 0)).level

    assert green > red > blue
    assert green == pytest.approx(149.685)


def test_luminance_spread_separates_a_silhouette_from_an_even_scene():
    """Two frames can share a brightness and differ in what a detector sees."""
    even = np.full((8, 8, 3), 128, dtype=np.uint8)

    silhouette = np.zeros((8, 8, 3), dtype=np.uint8)
    silhouette[:4] = 255

    assert luminance(even).level == pytest.approx(128.0)
    assert luminance(even).spread == pytest.approx(0.0)
    assert luminance(silhouette).level == pytest.approx(127.5)
    assert luminance(silhouette).spread == pytest.approx(127.5)


def test_visibility_counts_only_joints_at_or_above_the_floor():
    scores = [0.0] * 33
    for index, score in zip(
        ARM_LANDMARKS, [0.9, 0.5, 0.49, 0.2, 0.8, 0.1], strict=True
    ):
        scores[index] = score

    summary = visibility_summary(scores, floor=0.5)

    assert summary.located == 3
    assert summary.median == pytest.approx(0.495)


def test_visibility_reads_the_arm_joints_and_not_the_rest():
    """The gestures are signalled with the arms.

    A skeleton scored over all thirty-three points can look confident while
    every joint the task depends on was guessed.
    """
    scores = [0.95] * 33
    for index in ARM_LANDMARKS:
        scores[index] = 0.1

    summary = visibility_summary(scores)

    assert summary.located == 0
    assert summary.median == pytest.approx(0.1)


def test_visibility_refuses_a_topology_it_does_not_recognise():
    with pytest.raises(IndexError):
        visibility_summary([0.9] * 8)


def test_spread_indices_takes_the_centre_of_each_segment():
    assert spread_indices(24, 4) == [3, 9, 15, 21]
    assert spread_indices(10, 4) == [1, 3, 6, 8]


def test_spread_indices_returns_everything_when_asked_for_more_than_exists():
    assert spread_indices(3, 4) == [0, 1, 2]
    assert spread_indices(4, 4) == [0, 1, 2, 3]


def test_lighting_groups_put_the_darkest_first():
    levels = pd.Series([10.0, 200.0, 50.0, 150.0], index=["a", "b", "c", "d"])

    groups = lighting_groups(levels, groups=2)

    assert groups["a"] == 0
    assert groups["c"] == 0
    assert groups["d"] == 1
    assert groups["b"] == 1


def test_lighting_groups_stay_balanced_when_videos_share_a_brightness():
    """Cutting on brightness itself would leave a group holding four videos.

    A detection rate over four videos says nothing, so the groups are ranks.
    """
    levels = pd.Series([5.0] * 8, index=list("abcdefgh"))

    counts = lighting_groups(levels, groups=4).value_counts()

    assert sorted(counts.tolist()) == [2, 2, 2, 2]
