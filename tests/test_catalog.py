import re

import pytest
import torch

from benchmark import ENTRIES
from catalog import (
    CATALOG,
    LABELLED_REAL_VIDEOS,
    METHODS,
    NETWORKS,
    NO_REAL_VIDEOS,
    POSE,
    SEEDS,
    STEMS,
    UNLABELLED_REAL_VIDEOS,
    available_seeds,
    card,
    checkpoint_paths,
    find,
)
from model import backbone_of, head_width, shift_segments, stem_width

IMAGE_NETWORKS = [entry for entry in CATALOG if entry.network is not None]


def test_every_network_offers_every_method_once_and_the_pose_stands_apart():
    pairs = [(entry.network, entry.method) for entry in IMAGE_NETWORKS]

    assert sorted(pairs) == sorted(
        (network, method) for network in NETWORKS for method in METHODS
    )
    assert CATALOG[-1] is POSE
    assert POSE.network is None


def test_keys_are_unique_and_safe_in_a_folder_name():
    keys = [entry.key for entry in CATALOG]

    assert len(set(keys)) == len(keys)
    assert all(re.fullmatch(r"[a-z0-9-]+", key) for key in keys)


@pytest.mark.parametrize("entry", IMAGE_NETWORKS, ids=lambda entry: entry.key)
def test_only_a_difference_says_how_far_back_it_looks(entry):
    """The spacing is not in the checkpoint, so a wrong one here scores in silence.

    One spacing of five frames is what the record selected and what the
    fine-tuned network started from; three spacings are two, five and ten.
    """
    expected = {
        "difference": (5,),
        "fine_tuned": (5,),
        "spacings": (2, 5, 10),
        "prior": (2, 5, 10),
    }.get(entry.method, ())

    assert entry.spacings == expected
    assert entry.channels == 3 * (1 + len(expected))


def test_only_the_shift_reads_its_frames_in_clips():
    clipped = {entry.method: entry.clip for entry in CATALOG if entry.clip is not None}

    assert clipped == {"shift": 8}


def test_only_the_prior_corrects_its_decisions_and_it_reuses_the_spacing_network():
    """The correction changes the decision, not the network, so no file of its own."""
    assert {entry.method for entry in CATALOG if entry.prior} == {"prior"}
    for network in NETWORKS:
        assert STEMS[(network, "prior")] == STEMS[(network, "spacings")]


def test_a_configuration_that_used_the_real_videos_says_how():
    used = {entry.method: entry.real_videos for entry in CATALOG}

    assert used["fine_tuned"] == LABELLED_REAL_VIDEOS
    assert used["prior"] == UNLABELLED_REAL_VIDEOS
    assert all(
        used[method] == NO_REAL_VIDEOS
        for method in ("frames", "difference", "spacings", "shift", "pose")
    )


@pytest.mark.parametrize("entry", IMAGE_NETWORKS, ids=lambda entry: entry.key)
def test_every_latency_row_is_one_the_benchmark_times(entry):
    """A decision is costed on its own architecture, never on a borrowed one."""
    rows = {benchmark_entry.name: benchmark_entry for benchmark_entry in ENTRIES}

    (row,) = entry.latency_rows
    assert rows[row].channels == entry.channels
    assert (rows[row].shift == "offline") == (entry.clip is not None)
    assert row.startswith(entry.network)


def test_the_pose_pays_for_the_detector_and_then_the_classifier():
    assert POSE.latency_rows == ("pose_detector", "pose_classifier")


def test_a_wrong_key_names_the_ones_that_exist():
    assert find("resnet18-spacings").stem == "diff_multi"
    with pytest.raises(KeyError, match="resnet18-frames"):
        find("resnet18-everything")


def test_only_the_checkpoints_present_count_as_runs(tmp_path):
    entry = find("mobilenet-v3-small-shift")
    for seed in (0, 2):
        (tmp_path / f"{entry.stem}_s{seed}.pt").touch()

    assert available_seeds(entry, tmp_path) == [0, 2]
    assert [path.name for path in checkpoint_paths(entry, tmp_path)] == [
        f"{entry.stem}_s{seed}.pt" for seed in SEEDS
    ]


def test_the_card_says_when_the_runs_here_are_not_the_thesis_figures(tmp_path):
    entry = find("resnet18-frames")

    def runs() -> str:
        return dict(card(entry, tmp_path))["training runs"]

    assert runs().startswith("none of the 3")
    (tmp_path / f"{entry.stem}_s1.pt").touch()
    assert "not the thesis's figures" in runs()
    for seed in (0, 2):
        (tmp_path / f"{entry.stem}_s{seed}.pt").touch()
    assert runs() == "3 of 3 on this computer (results average the three)"


def test_the_card_quotes_the_thesis_only_when_given_what_it_measured(tmp_path):
    entry = find("efficientnet-b0-prior")

    assert [label for label, _ in card(entry, tmp_path)] == [
        "network",
        "sees",
        "learned from",
        "training runs",
    ]
    lines = dict(card(entry, tmp_path, in_the_thesis="75.33% per video"))
    assert lines["network"] == "EfficientNet-B0"
    assert lines["in the thesis"] == "75.33% per video"


@pytest.mark.skipif(
    not all(path.is_file() for path in checkpoint_paths(IMAGE_NETWORKS[0])),
    reason="the trained checkpoints are not on this computer",
)
@pytest.mark.parametrize("entry", IMAGE_NETWORKS, ids=lambda entry: entry.key)
def test_each_configuration_points_at_checkpoints_built_as_it_says(entry):
    """The network, the width of the stem, the clip and the idle class, per file.

    Read off the first seed, which is enough to catch a stem pointing at the
    wrong cell: no two methods on one network share all four.
    """
    path = checkpoint_paths(entry)[0]
    if not path.is_file():
        pytest.skip(f"{path.name} is not on this computer")
    weights = torch.load(path, map_location="cpu", weights_only=True)

    assert backbone_of(weights) == entry.network
    assert stem_width(weights) == entry.channels
    assert shift_segments(weights) == entry.clip
    assert head_width(weights) == 8
