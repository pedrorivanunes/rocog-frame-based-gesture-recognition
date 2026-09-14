import numpy as np
import pandas as pd
import pytest
import torch

from temporal import (
    FRAME_COUNTS,
    HEADS,
    batches,
    decisions,
    evaluate,
    pick,
    scene_of,
)


def _features(videos: int = 6, frames: int = 24, width: int = 8) -> torch.Tensor:
    return torch.arange(videos * frames * width, dtype=torch.float32).reshape(
        videos, frames, width
    )


def test_a_fixed_pick_takes_the_centre_of_each_segment():
    """Evaluation must be reproducible, so nothing there may draw."""
    assert pick(24, 4, None).tolist() == [3, 9, 15, 21]


def test_a_fixed_pick_is_the_same_every_time():
    assert pick(24, 8, None).tolist() == pick(24, 8, None).tolist()


def test_a_drawn_pick_stays_inside_its_segment():
    """The draw is the augmentation; leaving the segment would break coverage."""
    generator = np.random.default_rng(0)
    for _ in range(20):
        chosen = pick(24, 4, generator)
        assert ((chosen >= [0, 6, 12, 18]) & (chosen < [6, 12, 18, 24])).all()


def test_every_pick_size_stays_within_the_clip():
    for picks in FRAME_COUNTS:
        chosen = pick(24, picks, np.random.default_rng(1))
        assert len(chosen) == picks
        assert chosen.min() >= 0
        assert chosen.max() < 24


@pytest.mark.parametrize("name", sorted(HEADS))
def test_a_head_answers_one_class_per_video_at_any_length(name):
    """A head trained at one K and read at another would be an extrapolation."""
    head = HEADS[name](8, 7)
    for picks in (1, 4, 24):
        assert head(_features()[:, :picks]).shape == (6, 7)


def test_the_recurrent_head_carries_more_parameters_than_the_pooled_one():
    """Why the control is the shuffled recurrent head and not the pooled one."""
    pooled = sum(p.numel() for p in HEADS["linear"](512, 7).parameters())
    recurrent = sum(p.numel() for p in HEADS["gru"](512, 7).parameters())

    assert recurrent > 3 * pooled


def test_a_batch_carries_one_frame_count_for_all_of_its_rows():
    """Sequences of different lengths cannot be stacked into one tensor."""
    features, labels = _features(130), torch.zeros(130, dtype=torch.long)

    for sequence, target in batches(
        features, labels, np.random.default_rng(0), shuffle_frames=False
    ):
        assert sequence.shape[0] == target.shape[0]
        assert sequence.shape[1] in FRAME_COUNTS


def test_every_video_is_served_once_per_pass():
    features, labels = _features(130), torch.arange(130)
    seen = torch.cat(
        [
            target
            for _, target in batches(
                features, labels, np.random.default_rng(0), shuffle_frames=False
            )
        ]
    )

    assert sorted(seen.tolist()) == list(range(130))


def test_shuffling_keeps_the_frames_and_changes_their_order():
    """The control destroys the sequence without touching what is in it."""
    features, labels = _features(4), torch.zeros(4, dtype=torch.long)
    generator = np.random.default_rng(0)
    sequence, _ = next(batches(features, labels, generator, shuffle_frames=True))

    plain = next(
        iter(batches(features, labels, np.random.default_rng(0), shuffle_frames=False))
    )[0]
    assert sorted(sequence.flatten().tolist()) == sorted(plain.flatten().tolist())


def test_evaluation_of_a_head_is_deterministic():
    head = HEADS["gru"](8, 7)
    features, labels = _features(), torch.zeros(6, dtype=torch.long)

    first = evaluate(head, features, labels, 8, shuffle_frames=False)
    second = evaluate(head, features, labels, 8, shuffle_frames=False)

    assert torch.equal(first, second)


def test_probabilities_sum_to_one_per_video():
    head = HEADS["conv1d"](8, 7)
    features, labels = _features(), torch.zeros(6, dtype=torch.long)

    scored = evaluate(head, features, labels, 4, shuffle_frames=False)

    assert torch.allclose(scored.sum(dim=1), torch.ones(6), atol=1e-5)


def test_the_written_table_holds_one_row_per_video_and_frame_count():
    """A sequence head answers once per clip.

    A per-frame shape would invite later tools to average what is already an
    average.
    """
    head = HEADS["linear"](8, 7)
    features, labels = _features(), torch.arange(6) % 7
    names = dict(enumerate(["a", "b", "c", "d", "e", "f", "g"]))

    table = decisions(
        head, features, labels, np.array([f"v{i}" for i in range(6)]), names, False
    )

    assert len(table) == 6 * len(FRAME_COUNTS)
    assert sorted(table["frames"].unique()) == sorted(FRAME_COUNTS)
    assert [column for column in table.columns if column.startswith("p_")] == [
        f"p_{name}" for name in "abcdefg"
    ]


def test_the_scene_of_a_video_is_read_from_the_manifest():
    """The scene is a property of the dataset, not of a checkpoint.

    So it is looked up from a manifest rather than carried in the feature file.
    """
    manifest = pd.DataFrame(
        {
            "video_id": ["a", "a", "b", "c"],
            "group_id": ["s1", "s1", "s2", "s1"],
        }
    )

    assert scene_of(np.array(["c", "a"]), manifest).tolist() == ["s1", "s1"]
