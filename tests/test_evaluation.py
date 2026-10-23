from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from evaluation import (
    frame_metrics,
    load_scorer,
    manifest_places,
    parse_args,
    predict,
    probability_table,
    score_frames,
    with_neighbour_frames,
)
from model import build_model, insert_temporal_shift, load_shifted, set_segments


def test_parse_args_leaves_the_optional_settings_alone():
    args = parse_args(
        ["checkpoints/syn_ground_train.pt", "--manifest", "real_ground_test.csv"]
    )

    assert args.checkpoint == Path("checkpoints/syn_ground_train.pt")
    assert args.manifest == "real_ground_test.csv"
    assert args.validation_split is False
    assert args.adapt_bn is None
    assert args.output is None
    assert args.num_workers == 12


def test_parse_args_takes_a_manifest_to_adapt_the_statistics_on():
    """Off by default, because every table written before it was scored without."""
    adapted = parse_args(
        [
            "checkpoints/idle_s0.pt",
            "--manifest",
            "real_ground_test.csv",
            "--adapt-bn",
            "real_ground_train.csv",
        ]
    )

    assert adapted.adapt_bn == "real_ground_train.csv"


def test_parse_args_takes_the_validation_split():
    split = parse_args(
        [
            "checkpoints/es_none.pt",
            "--manifest",
            "syn_ground_train.csv",
            "--validation-split",
        ]
    )

    assert split.validation_split is True


def test_parse_args_requires_the_checkpoint():
    with pytest.raises(SystemExit):
        parse_args(["--manifest", "real_ground_test.csv"])


def test_parse_args_requires_the_manifest():
    """A default would make scoring the training frames the easy mistake."""
    with pytest.raises(SystemExit):
        parse_args(["checkpoints/es_none.pt"])


class NumberedFrames(Dataset):
    """Serves frames that carry their own position, so order can be traced."""

    def __init__(self, frames, frames_per_video=3):
        self.frames = frames
        self.frames_per_video = frames_per_video

    def __len__(self):
        return self.frames

    def __getitem__(self, index):
        frame = torch.full((3, 2, 2), float(index))
        return frame, index % 7, f"video{index // self.frames_per_video}"


class FirstPixel(nn.Module):
    """Scores every class from the frame's first pixel, carrying it through."""

    def forward(self, frames):
        return frames.reshape(len(frames), -1)[:, :1].repeat(1, 7)


def test_predict_preserves_the_order_the_loader_served():
    """A silent reorder would corrupt every analysis built on the scores."""
    loader = DataLoader(NumberedFrames(10), batch_size=4)

    logits, labels, video_ids = predict(FirstPixel(), loader, torch.device("cpu"))

    assert logits[:, 0].tolist() == [float(index) for index in range(10)]
    assert labels.tolist() == [index % 7 for index in range(10)]
    assert video_ids == [f"video{index // 3}" for index in range(10)]


def test_predict_leaves_the_model_in_evaluation_mode():
    """Batch normalization behaves differently between the two modes."""
    model = FirstPixel()
    model.train()

    predict(model, DataLoader(NumberedFrames(4), batch_size=2), torch.device("cpu"))

    assert not model.training


def test_frame_metrics_counts_every_frame():
    """Accuracy here is per frame, which is the quantity the project is about."""
    logits = torch.tensor([[9.0, 0.0], [0.0, 9.0], [9.0, 0.0], [9.0, 0.0]])
    labels = torch.tensor([0, 1, 1, 0])

    _, accuracy = frame_metrics(logits, labels, nn.CrossEntropyLoss())

    assert accuracy == 0.75


def gesture_manifest():
    """Two frames of one video and one of another, as extraction orders them."""
    return pd.DataFrame(
        {
            "video_id": ["videoA", "videoA", "videoB"],
            "frame_number": [4, 9, 2],
            "position": [0.1, 0.8, 0.5],
            "label": [0, 0, 1],
            "class_name": ["Halt", "Halt", "Rally"],
        }
    )


def test_probability_table_names_columns_by_class():
    """A column ordered by anything but the label would mislabel every score."""
    logits = torch.tensor([[9.0, 0.0], [9.0, 0.0], [0.0, 9.0]])

    table = probability_table(
        logits,
        ["videoA", "videoA", "videoB"],
        gesture_manifest(),
        {0: "Halt", 1: "Rally"},
    )

    assert list(table.columns) == [
        "video_id",
        "frame_number",
        "position",
        "label",
        "p_Halt",
        "p_Rally",
    ]
    assert table["p_Halt"].idxmax() == 0
    assert table["p_Rally"].idxmax() == 2


def test_probability_table_rows_sum_to_one():
    logits = torch.tensor([[2.0, 1.0], [0.0, 3.0], [1.0, 1.0]])

    table = probability_table(
        logits,
        ["videoA", "videoA", "videoB"],
        gesture_manifest(),
        {0: "Halt", 1: "Rally"},
    )

    assert table[["p_Halt", "p_Rally"]].sum(axis=1).round(6).tolist() == [1.0, 1.0, 1.0]


def test_probability_table_rejects_scores_out_of_order():
    """Misaligned scores would attach every probability to the wrong frame."""
    logits = torch.tensor([[9.0, 0.0], [9.0, 0.0], [0.0, 9.0]])

    with pytest.raises(RuntimeError):
        probability_table(
            logits,
            ["videoB", "videoA", "videoA"],
            gesture_manifest(),
            {0: "Halt", 1: "Rally"},
        )


def test_predict_does_not_hold_on_to_the_batches_it_was_served():
    """A loader with workers backs each batch by a file descriptor.

    Keeping the tensor keeps the descriptor, so a split of a quarter of a
    million frames — four thousand batches — runs past the open-file limit
    before it finishes. The check has to happen while the loop is still
    running: by the time predict returns, its own list is out of scope and a
    leak would look identical to none.
    """
    import gc
    import weakref

    still_held = []

    def loader():
        # The batch two iterations back is the one to ask about: predict's own
        # loop variable still points at the most recent one, whether or not it
        # also kept a copy, so checking that one would fail either way.
        references = []
        for _ in range(4):
            labels = torch.tensor([0, 1])
            references.append(weakref.ref(labels))
            yield torch.zeros(2, 3, 2, 2), labels, ["a", "b"]
            del labels
            gc.collect()
            if len(references) >= 2:
                still_held.append(references[-2]() is not None)

    class _Flat(nn.Module):
        def forward(self, batch):
            return batch.flatten(1)[:, :7]

    predict(_Flat(), loader(), torch.device("cpu"))

    assert still_held and not any(still_held)


def saved(model: nn.Module, path: Path) -> Path:
    """Write a network's state as training does, and hand back where."""
    torch.save(model.state_dict(), path)
    return path


def test_a_checkpoint_rebuilds_into_the_network_that_wrote_it(tmp_path):
    """Head, backbone and stem are read off the file, then every weight."""
    trained = build_model(8, "mobilenet_v3_small", 6, pretrained=False).eval()
    path = saved(trained, tmp_path / "trained.pt")

    scorer = load_scorer(path, torch.device("cpu"))

    frames = torch.randn(2, 6, 224, 224)
    assert (scorer.num_classes, scorer.backbone, scorer.in_channels) == (
        8,
        "mobilenet_v3_small",
        6,
    )
    assert (scorer.segments, scorer.recorded_segments, scorer.arm) == (None,) * 3
    assert torch.equal(scorer.model.eval()(frames), trained(frames))


def test_a_shifted_checkpoint_says_which_arm_and_clip_it_was_trained_with(tmp_path):
    trained = build_model(7, "resnet18", pretrained=False)
    insert_temporal_shift(trained, "resnet18", segments=8, shuffle=False, causal=True)
    path = saved(trained, tmp_path / "causal.pt")

    as_trained = load_scorer(path, torch.device("cpu"))
    shorter = load_scorer(path, torch.device("cpu"), clips=4)

    assert (as_trained.segments, as_trained.recorded_segments) == (8, 8)
    assert as_trained.arm == "causal"
    assert (shorter.segments, shorter.recorded_segments) == (4, 8)


def test_a_shifted_checkpoint_is_rebuilt_moving_the_share_it_trained_with(tmp_path):
    """A network that learned to read a quarter answers differently given an eighth."""
    trained = build_model(7, "resnet18", pretrained=False).eval()
    insert_temporal_shift(trained, "resnet18", segments=8, fold_divisor=4, causal=True)
    path = saved(trained, tmp_path / "wide.pt")
    narrow = build_model(7, "resnet18", pretrained=False).eval()
    insert_temporal_shift(narrow, "resnet18", segments=8, causal=True)
    load_shifted(narrow, trained.state_dict())

    scorer = load_scorer(path, torch.device("cpu"))

    clip = torch.randn(8, 3, 64, 64)
    for network in (trained, narrow, scorer.model.eval()):
        set_segments(network, 8)
    with torch.no_grad():
        expected, rebuilt, misread = trained(clip), scorer.model(clip), narrow(clip)
    assert scorer.fold_divisor == 4
    assert torch.allclose(rebuilt, expected, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(misread, expected, rtol=1e-3, atol=1e-4)


def test_a_clip_length_means_nothing_to_a_network_without_a_shift(tmp_path):
    """Its prediction for a frame does not depend on what arrived beside it."""
    plain = build_model(7, "mobilenet_v3_small", pretrained=False)
    path = saved(plain, tmp_path / "plain.pt")

    assert load_scorer(path, torch.device("cpu"), clips=4).segments is None


def scored_and_dense():
    """Two scored frames of one video, and the dense window they sit in."""
    dense = pd.DataFrame(
        {
            "video_id": ["v"] * 6,
            "frame_number": [10, 11, 12, 13, 14, 15],
            "path": [f"v_f{number}.jpg" for number in range(10, 16)],
        }
    )
    scored = dense[dense["frame_number"].isin([12, 15])].reset_index(drop=True)
    return scored, dense


def test_a_checkpoint_reading_only_the_frame_is_scored_on_the_rows_as_they_are():
    scored, dense = scored_and_dense()

    assert with_neighbour_frames(scored, 3, [5], dense=dense) is scored


def test_each_spacing_names_the_frame_that_far_back_within_the_window():
    """Frame 12 has no frame three back inside the window, so it takes the first."""
    scored, dense = scored_and_dense()

    named = with_neighbour_frames(scored, 9, [1, 3], dense=dense)

    assert named["neighbour_path"].tolist() == ["v_f11.jpg", "v_f14.jpg"]
    assert named["neighbour_path_1"].tolist() == ["v_f10.jpg", "v_f12.jpg"]


def test_the_anchor_differences_each_frame_against_the_one_scored_before_it():
    scored, _ = scored_and_dense()

    named = with_neighbour_frames(scored, 6, anchor=True)

    assert named["neighbour_path"].tolist() == ["v_f12.jpg", "v_f12.jpg"]


def test_a_checkpoint_reading_differences_has_to_be_told_where_they_came_from():
    scored, dense = scored_and_dense()

    with pytest.raises(ValueError, match="as spacings or as the anchor"):
        with_neighbour_frames(scored, 6, None, dense=dense)


def test_spacings_the_stem_was_not_widened_for_are_refused():
    """Left to the network, this would be a shape error after every frame loaded."""
    scored, dense = scored_and_dense()

    with pytest.raises(ValueError, match="trained on 1 spacing"):
        with_neighbour_frames(scored, 6, [2, 5], dense=dense)


def test_a_pass_served_in_order_has_nothing_to_put_back():
    scored, _ = scored_and_dense()

    assert manifest_places(scored, None) is None


def test_a_pass_served_in_clips_is_put_back_by_a_permutation():
    """Every frame is served once, so undoing the order loses none of them."""
    rows = pd.DataFrame(
        {"video_id": ["a"] * 8 + ["b"] * 8, "frame_number": list(range(8)) * 2}
    )

    places = manifest_places(rows, 4)

    assert sorted(places) == list(range(16))


class MeanBrightness(nn.Module):
    """Scores every class by how bright the frame is, so order can be read back."""

    def forward(self, frames):
        return frames.mean(dim=(1, 2, 3)).unsqueeze(1).repeat(1, 7)


def test_scores_come_back_in_the_order_the_manifest_lists_its_frames(tmp_path):
    """Brightness rises down the manifest, so the scores must rise with it."""
    rows = []
    for video, label in (("a", 0), ("b", 1)):
        for index in range(3):
            path = Path("frames") / f"{video}_{index}.jpg"
            (tmp_path / path).parent.mkdir(exist_ok=True)
            shade = 40 * index + (120 if video == "b" else 0)
            cv2.imwrite(str(tmp_path / path), np.full((256, 256, 3), shade, np.uint8))
            rows.append(
                {
                    "video_id": video,
                    "frame_number": index,
                    "label": label,
                    "path": str(path),
                }
            )

    logits, labels, video_ids = score_frames(
        MeanBrightness(),
        pd.DataFrame(rows),
        torch.device("cpu"),
        root=tmp_path,
        num_workers=0,
    )

    assert video_ids == ["a"] * 3 + ["b"] * 3
    assert labels.tolist() == [0, 0, 0, 1, 1, 1]
    assert logits[:, 0].tolist() == sorted(logits[:, 0].tolist())
