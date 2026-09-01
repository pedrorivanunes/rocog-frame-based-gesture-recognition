import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from evaluation import frame_metrics, predict, probability_table


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
