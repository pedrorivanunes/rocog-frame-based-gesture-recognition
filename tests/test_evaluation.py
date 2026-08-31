import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from evaluation import frame_metrics, predict


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
