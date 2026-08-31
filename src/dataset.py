"""Serve extracted frames to PyTorch, one manifest row at a time.

The manifest written during extraction is the index: each row names a frame on
disk and everything known about it. This module turns a row into the tensor a
model expects, and leaves selection to whoever builds the dataset — filtering
the manifest is how a run picks a domain, a split, a viewpoint or a subset.
"""

from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CROP_SIZE = 224


def train_transform() -> v2.Transform:
    """Pipeline used while training: the crop position is drawn at random.

    Random cropping is the only augmentation applied at this stage. It keeps the
    model from anchoring on absolute pixel positions and costs nothing, since the
    frames were stored at 256 precisely to leave this margin.
    """
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomCrop(CROP_SIZE),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def eval_transform() -> v2.Transform:
    """Pipeline used for validation and testing: the crop is fixed and central.

    Evaluation must be reproducible, so nothing here is random.
    """
    return v2.Compose(
        [
            v2.ToImage(),
            v2.CenterCrop(CROP_SIZE),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class FrameDataset(Dataset):
    """Frames listed in a manifest, ready for a model.

    One item is one frame, not one video: the model classifies frames
    independently, and shuffling frames from different videos into the same
    batch is what training wants. Video-level results are recovered afterwards
    by grouping predictions on the ``video_id`` each item carries.

    The transform is injected rather than fixed, because training and evaluation
    need different pipelines — a random crop against a fixed central one — and
    both should come from the same class.

    The rows are injected for the same reason: a run trains on one selection of
    the manifest and validates on another, and neither is a file on disk.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        data_root: Path,
        transform: v2.Transform,
    ):
        """Prepare to serve the frames a manifest lists.

        Args:
            manifest: The rows to serve, already selected. A frame of rows
                rather than a path to one: choosing the data is the caller's
                job, and every experiment here is a different choice — see
                ``splits``.
            data_root: Directory the manifest's ``path`` column is relative to.
            transform: Pipeline applied to every frame. Required rather than
                optional, so a missing one fails here instead of on the first read.
        """
        self.data_frame = manifest.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        """Count frames, not videos — the manifest holds 24 rows per video."""
        return len(self.data_frame)

    def __getitem__(self, index: int):
        """Return one frame, its label, and the video it came from.

        OpenCV reads images as BGR, so the frame is converted to RGB before the
        transform: the pretrained weights were learned on RGB, and feeding the
        channels reversed silently degrades every prediction.

        Returns:
            The transformed frame as a ``(3, 224, 224)`` float tensor, the class
            label, and the ``video_id`` needed to group predictions by video.
        """
        row = self.data_frame.iloc[index]
        frame_path = self.data_root / row["path"]
        frame = cv2.imread(str(frame_path))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.transform(frame)
        return frame, row["label"], row["video_id"]
