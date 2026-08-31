"""Serve extracted frames to PyTorch, one manifest row at a time.

The manifest written during extraction is the index: each row names a frame on
disk and everything known about it. This module turns a row into the tensor a
model expects, and leaves selection to whoever builds the dataset — filtering
the manifest is how a run picks a domain, a split, a viewpoint or a subset.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CROP_SIZE = 224
SAMPLER_SEED = 13


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


class SegmentSampler(Sampler[int]):
    """Draw a few frames per video each epoch, one from every temporal block.

    Extraction stored more frames per video than a single epoch needs, precisely
    so a run could draw a subset and draw it differently every time. An epoch
    over all 24 takes about 20 minutes on this machine; over 8 it takes under
    seven, and the model still meets the whole gesture window because the draw
    moves between epochs. It buys time, not information.

    The frames of a video sit consecutively in the manifest and in frame order,
    so the draw splits them into equal blocks and takes one frame from each.
    That is the reasoning extraction already applies to the gesture window:
    drawing without the constraint lets the chosen frames cluster in one part of
    the gesture, leaving a stretch of it unseen for that epoch. Solving the
    aliasing during extraction and reintroducing it here would undo the work.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        frames_per_video: int,
        seed: int = SAMPLER_SEED,
    ):
        """Group the manifest rows into the blocks each epoch draws from.

        Args:
            manifest: The rows the dataset serves. It has to be the same frame,
                because what this yields are positions into it.
            frames_per_video: Frames drawn per video per epoch, and therefore
                how many blocks each video is cut into.
            seed: Draws the frames. Its own generator, so that a run stays
                reproducible whatever else consumes randomness alongside it.

        Raises:
            RuntimeError: If videos hold differing numbers of frames, or if the
                stored count does not divide into equal blocks.
        """
        rows = manifest.reset_index(drop=True)
        positions = rows.groupby("video_id", sort=False).indices

        stored = {len(indices) for indices in positions.values()}
        if len(stored) != 1:
            raise RuntimeError(f"videos hold {sorted(stored)} frames, need one count")

        frames_stored = stored.pop()
        if frames_stored % frames_per_video:
            raise RuntimeError(
                f"{frames_stored} frames per video split unevenly into "
                f"{frames_per_video} blocks"
            )

        self.blocks = [
            indices.reshape(frames_per_video, -1) for indices in positions.values()
        ]
        self.frames_per_video = frames_per_video
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Count the frames one epoch draws, which is the length the loader reports."""
        return len(self.blocks) * self.frames_per_video

    def __iter__(self):
        """Draw one frame from every block of every video, in shuffled order.

        The order is shuffled because the blocks are built video by video, and
        batches drawn in that order would hold one gesture at a time.
        """
        drawn = np.array(
            [self.rng.choice(block) for video in self.blocks for block in video]
        )
        self.rng.shuffle(drawn)

        return iter(drawn.tolist())
