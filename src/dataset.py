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

from manifest import mask_path_for

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CROP_SIZE = 224
SAMPLER_SEED = 13

# The widely used ImageNet recipe, taken as published rather than tuned. Fitting
# these to the target domain would need measurements from it, and a source-only
# result may not look at the target at all — not even at unlabelled statistics.
# Tuning them on the source validation split is possible but answers a different
# question: that split ranks settings by what helps inside the rendered domain,
# while what a setting is worth here is decided by what survives the crossing.
# Taking a published default and saying it was not tuned is the honest option the
# protocol leaves, and saying so is part of the result.
PHOTOMETRIC_JITTER = {
    "brightness": 0.4,
    "contrast": 0.4,
    "saturation": 0.4,
    "hue": 0.1,
}

# The subject fills roughly half the frame's height, so a crop much below this
# starts cutting the arm away — and the arm is the signal. The ImageNet default
# reaches down to 0.08 of the area, which would be destructive here. Aspect
# stays close to square for the same reason: stretching one axis changes the
# apparent length of an extended arm.
GEOMETRIC_SCALE = (0.6, 1.0)
GEOMETRIC_RATIO = (0.9, 1.1)


# How often a frame's scene is replaced during training. Not one: a model that
# never sees the rendered terrain has no chance to learn what a plausible scene
# looks like, and half the batch keeping its own background costs nothing while
# hedging that. The value is a starting point, not a measurement.
BACKGROUND_PROBABILITY = 0.5


def solid_background(shape: tuple[int, int], generator=None) -> np.ndarray:
    """A background of one uniform colour, drawn at random.

    The bluntest possible scene: it carries no texture, no horizon and no
    objects, so a model cannot read anything from it. What survives training
    against it is whatever the person alone supports.

    Args:
        shape: ``(height, width)`` of the frame being composited.
        generator: Torch generator to draw from, or ``None`` for the global one,
            which a DataLoader seeds separately in every worker.

    Returns:
        A ``(height, width, 3)`` uint8 array of a single colour.
    """
    colour = torch.randint(0, 256, (3,), generator=generator, dtype=torch.uint8)

    return np.broadcast_to(colour.numpy(), (*shape, 3)).copy()


def noise_background(shape: tuple[int, int], generator=None) -> np.ndarray:
    """A background of independent random pixels.

    The opposite failure mode to a solid colour: maximal high-frequency detail
    with no structure at all. Between the two, a model that leans on the scene
    has nowhere left to lean.

    Args:
        shape: ``(height, width)`` of the frame being composited.
        generator: Torch generator to draw from, or ``None`` for the global one.

    Returns:
        A ``(height, width, 3)`` uint8 array of noise.
    """
    noise = torch.randint(0, 256, (*shape, 3), generator=generator, dtype=torch.uint8)

    return noise.numpy()


BACKGROUNDS = {"solid": solid_background, "noise": noise_background}


class BackgroundRandomiser:
    """Replace the scene behind the person, some of the time.

    The model scores far higher on rendered scenes it has never seen than on
    real footage. One reading of that gap is that what transfers is the pose and
    what does not is the scene, in which case a model held to the pose alone
    should lose less crossing over. Compositing the silhouette onto backgrounds
    that carry no information at all is the cheapest way to hold it there.

    Randomness comes from torch rather than numpy because a DataLoader seeds
    torch separately in each worker and does not always do the same for numpy's
    global generator — which would hand every worker the same backgrounds.

    Attributes:
        probability: Chance that a given frame has its scene replaced.
        kinds: Names of the background generators to draw between.
    """

    def __init__(
        self,
        probability: float = BACKGROUND_PROBABILITY,
        kinds: tuple[str, ...] = ("solid", "noise"),
    ):
        """Configure how often and with what to replace a scene.

        Args:
            probability: Chance a frame is composited, from 0 to 1. Zero leaves
                every frame untouched, which is how a run turns this off without
                a second code path.
            kinds: Which generators in ``BACKGROUNDS`` to draw between, uniformly.

        Raises:
            ValueError: If the probability falls outside 0 to 1, or a name is not
                a known generator.
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be between 0 and 1, got {probability}")
        unknown = set(kinds) - set(BACKGROUNDS)
        if unknown:
            raise ValueError(f"unknown background kinds: {sorted(unknown)}")
        if not kinds:
            raise ValueError("at least one background kind is needed")

        self.probability = probability
        self.kinds = kinds

    def __call__(
        self, frame: np.ndarray, silhouette: np.ndarray, generator=None
    ) -> np.ndarray:
        """Composite one frame's person onto a fresh background, or pass it through.

        Args:
            frame: The frame, ``(height, width, 3)``.
            silhouette: Boolean array, true on the person, same height and width.
            generator: Torch generator to draw from, or ``None`` for the global one.

        Returns:
            Either the frame unchanged, or the person over a new background.

        Raises:
            ValueError: If the silhouette does not cover the frame.
        """
        if silhouette.shape != frame.shape[:2]:
            raise ValueError(
                f"silhouette is {silhouette.shape}, frame is {frame.shape[:2]}"
            )
        if torch.rand((), generator=generator).item() >= self.probability:
            return frame

        index = int(torch.randint(len(self.kinds), (), generator=generator))
        background = BACKGROUNDS[self.kinds[index]](frame.shape[:2], generator)

        return np.where(silhouette[:, :, None], frame, background)


class RandomGamma:
    """Darken a rendered frame's midtones, the way outdoor footage is darkened.

    What separates the two domains photometrically is shadow. The renders carry
    almost no dark pixels; the real footage is full of them, and a model trained
    inside the rendered band has never seen the tones it meets on the other side.

    The jitter above cannot close that. Every factor ``ColorJitter`` draws is
    centred on one, so it widens a distribution without moving it, whatever
    magnitude it is given. Multiplying contrast does move it, but the wrong way
    round: most rendered pixels already sit in the bright half, so raising
    contrast drives them past white and destroys them.

    Gamma is the operation the gap asks for. It compresses the top of the scale
    and stretches the bottom, which is what creates the missing dark population,
    and by construction it cannot push a pixel past white. It darkens enough on
    its own to close the brightness gap too, so no separate brightness shift is
    needed.

    A range spanning untouched to heavily darkened is what keeps this
    source-only. Covering that breadth needs nothing measured from the target to
    justify, and it lands closer to the real distribution than a range fitted to
    it would.

    Attributes:
        gamma_range: Low and high bound the exponent is drawn between.
    """

    def __init__(self, gamma_range: tuple[float, float]):
        """Fix the range every draw comes from.

        Args:
            gamma_range: Low and high bound of the uniform draw. A gamma of 1.0
                leaves the frame untouched; above 1.0 darkens the midtones.

        Raises:
            ValueError: If a bound is not positive, or the two are out of order.
        """
        low, high = gamma_range
        if low <= 0 or high < low:
            raise ValueError(
                f"gamma range must be positive and ordered, got {gamma_range}"
            )
        self.gamma_range = (float(low), float(high))

    def __call__(self, image):
        """Apply one draw to one frame.

        Draws through torch's own generator rather than numpy's, which is what
        puts the draw under the per-worker seeding the loader already does — the
        same path ``ColorJitter`` takes.
        """
        low, high = self.gamma_range
        gamma = float(torch.empty(1).uniform_(low, high))
        return v2.functional.adjust_gamma(image, gamma)


def train_transform(
    photometric: bool = True,
    geometric: bool = False,
    gamma_shift: tuple[float, float] | None = None,
) -> v2.Transform:
    """Pipeline used while training: the crop position is drawn at random.

    Random cropping keeps the model from anchoring on absolute pixel positions
    and costs nothing, since the frames were stored at 256 precisely to leave
    this margin.

    Photometric jitter is the second augmentation. Rendered frames are
    photometrically narrow — measured on the source alone, their contrast spans
    a standard deviation of 31 to 48 across the split — and a model trained
    inside that band learns to depend on it. Widening the source distribution is
    the cheapest way to make an unseen target fall inside it, and it needs
    nothing from the target: the magnitudes are a published default, not a fit.

    Geometric jitter is the third: a horizontal flip and a crop that varies in
    scale rather than only in position. It is the pair the baseline paper names,
    and unlike the photometric one it does not aim at a measured gap — the
    subject occupies almost the same fraction of the frame in both domains, and
    the synthetic videos are never mirrored, the metadata's ``mirrored`` field
    being false in every file checked. It is a general regulariser, and the
    experiment is what it contributes next to the photometric one.

    Gamma is the fourth, and the only one aimed at a mismatch measured
    between the two domains rather than inside one. It runs after the jitter
    rather than instead of it, so the jitter's own spread survives underneath
    and exactly one thing changes against the runs it is compared to. See
    ``RandomGamma`` for why.

    Args:
        photometric: Whether to jitter brightness, contrast, saturation and hue.
            Off reproduces the earlier runs, which is what makes them comparable.
        geometric: Whether to flip horizontally and vary the crop's scale. The
            fixed-size random crop is replaced rather than added to, so that
            exactly one crop happens either way.
        gamma_shift: Range the gamma exponent is drawn from, or ``None`` to
            leave tone to the jitter alone, which is what every run before this
            option did.

    Returns:
        The pipeline, ready to apply to a frame.
    """
    crop = (
        v2.RandomResizedCrop(CROP_SIZE, scale=GEOMETRIC_SCALE, ratio=GEOMETRIC_RATIO)
        if geometric
        else v2.RandomCrop(CROP_SIZE)
    )

    steps = [v2.ToImage(), crop]
    if geometric:
        steps.append(v2.RandomHorizontalFlip())
    if photometric:
        steps.append(v2.ColorJitter(**PHOTOMETRIC_JITTER))
    if gamma_shift:
        steps.append(RandomGamma(gamma_shift))
    steps += [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    return v2.Compose(steps)


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
        background: BackgroundRandomiser | None = None,
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
            background: Replaces the scene behind the person on some frames.
                ``None`` serves frames as they were extracted, and is the only
                correct setting for evaluation: the real test footage has no
                segmentation to composite with, so a model has to meet its
                scenes intact. Compositing there would also measure the model on
                inputs no deployment ever produces.
        """
        self.data_frame = manifest.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform
        self.background = background

    def __len__(self) -> int:
        """Count frames, not videos — the manifest holds 24 rows per video."""
        return len(self.data_frame)

    def __getitem__(self, index: int):
        """Return one frame, its label, and the video it came from.

        OpenCV reads images as BGR, so the frame is converted to RGB before the
        transform: the pretrained weights were learned on RGB, and feeding the
        channels reversed silently degrades every prediction.

        A background swap happens before the transform, not after, because the
        silhouette is stored aligned to the extracted frame. The transform crops,
        so compositing afterwards would place a 256-wide mask over a 224-wide
        image.

        Returns:
            The transformed frame as a ``(3, 224, 224)`` float tensor, the class
            label, and the ``video_id`` needed to group predictions by video.

        Raises:
            RuntimeError: If a frame, or a silhouette a background swap needs,
                is missing from disk.
        """
        row = self.data_frame.iloc[index]
        frame_path = self.data_root / row["path"]
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"could not read frame {frame_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.background is not None:
            silhouette_path = self.data_root / mask_path_for(row["path"])
            silhouette = cv2.imread(str(silhouette_path), cv2.IMREAD_GRAYSCALE)
            if silhouette is None:
                raise RuntimeError(f"could not read silhouette {silhouette_path}")
            frame = self.background(frame, silhouette > 127)

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
