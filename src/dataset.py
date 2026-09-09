"""Serve extracted frames to PyTorch, one manifest row at a time.

The manifest written during extraction is the index: each row names a frame on
disk and everything known about it. This module turns a row into the tensor a
model expects, and leaves selection to whoever builds the dataset — filtering
the manifest is how a run picks a domain, a split, a viewpoint or a subset.
"""

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import v2

from manifest import FRAME_SIZE, mask_path_for, stored_size_for

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


# How often a training frame has the person's appearance replaced. Held as a
# constant rather than exposed, for the reason the sampler seeds are: within a
# sweep it is a control, and a control that becomes an option can be varied
# without ever showing up in a diff. Half, matching the background's default, so
# a model still meets the rendered person on the other half and cannot settle on
# "the person is a flat shape" — which would transfer no better than the texture
# it replaced.
TEXTURE_PROBABILITY = 0.5

TEXTURES = ("none", "blend", "replace", "everywhere")

# What the replacement is made of. Both by default, for the reason the background
# draws between the same two: a flat colour and pixel noise are opposite
# failures, and a model that meets both has nowhere left to lean. Naming one
# alone is how a run asks which of the two was doing the work.
TEXTURE_FILLS = {"both": ("solid", "noise"), "solid": ("solid",), "noise": ("noise",)}


class Augmentation(NamedTuple):
    """What training does to a frame before the model sees it.

    These travel together and always have. Every one of them is read by the
    training pipeline and by none of the evaluation one, which meets its frames
    as they are so that successive runs are measured against the same images.
    Carrying them as one value is what keeps a new treatment to a field added
    here, rather than to a parameter threaded through the command line, the
    loader builder and the transform that finally reads it.

    **Every treatment is off by default, and for a single reason:** each run
    recorded so far met its frames without it, and a default that applied it
    would make the runs after the option incomparable to the ones before. The
    jitter pair is on because it was on before either was an option, which is
    the same rule seen from the other side.

    Attributes:
        photometric: Whether to jitter brightness, contrast, saturation and hue.
        geometric: Whether to flip horizontally and vary the crop's scale. The
            fixed-size random crop is replaced rather than added to, so that
            exactly one crop happens either way.
        background: Chance of replacing the scene behind the person, 0 to 1.
            Zero builds no randomiser at all, so a run that does not ask for
            this never reads a silhouette.
        texture: Which replacement to apply inside the person, or ``none`` to
            leave the person as rendered. Needs a silhouette, like the
            background.
        texture_fill: What that replacement draws between. Separating it from
            the mode is how a run asks which of the two a result rests on.
        gamma_shift: Range the gamma exponent is drawn from, or ``None`` to
            leave tone to the jitter alone.
    """

    photometric: bool = True
    geometric: bool = True
    background: float = 0.0
    texture: str = "none"
    texture_fill: str = "both"
    gamma_shift: tuple[float, float] | None = None


# The standard run: the jitter pair and nothing else. Named so that a caller can
# take it without restating it, and so that "the default" is one value with one
# place to read it rather than a row of arguments repeated at every call.
DEFAULT_AUGMENTATION = Augmentation()


class TextureRandomiser:
    """Replace the appearance inside the person, some of the time.

    The background randomiser holds a model to the person by taking the scene
    away. This takes the next thing: what the person is made of, leaving only
    where the person is. ImageNet-trained networks are known to lean on texture
    rather than shape, and a rendered body and a photographed one differ in
    texture far more than in outline — so a model held to the outline has one
    less rendered cue to lean on.

    Three modes, and the third is the control. ``blend`` mixes the person toward
    a replacement by a random amount, so the model meets every degree of texture
    between untouched and gone. ``replace`` always goes all the way, so when it
    fires the person carries no interior detail at all. ``everywhere`` applies
    ``blend``'s mixture to the whole frame instead of the person alone: if the
    restriction to the silhouette is what matters, the two must come apart, and
    if any perturbation of this size would have done, they will not.

    ⚠️ ``everywhere`` is the stronger perturbation, not an equal one applied
    elsewhere — it covers the person as well as the scene. A draw near total
    leaves a frame with little left to read, which happens on a small share of
    the frames it fires on. That cost falls on the control, so it biases the
    comparison toward the treatment and has to be read with the result.

    Randomness comes from torch for the reason the background randomiser gives:
    a DataLoader seeds torch separately in each worker and does not always do the
    same for numpy's global generator.

    Attributes:
        mode: Which of ``blend``, ``replace`` or ``everywhere`` is in play.
        probability: Chance that a given frame is touched at all.
        kinds: Names of the fill generators to draw between.
    """

    def __init__(
        self,
        mode: str,
        probability: float = TEXTURE_PROBABILITY,
        kinds: tuple[str, ...] = ("solid", "noise"),
    ):
        """Configure what replaces the person, and how completely.

        Args:
            mode: One of ``TEXTURES`` other than ``none``. A run that wants this
                off builds no randomiser at all, rather than passing a mode that
                does nothing.
            probability: Chance a frame is touched, from 0 to 1. Defaulted rather
                than exposed on the command line: within a sweep it is a control,
                and every cell holds it at the same value. It is an argument at
                all so that a test can pin it.
            kinds: Which generators in ``BACKGROUNDS`` to draw the replacement
                from, uniformly. Both are used for the reason the background
                gives: a flat colour and pixel noise are opposite failures, and
                a model that leans on interior appearance has nowhere left to
                lean when it meets both.

        Raises:
            ValueError: If the mode is not one this knows, or a kind is not a
                known generator.
        """
        if mode not in TEXTURES or mode == "none":
            raise ValueError(f"texture mode {mode!r} is not one of {TEXTURES[1:]}")
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be between 0 and 1, got {probability}")
        unknown = set(kinds) - set(BACKGROUNDS)
        if unknown:
            raise ValueError(f"unknown fill kinds: {sorted(unknown)}")
        if not kinds:
            raise ValueError("at least one fill kind is needed")

        self.mode = mode
        self.probability = probability
        self.kinds = kinds

    def __call__(
        self, frame: np.ndarray, silhouette: np.ndarray, generator=None
    ) -> np.ndarray:
        """Replace one frame's person appearance, or pass the frame through.

        Args:
            frame: The frame, ``(height, width, 3)``. Composited already, if the
                run also randomises the background — see the dataset, which
                applies them in that order so that this one's effect on the
                scene survives instead of being painted over.
            silhouette: Boolean array, true on the person, same height and width.
            generator: Torch generator to draw from, or ``None`` for the global one.

        Returns:
            Either the frame unchanged, or the frame with the person's interior
            mixed toward a random fill.

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
        fill = BACKGROUNDS[self.kinds[index]](frame.shape[:2], generator)

        strength = (
            1.0
            if self.mode == "replace"
            else torch.rand((), generator=generator).item()
        )
        mixed = (1.0 - strength) * frame.astype(np.float32) + strength * fill
        mixed = mixed.round().astype(np.uint8)

        if self.mode == "everywhere":
            return mixed

        return np.where(silhouette[:, :, None], mixed, frame)


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
    augmentation: Augmentation = DEFAULT_AUGMENTATION,
    crop_size: int = CROP_SIZE,
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

    The background and texture treatments are not built here. Both composite a
    frame against its silhouette, which is a second file the transform is never
    handed — ``FrameDataset`` reads it and applies them before this pipeline
    runs. They are still part of the same value because they are part of the
    same question: what a training frame looks like.

    Args:
        augmentation: What to apply. The default applies the jitter pair alone,
            which is what every run did before any of the rest was an option.
        crop_size: Side the crop takes, in pixels. Follows the size the frames
            were stored at, leaving the same margin: the stored frame is what
            decides how much detail there is to crop from.

    Returns:
        The pipeline, ready to apply to a frame.
    """
    crop = (
        v2.RandomResizedCrop(crop_size, scale=GEOMETRIC_SCALE, ratio=GEOMETRIC_RATIO)
        if augmentation.geometric
        else v2.RandomCrop(crop_size)
    )

    steps = [v2.ToImage(), crop]
    if augmentation.geometric:
        steps.append(v2.RandomHorizontalFlip())
    if augmentation.photometric:
        steps.append(v2.ColorJitter(**PHOTOMETRIC_JITTER))
    if augmentation.gamma_shift:
        steps.append(RandomGamma(augmentation.gamma_shift))
    steps += [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    return v2.Compose(steps)


def eval_transform(crop_size: int = CROP_SIZE) -> v2.Transform:
    """Pipeline used for validation and testing: the crop is fixed and central.

    Evaluation must be reproducible, so nothing here is random.

    Args:
        crop_size: Side the crop takes, in pixels. Has to match what training
            used, or the model meets a field of view it never learnt on.
    """
    return v2.Compose(
        [
            v2.ToImage(),
            v2.CenterCrop(crop_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# How much of a stored frame a crop may leave behind. The crop is a margin:
# 224 out of 256 trims the border and leaves the subject where it was. Taking a
# third of the width instead is not a smaller resolution, it is a different
# field of view, and it is what naming the wrong tree or forgetting the option
# looks like — a run that trains, scores and reports a plausible number.
MINIMUM_CROP_SHARE = 0.5


def crop_fits(manifest: pd.DataFrame, transform: v2.Transform) -> None:
    """Refuse a crop that does not belong to the frames it will be taken from.

    The size a frame was stored at is readable from where it sits, and the size
    a pipeline crops to is readable from the pipeline, so the one mistake that
    nothing downstream would reveal can be caught before the first read rather
    than after the last epoch.

    Args:
        manifest: The rows about to be served. Only the first path is read.
        transform: The pipeline about to be applied. A pipeline holding no crop
            at all is left alone — the check is about disagreement, not about
            requiring one.

    Raises:
        ValueError: If the crop keeps less of the stored frame than
            ``MINIMUM_CROP_SHARE``, or is larger than the frame itself.
    """
    crops = [
        step.size[0]
        for step in getattr(transform, "transforms", [])
        if isinstance(step, (v2.RandomCrop, v2.CenterCrop, v2.RandomResizedCrop))
    ]
    if not crops or manifest.empty:
        return

    crop, stored = crops[0], stored_size_for(manifest["path"].iloc[0])
    if not stored * MINIMUM_CROP_SHARE <= crop <= stored:
        raise ValueError(
            f"a crop of {crop} does not belong to frames stored at {stored}: "
            f"the manifest points at data/frames"
            f"{'/' + str(stored) if stored != FRAME_SIZE else ''}/, "
            f"so the crop should be about {round(stored * CROP_SIZE / FRAME_SIZE)}"
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
        texture: TextureRandomiser | None = None,
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
            texture: Replaces the appearance inside the person on some frames.
                ``None`` for the same reason and with the same restriction: it
                needs a silhouette, and evaluation has none.

        Raises:
            ValueError: If the transform's crop is not a margin on the frames
                this manifest points at. See ``crop_fits``.
        """
        crop_fits(manifest, transform)

        self.data_frame = manifest.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform
        self.background = background
        self.texture = texture

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

        The scene is replaced before the person is, and the order is not
        arbitrary. The texture randomiser's control mode covers the whole frame,
        scene included; replacing the scene afterwards would paint that part of
        its work away and leave the control doing what the treatment does.

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

        if self.background is not None or self.texture is not None:
            silhouette_path = self.data_root / mask_path_for(row["path"])
            silhouette = cv2.imread(str(silhouette_path), cv2.IMREAD_GRAYSCALE)
            if silhouette is None:
                raise RuntimeError(f"could not read silhouette {silhouette_path}")
            person = silhouette > 127
            if self.background is not None:
                frame = self.background(frame, person)
            if self.texture is not None:
                frame = self.texture(frame, person)

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
        rows: np.ndarray | None = None,
    ):
        """Group the manifest rows into the blocks each epoch draws from.

        Args:
            manifest: The rows the dataset serves. It has to be the same frame,
                because what this yields are positions into it.
            frames_per_video: Frames drawn per video per epoch, and therefore
                how many blocks each video is cut into.
            seed: Draws the frames. Its own generator, so that a run stays
                reproducible whatever else consumes randomness alongside it.
            rows: Which of those rows this sampler draws from, as a boolean mask
                over them. ``None`` is all of them. A frame carrying two kinds
                of row — a video's gesture frames and the idle frames of the
                same video, stored at different counts — needs one sampler per
                kind, and each still has to name positions in the whole frame,
                since that is what the dataset serves.

        Raises:
            RuntimeError: If videos hold differing numbers of frames, or if the
                stored count does not divide into equal blocks.
        """
        frame = manifest.reset_index(drop=True)
        selected = (
            np.arange(len(frame)) if rows is None else np.flatnonzero(np.asarray(rows))
        )
        positions = frame.iloc[selected].groupby("video_id", sort=False).indices

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
            selected[indices].reshape(frames_per_video, -1)
            for indices in positions.values()
        ]
        self.frames_per_video = frames_per_video
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Count the frames one epoch draws, which is the length the loader reports."""
        return len(self.blocks) * self.frames_per_video

    def state_dict(self) -> dict:
        """Where the draw has got to, so a resumed run carries on rather than repeats.

        One generator serves every epoch, and each draws from where the last one
        left off — that is what makes an epoch meet different frames from the one
        before it. A run restarted from a saved model but a fresh generator would
        replay the first epoch's draw, and its later epochs would see a narrower
        slice of the stored frames than an uninterrupted run does.
        """
        return self.rng.bit_generator.state

    def load_state_dict(self, state: dict) -> None:
        """Put the draw back where it was.

        Args:
            state: What ``state_dict`` returned.
        """
        self.rng.bit_generator.state = state

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


class CombinedSampler(Sampler[int]):
    """Draw one epoch from several samplers and hand it back in a single order.

    An epoch that mixes rows stored at different counts per video needs one
    sampler for each. Eight frames drawn from a video's gesture and one drawn
    from the material before it are two draws over two sets of blocks, and a
    single sampler asked to do both would have to be told which rows are which —
    which is the dataset's business, not the sampler's.

    Shuffling again over the union is what makes this more than concatenation.
    Each sampler shuffles only its own draw, so without this an epoch would
    arrive as every gesture frame followed by every idle frame, and the batches
    at the end would hold nothing but bodies at rest — a batch normalisation
    layer meeting one class at a time is not the same layer.
    """

    def __init__(self, samplers: list[Sampler[int]], seed: int = SAMPLER_SEED):
        """Hold the samplers whose draws make up an epoch.

        Args:
            samplers: The samplers to draw from. Each must name positions in the
                same frame, since all of them index one dataset.
            seed: Shuffles the union. Its own generator, kept apart from the
                ones the samplers draw with so that adding a kind of row does
                not move the frames the others choose.
        """
        self.samplers = list(samplers)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Count the frames one epoch draws, which is the length the loader reports."""
        return sum(len(sampler) for sampler in self.samplers)

    def state_dict(self) -> dict:
        """Where every draw has got to: this sampler's shuffle and each part's.

        Three generators run here, not one — the two kinds of row draw from their
        own, and the shuffle that mixes them draws from a third. Saving only one
        would leave a resumed run half where it was.
        """
        return {
            "shuffle": self.rng.bit_generator.state,
            "parts": [sampler.state_dict() for sampler in self.samplers],
        }

    def load_state_dict(self, state: dict) -> None:
        """Put every draw back where it was.

        Args:
            state: What ``state_dict`` returned.
        """
        self.rng.bit_generator.state = state["shuffle"]
        for sampler, part in zip(self.samplers, state["parts"], strict=True):
            sampler.load_state_dict(part)

    def __iter__(self):
        """Draw from every sampler and shuffle the lot together."""
        drawn = np.concatenate(
            [
                np.fromiter(sampler, dtype=int, count=len(sampler))
                for sampler in self.samplers
            ]
        )
        self.rng.shuffle(drawn)

        return iter(drawn.tolist())
