"""Train a ResNet18 to classify RoCoG-v2 gesture frames.

Loads frames through the extraction manifest, fine-tunes ImageNet weights on the
seven gesture classes, keeps the checkpoint that validates best, and stops once
validation has stopped improving.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import (
    CROP_SIZE,
    DEFAULT_AUGMENTATION,
    SAMPLER_SEED,
    TEXTURE_FILLS,
    TEXTURES,
    Augmentation,
    BackgroundRandomiser,
    CombinedSampler,
    FrameDataset,
    SegmentSampler,
    TextureRandomiser,
    eval_transform,
    train_transform,
)
from device import describe, pick_device
from evaluation import frame_metrics, predict
from manifest import IDLE_LABEL
from model import (
    BACKBONES,
    DEFAULT_BACKBONE,
    FROZEN_STAGES,
    NUM_CLASSES,
    backbone_of,
    build_model,
    freeze,
    head_width,
)
from splits import (
    WINDOWS,
    add_idle_rows,
    sample_videos,
    select_frames,
    split_by_group,
    split_by_scene,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BATCH_SIZE = 64
FRAMES_PER_EPOCH = 8

# Idle frames drawn from each training video per epoch. Every video carries one
# gesture and every video carries idle material, so a video's frames land in one
# of seven classes while its idle frames all land in the eighth. Drawing one
# leaves the eighth class a little under the size of a gesture class; drawing two
# would put it at nearly twice. Extraction stores more than an epoch uses, here
# as everywhere else, so that the draw can move between epochs.
IDLE_FRAMES_PER_EPOCH = 1
LEARNING_RATE = 1e-4

# Defaults for the options parse_args exposes, so an argument-free run is the
# standard run.
SEED = 0
NUM_WORKERS = 12
MAX_EPOCHS = 15
PATIENCE = 3
# The rule behind every default below, and behind the ones DEFAULT_AUGMENTATION
# carries: an option arrives off. Each run recorded so far was made without it,
# and a default that turned it on would make every run after the option
# incomparable to every run before it. Turning one on is therefore always a
# deliberate act, visible in the command that made the run.
LABEL_SMOOTHING = 0.0
FREEZE = "none"
WINDOW = "full"
CHECKPOINT_NAME = "syn_ground_train.pt"
MANIFEST = "syn_ground_train.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line: which cell of the training grid to run.

    Every option defaults to the value the grid holds fixed, so a call with no
    arguments reproduces the standard run. What legitimately varies between
    cells is what these expose — the augmentation in play, the epoch budget,
    where the checkpoint lands. Batch size and the seeds stay as module
    constants on purpose: a control that becomes an option can be varied
    without ever showing up in a diff.

    ``--lr`` is the one control that had to be given up, and only because
    ``--freeze`` turned it into two controls sharing a name. Fine-tuning a
    pretrained network wants steps small enough not to undo what it already
    knows. Fitting a freshly initialised head on a backbone that cannot move is
    an ordinary optimisation from scratch, and the step that suits the first
    leaves the second still improving when the epoch budget runs out — measured
    here, and a run that has not converged measures nothing. One value cannot
    serve both. Every run prints the value it used, so the record still carries
    it.

    ``--num-workers`` is the exception, and worth stating plainly. It reads as a
    machine-capacity knob, but the loader seeds every worker separately and the
    training transforms draw inside them, so the worker count decides which
    random crop lands on which frame — measured, not assumed. It is exposed
    because a machine with fewer cores needs it, and it has to be held fixed
    across any set of runs meant to be compared.

    ``--seed`` fixes what this program controls, which is not the same as
    promising an identical result. On hardware whose kernels accumulate
    non-deterministically, two runs of one seed diverge anyway — measured here at
    a validation-loss spread wide enough to matter. Measure that spread on the
    machine at hand before reading any difference as the effect of a change.

    ``--freeze`` is the one option that changes what the run is measuring
    rather than what it is trained on. Fitting every layer to the source domain
    reshapes the features as well as the classifier, so a low score on another
    domain has two candidate causes that a single number cannot separate: a
    representation that never held the answer, or one that held it and was
    trained away. Holding the early stages fixed rules the second out by
    construction, at the price of a model that can fit the source less well.

    ``--texture`` carries its control in the same option, for the same reason
    ``--window`` does. Replacing what the person is made of is a strong
    perturbation, and a strong perturbation regularises whatever it touches, so a
    gain from ``blend`` alone would not say whether the silhouette mattered.
    ``everywhere`` applies the identical mixture to the whole frame; the pair
    coming apart is what would make the restriction the explanation.

    ``--window`` narrows the training frames without narrowing the ones a run is
    scored on, and the asymmetry is deliberate. Which frames a model learns from
    is a design choice available in the field; which frames it is judged on is
    not, because judging it on a chosen stretch of the clip would use where a
    frame sits to decide how much it counts. The option therefore carries its own
    control, ``scattered``, since narrowing the window also drops a third of the
    frames and the two effects would otherwise arrive together.

    ``--manifest`` and ``--validation-groups`` are what let a run train on a
    domain other than the synthetic one. Which rows are held out for validation
    cannot be inferred from the data: the synthetic manifest is split by scene,
    drawing one per viewpoint so that all six survive on both sides, and the real
    manifest has no viewpoint at all — its unit is the recorded subject. Naming
    the held-out groups keeps that choice visible in the command that produced a
    result, which matters because it is a control and not a knob.

    Args:
        argv: Arguments to parse. ``None`` reads ``sys.argv``.

    Returns:
        A namespace with ``manifest``, ``validation_groups``, ``seed``,
        ``photometric``, ``geometric``, ``background``, ``gamma_shift``,
        ``label_smoothing``, ``freeze``, ``texture``, ``texture_fill``,
        ``window``, ``lr``,
        ``max_epochs``, ``patience``, ``checkpoint_name``, ``num_workers`` and
        ``save_every_epoch``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--idle-manifest",
        default=None,
        help="frames of the pose a body holds before it gestures, a file name "
        "under data/manifests/. Adds an eighth class for them",
    )
    parser.add_argument(
        "--manifest",
        default=MANIFEST,
        help="frames to train on, a file name under data/manifests/",
    )
    parser.add_argument(
        "--validation-groups",
        nargs="+",
        default=None,
        metavar="GROUP",
        help="groups held out for validation, named rather than drawn. Omit for "
        "the synthetic manifest, which is split by scene, one per viewpoint; "
        "required for the real manifest, whose groups are the recorded subjects",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="repetition of a configuration. Moves the head's initial weights, "
        "the random transforms and the per-epoch frame draw together; the "
        "train/validation split stays fixed, being a control rather than noise",
    )
    parser.add_argument(
        "--photometric",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_AUGMENTATION.photometric,
        help="jitter brightness and contrast while training",
    )
    parser.add_argument(
        "--geometric",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_AUGMENTATION.geometric,
        help="flip and vary the crop scale while training",
    )
    parser.add_argument(
        "--background",
        type=float,
        default=DEFAULT_AUGMENTATION.background,
        metavar="PROBABILITY",
        help="chance of replacing a training frame's scene with a random solid "
        "colour or noise, from 0 to 1; evaluation is never composited",
    )
    parser.add_argument(
        "--gamma-shift",
        type=float,
        nargs=2,
        default=DEFAULT_AUGMENTATION.gamma_shift,
        metavar=("LOW", "HIGH"),
        help="draw a gamma exponent from this range while training, on top of "
        "the jitter. Above 1 darkens the midtones, which is where the rendered "
        "frames differ from photographed ones; a range wide enough to span from "
        "untouched to heavily darkened needs nothing measured from the target",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=LABEL_SMOOTHING,
        metavar="FRACTION",
        help="mass moved off the true class and spread over the other six "
        "while training, from 0 to 1. Validation is always scored against hard "
        "targets, so its loss stays comparable across values",
    )
    parser.add_argument(
        "--exclude-groups",
        nargs="+",
        metavar="GROUP",
        help="drop these groups from the manifest before anything else, so they "
        "reach neither training nor validation. Distinct from "
        "--validation-groups, which holds a group out of training but still "
        "lets it choose the stopping epoch: a group meant to be scored as "
        "unseen must not do that, or it has shaped the run it is measuring",
    )
    parser.add_argument(
        "--initial-weights",
        metavar="CHECKPOINT",
        help="start from these weights instead of ImageNet's, e.g. "
        "checkpoints/idle_s0.pt. The head's width and the backbone are read "
        "from the file rather than asked for, so a fine-tuning run cannot be "
        "paired with an architecture it was not written by",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        metavar="SHARE",
        help="train on this share of the TRAINING videos, drawn per class so "
        "that no gesture disappears and so that the fractions nest. Validation "
        "is never sampled. Whole videos, never a share of each one's frames",
    )
    parser.add_argument(
        "--backbone",
        choices=BACKBONES,
        default=DEFAULT_BACKBONE,
        help="which network to train. The two ResNet18s cost the same per frame "
        "and carry the same parameter count; the ibn variant normalises half of "
        "each shallow stage's channels by the instance instead of the batch, "
        "which drops appearance statistics the batch would have kept. The other "
        "two answer the same task at a fraction of the arithmetic per frame, "
        "which is what turns one cost figure into a curve. --freeze is defined "
        "only for the ResNets, whose stage names it addresses",
    )
    parser.add_argument(
        "--freeze",
        choices=sorted(FROZEN_STAGES),
        default=FREEZE,
        metavar="DEPTH",
        help="how far to hold the pretrained weights fixed: none trains the "
        "whole network, early keeps the first two stages, backbone keeps every "
        "stage and trains the head alone. Batch normalization statistics are "
        "held with the weights, so a frozen stage never adapts to the training "
        "domain",
    )
    parser.add_argument(
        "--texture",
        choices=TEXTURES,
        default=DEFAULT_AUGMENTATION.texture,
        metavar="MODE",
        help="replace what the person is made of while training: blend mixes "
        "the person toward a random fill by a random amount, replace always "
        "goes all the way, and everywhere applies blend's mixture to the whole "
        "frame instead, which is the control that says whether restricting it "
        "to the person is the point. Needs silhouettes, so training only",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=CROP_SIZE,
        metavar="PIXELS",
        help="side the crop takes from each stored frame. Follows the tree the "
        "manifest points at, leaving the same margin the default leaves on a "
        "256-pixel frame; a crop that is not a margin on those frames is "
        "refused rather than trained on",
    )
    parser.add_argument(
        "--texture-fill",
        choices=sorted(TEXTURE_FILLS),
        default=DEFAULT_AUGMENTATION.texture_fill,
        metavar="KIND",
        help="what the replacement is made of: both draws between a flat "
        "colour and pixel noise, and naming one alone asks which of the two "
        "carries whatever the mode is worth. Ignored without --texture",
    )
    parser.add_argument(
        "--window",
        choices=WINDOWS,
        default=WINDOW,
        metavar="EXTENT",
        help="which of each training video's frames to use: full keeps the "
        "annotated gesture window as extracted, middle drops the outermost "
        "frames at each end, where the pose is near neutral, and scattered "
        "keeps as many frames as middle does but spread over the whole window, "
        "which is middle's control. Training only: choosing frames by where "
        "they sit in the clip is temporal information, and evaluation may not "
        "have it",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help="Adam's step size. The default suits fine-tuning a pretrained "
        "network; a frozen backbone leaves a fresh head to fit from scratch, "
        "which wants a larger one to converge inside the epoch budget",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=MAX_EPOCHS,
        help="epoch cap; early stopping can end the run sooner",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
        help="epochs without a validation best before stopping",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=CHECKPOINT_NAME,
        help="file to write under checkpoints/",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="DataLoader subprocesses. Decides which random transform lands on "
        "which frame, so hold it fixed across runs being compared.",
    )
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="also write one checkpoint per epoch, not only the best; for the "
        "epoch-vs-real-accuracy diagnostic. Pair with a patience it never hits.",
    )
    return parser.parse_args(argv)


def checkpoint_name(base_name: str, seed: int, epoch: int | None = None) -> str:
    """Name a checkpoint after the run that produced it.

    ``es_none.pt`` at seed 2 becomes ``es_none_s2.pt``, and at epoch 3 of that
    seed ``es_none_s2_e03.pt``. The suffixes are not decoration: a grid runs the
    same configuration once per seed, so without them every repetition would
    write over the last one — and a checkpoint silently overwritten is a result
    that cannot be traced back. The epoch is zero-padded so the files sort in
    training order.

    Args:
        base_name: The ``--checkpoint-name`` the run was given.
        seed: The repetition this run is.
        epoch: One-based epoch number, or ``None`` for the run's best weights.

    Returns:
        The file name to write under ``checkpoints/``.
    """
    name = Path(base_name)
    marks = f"_s{seed}" + (f"_e{epoch:02d}" if epoch is not None else "")
    return f"{name.stem}{marks}{name.suffix}"


def build_loaders(
    train_manifest: pd.DataFrame,
    eval_manifest: pd.DataFrame,
    data_root: Path,
    num_workers: int = NUM_WORKERS,
    frames_per_epoch: int = FRAMES_PER_EPOCH,
    idle_frames_per_epoch: int = IDLE_FRAMES_PER_EPOCH,
    augmentation: Augmentation = DEFAULT_AUGMENTATION,
    crop_size: int = CROP_SIZE,
    seed: int = SEED,
) -> tuple[DataLoader, DataLoader]:
    """Build the training and evaluation loaders from two sets of manifest rows.

    Training draws a subset of each video's frames and drops the last partial
    batch, since batch normalization fails on a batch of one. Evaluation keeps
    every frame and its order, so that successive epochs are measured against
    exactly the same data. Both loaders pin their batches in page-locked memory,
    which speeds the copy to a GPU and costs nothing without one.

    Args:
        train_manifest: Rows listing the frames to train on.
        eval_manifest: Rows listing the frames to evaluate on. Disjoint from the
            training rows by construction — see ``splits``, which holds out
            whole groups precisely so that nothing straddles the boundary.
        data_root: Directory the manifests' ``path`` column is relative to.
        num_workers: Loader subprocesses. Zero avoids the ~16 s spawn cost, which
            is worth paying only for runs long enough to amortise it.
        idle_frames_per_epoch: Idle frames drawn from each training video per
            epoch. Read only when the training rows carry any, so a run without
            them is the run as it was before the class existed.
        frames_per_epoch: Frames drawn from each training video per epoch. The
            sampler shuffles, so the loader must not — passing a sampler and
            ``shuffle=True`` together is rejected by the DataLoader.
        augmentation: What training does to a frame before the model sees it.
            The training loader alone reads it; evaluation meets its frames as
            they are, so that successive runs are measured against the same
            images. Two of its treatments need a silhouette, which the real
            footage has none of - a second reason they never reach evaluation.
        crop_size: Side both pipelines crop to. Shared rather than augmented,
            because it is not a treatment: it says how much of a stored frame
            the model sees, and training and evaluation have to agree on that
            or the model meets a field of view it never learnt on.
        seed: Which repetition of a configuration this is. Offsets the sampler's
            own seed rather than replacing it, so that seed 0 keeps drawing the
            frames earlier runs drew and stays comparable to them.

    Returns:
        The training loader and the evaluation loader.
    """
    train_dataset = FrameDataset(
        train_manifest,
        data_root,
        transform=train_transform(augmentation, crop_size),
        background=(
            BackgroundRandomiser(augmentation.background)
            if augmentation.background
            else None
        ),
        texture=(
            TextureRandomiser(
                augmentation.texture, kinds=TEXTURE_FILLS[augmentation.texture_fill]
            )
            if augmentation.texture != "none"
            else None
        ),
    )
    eval_dataset = FrameDataset(
        eval_manifest, data_root, transform=eval_transform(crop_size)
    )

    # Two draws rather than one when the rows carry both kinds. A video holds a
    # different number of each, so counting them together would refuse to split
    # into equal blocks, and drawing them at one rate would let the class that
    # is stored least often decide the rate for all of them.
    rows = train_dataset.data_frame
    is_idle = (rows["label"] == IDLE_LABEL).to_numpy()
    sampler = (
        CombinedSampler(
            [
                SegmentSampler(
                    rows, frames_per_epoch, seed=SAMPLER_SEED + seed, rows=~is_idle
                ),
                SegmentSampler(
                    rows, idle_frames_per_epoch, seed=SAMPLER_SEED + seed, rows=is_idle
                ),
            ],
            seed=SAMPLER_SEED + seed,
        )
        if is_idle.any()
        else SegmentSampler(rows, frames_per_epoch, seed=SAMPLER_SEED + seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    return train_loader, eval_loader


class EarlyStopping:
    """Watch validation loss and say when training has stopped paying off.

    Augmentation moves where the optimum sits: the stronger the regularisation,
    the later a run peaks. A fixed epoch count therefore cannot be fair across
    configurations — measured here, training without augmentation turned upward
    at the fifth epoch while the geometric run was still improving at the last
    one, which made its result a floor rather than a peak. Letting each
    configuration run until it stops improving is what makes them comparable.

    Patience exists because the curve is not monotonic. The run without
    augmentation went 0.693, 0.699, 0.629: stopping at the first worse epoch
    would have discarded the best one, which came after.
    """

    def __init__(self, patience: int = PATIENCE):
        """Start with no history, so the first epoch always counts as best.

        Args:
            patience: Epochs without a new best before training should stop.
        """
        self.patience = patience
        self.best_loss = float("inf")
        self.epochs_without_improvement = 0

    def improved(self, loss: float) -> bool:
        """Record an epoch's validation loss and say whether it is the best yet.

        Args:
            loss: The epoch's validation loss.

        Returns:
            Whether this epoch beat every earlier one, which is when the caller
            should write a checkpoint.
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.epochs_without_improvement = 0
            return True

        self.epochs_without_improvement += 1
        return False

    @property
    def exhausted(self) -> bool:
        """Whether patience has run out and training should stop."""
        return self.epochs_without_improvement >= self.patience


# Options a resumed run may differ in without describing a different run. The
# worker count is machine bookkeeping; everything else defines the experiment,
# so a change in any of it means the saved progress belongs to another run.
INCIDENTAL_OPTIONS = {"num_workers"}


def save_atomically(state: dict, path: Path) -> None:
    """Write a checkpoint so that a crash cannot leave half of one behind.

    Saving takes long enough on a file this size for the machine to die partway
    through it, and a truncated checkpoint is worse than none: it exists, so
    everything downstream treats it as real and fails later, somewhere else.
    Writing beside the target and renaming afterwards makes the swap atomic, so
    the file is either the previous one or the new one and never a mixture.

    Args:
        state: What to save.
        path: Where it belongs. Its directory is created if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    beside = path.with_suffix(".partial")
    torch.save(state, beside)
    beside.replace(path)


def save_progress(
    path: Path,
    epochs_done: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    stopper: "EarlyStopping",
    sampler,
    arguments: argparse.Namespace,
) -> None:
    """Record everything a run would need to carry on from where it is.

    Four things move between epochs and all four have to travel together. The
    weights are the obvious one. The optimizer holds the running moments Adam
    accumulates, and a run restarted without them takes its next steps as if
    from a standing start. The stopper holds the best loss seen and how long
    ago, which is what decides when to give up. The sampler holds where the
    frame draw has got to.

    The options are recorded alongside them so that progress can be recognised
    as belonging to this run and no other. It is the same file name for every
    repetition of a configuration, and silently carrying on from a different
    one is the kind of mistake that shows up only as a number that will not
    reproduce.

    Args:
        path: Where to write. Overwritten atomically each epoch.
        epochs_done: How many epochs have finished, so the next one is this.
        model: The network, as it stands.
        optimizer: The optimizer, with its accumulated moments.
        stopper: The early-stopping state.
        sampler: The training sampler, which knows where its draw has got to.
        arguments: The parsed command line, kept for the identity check.
    """
    save_atomically(
        {
            "epochs_done": epochs_done,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": stopper.best_loss,
            "epochs_without_improvement": stopper.epochs_without_improvement,
            "sampler": sampler.state_dict(),
            "arguments": vars(arguments),
        },
        path,
    )


def load_progress(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    stopper: "EarlyStopping",
    sampler,
    arguments: argparse.Namespace,
) -> int:
    """Put a run back where it stopped, and say which epoch is next.

    What comes back is a continuation, not a reproduction. This machine's GPU is
    not deterministic — two identical runs of it already differ by more than
    most treatments do — so no amount of saved state would make a resumed run
    retrace the one that was interrupted. What the state buys is that the
    resumed run is drawn from the same distribution as an uninterrupted one
    rather than a narrower one: the draw carries on instead of replaying, and
    the optimizer keeps its momentum instead of restarting cold.

    Args:
        path: Where progress was written. Missing means there is none, which is
            the ordinary case and not an error.
        model: The network to load into, modified in place.
        optimizer: The optimizer to load into, modified in place.
        stopper: The early-stopping state to restore, modified in place.
        sampler: The training sampler to restore, modified in place.
        arguments: The parsed command line, checked against the saved one.

    Returns:
        The number of epochs already finished. Zero when there is nothing to
        resume, so a caller can always start its loop there.

    Raises:
        RuntimeError: If the saved progress was written by a run configured
            differently. Carrying on regardless would train one configuration
            on top of another and report the result as either.
    """
    if not path.exists():
        return 0

    saved = torch.load(path, map_location="cpu", weights_only=False)

    differences = sorted(
        option
        for option, value in vars(arguments).items()
        if option not in INCIDENTAL_OPTIONS and saved["arguments"].get(option) != value
    )
    if differences:
        raise RuntimeError(
            f"{path.name} was written by a run with a different "
            f"{', '.join(differences)}. Delete it to start this one over."
        )

    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    stopper.best_loss = saved["best_loss"]
    stopper.epochs_without_improvement = saved["epochs_without_improvement"]
    sampler.load_state_dict(saved["sampler"])

    return saved["epochs_done"]


def build_criteria(label_smoothing: float) -> tuple[nn.Module, nn.Module]:
    """Build the loss training minimises and the loss validation is scored with.

    Deliberately two objects rather than one. Smoothing spreads part of the
    target onto the wrong classes, which lifts the floor of the cross-entropy:
    a model answering perfectly no longer reaches zero. A validation loss
    carrying that offset would sit on a different scale for every value of the
    option — unreadable against ln(7), the loss of a uniform guess, and not
    comparable to any run recorded before the option existed.

    Early stopping reads the validation loss as well, which is the second
    reason. Sharing one criterion would let the treatment move the rule that
    picks the best epoch, and a sweep whose selection rule shifts with the value
    being swept measures two things at once.

    Args:
        label_smoothing: Mass moved off the true class while training, from 0
            to 1. Zero leaves training and validation identical.

    Returns:
        The training criterion and the validation criterion.
    """
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing), nn.CrossEntropyLoss()


def train_one_epoch(model, loader, criterion, optimizer, device, frozen=()) -> float:
    """Run one pass over the training data and return the mean loss.

    Measured against ln(7) ≈ 1.95, the loss of a model guessing uniformly across
    seven classes: a lower value means the model learned something.

    ``model.train()`` is recursive, which is why the frozen stages are named
    again here rather than only once at setup. Left alone they would come back
    into training mode at the top of every epoch and their batch normalization
    would resume re-estimating its statistics from these batches — a freeze that
    holds the weights and lets the normalisation drift is half a freeze, and the
    half that moves is the one the domain gap is made of.

    Args:
        model: The network to update.
        loader: Serves the epoch's training batches.
        criterion: The loss being minimised, smoothed targets included.
        optimizer: Applies the gradients. Sees only trainable parameters.
        device: Where the pass runs.
        frozen: Stages held at their pretrained values, from ``model.freeze``.

    Returns:
        Mean loss across the epoch's batches.
    """
    model.train()
    for stage in frozen:
        stage.eval()

    running_loss = 0.0
    for frames, labels, _ in loader:
        frames = frames.to(device)
        labels = labels.to(device)

        outputs = model(frames)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


if __name__ == "__main__":
    args = parse_args()

    # Covers the fresh classification head's initial weights and every random
    # transform; --seed also offsets the frame sampler, so one number moves all
    # the training stochasticity together and a repetition is named by it. The
    # train/validation split keeps its own fixed seed: which scenes are held out
    # is a control, and varying it would change the training data between runs
    # that are supposed to differ only in the treatment.
    torch.manual_seed(args.seed)

    manifest = pd.read_csv(PROJECT_ROOT / "data/manifests" / args.manifest)
    # Dropped before the split rather than after, so the excluded rows are absent
    # from both sides. split_by_group is reused for the checking it already does:
    # a name absent from the manifest stops the run instead of silently
    # excluding nothing.
    if args.exclude_groups:
        manifest, _ = split_by_group(manifest, args.exclude_groups)
    train_manifest, eval_manifest = (
        split_by_group(manifest, args.validation_groups)
        if args.validation_groups
        else split_by_scene(manifest)
    )
    # Training rows only, and after the split rather than before it. Validation
    # keeps the whole window on purpose: it is what every run recorded so far
    # was scored on, so narrowing it here would move the measurement along with
    # the treatment, and scoring on frames the model no longer trains on is the
    # honest question anyway.
    # Sampled before the frames are narrowed, and on the training side alone:
    # validation has to stay the same set of videos across the whole curve or
    # the points are scored against different questions.
    if args.fraction < 1:
        train_manifest = sample_videos(train_manifest, args.fraction, args.seed)
    train_manifest = select_frames(train_manifest, args.window, args.seed)

    held_out = sorted(eval_manifest["group_id"].unique())

    # Training only. Validation stays the seven-class question every run so far
    # was scored on, so the curve that picks a checkpoint keeps meaning what it
    # meant. The held-out groups are named rather than drawn again: the two
    # manifests cover the same videos, and a second draw could put a scene on
    # opposite sides of the boundary in the two of them.
    num_classes = NUM_CLASSES
    if args.idle_manifest:
        idle = pd.read_csv(PROJECT_ROOT / "data/manifests" / args.idle_manifest)
        idle_train, _ = split_by_group(idle, held_out)
        train_manifest = add_idle_rows(train_manifest, idle_train)
        num_classes = IDLE_LABEL + 1

    # Both read off the file rather than asked for, the way evaluation.py already
    # reads them: a fine-tuning run that met the wrong architecture would fail on
    # a shape mismatch at best, and a caller told to remember which checkpoint
    # carries which head will eventually pair the wrong two.
    initial_weights = None
    backbone = args.backbone
    if args.initial_weights:
        initial_weights = torch.load(
            PROJECT_ROOT / args.initial_weights, map_location="cpu"
        )
        num_classes = head_width(initial_weights)
        backbone = backbone_of(initial_weights)

    augmentation = Augmentation(
        photometric=args.photometric,
        geometric=args.geometric,
        background=args.background,
        texture=args.texture,
        texture_fill=args.texture_fill,
        gamma_shift=args.gamma_shift,
    )

    excluded = " ".join(args.exclude_groups) if args.exclude_groups else "none"
    print(
        f"manifest {args.manifest}  validation groups: {' '.join(held_out)}  "
        f"excluded: {excluded}"
    )
    print(
        f"fraction {args.fraction:g} -> "
        f"train {train_manifest['video_id'].nunique()} videos / "
        f"validation {eval_manifest['video_id'].nunique()} videos "
        f"({len(eval_manifest) / len(manifest):.1%} of frames)"
    )
    best_name = checkpoint_name(args.checkpoint_name, args.seed)
    print(
        f"seed {args.seed}  photometric {augmentation.photometric}  "
        f"geometric {augmentation.geometric}  "
        f"background {augmentation.background}  "
        f"label smoothing {args.label_smoothing}  "
        f"gamma shift {augmentation.gamma_shift}  "
        f"freeze {args.freeze}  "
        f"texture {augmentation.texture}/{augmentation.texture_fill}  "
        f"window {args.window} ({len(train_manifest)} training frames)  "
        f"crop {args.crop_size}  "
        f"idle {args.idle_manifest or 'off'} ({num_classes} classes)  "
        f"lr {args.lr}  "
        f"max epochs {args.max_epochs}  "
        f"patience {args.patience}  workers {args.num_workers}  ->  "
        f"checkpoints/{best_name}"
        f"{'  (+ one per epoch)' if args.save_every_epoch else ''}"
    )

    train_loader, eval_loader = build_loaders(
        train_manifest,
        eval_manifest,
        PROJECT_ROOT,
        args.num_workers,
        augmentation=augmentation,
        crop_size=args.crop_size,
        seed=args.seed,
    )

    device = pick_device()
    print(f"device: {describe(device)}")
    model = build_model(num_classes, backbone).to(device)
    if initial_weights is not None:
        model.load_state_dict(initial_weights)
        print(
            f"starting from {args.initial_weights}  ({backbone}, {num_classes} outputs)"
        )
    frozen = freeze(model, args.freeze)
    criterion, validation_criterion = build_criteria(args.label_smoothing)
    # Only the parameters that still train. Adam would skip a frozen one anyway,
    # having no gradient to apply, but naming them keeps the optimizer's state to
    # the size of what it actually updates. Under --freeze none this is every
    # parameter, so the runs recorded before the option are untouched.
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"training {sum(p.numel() for p in trainable):,} of "
        f"{sum(p.numel() for p in model.parameters()):,} parameters"
    )
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    stopper = EarlyStopping(args.patience)

    # Named beside the checkpoint it belongs to, and deleted when the run ends,
    # so its presence is exactly the statement "this run was cut off".
    progress_path = checkpoint_dir / f"{Path(best_name).stem}_progress.pt"
    epochs_done = load_progress(
        progress_path, model, optimizer, stopper, train_loader.sampler, args
    )
    if epochs_done:
        print(
            f"resuming after epoch {epochs_done}, best validation loss so far "
            f"{stopper.best_loss:.4f}"
        )

    for epoch in range(epochs_done, args.max_epochs):
        loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, frozen
        )

        logits, labels, _ = predict(model, eval_loader, device)
        eval_loss, eval_accuracy = frame_metrics(logits, labels, validation_criterion)

        print(
            f"epoch {epoch + 1}/{args.max_epochs}  train loss {loss:.4f}  "
            f"validation loss {eval_loss:.4f}  frame accuracy {eval_accuracy:.1%}"
        )

        if stopper.improved(eval_loss):
            save_atomically(model.state_dict(), checkpoint_dir / best_name)
            print("  best so far, checkpoint written")

        if args.save_every_epoch:
            save_atomically(
                model.state_dict(),
                checkpoint_dir
                / checkpoint_name(args.checkpoint_name, args.seed, epoch + 1),
            )

        # Last, so that progress is only claimed for an epoch whose checkpoint
        # is already on disk.
        save_progress(
            progress_path,
            epoch + 1,
            model,
            optimizer,
            stopper,
            train_loader.sampler,
            args,
        )

        if stopper.exhausted:
            print(f"stopped: {args.patience} epochs without improvement")
            break
    else:
        # Reaching the cap says only that patience never ran out, and with a
        # patience as large as the budget it never can — so the loop finishing
        # is not evidence that the run was still gaining. Whether it was is a
        # separate fact, and the stopper already holds it: an epoch count of
        # zero means the last epoch was the best one seen.
        past_best = stopper.epochs_without_improvement
        state = (
            "still improving" if past_best == 0 else f"{past_best} epochs past its best"
        )
        print(f"stopped: reached the {args.max_epochs}-epoch cap, {state}")

    # However the loop ended, it ended: what is left on disk is the best
    # checkpoint, and progress that describes a finished run would only mislead
    # the next one launched under the same name.
    progress_path.unlink(missing_ok=True)
