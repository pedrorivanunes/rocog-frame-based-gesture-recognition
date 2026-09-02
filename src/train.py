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
    SAMPLER_SEED,
    FrameDataset,
    SegmentSampler,
    eval_transform,
    train_transform,
)
from device import describe, pick_device
from evaluation import frame_metrics, predict
from model import build_model
from splits import split_by_scene

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BATCH_SIZE = 64
FRAMES_PER_EPOCH = 8

# Defaults for the options parse_args exposes, so an argument-free run is the
# standard run.
SEED = 0
NUM_WORKERS = 12
MAX_EPOCHS = 15
PATIENCE = 3
PHOTOMETRIC = True
GEOMETRIC = True
CHECKPOINT_NAME = "syn_ground_train.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line: which cell of the training grid to run.

    Every option defaults to the value the grid holds fixed, so a call with no
    arguments reproduces the standard run. What legitimately varies between
    cells is what these expose — the augmentation in play, the epoch budget,
    where the checkpoint lands. Batch size, learning rate and the seeds stay as
    module constants on purpose: a control that becomes an option can be varied
    without ever showing up in a diff.

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

    Args:
        argv: Arguments to parse. ``None`` reads ``sys.argv``.

    Returns:
        A namespace with ``seed``, ``photometric``, ``geometric``,
        ``max_epochs``, ``patience``, ``checkpoint_name``, ``num_workers`` and
        ``save_every_epoch``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
        default=PHOTOMETRIC,
        help="jitter brightness and contrast while training",
    )
    parser.add_argument(
        "--geometric",
        action=argparse.BooleanOptionalAction,
        default=GEOMETRIC,
        help="flip and vary the crop scale while training",
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
    photometric: bool = PHOTOMETRIC,
    geometric: bool = GEOMETRIC,
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
            training rows by construction — see ``splits.split_by_scene``.
        data_root: Directory the manifests' ``path`` column is relative to.
        num_workers: Loader subprocesses. Zero avoids the ~16 s spawn cost, which
            is worth paying only for runs long enough to amortise it.
        frames_per_epoch: Frames drawn from each training video per epoch. The
            sampler shuffles, so the loader must not — passing a sampler and
            ``shuffle=True`` together is rejected by the DataLoader.
        photometric: Whether training jitters brightness and contrast. Only the
            training pipeline is affected; evaluation stays fixed, so successive
            runs are measured against the same images.
        geometric: Whether training flips and varies the crop's scale. Same
            restriction — training only.
        seed: Which repetition of a configuration this is. Offsets the sampler's
            own seed rather than replacing it, so that seed 0 keeps drawing the
            frames earlier runs drew and stays comparable to them.

    Returns:
        The training loader and the evaluation loader.
    """
    train_dataset = FrameDataset(
        train_manifest, data_root, transform=train_transform(photometric, geometric)
    )
    eval_dataset = FrameDataset(eval_manifest, data_root, transform=eval_transform())

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=SegmentSampler(
            train_dataset.data_frame, frames_per_epoch, seed=SAMPLER_SEED + seed
        ),
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


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    """Run one pass over the training data and return the mean loss.

    Measured against ln(7) ≈ 1.95, the loss of a model guessing uniformly across
    seven classes: a lower value means the model learned something.

    Returns:
        Mean loss across the epoch's batches.
    """
    model.train()

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

    manifest = pd.read_csv(PROJECT_ROOT / "data/manifests/syn_ground_train.csv")
    train_manifest, eval_manifest = split_by_scene(manifest)

    held_out = sorted(eval_manifest["group_id"].unique())
    print(f"validation scenes: {' '.join(held_out)}")
    print(
        f"train {train_manifest['video_id'].nunique()} videos / "
        f"validation {eval_manifest['video_id'].nunique()} videos "
        f"({len(eval_manifest) / len(manifest):.1%} of frames)"
    )
    best_name = checkpoint_name(args.checkpoint_name, args.seed)
    print(
        f"seed {args.seed}  photometric {args.photometric}  "
        f"geometric {args.geometric}  max epochs {args.max_epochs}  "
        f"patience {args.patience}  workers {args.num_workers}  ->  "
        f"checkpoints/{best_name}"
        f"{'  (+ one per epoch)' if args.save_every_epoch else ''}"
    )

    train_loader, eval_loader = build_loaders(
        train_manifest,
        eval_manifest,
        PROJECT_ROOT,
        args.num_workers,
        photometric=args.photometric,
        geometric=args.geometric,
        seed=args.seed,
    )

    device = pick_device()
    print(f"device: {describe(device)}")
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    stopper = EarlyStopping(args.patience)

    for epoch in range(args.max_epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        logits, labels, _ = predict(model, eval_loader, device)
        eval_loss, eval_accuracy = frame_metrics(logits, labels, criterion)

        print(
            f"epoch {epoch + 1}/{args.max_epochs}  train loss {loss:.4f}  "
            f"validation loss {eval_loss:.4f}  frame accuracy {eval_accuracy:.1%}"
        )

        if stopper.improved(eval_loss):
            torch.save(model.state_dict(), checkpoint_dir / best_name)
            print("  best so far, checkpoint written")

        if args.save_every_epoch:
            torch.save(
                model.state_dict(),
                checkpoint_dir
                / checkpoint_name(args.checkpoint_name, args.seed, epoch + 1),
            )

        if stopper.exhausted:
            print(f"stopped: {args.patience} epochs without improvement")
            break
    else:
        print(f"stopped: reached the {args.max_epochs}-epoch cap, still improving")
