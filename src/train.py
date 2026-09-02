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

from dataset import FrameDataset, SegmentSampler, eval_transform, train_transform
from device import describe, pick_device
from evaluation import frame_metrics, predict
from model import build_model
from splits import split_by_scene

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BATCH_SIZE = 64
TRAIN_SEED = 0
FRAMES_PER_EPOCH = 8

# Defaults for the options parse_args exposes: the values the experiment log
# fixes for the grid, so an argument-free run is the standard run.
NUM_WORKERS = 12
MAX_EPOCHS = 15
PATIENCE = 3
PHOTOMETRIC = True
GEOMETRIC = True
CHECKPOINT_NAME = "syn_ground_train.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line: which cell of the training grid to run.

    Every option defaults to the value fixed for the grid, so a call with no
    arguments reproduces the standard run. What legitimately varies between
    cells is what these expose — the augmentation in play, the epoch budget,
    where the checkpoint lands. The controls the experiment log holds constant
    (batch size, learning rate, seeds) stay as module constants on purpose: an
    option can be varied without ever showing up in a diff.

    Args:
        argv: Arguments to parse. ``None`` reads ``sys.argv``.

    Returns:
        A namespace with ``photometric``, ``geometric``, ``max_epochs``,
        ``patience``, ``checkpoint_name`` and ``num_workers``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
        help="DataLoader subprocesses",
    )
    return parser.parse_args(argv)


def build_loaders(
    train_manifest: pd.DataFrame,
    eval_manifest: pd.DataFrame,
    data_root: Path,
    num_workers: int = NUM_WORKERS,
    frames_per_epoch: int = FRAMES_PER_EPOCH,
    photometric: bool = PHOTOMETRIC,
    geometric: bool = GEOMETRIC,
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
        sampler=SegmentSampler(train_dataset.data_frame, frames_per_epoch),
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

    # The split and the frame sampler carry their own generators; this one covers
    # what is left — the fresh classification head's initial weights and every
    # random transform. Without it a run cannot be repeated, and two runs of one
    # configuration differ by an amount that is not distinguishable from the
    # difference between two configurations.
    torch.manual_seed(TRAIN_SEED)

    manifest = pd.read_csv(PROJECT_ROOT / "data/manifests/syn_ground_train.csv")
    train_manifest, eval_manifest = split_by_scene(manifest)

    held_out = sorted(eval_manifest["group_id"].unique())
    print(f"validation scenes: {' '.join(held_out)}")
    print(
        f"train {train_manifest['video_id'].nunique()} videos / "
        f"validation {eval_manifest['video_id'].nunique()} videos "
        f"({len(eval_manifest) / len(manifest):.1%} of frames)"
    )
    print(
        f"photometric {args.photometric}  geometric {args.geometric}  "
        f"max epochs {args.max_epochs}  patience {args.patience}  "
        f"workers {args.num_workers}  ->  checkpoints/{args.checkpoint_name}"
    )

    train_loader, eval_loader = build_loaders(
        train_manifest,
        eval_manifest,
        PROJECT_ROOT,
        args.num_workers,
        photometric=args.photometric,
        geometric=args.geometric,
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
            torch.save(model.state_dict(), checkpoint_dir / args.checkpoint_name)
            print("  best so far, checkpoint written")

        if stopper.exhausted:
            print(f"stopped: {args.patience} epochs without improvement")
            break
    else:
        print(f"stopped: reached the {args.max_epochs}-epoch cap, still improving")
