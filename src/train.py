"""Train a ResNet18 to classify RoCoG-v2 gesture frames.

Loads frames through the extraction manifest, fine-tunes ImageNet weights on the
seven gesture classes, and writes a checkpoint after every epoch.
"""

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from dataset import FrameDataset, SegmentSampler, eval_transform, train_transform
from splits import split_by_scene

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NUM_CLASSES = 7
BATCH_SIZE = 64
NUM_WORKERS = 8
EPOCHS = 5
FRAMES_PER_EPOCH = 8


def build_loaders(
    train_manifest: pd.DataFrame,
    eval_manifest: pd.DataFrame,
    data_root: Path,
    num_workers: int = NUM_WORKERS,
    frames_per_epoch: int = FRAMES_PER_EPOCH,
) -> tuple[DataLoader, DataLoader]:
    """Build the training and evaluation loaders from two sets of manifest rows.

    Training draws a subset of each video's frames and drops the last partial
    batch, since batch normalization fails on a batch of one. Evaluation keeps
    every frame and its order, so that successive epochs are measured against
    exactly the same data.

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

    Returns:
        The training loader and the evaluation loader.
    """
    train_dataset = FrameDataset(train_manifest, data_root, transform=train_transform())
    eval_dataset = FrameDataset(eval_manifest, data_root, transform=eval_transform())

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=SegmentSampler(train_dataset.data_frame, frames_per_epoch),
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return train_loader, eval_loader


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Build a ResNet18 with ImageNet weights and a fresh classification head.

    The convolutional layers keep what they learned on ImageNet — edges, textures,
    shapes. Only the final layer is replaced, mapping the 512 features it produces
    to the gesture classes instead of ImageNet's 1000 categories. That mapping is
    what training has to learn.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


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
    manifest = pd.read_csv(PROJECT_ROOT / "data/manifests/syn_ground_train.csv")
    train_manifest, eval_manifest = split_by_scene(manifest)

    held_out = sorted(eval_manifest["group_id"].unique())
    print(f"validation scenes: {' '.join(held_out)}")
    print(
        f"train {train_manifest['video_id'].nunique()} videos / "
        f"validation {eval_manifest['video_id'].nunique()} videos "
        f"({len(eval_manifest) / len(manifest):.1%} of frames)"
    )

    train_loader, eval_loader = build_loaders(
        train_manifest,
        eval_manifest,
        PROJECT_ROOT,
        NUM_WORKERS,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"epoch {epoch + 1}/{EPOCHS}  loss {loss:.4f}")
        torch.save(model.state_dict(), checkpoint_dir / "syn_ground_train.pt")
