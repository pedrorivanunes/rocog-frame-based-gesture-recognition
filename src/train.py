"""Train a ResNet18 to classify RoCoG-v2 gesture frames.

Loads frames through the extraction manifest, fine-tunes ImageNet weights on the
seven gesture classes, and writes a checkpoint after every epoch.
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from dataset import FrameDataset, eval_transform, train_transform

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NUM_CLASSES = 7
BATCH_SIZE = 64
NUM_WORKERS = 8
EPOCHS = 5


def build_loaders(
    train_manifest: Path,
    eval_manifest: Path,
    data_root: Path,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, DataLoader]:
    """Build the training and evaluation loaders from two manifests.

    Training shuffles and drops the last partial batch, since batch normalization
    fails on a batch of one. Evaluation keeps every sample and its order.

    Args:
        train_manifest: Manifest listing the frames to train on.
        eval_manifest: Manifest listing the frames to evaluate on.
        data_root: Directory the manifests' ``path`` column is relative to.
        num_workers: Loader subprocesses. Zero avoids the ~16 s spawn cost, which
            is worth paying only for runs long enough to amortise it.

    Returns:
        The training loader and the evaluation loader.
    """
    train_dataset = FrameDataset(train_manifest, data_root, transform=train_transform())
    eval_dataset = FrameDataset(eval_manifest, data_root, transform=eval_transform())

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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
    train_loader, eval_loader = build_loaders(
        PROJECT_ROOT / "data/manifests/syn_ground_train.csv",
        PROJECT_ROOT / "data/manifests/real_ground_test.csv",
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
