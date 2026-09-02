"""Run a trained model over a set of frames and report what it predicted.

Inference and analysis are deliberately kept apart. A pass over the model is the
expensive half and happens once per model and set of rows; the questions asked
of its output are many — loss, accuracy per frame, aggregation over a growing
number of frames, confusion between classes — and every one of them is
arithmetic over the same numbers. Returning the raw scores instead of a single
figure is what lets the expensive half run once and the cheap half run often.
"""

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = "syn_ground_train.csv"
CHECKPOINT = "syn_ground_train.pt"
HOLD_OUT_VALIDATION = True
BATCH_SIZE = 64
NUM_WORKERS = 8


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Score every frame a loader serves, without updating the model.

    The scores are returned before the softmax, as logits, because both things
    built on top of them want that form: the loss is defined on logits, and
    class probabilities are one softmax away. Normalising here would only force
    the loss to undo it.

    They come back on the CPU. The caller keeps a whole split's worth of scores,
    and accumulating them on the GPU would compete with the memory training
    needs on the next epoch.

    Args:
        model: The network to run. Switched to evaluation mode, which fixes the
            batch normalization statistics instead of updating them from the
            batch at hand.
        loader: Serves the frames to score. Its order is preserved in the
            result, so the rows can be matched back to a manifest.
        device: Where the forward pass runs.

    Returns:
        The ``(frames, classes)`` logits, the matching labels, and the video
        each frame came from, all in the order the loader served them.
    """
    model.eval()

    logits = []
    labels = []
    video_ids = []

    with torch.no_grad():
        for frames, batch_labels, batch_video_ids in loader:
            logits.append(model(frames.to(device)).cpu())
            labels.append(batch_labels)
            video_ids.extend(batch_video_ids)

    return torch.cat(logits), torch.cat(labels), video_ids


def frame_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Reduce a pass over the model to a loss and a per-frame accuracy.

    Both are computed over every frame at once rather than averaged across
    batches. The evaluation loader keeps its last partial batch, so a mean of
    batch means would weight the frames in that batch more heavily than the
    rest.

    Accuracy here counts frames, not videos: it says how often a single frame is
    read correctly on its own, which is the quantity the whole project is about.
    The video-level figure comes later, from aggregating these same scores.

    Args:
        logits: Scores as returned by ``predict``.
        labels: The true class of each frame.
        criterion: Loss function, the same one training minimises, so that the
            two numbers can be read against each other.

    Returns:
        The mean loss and the fraction of frames classified correctly.
    """
    loss = criterion(logits, labels).item()
    accuracy = (logits.argmax(dim=1) == labels).float().mean().item()

    return loss, accuracy


def probability_table(
    logits: torch.Tensor,
    video_ids: list[str],
    manifest: pd.DataFrame,
    class_names: dict[int, str],
) -> pd.DataFrame:
    """Turn one pass over the model into the table every later analysis reads.

    This is the artefact that separates the expensive half of evaluation from
    the cheap one. Scoring frames needs a GPU and minutes; the questions asked
    of the scores — aggregating over a growing number of frames, comparing
    strategies, confusion between classes, where in the gesture the information
    sits — are arithmetic over these same numbers, and there are dozens of them.
    Writing the scores down once is what keeps the dozens free.

    Probabilities are stored rather than logits because every one of those
    questions is posed in probability space: averaging predictions, ranking
    frames by confidence, reading a class score. A mean of logits is not a mean
    of probabilities, and storing the form the analysis does not use would
    invite the wrong one.

    Only the columns that vary frame by frame are carried. Class, viewpoint and
    scene belong to the video, so they join back on ``video_id`` alone — which
    matters, because a frame is not uniquely identified by its video and frame
    number: short clips round two segments onto the same frame in about 1.7% of
    rows, and joining on that pair would multiply them.

    Args:
        logits: Scores as returned by ``predict``.
        video_ids: The video each score came from, also from ``predict``. Used
            to check the alignment rather than to fill a column.
        manifest: The rows that were scored, in the order they were served.
        class_names: Label to gesture name, so the columns say what they hold.

    Returns:
        One row per frame: the video it came from, where in the gesture it sits,
        its true label, and one probability column per gesture.

    Raises:
        RuntimeError: If the scores are not in the manifest's order, which would
            silently attach every probability to the wrong frame.
    """
    rows = manifest.reset_index(drop=True)
    if video_ids != rows["video_id"].tolist():
        raise RuntimeError("scores are out of manifest order; the table would be wrong")

    table = rows[["video_id", "frame_number", "position", "label"]].copy()
    probabilities = logits.softmax(dim=1).numpy()
    for label, name in sorted(class_names.items()):
        table[f"p_{name}"] = probabilities[:, label]

    return table


if __name__ == "__main__":
    from dataset import FrameDataset, eval_transform
    from device import describe, pick_device
    from manifest import load_class_names
    from model import build_model
    from splits import split_by_scene

    manifest = pd.read_csv(PROJECT_ROOT / "data/manifests" / MANIFEST)
    split_name = Path(MANIFEST).stem
    if HOLD_OUT_VALIDATION:
        _, manifest = split_by_scene(manifest)
        split_name = f"{split_name}_validation"

    class_names = load_class_names(PROJECT_ROOT / "data" / "class_dict.json")
    device = pick_device()
    print(f"device: {describe(device)}")
    model = build_model().to(device)
    model.load_state_dict(
        torch.load(PROJECT_ROOT / "checkpoints" / CHECKPOINT, map_location=device)
    )

    loader = DataLoader(
        FrameDataset(manifest, PROJECT_ROOT, eval_transform()),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    logits, labels, video_ids = predict(model, loader, device)
    loss, accuracy = frame_metrics(logits, labels, nn.CrossEntropyLoss())
    table = probability_table(logits, video_ids, manifest, class_names)

    predictions_dir = PROJECT_ROOT / "data" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    output = predictions_dir / f"{Path(CHECKPOINT).stem}__{split_name}.csv"
    table.to_csv(output, index=False)

    print(f"scored {len(table)} frames from {table['video_id'].nunique()} videos")
    print(f"frame loss {loss:.4f}  frame accuracy {accuracy:.1%}")
    print(f"written to {output.relative_to(PROJECT_ROOT)}")
