"""Run a trained model over a set of frames and report what it predicted.

Inference and analysis are deliberately kept apart. A pass over the model is the
expensive half and happens once per model and set of rows; the questions asked
of its output are many — loss, accuracy per frame, aggregation over a growing
number of frames, confusion between classes — and every one of them is
arithmetic over the same numbers. Returning the raw scores instead of a single
figure is what lets the expensive half run once and the cheap half run often.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader


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
