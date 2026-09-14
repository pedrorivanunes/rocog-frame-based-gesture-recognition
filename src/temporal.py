"""Read a video as an ordered sequence of frames instead of a bag of them.

Every decision this project has reported is invariant to the order of the
frames it aggregates: averaging class probabilities cannot use time, not by
choice but by construction. That property is what makes the frame classifier a
clean probe, and it is also the ceiling the probe cannot see past. This is the
first step over that line, and it is deliberately the cheapest one.

Nothing about the appearance half changes. The backbone is a trained checkpoint
run once to produce, for each frame, the vector its classifier would have read.
A small head then learns to turn a sequence of those vectors into one answer.
Because the features are fixed, any difference between an order-aware head and
an order-blind one is order, and nothing else.

⚠️ THE CONTROL IS THE POINT, AND IT IS NOT THE LINEAR HEAD. A recurrent head
carries far more parameters than a linear one, so beating the linear head would
confound order with capacity. The control that isolates order is the same
recurrent head fed the same frames **shuffled**: identical architecture,
identical capacity, identical training, with only the sequence destroyed. The
linear head is kept as a second, weaker reference — it says what a sequence
model buys over no sequence model at all.

⚠️ THE HEAD TRAINS ON EXACTLY WHAT THE ORIGINAL HEAD TRAINED ON. The classifier
this replaces was fitted, jointly with the backbone, on the synthetic training
split and stopped on the held-out scenes. Giving the new head less would not
make the comparison cleaner; it would hand the treatment a handicap the control
never had, and any difference would be data rather than order. So the split is
the same one, drawn by the same function, and the stopping scenes are the same
scenes.

An earlier version of this fitted the head on the held-out scenes alone, on the
reasoning that the backbone answers its own training videos at 99% and a head
fitted on features that good would learn that one frame is always enough. That
traded the result for a secondary concern, and the first measurements weakened
the concern itself: on held-out scenes the heads climb from 94.5% at one frame
to 98.6% at twenty-four, so there is room to learn to combine frames, and the
shuffled control learns it just as well as the ordered one.

Everything stays source-only: no real video, labelled or not, enters training.

Run from the project root::

    python src/temporal.py --features idle_s0 --head gru --seed 0
    python src/temporal.py --features idle_s0 --head gru --shuffle --seed 0
    python src/temporal.py --features idle_s0 --head linear --seed 0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FRAME_COUNTS = (1, 2, 4, 8, 16, 24)
HIDDEN = 128
BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 20
LEARNING_RATE = 1e-3


class MeanHead(nn.Module):
    """Average the frames, then classify. Order-blind by construction."""

    def __init__(self, width: int, classes: int) -> None:
        """Build the pooled classifier for features of the given width."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, classes)
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Score one batch of ``(videos, frames, width)`` sequences."""
        return self.net(sequence.mean(dim=1))


class GRUHead(nn.Module):
    """Read the frames in order, carrying a summary of what came before.

    Unidirectional on purpose. A backward pass would see the whole clip before
    answering, which no deployment that emits a decision as frames arrive can
    do, and the cost argument this project makes is about exactly that setting.
    """

    def __init__(self, width: int, classes: int) -> None:
        """Build the recurrent classifier for features of the given width."""
        super().__init__()
        self.rnn = nn.GRU(width, HIDDEN, batch_first=True)
        self.out = nn.Linear(HIDDEN, classes)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Score one batch of ``(videos, frames, width)`` sequences."""
        _, last = self.rnn(sequence)
        return self.out(last.squeeze(0))


class Conv1dHead(nn.Module):
    """Look at short windows of frames, then pool over the whole clip.

    Between the other two: it sees local order within a window of three frames
    and ignores it beyond, so where it lands says whether what matters is local
    or spans the gesture.
    """

    def __init__(self, width: int, classes: int) -> None:
        """Build the convolutional classifier for features of the given width."""
        super().__init__()
        self.conv = nn.Conv1d(width, HIDDEN, kernel_size=3, padding=1)
        self.out = nn.Linear(HIDDEN, classes)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Score one batch of ``(videos, frames, width)`` sequences."""
        activated = torch.relu(self.conv(sequence.transpose(1, 2)))
        return self.out(activated.mean(dim=2))


HEADS = {"linear": MeanHead, "gru": GRUHead, "conv1d": Conv1dHead}


def scene_of(video_ids: np.ndarray, manifest: pd.DataFrame) -> np.ndarray:
    """The scene each video belongs to, looked up from the manifest.

    Kept out of the feature file on purpose: the features are what a checkpoint
    produced, and which scene a video came from is a property of the dataset
    that any manifest can answer.
    """
    lookup = manifest.drop_duplicates("video_id").set_index("video_id")["group_id"]

    return lookup.reindex(video_ids).to_numpy()


def pick(frames: int, picks: int, generator: np.random.Generator | None) -> np.ndarray:
    """Choose which frames a decision sees.

    Spread over the clip in either case, because that is the protocol every
    number in the record was produced under. The generator makes the position
    within each segment random, which is the augmentation that stops the head
    from fitting one fixed set of frame indices.
    """
    edges = np.linspace(0, frames, picks + 1)
    if generator is None:
        return ((edges[:-1] + edges[1:]) / 2).astype(int)

    spans = np.maximum(np.diff(edges).astype(int), 1)
    return (edges[:-1].astype(int) + generator.integers(0, spans)).clip(0, frames - 1)


def batches(
    features: torch.Tensor,
    labels: torch.Tensor,
    generator: np.random.Generator,
    shuffle_frames: bool,
):
    """Serve shuffled batches, each at one frame count drawn for that batch.

    Drawing K per batch rather than fixing it is what keeps the head usable at
    every K the accuracy curve reports. A head trained at one length and read at
    another would be an extrapolation reported as a measurement.
    """
    order = generator.permutation(len(features))
    for start in range(0, len(order), BATCH_SIZE):
        rows = order[start : start + BATCH_SIZE]
        picks = int(generator.choice(FRAME_COUNTS))
        chosen = features[rows][:, pick(features.shape[1], picks, generator)]
        if shuffle_frames:
            chosen = chosen[:, generator.permutation(chosen.shape[1])]
        yield chosen, labels[rows]


def evaluate(
    head: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    picks: int,
    shuffle_frames: bool,
    generator: np.random.Generator | None = None,
) -> torch.Tensor:
    """Probabilities for every video at one frame count, with no gradient."""
    head.eval()
    chosen = features[:, pick(features.shape[1], picks, None)]
    if shuffle_frames:
        stream = generator or np.random.default_rng(0)
        chosen = chosen[:, stream.permutation(chosen.shape[1])]
    with torch.no_grad():
        return head(chosen).softmax(dim=1)


def train_head(
    head: nn.Module,
    train: tuple[torch.Tensor, torch.Tensor],
    held: tuple[torch.Tensor, torch.Tensor],
    generator: np.random.Generator,
    shuffle_frames: bool,
) -> nn.Module:
    """Fit the head, stopping on scenes it did not fit.

    The stopping score averages accuracy over every frame count rather than
    taking one, because the head is meant to serve the whole curve and a
    checkpoint chosen at K=24 alone would be chosen for the easiest point.
    """
    optimiser = torch.optim.Adam(head.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_score, best_state, waited, stopped = -1.0, None, 0, 0
    for epoch in range(MAX_EPOCHS):
        stopped = epoch + 1
        head.train()
        for sequence, target in batches(*train, generator, shuffle_frames):
            optimiser.zero_grad()
            criterion(head(sequence), target).backward()
            optimiser.step()

        scores = [
            (evaluate(head, *held, picks, shuffle_frames).argmax(1) == held[1])
            .float()
            .mean()
            .item()
            for picks in FRAME_COUNTS
        ]
        score = float(np.mean(scores))
        if score > best_score:
            best_score, waited = score, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break

    print(f"  stopped at epoch {stopped}, held-out mean accuracy {best_score:.4f}")
    head.load_state_dict(best_state)

    return head


def decisions(
    head: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    video_ids: np.ndarray,
    class_names: dict[int, str],
    shuffle_frames: bool,
) -> pd.DataFrame:
    """One row per video and frame count: what the head decided, and from what.

    Deliberately not the per-frame shape the rest of the record uses. A head
    that reads a sequence produces one answer per clip, and writing it as though
    it were per-frame would invite tools to average what is already an average.
    """
    rows = []
    for picks in FRAME_COUNTS:
        probabilities = evaluate(head, features, labels, picks, shuffle_frames).numpy()
        table = pd.DataFrame(
            {"video_id": video_ids, "label": labels.numpy(), "frames": picks}
        )
        for label, name in sorted(class_names.items()):
            table[f"p_{name}"] = probabilities[:, label]
        rows.append(table)

    return pd.concat(rows, ignore_index=True)


def load(stem: str, split: str) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Read one feature file written by ``evaluation.py --features``."""
    path = PROJECT_ROOT / "data/features" / f"{stem}__{split}.npz"
    if not path.exists():
        raise SystemExit(f"missing {path.relative_to(PROJECT_ROOT)}; extract it first")
    held = np.load(path)

    return (
        torch.from_numpy(held["features"]),
        torch.from_numpy(held["labels"]).long(),
        held["video_ids"],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--features",
        required=True,
        help="checkpoint stem the features were written under, e.g. idle_s0.",
    )
    parser.add_argument("--head", choices=sorted(HEADS), default="gru")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="destroy the order while keeping the architecture. This is the "
        "control that isolates order from capacity.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--score",
        nargs="+",
        default=["real_ground_test", "syn_ground_train"],
        help="which feature files to score, by split name. The source split is "
        "restricted to the scenes the head was stopped on, never fitted on.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Fit one head on held-out synthetic scenes and score it where asked."""
    from manifest import load_class_names, with_idle_class
    from splits import split_by_scene

    args = parse_args(argv)
    torch.manual_seed(args.seed)
    generator = np.random.default_rng(args.seed)

    source = "syn_ground_train"
    features, labels, video_ids = load(args.features, source)

    # The same boundary the backbone was trained across, drawn by the same
    # function from the same manifest. Anything else would give the head a
    # different dataset from the classifier it is being read against.
    manifest = pd.read_csv(PROJECT_ROOT / "data/manifests/syn_ground_train.csv")
    _, validation = split_by_scene(manifest)
    held_videos = set(validation["video_id"].unique())
    is_held = np.array([video in held_videos for video in video_ids])
    scenes = scene_of(video_ids, manifest)
    fit_scenes = len(pd.unique(scenes[~is_held]))
    stop_scenes = len(pd.unique(scenes[is_held]))

    class_names = load_class_names(PROJECT_ROOT / "data" / "class_dict.json")
    classes = int(labels.max()) + 1
    if classes > len(class_names):
        class_names = with_idle_class(class_names)

    name = f"temporal_{args.head}{'_shuffled' if args.shuffle else ''}"
    print(f"{name}  from {args.features}  seed {args.seed}")
    print(
        f"  {(~is_held).sum()} videos over {fit_scenes} scenes to fit, "
        f"{is_held.sum()} over {stop_scenes} to stop on"
    )

    head = HEADS[args.head](features.shape[2], classes)
    print(f"  {sum(p.numel() for p in head.parameters()):,} parameters")
    head = train_head(
        head,
        (features[~is_held], labels[~is_held]),
        (features[is_held], labels[is_held]),
        generator,
        args.shuffle,
    )

    predictions = PROJECT_ROOT / "data" / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    for split in args.score:
        scored = load(args.features, split)
        if split == source:
            # Scoring the source means scoring the held-out scenes, never the
            # ones the head was fitted on. Without this the sanity check the
            # whole reading rests on would be measured on training data.
            scored = (scored[0][is_held], scored[1][is_held], scored[2][is_held])
        table = decisions(head, *scored, class_names, args.shuffle)
        output = predictions / f"{name}_{args.features}__{split}__byK.csv"
        table.to_csv(output, index=False)
        top = table[table["frames"] == max(FRAME_COUNTS)]
        columns = [f"p_{n}" for _, n in sorted(class_names.items())]
        accuracy = (top[columns].to_numpy().argmax(1) == top["label"]).mean()
        print(f"  {split}: K=24 accuracy {accuracy:.1%}  ->  {output.name}")


if __name__ == "__main__":
    main()
