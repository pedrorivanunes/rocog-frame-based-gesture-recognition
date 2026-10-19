"""Classify gestures from where the joints are, frame by frame.

The appearance models read pixels. This branch reads the body: the joints the
pose detector found in each stored frame, extracted once by
``pose_keypoints.py``. Joints carry no texture, lighting or rendering, so they
are the same quantity whether a frame was filmed or synthesised, and a branch
that reads only them asks whether the gap between the two domains survives
when appearance is taken away.

Three inputs, each adding one thing to the last:

    position    where the arm joints are, relative to the body
    motion      the same, plus how far each joint moved since the previous
                stored frame of the video
    visibility  the same, plus the detector's confidence in each joint

The first is the branch's counterpart of a model reading still frames; the
second of one reading frame differences; the third lets the classifier learn
which joints the detector guessed rather than saw, which happens often in the
synthetic views that do not face the camera and almost never in the real ones.

Everything except the classifier mirrors how the appearance models are trained,
because the point is a comparison: the same synthetic videos, the same scenes
held out for choosing the checkpoint, the same frames of the gesture window,
the same idle frames as an eighth class, the same frames drawn each epoch, the
same horizontal mirroring, the same label smoothing, and the same rule for
which epoch to keep. Nothing from the real domain enters training.

The classifier is a small network of fully connected layers. What it reads is a
few dozen numbers with no grid to them, so the convolutions an image needs have
nothing to act on. Its settings are its own and are fixed here rather than
tuned: a pretrained image network's learning rate and epoch budget describe a
different optimisation.

The tables it writes have the columns and names ``evaluation.py`` writes, so
every reading that accepts an appearance cell accepts these.

Run from the project root::

    python src/pose_train.py --input position --seed 0
    python src/pose_train.py --input motion --seed 0
    python src/pose_train.py --input visibility --seed 0
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from dataset import SAMPLER_SEED, CombinedSampler, SegmentSampler
from evaluation import probability_table
from manifest import IDLE_LABEL, load_class_names, with_idle_class
from splits import add_idle_rows, select_frames, split_by_group, split_by_scene
from train import (
    FRAMES_PER_EPOCH,
    IDLE_FRAMES_PER_EPOCH,
    EarlyStopping,
    build_criteria,
    checkpoint_name,
    save_atomically,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KEYPOINTS_DIR = PROJECT_ROOT / "data/pose/keypoints"
PREDICTIONS_DIR = PROJECT_ROOT / "data/predictions"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

GESTURE_MANIFEST = "syn_ground_train.csv"
IDLE_MANIFEST = "syn_ground_train_idle.csv"
REAL_MANIFESTS = ("real_ground_test.csv", "real_ground_train.csv")

# The arm joints and the hips, in BlazePose numbering, and the names the stored
# file must give them. The hips are there to measure the torso with; the
# gestures are signalled with the arms. Checked against the file rather than
# trusted, because a numbering that drifted would train on the wrong joints
# without complaint.
JOINTS = (11, 12, 13, 14, 15, 16, 23, 24)
JOINT_NAMES = (
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
)
# Each joint's partner on the other side of the body, by position in JOINTS. A
# mirrored frame shows the left arm where the right one was, and the detector
# names joints by the body's own side, so mirroring swaps the names as well as
# negating the horizontal coordinate.
MIRROR = (1, 0, 3, 2, 5, 4, 7, 6)

# The input each named branch reads, and the name its files are written under.
INPUTS = {"position": "pose", "motion": "pose_motion", "visibility": "pose_vis"}

# The base cell's training choices that are not about the network.
WINDOW = "middle"
LABEL_SMOOTHING = 0.1
FLIP_PROBABILITY = 0.5

# The classifier's own settings, the ones this project already used for a small
# head trained on stored per-frame features. Two hidden layers rather than that
# head's one: those features had been shaped for classification by the network
# that produced them, and joint coordinates have not.
HIDDEN = 128
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 200
PATIENCE = 20


class Joints:
    """The stored joints of one manifest, ready to be turned into inputs.

    Positions are measured from the midpoint of the shoulders and in units of
    the torso, the distance from that midpoint to the midpoint of the hips. That
    removes where the signaller stands and how far from the camera, which say
    nothing about the gesture. The torso rather than the width of the shoulders,
    because the shoulders shorten as the body turns away and the torso does not.

    Motion is measured between distinct stored frames, later minus earlier: a
    frame's own position minus that of the previous stored frame of the same
    video. The first frame of a video has no previous one and takes the interval
    to the next frame instead, so that no row carries a motion of zero merely for
    coming first; a constant that marked first frames would mark the idle class
    far more often than any gesture, since its videos store four frames, not
    twenty-four.
    """

    def __init__(self, manifest_name: str):
        """Load and normalise the joints stored for a manifest.

        Args:
            manifest_name: File name under ``data/manifests/``.

        Raises:
            RuntimeError: If the stored file does not match the manifest row for
                row, or names the joints differently.
        """
        manifest = pd.read_csv(PROJECT_ROOT / "data/manifests" / manifest_name)
        with np.load(KEYPOINTS_DIR / f"{Path(manifest_name).stem}.npz") as stored:
            if list(stored["path"]) != manifest["path"].tolist():
                raise RuntimeError(f"{manifest_name}: keypoints not aligned with rows")
            names = tuple(stored["names"][list(JOINTS)])
            if names != JOINT_NAMES:
                raise RuntimeError(f"{manifest_name}: joints are named {names}")
            image = stored["image"][:, list(JOINTS)]

        xy = image[..., :2]
        shoulders = xy[:, 0:2].mean(axis=1)
        hips = xy[:, 6:8].mean(axis=1)
        torso = np.linalg.norm(shoulders - hips, axis=1)
        self.position = (xy - shoulders[:, None]) / torso[:, None, None]
        self.visibility = image[..., 3]
        earlier, later = interval_ends(manifest)
        self.motion = self.position[later] - self.position[earlier]
        # A frame without a person has no joints; a motion measured against
        # such a frame has none either, but the position of the frame itself
        # still does. Kept apart so that each input loses only what it lacks.
        self.found = ~np.isnan(xy[:, 0, 0])
        self.found_interval = self.found & self.found[earlier] & self.found[later]
        # Paths repeat where a short clip rounded two samples onto one frame;
        # either row holds the same frame's joints.
        self.row_of = dict(zip(manifest["path"], range(len(manifest)), strict=True))

    def rows(self, paths: pd.Series) -> np.ndarray:
        """Positions in this manifest of the frames at the given paths."""
        return np.array([self.row_of[path] for path in paths])

    def usable(self, rows: np.ndarray, kind: str) -> np.ndarray:
        """Which of the given rows have everything the input needs."""
        return (self.found if kind == "position" else self.found_interval)[rows]


def interval_ends(manifest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """For every row, the two stored frames its motion is measured between.

    Args:
        manifest: One row per stored frame, with ``video_id`` and
            ``frame_number``.

    Returns:
        Row positions of the earlier and the later frame of each row's interval.
        For a video holding a single distinct frame both are the row itself,
        which makes its motion zero.
    """
    earlier = np.arange(len(manifest))
    later = np.arange(len(manifest))
    frames = manifest["frame_number"].to_numpy()
    for positions in manifest.groupby("video_id", sort=False).indices.values():
        ordered = positions[np.argsort(frames[positions], kind="stable")]
        numbers = frames[ordered]
        distinct, first = np.unique(numbers, return_index=True)
        if len(distinct) == 1:
            continue
        rank = np.searchsorted(distinct, numbers)
        earlier[ordered] = np.where(
            rank > 0, ordered[first[np.maximum(rank - 1, 0)]], ordered
        )
        later[ordered] = np.where(rank > 0, ordered, ordered[first[1]])
    return earlier, later


def width(kind: str) -> int:
    """Numbers per input row for the given input.

    Two coordinates per joint, two more for its motion, one for its confidence.
    """
    per_joint = {"position": 2, "motion": 4, "visibility": 5}[kind]
    return per_joint * len(JOINTS)


def inputs(
    joints: Joints, rows: np.ndarray, kind: str, flip: np.ndarray | None = None
) -> np.ndarray:
    """Assemble the input of each requested row, mirrored where asked.

    Args:
        joints: The manifest the rows belong to.
        rows: Positions in that manifest.
        kind: Which input, a key of ``INPUTS``.
        flip: Which rows to mirror, or ``None`` for none.

    Returns:
        One row of inputs per requested row.
    """
    position = joints.position[rows].copy()
    parts = [position]
    if kind in ("motion", "visibility"):
        parts.append(joints.motion[rows].copy())
    if kind == "visibility":
        parts.append(joints.visibility[rows][..., None].copy())
    if flip is not None and flip.any():
        for part in parts:
            part[flip] = part[flip][:, list(MIRROR)]
        # Horizontal coordinates change sign about the shoulders' midpoint;
        # confidence has no direction.
        for part in parts[: 2 if kind != "position" else 1]:
            part[flip, :, 0] *= -1
    return np.concatenate([part.reshape(len(rows), -1) for part in parts], axis=1)


class JointClassifier(nn.Module):
    """Fully connected layers over standardised joint inputs.

    The standardisation travels inside the model, as buffers set from the
    training rows, so a checkpoint scores new rows the way it was trained
    without anything else having to remember the statistics.
    """

    def __init__(self, width: int, classes: int):
        """Build the network for inputs of the given width.

        Args:
            width: Numbers per input row.
            classes: Outputs, the gestures and the idle class.
        """
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("scale", torch.ones(width))
        self.net = nn.Sequential(
            nn.Linear(width, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Score a batch of input rows."""
        return self.net((features - self.mean) / self.scale)


def training_rows(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The rows an appearance run trains and validates on, chosen the same way.

    Args:
        seed: The repetition, which the frame selection is given as the
            appearance runs give it.

    Returns:
        The training rows, gestures then idle, numbered from zero; and the
        validation rows, every frame of the held-out scenes.
    """
    manifest = pd.read_csv(PROJECT_ROOT / "data/manifests" / GESTURE_MANIFEST)
    train_rows, validation_rows = split_by_scene(manifest)
    train_rows = select_frames(train_rows, WINDOW, seed)
    held_out = sorted(validation_rows["group_id"].unique())
    idle = pd.read_csv(PROJECT_ROOT / "data/manifests" / IDLE_MANIFEST)
    idle_train, _ = split_by_group(idle, held_out)
    idle_train = idle_train[idle_train["video_id"].isin(train_rows["video_id"])]
    return add_idle_rows(train_rows, idle_train), validation_rows


def train(kind: str, seed: int, max_epochs: int = MAX_EPOCHS) -> JointClassifier:
    """Fit the classifier on the synthetic videos and keep its best epoch.

    Args:
        kind: Which input, a key of ``INPUTS``.
        seed: The repetition.
        max_epochs: Upper bound on epochs; patience usually ends the run first.

    Returns:
        The model at the epoch with the lowest validation loss.
    """
    torch.manual_seed(seed)
    flips = np.random.default_rng(seed)
    gestures, idle = Joints(GESTURE_MANIFEST), Joints(IDLE_MANIFEST)
    rows, validation_rows = training_rows(seed)

    # Every training row's input, unmirrored, and whether it has one. The two
    # kinds of row come from two manifests, so each is looked up in its own.
    is_idle = (rows["label"] == IDLE_LABEL).to_numpy()
    source_rows = np.empty(len(rows), dtype=int)
    source_rows[~is_idle] = gestures.rows(rows.loc[~is_idle, "path"])
    source_rows[is_idle] = idle.rows(rows.loc[is_idle, "path"])
    usable = np.empty(len(rows), dtype=bool)
    usable[~is_idle] = gestures.usable(source_rows[~is_idle], kind)
    usable[is_idle] = idle.usable(source_rows[is_idle], kind)
    labels = torch.as_tensor(rows["label"].to_numpy())

    def batch_inputs(positions: np.ndarray, flip: np.ndarray | None) -> torch.Tensor:
        idle_part = is_idle[positions]
        out = np.empty((len(positions), width(kind)), dtype=np.float32)
        for source, mask in ((gestures, ~idle_part), (idle, idle_part)):
            if mask.any():
                out[mask] = inputs(
                    source,
                    source_rows[positions[mask]],
                    kind,
                    None if flip is None else flip[mask],
                )
        return torch.as_tensor(out)

    every = np.flatnonzero(usable)
    reference = batch_inputs(every, None)
    model = JointClassifier(reference.shape[1], IDLE_LABEL + 1)
    model.mean.copy_(reference.mean(dim=0))
    model.scale.copy_(reference.std(dim=0).clamp_min(1e-6))

    validation_source = gestures.rows(validation_rows["path"])
    validation_usable = gestures.usable(validation_source, kind)
    validation_inputs = torch.as_tensor(
        inputs(gestures, validation_source[validation_usable], kind),
        dtype=torch.float32,
    )
    validation_labels = torch.as_tensor(
        validation_rows["label"].to_numpy()[validation_usable]
    )

    sampler = CombinedSampler(
        [
            SegmentSampler(
                rows, FRAMES_PER_EPOCH, seed=SAMPLER_SEED + seed, rows=~is_idle
            ),
            SegmentSampler(
                rows, IDLE_FRAMES_PER_EPOCH, seed=SAMPLER_SEED + seed, rows=is_idle
            ),
        ],
        seed=SAMPLER_SEED + seed,
    )
    criterion, validation_criterion = build_criteria(LABEL_SMOOTHING)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    stopper = EarlyStopping(PATIENCE)
    best_state = None

    print(
        f"input {kind}  seed {seed}  width {reference.shape[1]}  "
        f"training rows {len(rows)} ({(~usable).sum()} without a person)  "
        f"validation rows {len(validation_rows)} "
        f"({(~validation_usable).sum()} without a person)  "
        f"held-out scenes {' '.join(sorted(validation_rows['group_id'].unique()))}",
        flush=True,
    )
    for epoch in range(max_epochs):
        start = time.perf_counter()
        drawn = np.fromiter(iter(sampler), dtype=int)
        drawn = drawn[usable[drawn]]
        model.train()
        total, count = 0.0, 0
        # Whole batches only, as the appearance runs draw them.
        for first in range(0, len(drawn) - BATCH_SIZE + 1, BATCH_SIZE):
            positions = drawn[first : first + BATCH_SIZE]
            flip = flips.random(len(positions)) < FLIP_PROBABILITY
            optimiser.zero_grad()
            loss = criterion(model(batch_inputs(positions, flip)), labels[positions])
            loss.backward()
            optimiser.step()
            total += loss.item() * len(positions)
            count += len(positions)

        model.eval()
        with torch.no_grad():
            logits = model(validation_inputs)
            validation_loss = validation_criterion(logits, validation_labels).item()
            accuracy = (logits.argmax(dim=1) == validation_labels).float().mean()
        print(
            f"epoch {epoch + 1}/{max_epochs}  train loss {total / count:.4f}  "
            f"validation loss {validation_loss:.4f}  frame accuracy {accuracy:.1%}"
            f"  {time.perf_counter() - start:.1f} s",
            flush=True,
        )
        if stopper.improved(validation_loss):
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print("  best so far", flush=True)
        if stopper.exhausted:
            print(f"stopped: {PATIENCE} epochs without improvement", flush=True)
            break
    else:
        print(f"stopped: reached the {max_epochs}-epoch cap", flush=True)

    model.load_state_dict(best_state)
    return model


def score(model: JointClassifier, kind: str, name: str) -> None:
    """Write the probability tables for every split an appearance cell is read on.

    A frame without a person gets no score from the model and is written with
    the uniform distribution: it adds the same amount to every class of its
    video's sum, so the video's decision is the one the other frames make, and
    as a frame it counts as a miss.

    Args:
        model: The trained classifier.
        kind: The input it reads.
        name: The file stem, the branch's name with the seed.
    """
    class_names = with_idle_class(
        load_class_names(PROJECT_ROOT / "data" / "class_dict.json")
    )
    gestures = Joints(GESTURE_MANIFEST)
    _, validation_rows = split_by_scene(
        pd.read_csv(PROJECT_ROOT / "data/manifests" / GESTURE_MANIFEST)
    )
    splits = [("syn_ground_train_validation", gestures, validation_rows)]
    for manifest_name in REAL_MANIFESTS:
        splits.append(
            (
                Path(manifest_name).stem,
                Joints(manifest_name),
                pd.read_csv(PROJECT_ROOT / "data/manifests" / manifest_name),
            )
        )

    model.eval()
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for split, joints, rows in splits:
        source = joints.rows(rows["path"])
        usable = joints.usable(source, kind)
        logits = torch.zeros(len(rows), IDLE_LABEL + 1)
        with torch.no_grad():
            logits[torch.as_tensor(usable)] = model(
                torch.as_tensor(inputs(joints, source[usable], kind))
            )
        table = probability_table(logits, rows["video_id"].tolist(), rows, class_names)
        output = PREDICTIONS_DIR / f"{name}__{split}.csv"
        table.to_csv(output, index=False)
        accuracy = (logits.argmax(dim=1).numpy() == rows["label"].to_numpy()).mean()
        print(
            f"{split}: {len(rows)} frames, {(~usable).sum()} without a person, "
            f"frame accuracy {accuracy:.1%}  ->  {shown(output)}",
            flush=True,
        )


def shown(path: Path) -> Path:
    """A path as it reads best in the log: relative to the project if inside it."""
    return path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line: which input, and which repetition.

    Args:
        argv: Arguments to parse, or ``None`` for the process's own.

    Returns:
        A namespace with ``input`` and ``seed``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", choices=tuple(INPUTS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    started = time.perf_counter()
    trained = train(args.input, args.seed)
    name = Path(checkpoint_name(f"{INPUTS[args.input]}.pt", args.seed)).stem
    save_atomically(trained.state_dict(), CHECKPOINT_DIR / f"{name}.pt")
    score(trained, args.input, name)
    print(f"time taken: {(time.perf_counter() - started) / 60:.1f} min")
