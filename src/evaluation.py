"""Run a trained model over a set of frames and report what it predicted.

Inference and analysis are deliberately kept apart. A pass over the model is the
expensive half and happens once per model and set of rows; the questions asked
of its output are many — loss, accuracy per frame, aggregation over a growing
number of frames, confusion between classes — and every one of them is
arithmetic over the same numbers. Returning the raw scores instead of a single
figure is what lets the expensive half run once and the cheap half run often.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

# The two runtime imports not left to the main block below. CROP_SIZE because
# parse_args has to name its default and parse_args is read at import time; the
# shift helpers because ``predict`` itself calls them, and training imports
# ``predict`` to score its validation split every epoch.
from dataset import CROP_SIZE
from model import reset_shuffle, set_segments

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BATCH_SIZE = 64

# Default for the one option parse_args gives a default to. Worker count is
# safe to vary here, unlike in training: eval_transform is deterministic, so no
# random draw depends on how the frames were split across processes.
NUM_WORKERS = 12


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line: which checkpoint to score against which frames.

    The manifest is required rather than defaulted. The obvious default would be
    the training manifest, and scoring that without ``--validation-split`` would
    quietly measure the model on the frames it was fitted to — the one mistake
    this script must not make easy. Naming the rows is therefore always a
    deliberate act.

    Args:
        argv: Arguments to parse. ``None`` reads ``sys.argv``.

    Returns:
        A namespace with ``checkpoint`` (path to the trained weights),
        ``manifest`` (a file name under data/manifests/), ``validation_split``
        (score only the manifest's held-out scenes, not all of it), ``adapt_bn``
        (a manifest to re-estimate the normalisation statistics on first, or
        ``None`` to score the checkpoint as it was trained), ``output`` (the
        table's name under data/predictions/, or ``None`` to derive it from the
        checkpoint and manifest), ``crop_size`` (the side to crop each stored
        frame to) and ``num_workers``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="trained weights to score, e.g. checkpoints/syn_ground_train.pt",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="frames to score, a file name under data/manifests/",
    )
    parser.add_argument(
        "--validation-split",
        action="store_true",
        help="score only the held-out validation scenes of the manifest",
    )
    parser.add_argument(
        "--difference-gain",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="multiply each difference channel by this, matching what the "
        "checkpoint was trained with. A plain state dict has nowhere to record "
        "it, so it is asked for — and it goes into the table's name, because a "
        "checkpoint scored at the wrong gain would otherwise write over the "
        "table scored at the right one",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        type=int,
        metavar="VIEW",
        help="keep only these synthetic camera positions before scoring, and "
        "before the validation split so the held-out scenes are the ones the "
        "matching training run held out. A checkpoint trained on a subset of "
        "the viewpoints has to be scored on the same subset, or its origin "
        "figure is read off angles it never saw",
    )
    neighbours = parser.add_mutually_exclusive_group()
    neighbours.add_argument(
        "--neighbour-stride",
        nargs="+",
        type=int,
        default=None,
        metavar="STRIDE",
        help="how many frames back the difference channels were taken from, "
        "for a checkpoint trained to read them. Asked for rather than read off "
        "the file because a plain state dict has nowhere to record it; scoring "
        "such a checkpoint without it is refused rather than guessed.",
    )
    neighbours.add_argument(
        "--neighbour-anchor",
        action="store_true",
        help="the same, for a checkpoint trained to difference each frame "
        "against the one scored before it rather than one a fixed distance "
        "back. Must match how the checkpoint was trained.",
    )
    parser.add_argument(
        "--features",
        action="store_true",
        help="write the penultimate features instead of class probabilities, "
        "for a head that reads a sequence of frames rather than one frame.",
    )
    parser.add_argument(
        "--adapt-bn",
        metavar="MANIFEST",
        help="re-estimate the batch normalisation statistics on these frames "
        "before scoring, a file name under data/manifests/. Reads no label, but "
        "reads the target domain: a result produced this way is not source-only "
        "and its baseline is the published adaptation row, not the source-only "
        "one. Naming the target's training split keeps the scored subjects "
        "unseen; naming the scored manifest itself is the transductive variant",
    )
    parser.add_argument(
        "--output",
        help="table name under data/predictions/; derived from the checkpoint "
        "and manifest when omitted",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=CROP_SIZE,
        metavar="PIXELS",
        help="side the crop takes from each stored frame. Has to be the one the "
        "checkpoint was trained with, or the model meets a field of view it "
        "never learnt on; a crop that is not a margin on these frames is "
        "refused rather than scored",
    )
    parser.add_argument(
        "--clips",
        type=int,
        metavar="FRAMES",
        help="score a model that carries a shift at this clip length instead of "
        "the one it trained at, which is what the accuracy-against-K curve "
        "needs: at K frames aggregated, a decision that read K frames is the "
        "honest comparison against an order-blind model's K. Omit to use the "
        "recorded length, which is the default and the right one for a single "
        "figure. Has no effect on a checkpoint without a shift",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="DataLoader subprocesses",
    )
    return parser.parse_args(argv)


def feature_cube(
    features: torch.Tensor, manifest: pd.DataFrame, video_ids: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Stack a pass of penultimate features into one array per video.

    The stacking reuses the probability table's own grouping rather than
    repeating it, by cubing a column of row positions and then gathering. That
    keeps the checks that matter — equal frame counts, one label per video — in
    one place, and avoids building a five-hundred-column frame to throw away.

    Args:
        features: ``(frames, width)`` as returned by ``predict`` on a model
            whose head was stripped.
        manifest: The rows that were scored, in the order they were served.
        video_ids: The video each row came from, used to check the alignment.

    Returns:
        The ``(videos, frames, width)`` features, each video's label, and the
        video ids, all in the same order.

    Raises:
        RuntimeError: If the features are not in the manifest's order.
    """
    from aggregation import cube_by_video

    rows = manifest.reset_index(drop=True)
    if video_ids != rows["video_id"].tolist():
        raise RuntimeError(
            "features are out of manifest order; the cube would be wrong"
        )

    order = rows[["video_id", "label"]].copy()
    order["row"] = np.arange(len(order))
    positions, labels, ids = cube_by_video(order, ["row"])

    return features.numpy()[positions[:, :, 0]], labels, ids


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    clips: int | None = None,
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
        clips: Frames per clip the loader serves, when it serves clips at all.
            A model carrying a shift needs it: its prediction for a frame
            depends on which frames arrived beside it, so the number is part of
            what is being measured rather than a detail of batching. A single
            value, unlike training, because an evaluation split holds one kind
            of row.

    Returns:
        The ``(frames, classes)`` logits, the matching labels, and the video
        each frame came from, all in the order the loader served them.
    """
    model.eval()
    # Before the pass, not during it, so that two passes over the same frames
    # return the same numbers. The control draws a permutation per batch, and
    # left running it would depend on how many batches training happened to
    # consume first, putting that in the epoch-to-epoch curve as noise.
    reset_shuffle(model)
    if clips:
        set_segments(model, clips)

    logits = []
    labels = []
    video_ids = []

    with torch.no_grad():
        for frames, batch_labels, batch_video_ids in loader:
            logits.append(model(frames.to(device)).cpu())
            # Copied off the shared memory the workers hand it over on. A
            # loader with workers backs every batch it serves by a file
            # descriptor, and keeping the tensor keeps the descriptor: a split
            # of a quarter of a million frames is four thousand batches, which
            # is past the usual open-file limit. The scores do not need this
            # because moving them off the device allocates afresh.
            labels.append(batch_labels.clone())
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
    from dataset import FrameDataset, clip_batches, eval_transform, restore_order
    from device import describe, pick_device
    from manifest import dense_name_for, load_class_names, with_idle_class
    from model import (
        DEFAULT_BACKBONE,
        adapt_batchnorm,
        backbone_of,
        build_model,
        head_width,
        insert_temporal_shift,
        shift_segments,
        shift_shuffled,
        stem_width,
    )
    from splits import (
        keep_views,
        split_by_scene,
        with_neighbours,
        with_previous_anchor,
    )

    args = parse_args()
    print(
        f"checkpoint {args.checkpoint}  manifest {args.manifest}"
        f"{'  (validation split)' if args.validation_split else ''}"
    )

    manifest = pd.read_csv(PROJECT_ROOT / "data/manifests" / args.manifest)
    split_name = Path(args.manifest).stem
    # Before the split, as training does it, so that the held-out scenes are the
    # ones the matching run held out. The viewpoints go into the table's name for
    # the same reason --adapt-bn does: the same checkpoint scored on the same
    # manifest with and without them is two different measurements, and a name
    # that hid the difference would let one overwrite the other in silence.
    if args.views:
        manifest = keep_views(manifest, args.views)
        split_name = f"{split_name}_views{''.join(str(v) for v in args.views)}"
    if args.validation_split:
        _, manifest = split_by_scene(manifest)
        split_name = f"{split_name}_validation"
    # The gain leaves no trace in the weights, so it leaves one here. A table
    # named without it and a table named with it are two measurements, and the
    # one that would be silently wrong is the one scored at the default.
    if args.difference_gain != 1.0:
        split_name = f"{split_name}_gain{args.difference_gain:g}"

    class_names = load_class_names(PROJECT_ROOT / "data" / "class_dict.json")
    device = pick_device()
    print(f"device: {describe(device)}")

    # How wide the head is belongs to the checkpoint, not to the run scoring it.
    # One trained with the idle class carries an output the dataset's own
    # vocabulary has no name for, and building the network at the wrong width
    # fails on a shape mismatch rather than on a wrong number.
    weights = torch.load(args.checkpoint, map_location=device)
    num_classes = head_width(weights)
    if num_classes > len(class_names):
        class_names = with_idle_class(class_names)
    # Which backbone wrote the file is read off the file for the same reason the
    # head's width is: a caller asked to remember would eventually pair the
    # wrong two, and the failure is a shape mismatch at best.
    backbone = backbone_of(weights)
    in_channels = stem_width(weights)
    if in_channels > 3:
        if args.neighbour_stride is None and not args.neighbour_anchor:
            raise SystemExit(
                f"{args.checkpoint.name} reads {in_channels} channels; "
                "--neighbour-stride or --neighbour-anchor says which frame the "
                "extra ones came from"
            )
        if args.neighbour_anchor:
            manifest = with_previous_anchor(manifest)
            print("differencing against the frame scored before it")
        else:
            dense_name = dense_name_for(args.manifest)
            dense = pd.read_csv(PROJECT_ROOT / "data/manifests" / dense_name)
            # The count has to match what the stem was widened to, or the
            # batch and the network disagree on how many channels a frame has.
            # Checked here rather than left to the shape error, which arrives
            # after the manifest, the dense pass and the model are all loaded.
            expected = 3 * (1 + len(args.neighbour_stride))
            if expected != in_channels:
                raise SystemExit(
                    f"{args.checkpoint.name} reads {in_channels} channels, so "
                    f"it was trained on {in_channels // 3 - 1} spacing(s); "
                    f"{len(args.neighbour_stride)} were given"
                )
            manifest = with_neighbours(manifest, args.neighbour_stride, dense)
            spacings = " and ".join(str(s) for s in args.neighbour_stride)
            print(f"differencing against the frames {spacings} back")

    model = build_model(num_classes, backbone, in_channels).to(device)
    # Read off the file for the same reason the head's width and the backbone
    # are: a model whose features were learned with neighbours visible predicts
    # differently without them, so scoring one as though it had no shift would
    # report a number for a network that never existed. Nothing here is a flag
    # a caller could forget.
    # The override only applies to a checkpoint that has a shift. Without one a
    # frame's prediction does not depend on what arrived beside it, so clip
    # length changes no number, and inserting a shift a run never trained with
    # would score a network that never existed.
    recorded = shift_segments(weights)
    segments = None if recorded is None else (args.clips or recorded)
    if recorded is None and args.clips:
        print(f"--clips ignored: {args.checkpoint.name} carries no shift")
    if segments is not None:
        insert_temporal_shift(
            model,
            backbone,
            segments=segments,
            shuffle=shift_shuffled(weights),
        )
    model.load_state_dict(weights)
    if backbone != DEFAULT_BACKBONE:
        print(f"backbone: {backbone}")
    if segments is not None:
        arm = "shuffled" if shift_shuffled(weights) else "ordered"
        print(
            f"temporal shift: {arm}, clips of {segments}"
            + (f"  (trained at {recorded})" if recorded != segments else "")
        )
        # Only an overridden length goes into the name, and the asymmetry is the
        # point. The recorded one needs no mark because the weights carry it, so
        # the default table keeps the name every other cell's does and the
        # canonical reader finds it. A length the caller chose has nothing in the
        # file to disambiguate it, and the curve scores one checkpoint at several
        # of them, so those must not write over each other.
        if segments != recorded:
            split_name = f"{split_name}_clips{segments}"

    def frame_loader(rows: pd.DataFrame) -> DataLoader:
        """Serve a manifest's frames exactly as scoring will meet them.

        Used for the adaptation pass as well as for scoring, and that is the
        point: statistics estimated from augmented frames would describe images
        the model never sees at evaluation.
        """
        dataset = FrameDataset(
            rows,
            PROJECT_ROOT,
            eval_transform(args.crop_size),
            difference_gain=args.difference_gain,
        )
        if segments is not None:
            return DataLoader(
                dataset,
                batch_sampler=clip_batches(rows, segments, BATCH_SIZE),
                num_workers=args.num_workers,
            )
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=args.num_workers,
        )

    # Where each row of a window-spanning pass belongs in the manifest. Computed
    # once here rather than from the loader, because both passes below need it
    # and because a second loader built only to read its sampler is a second
    # thing that can disagree with the first.
    places = (
        restore_order(clip_batches(manifest, segments, BATCH_SIZE))
        if segments is not None
        else None
    )

    if args.adapt_bn:
        adaptation_rows = pd.read_csv(PROJECT_ROOT / "data/manifests" / args.adapt_bn)
        seen = adapt_batchnorm(model, frame_loader(adaptation_rows), device)
        transductive = args.adapt_bn == args.manifest and not args.validation_split
        print(
            f"batch norm re-estimated on {seen} frames from {args.adapt_bn}"
            + ("  (transductive: the scored rows themselves)" if transductive else "")
        )

    if args.features:
        from model import strip_head

        width = strip_head(model, backbone)
        features, _, video_ids = predict(
            model, frame_loader(manifest), device, segments
        )
        if places is not None:
            features = features[places]
            video_ids = [video_ids[place] for place in places]
        cube, video_labels, ids = feature_cube(features, manifest, video_ids)

        features_dir = PROJECT_ROOT / "data" / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        output = features_dir / f"{args.checkpoint.stem}__{split_name}.npz"
        np.savez_compressed(
            output,
            features=cube.astype(np.float32),
            labels=video_labels,
            video_ids=np.array(ids),
        )
        print(f"{cube.shape[0]} videos x {cube.shape[1]} frames x {width} features")
        print(f"written to {output.relative_to(PROJECT_ROOT)}")
        raise SystemExit(0)

    logits, labels, video_ids = predict(model, frame_loader(manifest), device, segments)
    if places is not None:
        # Window-spanning clips do not arrive in the manifest's order. Moving the
        # three arrays back by the one permutation is what lets the table be
        # built at all: it refuses scores out of that order, and that refusal is
        # the only thing standing between a permuted pass and every probability
        # attached to the wrong frame.
        logits = logits[places]
        labels = labels[places]
        video_ids = [video_ids[place] for place in places]
    loss, accuracy = frame_metrics(logits, labels, nn.CrossEntropyLoss())
    table = probability_table(logits, video_ids, manifest, class_names)

    predictions_dir = PROJECT_ROOT / "data" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    # An adapted model is a different model, so its table must not land on the
    # name the unadapted one already wrote. Where the statistics came from goes
    # into the name too: the inductive and transductive variants differ in
    # nothing else, and a name that hid it would let one silently overwrite the
    # other.
    adapted = f"_adabn_{Path(args.adapt_bn).stem}" if args.adapt_bn else ""
    output = predictions_dir / (
        args.output or f"{args.checkpoint.stem}{adapted}__{split_name}.csv"
    )
    table.to_csv(output, index=False)

    print(f"scored {len(table)} frames from {table['video_id'].nunique()} videos")
    print(f"frame loss {loss:.4f}  frame accuracy {accuracy:.1%}")
    print(f"written to {output.relative_to(PROJECT_ROOT)}")
