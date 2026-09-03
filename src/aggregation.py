"""Turn per-frame probabilities into a per-video decision, and measure the cost.

This is the module the research question is written in. A frame classifier does
not answer "which gesture is this" — it answers that for one frame at a time,
and something has to combine those answers. How many frames that something needs
before the gesture is recoverable *is* the measurement: a gesture whose accuracy
is already flat at one frame is separable by pose alone, one that climbs with
more frames needs the accumulation, and one that never climbs needs information
no still frame carries.

So the curve of accuracy against the number of aggregated frames is not a tuning
exercise. It is the instrument, and it is read per class, because the whole point
is that different gestures sit at different places on it.

Two ways of choosing which frames are offered, and they answer different halves
of the question. Spreading the frames evenly over the gesture asks how much is
recoverable when the coverage is as good as it gets. Drawing them at random and
repeating asks how much the *choice* matters — and the spread that comes back is
itself a result, because a gesture whose accuracy swings wildly with which frames
were seen is a gesture whose discriminative information is unevenly distributed
in time.

Everything here is arithmetic over a probability table that already exists on
disk. No model, no GPU, no I/O inside the functions: the expensive half ran once
in ``evaluation.py``, and this half is meant to run as often as a question comes
up.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Frame counts to aggregate over. The last one is filled in from the table so
# that "all of them" is always on the curve, whatever the extraction wrote.
FRAME_COUNTS = (1, 2, 4, 8, 16)

# Draws per frame count when frames are picked at random. The estimate of the
# spread, not of the mean, is what sets this: a mean settles within a few dozen
# draws, a standard deviation wants more, and the whole computation is arithmetic
# over an array that fits in cache. Raising it costs milliseconds.
REPEATS = 200

# Fixed so that a curve can be recomputed and compared. This seed only decides
# which frames are drawn during analysis; it has nothing to do with the training
# seed, which is a property of the model being read here.
SEED = 11


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line: which probability tables to read.

    Several tables are accepted at once because the interesting comparisons are
    between them — the same configuration at three seeds, or the same model on
    the synthetic and the real domain. Each keeps its own rows in the output,
    identified by file name, and averaging across them is left to the reader.

    Args:
        argv: Arguments to parse. ``None`` reads ``sys.argv``.

    Returns:
        A namespace with ``tables`` (paths to probability tables), ``output``
        (where to write the curve, or ``None`` to only print), ``repeats`` and
        ``seed``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "tables",
        type=Path,
        nargs="+",
        help="probability tables written by evaluation.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="where to write the curve as CSV; printed only when omitted",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=REPEATS,
        help="random frame draws per frame count",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="seed for the random frame draws, so a curve can be reproduced",
    )
    return parser.parse_args(argv)


def probability_cube(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Reshape a probability table into one array per video.

    Every later function indexes frames within a video, which a long table of one
    row per frame cannot do without grouping again each time. Paying for the
    reshape once turns the rest of the module into array arithmetic.

    The frames keep the order the table stores them in, which ``evaluation.py``
    inherits from the manifest, which the extraction wrote in ascending position
    through the gesture. That order is what makes "spread the picks evenly over
    the gesture" mean what it says.

    Args:
        table: One row per frame, as written by ``evaluation.py``: ``video_id``,
            ``label`` and one ``p_<class>`` column per gesture.

    Returns:
        The ``(videos, frames, classes)`` probabilities, the true label of each
        video, and the video ids, all in the same order.

    Raises:
        ValueError: If the videos do not all carry the same number of frames, or
            if a video's rows disagree about its label. Either would mean the
            frames cannot be stacked into one array, and reshaping regardless
            would silently misalign them.
    """
    class_columns = [column for column in table.columns if column.startswith("p_")]
    if not class_columns:
        raise ValueError("no p_<class> columns; is this a probability table?")

    grouped = table.groupby("video_id", sort=False)

    frame_counts = grouped.size().unique()
    if len(frame_counts) != 1:
        raise ValueError(
            f"videos carry different frame counts {sorted(frame_counts)}; "
            "they cannot be stacked into one array"
        )

    labels_per_video = grouped["label"].nunique()
    if (labels_per_video != 1).any():
        disagreeing = labels_per_video[labels_per_video != 1].index.tolist()
        raise ValueError(f"videos with more than one label: {disagreeing}")

    video_ids = list(grouped.groups)
    cube = (
        table[class_columns]
        .to_numpy()
        .reshape(len(video_ids), int(frame_counts[0]), len(class_columns))
    )

    return cube, grouped["label"].first().to_numpy(), video_ids


def segment_indices(frame_count: int, picks: int) -> np.ndarray:
    """Choose ``picks`` frames spread evenly across a gesture.

    The gesture is cut into as many equal segments as there are picks, and the
    middle frame of each is taken. This is the test-time half of the sampling
    the extraction already uses: dividing first and then picking inside each
    division keeps the picks from clustering, and taking the centre rather than
    a random offset makes the result reproducible.

    Picking every ``frame_count / picks``-th frame instead would drift towards
    one end and, worse, would resonate with a repeated gesture — landing on the
    same phase of every repetition, which is the failure the segment scheme
    exists to avoid.

    Args:
        frame_count: Frames available per video.
        picks: How many to choose. Must not exceed ``frame_count``.

    Returns:
        The chosen indices, ascending and distinct.

    Raises:
        ValueError: If ``picks`` is not between 1 and ``frame_count``.
    """
    if not 1 <= picks <= frame_count:
        raise ValueError(f"cannot pick {picks} of {frame_count} frames")

    return ((np.arange(picks) + 0.5) * frame_count / picks).astype(int)


def random_indices(
    frame_count: int, picks: int, generator: np.random.Generator
) -> np.ndarray:
    """Choose ``picks`` frames at random, without replacement.

    The counterpart to ``segment_indices``. Repeated draws measure how much the
    choice of frames matters, which is the direct reading of whether a gesture's
    discriminative information sits evenly along it or in a few moments.

    Args:
        frame_count: Frames available per video.
        picks: How many to choose.
        generator: Source of randomness, passed in rather than created here so
            that a whole curve reproduces from one seed.

    Returns:
        The chosen indices, ascending and distinct.

    Raises:
        ValueError: If ``picks`` is not between 1 and ``frame_count``.
    """
    if not 1 <= picks <= frame_count:
        raise ValueError(f"cannot pick {picks} of {frame_count} frames")

    return np.sort(generator.choice(frame_count, size=picks, replace=False))


def aggregate_mean(probabilities: np.ndarray) -> np.ndarray:
    """Decide each video by averaging its frames' probabilities.

    The default strategy, and one a deployed system could actually run: it reads
    only what the model outputs, never the true label. That distinction matters,
    because selecting frames by whether they were classified correctly would use
    the answer to choose the input and inflate the score for free.

    Averaging probabilities rather than logits is deliberate and is why the table
    stores probabilities. A mean of logits weights a frame by how extreme its
    scores happen to be, which lets one very confident frame overrule the rest.

    Args:
        probabilities: ``(videos, frames, classes)`` scores.

    Returns:
        The predicted class of each video.
    """
    return probabilities.mean(axis=1).argmax(axis=1)


def aggregate_vote(probabilities: np.ndarray) -> np.ndarray:
    """Decide each video by majority vote over its frames' own predictions.

    The other strategy that stays label-free, and so equally deployable. It
    differs from the mean in what it throws away: each frame contributes one vote
    regardless of how sure it was, so a single confident frame cannot carry a
    video, and a class that is weakly favoured by many frames wins. Comparing the
    two says whether the model's errors are a few loud frames or a broad drift.

    Ties are broken by the summed probability of the tied classes, not by class
    index. The obvious implementation — ``bincount().argmax()`` — hands every tie
    to the lowest-numbered class, which would be a standing thumb on the scale
    for ``Advance``. In a project measuring which classes a model collapses into,
    an artefact shaped exactly like the thing being measured is not acceptable.

    Args:
        probabilities: ``(videos, frames, classes)`` scores.

    Returns:
        The predicted class of each video.
    """
    class_count = probabilities.shape[2]
    frame_votes = probabilities.argmax(axis=2)

    counts = np.stack(
        [(frame_votes == label).sum(axis=1) for label in range(class_count)], axis=1
    )
    # Probability mass lands far below one vote, so it only ever separates
    # classes the votes left level.
    tiebreak = probabilities.sum(axis=1) / (probabilities.shape[1] * class_count + 1)

    return (counts + tiebreak).argmax(axis=1)


AGGREGATIONS = {"mean": aggregate_mean, "vote": aggregate_vote}


def accuracy_by_class(
    predictions: np.ndarray, labels: np.ndarray, class_count: int
) -> np.ndarray:
    """Score predictions once overall and once per class.

    Per class means recall — of the videos that truly are this gesture, how many
    came back right. That is the quantity the research question is posed in: a
    curve per gesture, not one curve for the set. The overall figure is kept
    alongside because a mean of per-class recalls is not the overall accuracy
    when the classes are unbalanced, and quoting one for the other is an easy
    mistake to make later.

    Args:
        predictions: Predicted class per video.
        labels: True class per video.
        class_count: How many gestures exist, so that a class absent from these
            videos still gets a column rather than being silently dropped.

    Returns:
        ``class_count + 1`` accuracies: the overall one first, then one per
        class in label order. A class with no videos here comes back as NaN,
        which propagates rather than reading as a zero score.
    """
    scores = np.empty(class_count + 1)
    scores[0] = (predictions == labels).mean()

    for label in range(class_count):
        of_this_class = labels == label
        scores[label + 1] = (
            (predictions[of_this_class] == label).mean()
            if of_this_class.any()
            else np.nan
        )

    return scores


def accuracy_curve(
    cube: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    repeats: int = REPEATS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Measure accuracy against the number of frames aggregated.

    The whole instrument. For each frame count, each aggregation strategy and
    each of the two ways of choosing frames, this reports how often the video is
    read correctly — overall and per gesture.

    The two selections are not alternatives to pick between. ``segments`` spreads
    the picks over the gesture and gives one number, the best a system gets from
    that many frames. ``random`` draws repeatedly and reports the mean and the
    spread across draws; that spread is a measurement in its own right, of how
    unevenly the discriminative information is distributed along the gesture.

    At the largest frame count the two coincide by construction — there is only
    one way to take all of them — so ``random`` reports a spread of zero there,
    which is a property of the question and not a bug.

    Args:
        cube: ``(videos, frames, classes)`` probabilities.
        labels: True class per video.
        class_names: Gesture names in label order, used to name the rows.
        repeats: Random draws per frame count.
        seed: Seed for those draws.

    Returns:
        One row per frame count, aggregation, selection and scope, with columns
        ``frames``, ``aggregation``, ``selection``, ``scope``, ``accuracy`` and
        ``spread``. ``spread`` is the standard deviation across random draws,
        and is NaN for the deterministic selection, which has nothing to vary.
    """
    frame_count = cube.shape[1]
    counts = [count for count in FRAME_COUNTS if count < frame_count] + [frame_count]
    generator = np.random.default_rng(seed)
    scopes = ["overall", *class_names]

    rows = []
    for picks in counts:
        for name, aggregate in AGGREGATIONS.items():
            spread_out = accuracy_by_class(
                aggregate(cube[:, segment_indices(frame_count, picks), :]),
                labels,
                len(class_names),
            )
            drawn = np.stack(
                [
                    accuracy_by_class(
                        aggregate(
                            cube[:, random_indices(frame_count, picks, generator), :]
                        ),
                        labels,
                        len(class_names),
                    )
                    for _ in range(repeats)
                ]
            )

            for index, scope in enumerate(scopes):
                rows.append(
                    {
                        "frames": picks,
                        "aggregation": name,
                        "selection": "segments",
                        "scope": scope,
                        "accuracy": spread_out[index],
                        "spread": np.nan,
                    }
                )
                rows.append(
                    {
                        "frames": picks,
                        "aggregation": name,
                        "selection": "random",
                        "scope": scope,
                        "accuracy": np.nanmean(drawn[:, index]),
                        "spread": np.nanstd(drawn[:, index], ddof=1),
                    }
                )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from manifest import load_class_names

    args = parse_args()
    class_names = load_class_names(PROJECT_ROOT / "data" / "class_dict.json")
    ordered_names = [name for _, name in sorted(class_names.items())]

    curves = []
    for path in args.tables:
        cube, labels, video_ids = probability_cube(pd.read_csv(path))
        curve = accuracy_curve(
            cube, labels, ordered_names, repeats=args.repeats, seed=args.seed
        )
        curve.insert(0, "table", path.stem)
        curves.append(curve)

        print(f"\n{path.stem}  —  {len(video_ids)} videos, {cube.shape[1]} frames each")
        overall = curve[(curve.scope == "overall") & (curve.selection == "random")]
        for name in AGGREGATIONS:
            row = overall[overall.aggregation == name]
            drawn = "  ".join(
                f"K={frames:<3}{accuracy:6.1%} ±{spread:.1%}"
                for frames, accuracy, spread in zip(
                    row.frames, row.accuracy, row.spread, strict=True
                )
            )
            print(f"  {name:5s} {drawn}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(curves, ignore_index=True).to_csv(args.output, index=False)
        print(f"\nwritten to {args.output}")
