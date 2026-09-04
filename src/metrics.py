"""Read a probability table as a description of how a model fails.

Overall accuracy says how often a model is right and nothing about the shape of
being wrong, and the shape is where this problem lives: a model that spreads its
errors evenly and one that funnels every uncertain frame into a single class
score the same and are not the same model. The second has a prior, not a
percept, and only a breakdown by class shows it.

Three views, each answering something the others cannot. A confusion matrix says
which gesture is taken for which. A class report says whether a class is being
missed or over-claimed, which the matrix's rows alone cannot separate. A
prediction trace keeps the frames in order and shows a video as the sequence of
guesses it actually produced — the only view of the three that survives
aggregation, and the only one that can show a gesture whose trajectory passes
through the resting pose of another.

Everything is arithmetic over a table that already exists on disk, so asking a
new question of a scored model costs nothing.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aggregation import aggregate_mean, probability_cube

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Multiples of its own standard error a difference must clear to be marked when
# a matrix of differences is printed. Two is a screening threshold, not a test:
# at three repetitions the error has two degrees of freedom, so it says where to
# look rather than what is established.
MARKER_THRESHOLD = 2.0

# Stands in for a figure that has no value rather than a value of zero. Precision
# is undefined for a class a run never guessed, and a collapse is exactly when
# that happens -- printing it as 0.0 would read as a measured failure, and as nan
# it would read as a bug.
UNDEFINED = "--"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line: which probability tables to describe.

    Several tables are accepted so that repetitions of one configuration can be
    read together. A single run's per-class numbers move by more than the
    differences being looked for, so a matrix from one table invites conclusions
    it cannot support; averaging over seeds and reporting the spread is what
    makes the view honest.

    Args:
        argv: Arguments to parse. ``None`` reads ``sys.argv``.

    A second group can be named with ``--against``, which turns the run into a
    comparison. Two configurations differ by less than a single run of either
    moves, so the difference has to be reported with its own uncertainty rather
    than by putting two tables side by side and reading the columns.

    Returns:
        A namespace with ``tables``, ``against``, ``level``, ``threshold`` and
        ``traces``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "tables",
        type=Path,
        nargs="+",
        help="probability tables written by evaluation.py; averaged when several",
    )
    parser.add_argument(
        "--level",
        choices=("frame", "video"),
        default="frame",
        help="score each frame on its own, or each video after averaging its "
        "frames; frame has 24 times the data, video is the reported figure",
    )
    parser.add_argument(
        "--against",
        type=Path,
        nargs="+",
        default=None,
        metavar="TABLE",
        help="a second group to compare the first against; the difference is "
        "taken per repetition when both groups hold the same number",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=MARKER_THRESHOLD,
        help="standard errors a difference must clear to be marked with a star",
    )
    parser.add_argument(
        "--traces",
        type=int,
        default=0,
        metavar="N",
        help="also print the frame-by-frame guesses for N videos per class",
    )
    return parser.parse_args(argv)


def predictions(
    table: pd.DataFrame, level: str = "frame"
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a probability table to what was true and what was guessed.

    The single place the two levels differ, so that every view below reads the
    same numbers and a change of level cannot mean one thing in one view and
    something else in another.

    Args:
        table: One row per frame, as written during evaluation.
        level: ``"frame"`` scores each frame alone; ``"video"`` averages a
            video's frames first, which is the strategy the reported figures use.

    Returns:
        The true and predicted class of each item, as label indices.

    Raises:
        ValueError: If the level is not one of the two.
    """
    if level not in ("frame", "video"):
        raise ValueError(f"level must be 'frame' or 'video', got {level!r}")

    cube, video_labels, _ = probability_cube(table)
    if level == "video":
        return video_labels, aggregate_mean(cube)

    frames = cube.reshape(-1, cube.shape[2])
    labels = np.repeat(video_labels, cube.shape[1])

    return labels, frames.argmax(axis=1)


def confusion_matrix(
    table: pd.DataFrame, class_names: list[str], level: str = "frame"
) -> pd.DataFrame:
    """Count which gesture was taken for which, as a share of each true class.

    Rows are normalised rather than left as counts because the question is
    almost always "where does this class go", and raw counts make classes with
    more items look worse at a glance. The diagonal is then recall.

    Args:
        table: One row per frame.
        class_names: Gesture names in label order, naming rows and columns.
        level: Passed to ``predictions``.

    Returns:
        A square frame indexed by true class, with predicted classes as columns
        and percentages of each row. A true class absent from the table comes
        back as NaN rather than zero, which would read as a scored failure.
    """
    truth, guess = predictions(table, level)
    counts = np.zeros((len(class_names), len(class_names)))
    for actual, predicted in zip(truth, guess, strict=True):
        counts[actual, predicted] += 1

    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        shares = np.where(totals > 0, counts / totals * 100, np.nan)

    return pd.DataFrame(shares, index=class_names, columns=class_names)


def class_report(
    table: pd.DataFrame, class_names: list[str], level: str = "frame"
) -> pd.DataFrame:
    """Report recall, precision and how much of the output each class takes.

    Recall alone cannot tell a class that is understood from a class that is
    being guessed constantly: a sink that swallows a third of the output will
    have high recall on itself for no better reason than volume. Precision and
    the share of predictions are what separate the two, and the share is what
    makes a collapse visible without reading the whole matrix.

    Args:
        table: One row per frame.
        class_names: Gesture names in label order.
        level: Passed to ``predictions``.

    Returns:
        A row per class with ``recall``, ``precision``, ``support`` (how many
        items truly are this class) and ``share`` (percentage of all guesses
        that named it). Recall and precision are NaN where undefined — a class
        with no items and a class never guessed respectively — rather than zero,
        which would read as a measured failure.
    """
    truth, guess = predictions(table, level)
    rows = {}
    for label, name in enumerate(class_names):
        actual = truth == label
        predicted = guess == label
        rows[name] = {
            "recall": (guess[actual] == label).mean() * 100 if actual.any() else np.nan,
            "precision": (truth[predicted] == label).mean() * 100
            if predicted.any()
            else np.nan,
            "support": int(actual.sum()),
            "share": predicted.mean() * 100,
        }

    return pd.DataFrame(rows).T


def prediction_imbalance(
    table: pd.DataFrame, class_names: list[str], level: str = "frame"
) -> float:
    """Measure how far the guesses are from using every class equally.

    A single number for "is this model collapsing", so that runs can be compared
    without reading seven rows each. It says nothing about being right: a model
    can be perfectly balanced and wrong throughout. It is the companion to
    accuracy, not a substitute.

    Args:
        table: One row per frame.
        class_names: Gesture names in label order.
        level: Passed to ``predictions``.

    Returns:
        Mean absolute deviation, in percentage points, between each class's
        share of the guesses and an equal share. Zero means every class was
        named equally often.
    """
    _, guess = predictions(table, level)
    shares = np.array(
        [(guess == label).mean() * 100 for label in range(len(class_names))]
    )

    return float(np.abs(shares - 100 / len(class_names)).mean())


def prediction_trace(table: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    """Lay out each video as the sequence of guesses its frames produced.

    The one view here that does not aggregate. A matrix says a gesture is taken
    for two others and cannot say whether that happens at once or in turn, and
    the difference matters: a gesture held in one pose that is misread is a
    different failure from a gesture whose arm sweeps through the resting pose of
    another on its way up. The second shows here as a run of one class in the
    middle of a video and a different one at the edges, and shows nowhere else.

    Frames keep the order the table stores them in, which is ascending position
    through the gesture.

    Args:
        table: One row per frame.
        class_names: Gesture names in label order.

    Returns:
        A row per video, indexed by ``video_id``: ``truth`` naming its gesture,
        then one column per frame position holding the class guessed there.
    """
    cube, labels, video_ids = probability_cube(table)
    guesses = cube.argmax(axis=2)

    trace = pd.DataFrame(
        guesses,
        index=pd.Index(video_ids, name="video_id"),
        columns=range(cube.shape[1]),
    )
    trace.insert(0, "truth", [class_names[label] for label in labels])

    return trace


def mean_and_spread(
    frames: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average the same view over repetitions, and say how much it moved.

    Per-class figures from a single run move by more than the differences
    usually being looked for, so a view built from one table can support a
    conclusion the data does not. Reporting the spread beside the mean is what
    stops that, and a spread wider than the effect is itself the finding.

    Args:
        frames: The same view computed from each repetition. All must share an
            index and columns.

    A quantity undefined in some repetitions is averaged over the ones where it
    exists. Precision is undefined for a class a run never guessed, and letting
    one such run turn the average into NaN would erase the column exactly when a
    collapse is the thing being looked at. The share of predictions stays beside
    it to show how often that happened.

    Returns:
        The element-wise mean and sample standard deviation. The spread is NaN
        throughout when given a single frame, which is the honest reading of one
        measurement rather than a claim of no variation.

    Raises:
        ValueError: If no frames are given, or they disagree in shape.
    """
    if not frames:
        raise ValueError("no frames to average")
    first = frames[0]
    for frame in frames[1:]:
        if not (
            frame.index.equals(first.index) and frame.columns.equals(first.columns)
        ):
            raise ValueError("frames disagree in index or columns")

    stacked = np.stack([frame.to_numpy(dtype=float) for frame in frames])
    spread = (
        stacked.std(axis=0, ddof=1) if len(frames) > 1 else np.full(first.shape, np.nan)
    )

    return (
        pd.DataFrame(
            np.nanmean(stacked, axis=0), index=first.index, columns=first.columns
        ),
        pd.DataFrame(spread, index=first.index, columns=first.columns),
    )


def compare(
    before: list[pd.DataFrame],
    after: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Difference between two sets of repetitions, with its own noise beside it.

    Two views read side by side answer what neither answers alone: what a change
    did to each class. An overall score can hold still while a model trades one
    class for another, and only the difference shows it. Done by hand, cell by
    cell, that comparison is slow and invites reading a movement that is only a
    seed changing its mind.

    Whatever view is handed in is what comes back differenced, so one function
    serves a confusion matrix, a class report, or anything else of the same
    shape. That is also why the inputs are views rather than probability tables:
    building a second path that scored tables itself would let the two
    comparisons drift apart.

    The difference is taken **per repetition** whenever both sides have the same
    number of them, and averaged afterwards. A seed fixes the head's initial
    weights and the per-epoch frame draw together, so same-seed runs start from
    the same place and see the same frames; differencing them first removes that
    shared variation instead of letting it inflate both sides. When the counts
    disagree there is nothing to pair, and the means are differenced instead —
    correct, but blunter, because the run-to-run spread stays in the answer.

    Args:
        before: The view computed from each repetition of the baseline.
        after: The same view from each repetition of the treatment. Paired with
            ``before`` position by position, so the two lists have to name their
            repetitions in the same order.

    Returns:
        The element-wise difference, and its **standard error** — not the
        standard deviation that ``mean_and_spread`` reports. A difference is
        read against how far it could have moved by chance, and the standard
        error is the quantity that answers that; returning the deviation would
        leave every caller dividing by the same square root.

        The error is NaN throughout when either side has a single repetition,
        which is the honest reading of one measurement.

    Raises:
        ValueError: If either side is empty, or the views disagree in shape.
    """
    if not before or not after:
        raise ValueError("both sides need at least one view")
    first = before[0]
    for frame in [*before[1:], *after]:
        if not (
            frame.index.equals(first.index) and frame.columns.equals(first.columns)
        ):
            raise ValueError("views disagree in index or columns")

    if len(before) == len(after):
        stacked = np.stack(
            [
                treated.to_numpy(dtype=float) - base.to_numpy(dtype=float)
                for base, treated in zip(before, after, strict=True)
            ]
        )
        delta = np.nanmean(stacked, axis=0)
        error = (
            stacked.std(axis=0, ddof=1) / np.sqrt(len(stacked))
            if len(stacked) > 1
            else np.full(first.shape, np.nan)
        )
    else:
        base_mean, base_spread = mean_and_spread(before)
        treated_mean, treated_spread = mean_and_spread(after)
        delta = treated_mean.to_numpy() - base_mean.to_numpy()
        # Unpaired: the two spreads are independent, so their errors add in
        # quadrature. A side of one repetition contributes NaN, which carries
        # through and says the difference cannot be judged.
        error = np.sqrt(
            base_spread.to_numpy() ** 2 / len(before)
            + treated_spread.to_numpy() ** 2 / len(after)
        )

    return (
        pd.DataFrame(delta, index=first.index, columns=first.columns),
        pd.DataFrame(error, index=first.index, columns=first.columns),
    )


def abbreviate(names: list[str], width: int = 3) -> list[str]:
    """Shorten class names to fit a terminal column, keeping them distinct.

    Truncation alone is not enough here: two of the gestures agree on their
    first several letters, and a matrix whose rows cannot be told apart is worse
    than no matrix. Names written in camel case keep their capitals, which is
    what separates them.

    Args:
        names: The names to shorten.
        width: Characters to aim for. Widened as needed to stay distinct.

    Returns:
        One short name per input, in the same order, all different.
    """

    def short(name: str, size: int) -> str:
        capitals = [character for character in name if character.isupper()]
        if len(capitals) >= size:
            return "".join(capitals[:size])
        if len(capitals) >= 2:
            return name[: size - len(capitals) + 1] + "".join(capitals[1:])
        return name[:size]

    size = width
    while size <= max(len(name) for name in names):
        shortened = [short(name, size) for name in names]
        if len(set(shortened)) == len(names):
            return shortened
        size += 1

    return list(names)


def format_matrix(
    mean: pd.DataFrame, spread: pd.DataFrame | None = None, width: int = 6
) -> str:
    """Render a confusion matrix for a terminal, abbreviating the class names.

    Args:
        mean: The matrix to show.
        spread: Standard deviations to append to the diagonal, or ``None``.
        width: Column width in characters.

    Returns:
        The rendered table, rows being the true class.
    """
    short = abbreviate(list(mean.columns), width - 3)
    lines = ["".join(f"{'':>7}") + "".join(f"{name:>{width}}" for name in short)]
    for row in range(len(mean.index)):
        cells = "".join(
            f"{mean.iloc[row, column]:{width}.0f}" for column in range(len(short))
        )
        note = ""
        if spread is not None and not np.isnan(spread.iloc[row, row]):
            note = f"   recall {mean.iloc[row, row]:.0f} ± {spread.iloc[row, row]:.0f}"
        lines.append(f"{short[row]:>7}{cells}{note}")

    return "\n".join(lines)


def format_class_report(
    report: pd.DataFrame, spread: pd.DataFrame | None = None, signed: bool = False
) -> str:
    """Render a class report as aligned columns, with spreads where they exist.

    Args:
        report: Rows per class, as ``class_report`` returns them, possibly
            already averaged over repetitions.
        spread: Uncertainty of each figure, or ``None`` to print bare numbers.
        signed: Whether to force a sign, which is what a difference needs and a
            level does not.

    Returns:
        The rendered block, one line per class, headed by its column names.
    """
    sign = "+" if signed else ""
    # Widths track the cells below: a figure, its unit, and room for a spread.
    lines = [f"{'':16}{'recall':>12}{'precision':>12}{'support':>9}{'share':>12}"]
    for name in report.index:
        row = report.loc[name]
        cells = ""
        for column in ("recall", "precision", "support", "share"):
            if column == "support":
                cells += f"{row[column]:{sign}9.0f}"
                continue
            note = ""
            if spread is not None and not np.isnan(spread.loc[name, column]):
                note = f"±{spread.loc[name, column]:.0f}"
            figure = (
                f"{row[column]:{sign}6.1f}%"
                if not np.isnan(row[column])
                else f"{UNDEFINED:>7}"
            )
            cells += f"{figure}{note:>5}"
        lines.append(f"{name:16}{cells}")

    return "\n".join(lines)


def format_delta(
    delta: pd.DataFrame,
    error: pd.DataFrame,
    threshold: float = MARKER_THRESHOLD,
    width: int = 7,
) -> str:
    """Render a differenced matrix, marking the cells that clear the noise.

    A seven-class difference is forty-nine signed numbers, and most of them are
    one item changing its mind: with fourteen videos in a class, a single video
    moves that row by seven percentage points. Printed unmarked, the few cells
    that mean something sit in a wall of cells that do not, and the eye has no
    way to tell which is which.

    So every cell is shown — hiding them would be deciding for the reader what
    is worth seeing — and the ones whose difference exceeds ``threshold`` times
    its own standard error carry a mark.

    ⚠️ The mark is a screening aid, not a test. With three repetitions the error
    rests on two degrees of freedom, and "twice the standard error" is nowhere
    near a p-value. It says where to look, not what to conclude.

    Args:
        delta: The differenced view, classes on both axes.
        error: Its standard error, same shape.
        threshold: Multiples of the standard error a cell must clear to be
            marked. Raise it to be shown less, lower it while exploring.
        width: Column width in characters, one of which the mark takes.

    Returns:
        The rendered table, rows being the true class, with each row's own
        difference in recall spelled out alongside.
    """
    short = abbreviate(list(delta.columns), width - 4)
    lines = [f"{'':>7}" + "".join(f"{name:>{width}}" for name in short)]
    for row in range(len(delta.index)):
        cells = ""
        for column in range(len(short)):
            moved = delta.iloc[row, column]
            spread = error.iloc[row, column]
            readable = not np.isnan(spread) and spread > 0
            marked = readable and abs(moved) >= threshold * spread
            figure = (
                f"{moved:+{width - 1}.0f}"
                if not np.isnan(moved)
                else f"{UNDEFINED:>{width - 1}}"
            )
            cells += figure + ("*" if marked else " ")
        note = f"   recall {delta.iloc[row, row]:+.0f}"
        if not np.isnan(error.iloc[row, row]):
            note += f" ± {error.iloc[row, row]:.0f}"
        lines.append(f"{short[row]:>7}{cells}{note}")

    return "\n".join(lines)


if __name__ == "__main__":
    from manifest import load_class_names

    args = parse_args()
    names = load_class_names(PROJECT_ROOT / "data" / "class_dict.json")
    class_names = [name for _, name in sorted(names.items())]

    def views(paths):
        """Score a group of tables into the three views a run reports."""
        tables = [pd.read_csv(path) for path in paths]
        return (
            [confusion_matrix(table, class_names, args.level) for table in tables],
            [class_report(table, class_names, args.level) for table in tables],
            [prediction_imbalance(table, class_names, args.level) for table in tables],
        )

    def describe(paths, matrices, reports, imbalances, title):
        """Print one group's matrix, class report and imbalance."""
        print(f"\n{title}: {len(paths)} table(s), scored per {args.level}")
        for path in paths:
            print(f"  {path.stem}")

        matrix, matrix_spread = mean_and_spread(matrices)
        print("\nconfusion, % of each true class (rows are truth)\n")
        print(format_matrix(matrix, matrix_spread))

        report, report_spread = mean_and_spread(reports)
        print("\nper class")
        print(format_class_report(report, report_spread))

        print(
            f"\nimbalance {np.mean(imbalances):.1f} points from an equal share"
            f"{f' ± {np.std(imbalances, ddof=1):.1f}' if len(imbalances) > 1 else ''}"
        )

    matrices, reports, imbalances = views(args.tables)
    describe(
        args.tables,
        matrices,
        reports,
        imbalances,
        "baseline" if args.against else "tables",
    )

    if args.against:
        other_matrices, other_reports, other_imbalances = views(args.against)
        describe(
            args.against, other_matrices, other_reports, other_imbalances, "treatment"
        )

        paired = len(args.tables) == len(args.against)
        print(
            f"\n\ndifference, treatment - baseline, "
            f"{'paired by repetition' if paired else 'unpaired'}"
        )
        print(
            f"a star marks a cell at least {args.threshold:g}x its own standard "
            "error from zero:"
        )
        print("where to look, not what is established")

        matrix_delta, matrix_error = compare(matrices, other_matrices)
        print("\nconfusion, percentage points\n")
        print(format_delta(matrix_delta, matrix_error, args.threshold))

        report_delta, report_error = compare(reports, other_reports)
        # Support counts the items of each true class, which both sides scored,
        # so a non-zero column here means the two groups did not read the same
        # rows and nothing below it can be compared.
        print("\nper class, percentage points (support must be zero)")
        print(format_class_report(report_delta, report_error, signed=True))

        moved = np.mean(other_imbalances) - np.mean(imbalances)
        print(f"\nimbalance {moved:+.1f} points")

    if args.traces:
        trace = prediction_trace(pd.read_csv(args.tables[0]), class_names)
        print(f"\nframe-by-frame guesses, {args.tables[0].stem}")
        print("  " + "  ".join(f"{i}={name}" for i, name in enumerate(class_names)))
        for gesture in class_names:
            for video_id, row in (
                trace[trace.truth == gesture].head(args.traces).iterrows()
            ):
                sequence = "".join(str(int(value)) for value in row[1:])
                print(f"  {gesture[:13]:13} {sequence}  {video_id[:26]}")
