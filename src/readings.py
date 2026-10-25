"""Read a configuration's scores into the figures the thesis reports for it.

The program scores videos and then has to say what the scores mean: how often a
frame was right, how often a video was, how that changes with the number of
frames a decision waits for, and which gestures were taken for which. The thesis
computed all of that from probability tables, and this module is that
arithmetic, applied the same way to a table scored a minute ago and to one
stored when the thesis was written. One function for both is what makes "now"
and "reported" comparable: two implementations of one figure agree only until
one of them changes.

Three readings exist, because three kinds of configuration do.

* Most decide each frame alone, and a video is the mean of its frames'
  probabilities over the seven gestures. A network trained with the idle class
  has an eighth output, which is dropped without renormalising the other seven:
  each frame would be rescaled by a different factor, and the mean over a video
  could then pick a different gesture than the thesis's reading does.
* A network that exchanges channels between frames decides per clip, the mean of
  the frames it read together. Those frames are not independent, so its accuracy
  per frame no longer describes its decisions, and its accuracy at another clip
  length needs another scoring, since what a frame says depends on which frames
  arrived beside it.
* A configuration with the prior correction divides each decision's
  probabilities by how often its network named each gesture on the real
  training videos, estimated once on those videos and then frozen.

Everything here is arithmetic over arrays, except the few functions that say
they read the stored tables or the latency files.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aggregation import probability_cube, random_indices, segment_indices
from catalog import SEEDS, Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = PROJECT_ROOT / "data/predictions"
BENCHMARKS = PROJECT_ROOT / "data/benchmarks"

# The seven gestures. Any output past them is the idle class.
GESTURES = 7
TEST_SPLIT = "real_ground_test"
TRAIN_SPLIT = "real_ground_train"

# Frames a decision aggregates, the axis of the accuracy curve. The stored
# videos hold 24, so the last point is the whole video.
FRAME_COUNTS = (1, 2, 4, 8, 16, 24)
# A clip takes one frame from each equal block of the stored frames, so its
# length has to divide them, and 16 does not divide 24.
CLIP_LENGTHS = (1, 2, 4, 8, 12, 24)

# Random draws per point of the curve, and the seeds of the two random readings.
# Fixed so that a figure computed twice comes out the same; they decide which
# frames are drawn and which labels are shuffled, nothing about any model.
DRAWS = 200
CURVE_SEED = 11
NULL_SEED = 3

# The latency session the program reads, and the three ways each row was timed.
LATENCY_SESSION = "catalog_"
SETTINGS = {"cpu_1t": "1 CPU thread", "cpu_0t": "every CPU core", "cuda_1t": "GPU"}


def gesture_cube(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """A probability table as ``(videos, frames, gestures)``, the idle output dropped.

    Args:
        table: One row per frame, as scoring writes it.

    Returns:
        The probabilities over the seven gestures, each video's label, and the
        video ids, all in the table's order.
    """
    cube, labels, video_ids = probability_cube(table)
    return cube[:, :, :GESTURES], labels, video_ids


def clip_partition(frames: int, length: int) -> np.ndarray:
    """Which stored frames form each clip a shifted network read together.

    One frame from each of ``length`` equal blocks, the construction scoring
    uses when it batches a shifted network's clips, so a clip spans the gesture
    rather than a third of it. Written here as a reshape and a transpose where
    scoring writes a loop: the two have to agree, and a test holds them to it.

    Args:
        frames: Stored frames per video.
        length: Frames per clip.

    Returns:
        Positions as ``(clips, length)``, a row per clip, in temporal order.

    Raises:
        ValueError: If the frames do not divide into whole clips.
    """
    if frames % length:
        raise ValueError(f"{frames} frames do not split into clips of {length}")

    return np.arange(frames).reshape(length, frames // length).T


def video_scores(cube: np.ndarray, marginal: np.ndarray | None = None) -> np.ndarray:
    """What a video's decision is taken from: its frames' mean, corrected if asked.

    Args:
        cube: ``(videos, frames, gestures)`` probabilities.
        marginal: How often the network named each gesture where the correction
            was estimated. Dividing by it lifts the gestures the network
            under-names; ``None`` leaves the mean as it is.

    Returns:
        ``(videos, gestures)`` scores, whose largest entry is the decision.
    """
    scores = cube.mean(axis=1)
    if marginal is not None:
        scores = scores / np.clip(marginal, 1e-12, None)
    return scores


def prior_marginal(cube: np.ndarray) -> np.ndarray:
    """How often a network named each gesture over a set of videos, without labels.

    The mean of the videos' mean probabilities: each video weighs the same,
    whatever its frame count. Read on the real training videos, never on the
    videos being decided, so that a video arriving alone can be decided too.
    """
    return cube.mean(axis=1).mean(axis=0)


def clip_decisions(cube: np.ndarray, length: int) -> np.ndarray:
    """The gesture each clip of ``length`` frames decides, as ``(videos, clips)``."""
    clips = cube[:, clip_partition(cube.shape[1], length)]
    return clips.mean(axis=2).argmax(axis=2)


def frame_accuracy(cube: np.ndarray, labels: np.ndarray) -> float:
    """Share of frames whose most probable gesture is the video's."""
    return float((cube.argmax(axis=2) == labels[:, None]).mean())


def confidence_auroc(cube: np.ndarray, labels: np.ndarray) -> float:
    """Whether the network is surer of the frames it gets right than of the rest.

    The probability that a right frame, drawn at random, carries a higher top
    probability than a wrong one. It is a different question from accuracy, and
    the one that decides whether any scheme that trusts confident frames could
    work. Ties count half, as in the rank form of the statistic.

    Returns:
        The AUROC, or NaN when every frame is right or every one is wrong.
    """
    frames = cube.reshape(-1, cube.shape[2])
    right = frames.argmax(axis=1) == np.repeat(labels, cube.shape[1])
    if right.all() or not right.any():
        return float("nan")
    ranks = pd.Series(frames.max(axis=1)).rank().to_numpy()
    positives, negatives = right.sum(), (~right).sum()
    above = ranks[right].sum() - positives * (positives + 1) / 2
    return float(above / (positives * negatives))


def oracle_and_null(
    cube: np.ndarray,
    labels: np.ndarray,
    draws: int = DRAWS,
    seed: int = NULL_SEED,
) -> tuple[float, float]:
    """The ceiling if the right frame could be picked, and what chance gives it.

    The oracle reads the label to choose the frame, so it is never a result: it
    bounds what any rule choosing frames could reach. It needs its null beside
    it, because with seven gestures and twenty-four frames a network that knows
    nothing still lands on the true gesture somewhere; giving each video another
    video's label measures how often.
    """
    per_frame = cube.argmax(axis=2)
    oracle = (per_frame == labels[:, None]).any(axis=1).mean()
    generator = np.random.default_rng(seed)
    shuffled = [
        (per_frame == generator.permutation(labels)[:, None]).any(axis=1).mean()
        for _ in range(draws)
    ]
    return float(oracle), float(np.mean(shuffled))


def ranked_share(scores: np.ndarray, labels: np.ndarray, depth: int) -> float:
    """Share of videos whose true gesture is among the ``depth`` best scored.

    It separates a network that cannot tell two gestures apart from one that can
    and ranks them badly: a gesture ranked second is visible to it, one outside
    the top three of seven is not.
    """
    order = np.argsort(-scores, axis=1)[:, :depth]
    return float((order == labels[:, None]).any(axis=1).mean())


def accuracy_by_k(
    cube: np.ndarray,
    labels: np.ndarray,
    marginal: np.ndarray | None = None,
    counts: tuple[int, ...] = FRAME_COUNTS,
    draws: int = DRAWS,
    seed: int = CURVE_SEED,
) -> dict[int, tuple[float, float, float]]:
    """Video accuracy against the number of frames a decision waits for.

    Two selections, because they answer two questions. Spread over the gesture,
    one frame from each equal stretch, is the best that many frames can do and
    is the curve the thesis reads. Drawn at random, repeatedly, says how much
    the choice of frames matters, and its spread is a finding about the gesture
    rather than noise to average away.

    Returns:
        For each count the stored videos allow: the spread accuracy, then the
        mean and the sample deviation of the random draws.
    """
    available = cube.shape[1]
    generator = np.random.default_rng(seed)
    curve = {}
    for picks in counts:
        if picks > available:
            continue
        spread = video_scores(cube[:, segment_indices(available, picks)], marginal)
        drawn = [
            (
                video_scores(
                    cube[:, random_indices(available, picks, generator)], marginal
                ).argmax(axis=1)
                == labels
            ).mean()
            for _ in range(draws)
        ]
        curve[picks] = (
            float((spread.argmax(axis=1) == labels).mean()),
            float(np.mean(drawn)),
            float(np.std(drawn, ddof=1)),
        )
    return curve


def by_gesture(
    decisions: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recall and precision per gesture, and the confusion matrix they come from.

    Args:
        decisions: The gesture each decision named.
        labels: The true gesture of each, in the same shape.

    Returns:
        Recall and precision per gesture, NaN for a gesture with nothing to
        divide by, and the counts with true gestures as rows.
    """
    confusion = np.zeros((GESTURES, GESTURES), dtype=int)
    np.add.at(confusion, (labels.ravel(), decisions.ravel()), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.diag(confusion) / confusion.sum(axis=1)
        precision = np.diag(confusion) / confusion.sum(axis=0)
    return recall, precision, confusion


@dataclass(frozen=True)
class Reading:
    """What one run's table says, every share a fraction of one.

    Attributes:
        video_ids: The videos read, in the table's order.
        labels: Their true gestures.
        frame: Share of frames right.
        video: Share of videos right, deciding each by its frames' mean, after
            the correction when the configuration has one.
        top3: Share of videos whose gesture is among the three best scored.
        auroc: Whether right frames are the confident ones.
        oracle: Share of videos with at least one right frame.
        null: The same, with the labels shuffled between videos.
        curve: ``accuracy_by_k`` of the configuration's decision.
        decisions: One per video, or one per clip for a shifted network, by the
            configuration's own rule.
        decided: The true gesture of each decision, in the same shape.
        clip: Share of clips right at the length the network trained on, for a
            shifted network; ``None`` otherwise.
    """

    video_ids: list[str]
    labels: np.ndarray
    frame: float
    video: float
    top3: float
    auroc: float
    oracle: float
    null: float
    curve: dict[int, tuple[float, float, float]]
    decisions: np.ndarray
    decided: np.ndarray
    clip: float | None


def read(
    table: pd.DataFrame,
    marginal: np.ndarray | None = None,
    clip: int | None = None,
) -> Reading:
    """Every figure one run's table gives, read the configuration's way.

    Args:
        table: One row per frame, as scoring writes it.
        marginal: The frozen correction, for a configuration that has one.
        clip: The clip length a shifted network trained on, for one that has a
            shift. Its decisions are then the clips', and its per-frame and
            per-video figures are kept as the reading of the same scores that
            treats the frames as independent, which they are not.

    Returns:
        The figures, with the decisions they rest on.
    """
    cube, labels, video_ids = gesture_cube(table)
    scores = video_scores(cube, marginal)
    oracle, null = oracle_and_null(cube, labels)
    if clip is None:
        decisions, decided, clip_share = scores.argmax(axis=1), labels, None
    else:
        decisions = clip_decisions(cube, clip)
        decided = np.repeat(labels[:, None], decisions.shape[1], axis=1)
        clip_share = float((decisions == decided).mean())

    return Reading(
        video_ids=video_ids,
        labels=labels,
        frame=frame_accuracy(cube, labels),
        video=float((scores.argmax(axis=1) == labels).mean()),
        top3=ranked_share(scores, labels, 3),
        auroc=confidence_auroc(cube, labels),
        oracle=oracle,
        null=null,
        curve=accuracy_by_k(cube, labels, marginal),
        decisions=decisions,
        decided=decided,
        clip=clip_share,
    )


def clip_curve(tables: dict[int, pd.DataFrame]) -> dict[int, float]:
    """A shifted network's decision accuracy against its clip length.

    Each length is a scoring of its own, because a frame's scores depend on the
    frames that shared its clip; reading another length off one table would
    decide on clips the network never saw.

    Args:
        tables: One run's table at each clip length it was scored at.

    Returns:
        Share of clips right at each length.
    """
    curve = {}
    for length, table in sorted(tables.items()):
        cube, labels, _ = gesture_cube(table)
        decisions = clip_decisions(cube, length)
        curve[length] = float((decisions == labels[:, None]).mean())
    return curve


def mean_and_deviation(values: list[float]) -> tuple[float, float]:
    """Mean over runs and the sample deviation, NaN for a deviation of one run.

    Sample rather than population, as every spread in the thesis is: the runs
    are three draws of a training that could have gone otherwise.
    """
    array = np.asarray(values, dtype=float)
    deviation = float(array.std(ddof=1)) if len(array) > 1 else float("nan")
    return float(array.mean()), deviation


@dataclass(frozen=True)
class Summary:
    """A configuration's figures over its runs, in percent, as the thesis quotes them.

    The same summary is made of what was scored now and of what was stored when
    the thesis was written, so the two columns of a comparison are one kind of
    thing.

    Attributes:
        runs: How many runs it averages.
        frame: Mean and sample deviation of the share of frames right.
        video: The same for videos, after the correction when there is one.
        top3: The same for the true gesture among the three best scored.
        auroc: The same for the confidence AUROC, in its own unit.
        oracle: Mean share of videos with a right frame.
        null: Mean of the same with the labels shuffled.
        clip: Mean and deviation of the share of clips right, for a shifted
            network; ``None`` otherwise.
        raw_video: Mean and deviation of the video share before the correction,
            for a configuration that has one; ``None`` otherwise.
        curve: The runs' mean of each point of ``accuracy_by_k``.
        recall: The runs' mean recall per gesture.
        precision: The runs' mean precision per gesture.
        confusion: The runs' decisions counted together, true gestures as rows.
    """

    runs: int
    frame: tuple[float, float]
    video: tuple[float, float]
    top3: tuple[float, float]
    auroc: tuple[float, float]
    oracle: float
    null: float
    clip: tuple[float, float] | None
    raw_video: tuple[float, float] | None
    curve: dict[int, tuple[float, float, float]]
    recall: np.ndarray
    precision: np.ndarray
    confusion: np.ndarray


def _mean_where_defined(arrays: list[np.ndarray]) -> np.ndarray:
    """The mean over runs of each entry, over the runs where it is defined.

    A gesture none of a run's videos showed has no recall in that run, and the
    mean of the others is the right figure for it; one no run showed stays
    undefined rather than becoming a zero.
    """
    stacked = np.array(arrays, dtype=float)
    counted = (~np.isnan(stacked)).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counted > 0, np.nansum(stacked, axis=0) / counted, np.nan)


def summarize(readings: list[Reading], raw: list[Reading] | None = None) -> Summary:
    """Average one configuration's runs into the figures a reader is shown.

    Args:
        readings: One reading per run, the configuration's way.
        raw: For a corrected configuration, the same runs read without the
            correction, so that what it adds can be shown beside it.

    Returns:
        The summary, every share in percent.
    """

    def percent(values: list[float]) -> tuple[float, float]:
        return mean_and_deviation([value * 100 for value in values])

    per_gesture = [by_gesture(each.decisions, each.decided) for each in readings]
    recall = _mean_where_defined([recall for recall, _, _ in per_gesture]) * 100
    precision = _mean_where_defined([prec for _, prec, _ in per_gesture]) * 100
    curve = {
        picks: tuple(
            float(np.mean([each.curve[picks][i] for each in readings]) * 100)
            for i in range(3)
        )
        for picks in readings[0].curve
    }
    shifted = readings[0].clip is not None
    return Summary(
        runs=len(readings),
        frame=percent([each.frame for each in readings]),
        video=percent([each.video for each in readings]),
        top3=percent([each.top3 for each in readings]),
        auroc=mean_and_deviation([each.auroc for each in readings]),
        oracle=float(np.mean([each.oracle for each in readings]) * 100),
        null=float(np.mean([each.null for each in readings]) * 100),
        clip=percent([each.clip for each in readings]) if shifted else None,
        raw_video=percent([each.video for each in raw]) if raw else None,
        curve=curve,
        recall=recall,
        precision=precision,
        confusion=sum(confusion for _, _, confusion in per_gesture),
    )


def stored_table(
    stem: str,
    seed: int,
    split: str = TEST_SPLIT,
    length: int | None = None,
    trained: int | None = None,
    root: Path = PREDICTIONS,
) -> Path:
    """Where scoring stored one run's table, the name built as scoring builds it.

    A shifted network scored at the length it trained on carries no mark; any
    other length carries ``_clips{length}``, so that the scorings of one file
    at several lengths cannot overwrite each other.
    """
    suffix = "" if length is None or length == trained else f"_clips{length}"
    return root / f"{stem}_s{seed}__{split}{suffix}.csv"


def stored_marginal(
    configuration: Configuration, seed: int, root: Path = PREDICTIONS
) -> np.ndarray:
    """The frozen correction of one run: its marginal on the real training videos.

    Read from the table stored when the run was scored on those videos, which is
    the estimate the thesis's figure used; computing it again from fresh scores
    would be a second estimate of the same thing, and not the reported one.
    """
    path = stored_table(configuration.stem, seed, TRAIN_SPLIT, root=root)
    cube, _, _ = gesture_cube(pd.read_csv(path))
    return prior_marginal(cube)


def reported(configuration: Configuration, root: Path = PREDICTIONS) -> Summary | None:
    """The thesis's figures for a configuration, from the tables stored for it.

    Read with the same functions as a run scored now, so the two can only
    differ where the scores do. ``None`` when no run's table is on this
    computer.
    """
    readings, raw = [], []
    for seed in SEEDS:
        path = stored_table(configuration.stem, seed, root=root)
        if not path.is_file():
            continue
        table = pd.read_csv(path)
        marginal = (
            stored_marginal(configuration, seed, root) if configuration.prior else None
        )
        readings.append(read(table, marginal, configuration.clip))
        if marginal is not None:
            raw.append(read(table))
    if not readings:
        return None
    return summarize(readings, raw or None)


def in_the_thesis(configuration: Configuration, figures: Summary) -> str:
    """The card's line for what the thesis measured, worded for its reading."""
    where = "on the 100 real test videos"
    if figures.clip is not None:
        return (
            f"{figures.frame[0]:.2f}% per frame · {figures.clip[0]:.2f}% per"
            f" decision on {configuration.clip} frames, {where}"
        )
    if figures.raw_video is not None:
        return (
            f"{figures.video[0]:.2f}% per video with the correction"
            f" ({figures.raw_video[0]:.2f}% without), {where}"
        )
    return (
        f"{figures.frame[0]:.2f}% per frame · {figures.video[0]:.2f}% per video,"
        f" {where}"
    )


def frame_agreement(now: pd.DataFrame, stored: pd.DataFrame) -> tuple[int, int, float]:
    """How many frames scored now name the gesture the stored table named.

    Frames are matched by video and by their place among that video's frames,
    which both tables keep in the manifest's order. Not by frame number alone:
    in short videos two stored frames can share one, and a join on it would
    count them twice. A run over some of the test videos is compared on the
    frames it has. The largest difference in any probability comes back beside
    the count, because two tables can agree on every answer and still not hold
    the same scores.

    Returns:
        Frames naming the same gesture, frames compared, and the largest
        absolute difference in a gesture's probability.

    Raises:
        ValueError: If a frame scored now has no counterpart in the stored
            table, or sits where the stored table has another frame, either of
            which would mean the two are not the same frames.
    """

    def by_place(table: pd.DataFrame) -> pd.DataFrame:
        place = table.groupby("video_id", sort=False).cumcount()
        return table.assign(place=place.to_numpy()).set_index(["video_id", "place"])

    first, second = by_place(now), by_place(stored)
    if not first.index.isin(second.index).all():
        raise ValueError("some frames scored now are not in the stored table")
    second = second.loc[first.index]
    if (first["frame_number"].to_numpy() != second["frame_number"].to_numpy()).any():
        raise ValueError("the two tables hold different frames at the same place")

    columns = [column for column in now.columns if column.startswith("p_")][:GESTURES]
    ours, theirs = first[columns].to_numpy(), second[columns].to_numpy()
    same = int((ours.argmax(axis=1) == theirs.argmax(axis=1)).sum())
    return same, len(ours), float(np.abs(ours - theirs).max())


def latency_by_k(
    configuration: Configuration,
    session: str = LATENCY_SESSION,
    root: Path = BENCHMARKS,
) -> dict[str, dict[int, float]]:
    """How long one decision took at each K, read from an idle-machine session.

    Never timed here: a number measured while the program runs would carry
    whatever else the computer was doing. A configuration costs what its rows
    cost together, which matters for the pose, whose decision pays the detector
    and then the classifier. A K is reported only where every row has it.

    Args:
        configuration: The one being costed.
        session: Prefix of the session's canonical files.
        root: Where those files are.

    Returns:
        Milliseconds per decision, by setting and then by K. A setting the
        session has no file for, or no row for, is left out.
    """
    table = {}
    for setting in SETTINGS:
        path = root / f"latency_{session}{setting}_isolated.csv"
        if not path.is_file():
            continue
        with path.open() as handle:
            rows = [row for row in csv.DictReader(handle) if row["level"] == "decision"]
        per_row = []
        for name in configuration.latency_rows:
            found = {
                int(row["frames"]): float(row["median_ms"])
                for row in rows
                if row["model"] == name
            }
            per_row.append(found)
        common = set.intersection(*(set(found) for found in per_row))
        if common:
            table[setting] = {
                k: sum(found[k] for found in per_row) for k in sorted(common)
            }
    return table
