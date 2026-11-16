"""Recognise RoCoG-v2 gestures with a trained configuration, and see how it did.

A menu in the terminal, so that running a model needs no script: choose a
configuration and read what it is, choose videos, and get the figures the thesis
reports for it, computed on those videos and set beside what the thesis measured
on the same ones. A video from outside the dataset carries no label, so for it
the program says what each run decided and how sure it was. Every table scored
is kept under ``results/``, in the format of the project's own tables.

Run from the project root::

    ./rocog                                                 # the menu
    ./rocog --configuration resnet18-spacings --videos test # the same, no menu
    ./rocog --configuration resnet18-shift --videos subject:S04 --limit 5
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from catalog import (  # noqa: E402
    CATALOG,
    NETWORKS,
    NO_REAL_VIDEOS,
    Configuration,
    card,
    find,
)
from frame_cache import EXTERNAL, VideoSource, dataset_videos  # noqa: E402
from manifest import load_class_names, video_metadata  # noqa: E402
from readings import (  # noqa: E402
    BENCHMARKS,
    CLIP_LENGTHS,
    GESTURES,
    PREDICTIONS,
    SETTINGS,
    Cost,
    Summary,
    clip_curve,
    frame_agreement,
    gesture_cube,
    in_the_thesis,
    latency_by_k,
    read,
    reported,
    stored_table,
    summarize,
    video_scores,
)

DATA = PROJECT_ROOT / "data"
TEST_ANNOTATIONS = DATA / "annotations" / "real_ground_test.txt"
TEST_SPLIT = "real_ground_test"
RESULTS = PROJECT_ROOT / "results"


# ------------------------------------------------------------------ the videos


@dataclass(frozen=True)
class Selection:
    """The videos a run is asked about, and what can be said of them.

    Attributes:
        label: What was chosen, in words, for the report.
        slug: The same, short, for the names of the saved tables.
        sources: The videos, in the order they are scored.
        labelled: Whether their gestures are known, without which there is no
            accuracy to compute, only decisions.
        tested: Whether they are real test videos, the ones the thesis's
            stored tables cover, so that what it measured on them can be read.
    """

    label: str
    slug: str
    sources: list[VideoSource]
    labelled: bool = True
    tested: bool = True


def test_videos(root: Path = PROJECT_ROOT) -> list[VideoSource]:
    """The 100 real test videos, in the order the stored tables list them.

    The same order keeps every batch the same as when the tables were written,
    which on a GPU is part of what reproducing them exactly means.
    """
    sources = dataset_videos(
        root / "data/annotations/real_ground_test.txt",
        root / "data",
        root / "data/manifests",
    )
    manifest = root / "data/manifests" / f"{TEST_SPLIT}.csv"
    if manifest.exists():
        order = pd.read_csv(manifest, usecols=["video_id"])["video_id"].unique()
        rank = {video: place for place, video in enumerate(order)}
        sources.sort(key=lambda source: rank.get(source.path.stem, len(rank)))
    return sources


def subject_of(source: VideoSource) -> str:
    """The person filmed, read off the file's name as the dataset names it."""
    return video_metadata(source.path, source.domain).group_id


def all_tested(sources: list[VideoSource]) -> Selection:
    """Every real test video."""
    return Selection(f"all {len(sources)} real test videos", TEST_SPLIT, sources)


def of_gesture(sources: list[VideoSource], label: int, name: str) -> Selection:
    """The real test videos of one gesture."""
    chosen = [source for source in sources if source.label == label]
    return Selection(
        f"the {len(chosen)} real test videos of {name}", f"{TEST_SPLIT}_{name}", chosen
    )


def of_subject(sources: list[VideoSource], subject: str) -> Selection:
    """The real test videos of one person."""
    chosen = [source for source in sources if subject_of(source) == subject]
    return Selection(
        f"the {len(chosen)} real test videos of subject {subject}",
        f"{TEST_SPLIT}_{subject}",
        chosen,
    )


def picked(sources: list[VideoSource]) -> Selection:
    """Real test videos chosen one by one."""
    count = len(sources)
    noun = "video" if count == 1 else "videos"
    return Selection(f"{count} real test {noun}", f"{TEST_SPLIT}_picked", sources)


def from_file(path: Path) -> Selection:
    """A video from anywhere on this computer, whose gesture nobody stated."""
    return Selection(
        f"the video {path.name}",
        "external",
        [VideoSource(path, EXTERNAL)],
        labelled=False,
        tested=False,
    )


# ------------------------------------------------------------- what came back


@dataclass
class Outcome:
    """A run, read: the figures now, the thesis's on the same videos, the costs.

    Attributes:
        configuration: What ran.
        selection: On which videos.
        tables: Each run's table, by seed.
        lengths: For a shifted network, each run's table at every clip length
            scored, by seed and length.
        marginals: For a corrected configuration, each run's frozen marginal.
        now: The summary of the tables, ``None`` for unlabelled videos.
        thesis: What the stored tables say about the same videos, ``None`` when
            they are not test videos or the tables are not on this computer.
        agreement: Frames naming the same gesture as the stored tables, frames
            compared, and the largest probability gap, over every run.
        curve_now: For a shifted network, its decision accuracy at each clip
            length, mean over the runs, in percent.
        curve_thesis: The same from the stored tables.
        latency: What one decision cost, by setting and K.
        seconds: How long the scoring took.
    """

    configuration: Configuration
    selection: Selection
    tables: dict[int, pd.DataFrame]
    lengths: dict[int, dict[int, pd.DataFrame]] = field(default_factory=dict)
    marginals: dict[int, np.ndarray] = field(default_factory=dict)
    now: Summary | None = None
    thesis: Summary | None = None
    agreement: tuple[int, int, float] | None = None
    curve_now: dict[int, float] = field(default_factory=dict)
    curve_thesis: dict[int, float] = field(default_factory=dict)
    latency: dict[str, dict[int, Cost]] = field(default_factory=dict)
    seconds: float = 0.0


def _mean_curve(curves: list[dict[int, float]]) -> dict[int, float]:
    """Each length's mean over the runs that have it, in percent."""
    lengths = sorted(set().union(*curves)) if curves else []
    return {
        length: float(np.mean([curve[length] for curve in curves if length in curve]))
        * 100
        for length in lengths
    }


def evaluate(
    configuration: Configuration,
    selection: Selection,
    tables: dict[int, pd.DataFrame],
    lengths: dict[int, dict[int, pd.DataFrame]] | None = None,
    marginals: dict[int, np.ndarray] | None = None,
    predictions: Path = PREDICTIONS,
    benchmarks: Path = BENCHMARKS,
) -> Outcome:
    """Read every table a run produced, and the stored ones for the same videos.

    The stored tables are cut to the chosen videos before they are read, so the
    two columns of every comparison describe the same frames.
    """
    lengths, marginals = lengths or {}, marginals or {}
    outcome = Outcome(configuration, selection, tables, lengths, marginals)
    outcome.latency = latency_by_k(configuration, root=benchmarks)
    if not selection.labelled:
        return outcome

    def readings_of(source: dict[int, pd.DataFrame]):
        readings, raw = [], []
        for seed, table in sorted(source.items()):
            readings.append(read(table, marginals.get(seed), configuration.clip))
            if configuration.prior:
                raw.append(read(table))
        return summarize(readings, raw or None)

    outcome.now = readings_of(tables)
    outcome.curve_now = _mean_curve([clip_curve(lengths[seed]) for seed in lengths])
    if not selection.tested:
        return outcome

    chosen = {source.path.stem for source in selection.sources}

    def stored(seed: int, length: int | None = None) -> pd.DataFrame | None:
        path = stored_table(
            configuration.stem,
            seed,
            length=length,
            trained=configuration.clip,
            root=predictions,
        )
        if not path.is_file():
            return None
        table = pd.read_csv(path)
        return table[table["video_id"].isin(chosen)].reset_index(drop=True)

    thesis_tables = {
        seed: table for seed in tables if (table := stored(seed)) is not None
    }
    if not thesis_tables:
        return outcome
    outcome.thesis = readings_of(thesis_tables)
    same = compared = 0
    gap = 0.0
    for seed, table in tables.items():
        if seed in thesis_tables:
            agree, count, largest = frame_agreement(table, thesis_tables[seed])
            same, compared, gap = same + agree, compared + count, max(gap, largest)
    outcome.agreement = (same, compared, gap)
    if lengths:
        stored_curves = []
        for seed, by_length in lengths.items():
            found = {length: stored(seed, length) for length in by_length}
            if all(table is not None for table in found.values()):
                stored_curves.append(clip_curve(found))
        outcome.curve_thesis = _mean_curve(stored_curves)
    return outcome


# ------------------------------------------------------------- what is shown


def _share(value: tuple[float, float] | None, runs: int) -> str:
    """A mean and its deviation over runs, or the mean alone for one run."""
    if value is None:
        return "—"
    mean, deviation = value
    return f"{mean:.2f}" if runs < 2 else f"{mean:.2f} ± {deviation:.2f}"


def figures_rows(outcome: Outcome) -> list[tuple[str, str, str]]:
    """The headline figures, now and in the thesis, one row per figure."""
    now, thesis = outcome.now, outcome.thesis
    if now is None:
        return []
    runs_now, runs_thesis = now.runs, thesis.runs if thesis else 0

    def row(label: str, pick) -> tuple[str, str, str]:
        mine = pick(now)
        theirs = pick(thesis) if thesis else None
        return label, _share(mine, runs_now), _share(theirs, runs_thesis)

    rows = [row("per frame (%)", lambda summary: summary.frame)]
    if outcome.configuration.clip:
        length = outcome.configuration.clip
        rows.append(
            row(f"per decision on {length} frames (%)", lambda summary: summary.clip)
        )
        rows.append(row("per video, mean of its frames (%)", lambda s: s.video))
    elif outcome.configuration.prior:
        rows.append(row("per video, corrected (%)", lambda summary: summary.video))
        rows.append(row("per video, before the correction (%)", lambda s: s.raw_video))
    else:
        rows.append(row("per video (%)", lambda summary: summary.video))
    rows.append(row("true gesture in the top 3 (%)", lambda summary: summary.top3))
    return rows


RANGE_NOTE = (
    "in parentheses: the fastest and slowest timing pass, where they lie more"
    " than 10% of the median apart; that cost is a range, not a value"
)
# Three spacings prepare four frames for every frame a decision reads, and from
# this K that passes the 32 MiB above which the memory allocator takes memory
# fresh from the system on every call and hands it back after. On the GPU,
# where the network is fast, paying for that memory is most of the decision.
ALLOCATOR_K = 16
ALLOCATOR_NOTE = (
    f"GPU from K = {ALLOCATOR_K}: most of this time is memory, not the network;"
    " three spacings prepare four frames for every frame decided, and past 32 MiB"
    " that memory comes fresh from the system on each decision"
)


def _cost(cost: Cost | None) -> str:
    """A cost's median, with its fastest and slowest pass when they spread."""
    if cost is None:
        return "—"
    shown = f"{cost.median:.1f}"
    if cost.unstable:
        shown += f" ({cost.fastest:.1f}–{cost.slowest:.1f})"
    return shown


def k_rows(outcome: Outcome) -> tuple[list[str], list[list[str]], list[str]]:
    """Accuracy and the cost of one decision at each K, as a table and its notes.

    A shifted network's K is its clip length, each scored on its own; every
    other configuration's K is how many of the stored frames a decision
    averages, spread over the gesture. Videos without labels have no accuracy,
    and get the costs alone.

    A cost is the median of its timing passes, with the fastest and slowest
    beside it where they spread so far that the median alone would read as more
    exact than it is. The notes are what the costs shown need to be read right:
    what those ranges are, and where a GPU time is mostly memory.

    Returns:
        The header, one row per K, and the notes to print under the table.
    """
    settings = [setting for setting in SETTINGS if setting in outcome.latency]
    costs = [f"{SETTINGS[setting]} (ms)" for setting in settings]
    configuration, now, thesis = outcome.configuration, outcome.now, outcome.thesis
    if now is None:
        ks = sorted(set().union(*(outcome.latency[s] for s in settings)))
        header, accuracy, reported_k = ["K", *costs], None, None
    elif configuration.clip:
        accuracy = outcome.curve_now or {configuration.clip: now.clip[0]}
        reported_k = outcome.curve_thesis or (
            {configuration.clip: thesis.clip[0]} if thesis else {}
        )
        ks = sorted(accuracy)
        header = ["K", "accuracy now (%)", "in the thesis (%)", *costs]
    else:
        accuracy = {k: point[0] for k, point in now.curve.items()}
        reported_k = (
            {k: point[0] for k, point in thesis.curve.items()} if thesis else {}
        )
        ks = sorted(accuracy)
        header = ["K", "accuracy now (%)", "in the thesis (%)", *costs]

    rows, shown = [], []
    for k in ks:
        line = [str(k)]
        if accuracy is not None:
            line.append(f"{accuracy[k]:.2f}" if k in accuracy else "—")
            line.append(f"{reported_k[k]:.2f}" if k in reported_k else "—")
        for setting in settings:
            cost = outcome.latency[setting].get(k)
            line.append(_cost(cost))
            if cost is not None:
                shown.append((setting, k, cost))
        rows.append(line)

    notes = []
    if any(cost.unstable for _, _, cost in shown):
        notes.append(RANGE_NOTE)
    if len(configuration.spacings) == 3 and any(
        setting == "cuda_1t" and k >= ALLOCATOR_K for setting, k, _ in shown
    ):
        notes.append(ALLOCATOR_NOTE)
    return header, rows, notes


def gesture_rows(outcome: Outcome, names: list[str]) -> list[tuple[str, str, str]]:
    """Recall and precision per gesture, mean over the runs, in percent."""
    if outcome.now is None:
        return []

    def shown(value: float) -> str:
        return "—" if np.isnan(value) else f"{value:.1f}"

    return [
        (name, shown(outcome.now.recall[index]), shown(outcome.now.precision[index]))
        for index, name in enumerate(names)
    ]


def decisions_by_video(
    outcome: Outcome, names: list[str]
) -> list[tuple[str, str, list[str], list[float]]]:
    """Each video's gesture, and what each run decided for it and how surely.

    The decision is the configuration's own: corrected where it has a
    correction, and for a shifted network the mean of its frames, since the
    clips of one video can disagree with each other.
    """
    rows = {}
    for seed, table in sorted(outcome.tables.items()):
        cube, labels, video_ids = gesture_cube(table)
        scores = video_scores(cube, outcome.marginals.get(seed))
        shares = scores / scores.sum(axis=1, keepdims=True)
        for index, video in enumerate(video_ids):
            truth = names[labels[index]] if 0 <= labels[index] < GESTURES else "?"
            entry = rows.setdefault(video, (truth, [], []))
            entry[1].append(names[int(shares[index].argmax())])
            entry[2].append(float(shares[index].max()))
    return [(video, truth, said, sure) for video, (truth, said, sure) in rows.items()]


def report(console, outcome: Outcome, names: list[str]) -> None:
    """Print everything a run says, in the order a reader needs it."""
    from rich.table import Table

    configuration, selection = outcome.configuration, outcome.selection
    console.print(
        f"[bold]{configuration.label}[/bold] on {selection.label}"
        f" · {len(outcome.tables)} run(s) · scored in {outcome.seconds:.0f} s"
    )

    rows = figures_rows(outcome)
    if rows:
        table = Table(title="How it did", title_justify="left")
        for column in ("", "now", "in the thesis"):
            table.add_column(column, justify="right" if column else "left")
        for row in rows:
            table.add_row(*row)
        console.print(table)
        if outcome.agreement:
            same, compared, gap = outcome.agreement
            console.print(
                f"same answer as the thesis on {same:,} of {compared:,} frames"
                f" (largest difference in a probability: {gap:.1e})"
            )
        elif selection.tested:
            console.print(
                "the thesis's tables for these videos are not on this computer"
            )

    header, rows, notes = k_rows(outcome)
    if rows:
        title = "Accuracy and cost by K" if outcome.now else "Cost by K"
        table = Table(title=title, title_justify="left")
        for column in header:
            table.add_column(column, justify="right")
        for row in rows:
            table.add_row(*row)
        console.print(table)
        for note in notes:
            console.print(note)
    if not outcome.latency:
        console.print("latency: not measured yet for this configuration")

    if outcome.now is not None:
        table = Table(title="By gesture", title_justify="left")
        for column, justify in (
            ("gesture", "left"),
            ("recall (%)", "right"),
            ("precision (%)", "right"),
        ):
            table.add_column(column, justify=justify)
        for row in gesture_rows(outcome, names):
            table.add_row(*row)
        console.print(table)

        table = Table(
            title="Confusion, every run's decisions (rows: true gesture)",
            title_justify="left",
        )
        table.add_column("")
        for name in names:
            table.add_column(name[:5], justify="right")
        for index, name in enumerate(names):
            counts = outcome.now.confusion[index]
            table.add_row(name, *(str(count) for count in counts))
        console.print(table)

    videos = decisions_by_video(outcome, names)
    if selection.labelled:
        wrong = [row for row in videos if any(said != row[1] for said in row[2])]
        title = f"Videos some run got wrong ({len(wrong)} of {len(videos)})"
        shown = wrong
    else:
        title, shown = "What each run decided", videos
    if shown:
        table = Table(title=title, title_justify="left")
        table.add_column("video")
        if selection.labelled:
            table.add_column("gesture")
        table.add_column("each run decided")
        for video, truth, said, sure in shown:
            decided = " · ".join(
                f"{name} ({share:.2f})" for name, share in zip(said, sure, strict=True)
            )
            table.add_row(video, *([truth] if selection.labelled else []), decided)
        console.print(table)


# ------------------------------------------------------------- what is kept


def save(
    outcome: Outcome,
    text: str,
    root: Path = RESULTS,
    when: datetime | None = None,
) -> Path:
    """Keep a run's tables and its report, under a folder of its own.

    The tables have the columns and names of the project's own, so anything
    that reads those reads these.
    """
    when = when or datetime.now()
    folder = root / f"{when:%Y-%m-%d_%H%M%S}_{outcome.configuration.key}"
    folder.mkdir(parents=True, exist_ok=False)
    configuration, slug = outcome.configuration, outcome.selection.slug
    for seed, table in outcome.tables.items():
        table.to_csv(folder / f"{configuration.stem}_s{seed}__{slug}.csv", index=False)
    for seed, by_length in outcome.lengths.items():
        for length, table in by_length.items():
            if length == configuration.clip:
                continue
            name = f"{configuration.stem}_s{seed}__{slug}_clips{length}.csv"
            table.to_csv(folder / name, index=False)
    (folder / "report.txt").write_text(text)
    return folder


# ------------------------------------------------------------------ the menu


def _description(configuration: Configuration, reason: str | None) -> str:
    """The card shown under the menu while a configuration is highlighted."""
    lines = card(configuration)
    figures = reported(configuration)
    if figures is not None:
        lines.append(("in the thesis", in_the_thesis(configuration, figures)))
    if reason:
        lines.append(("not here", reason))
    return "\n".join(f"  {label:<14} {text}" for label, text in lines)


def ask_configuration(why_not, **prompt) -> Configuration | None:
    """The configuration to run, chosen from the catalog grouped by network.

    Args:
        why_not: Says why a configuration cannot run here, or ``None``.
        prompt: Passed to the prompt, which is how a test types into it.
    """
    import questionary
    from questionary import Choice, Separator

    choices = []
    for network, title in [*NETWORKS.items(), (None, "Without an image network")]:
        choices.append(Separator(f"── {title} "))
        for configuration in CATALOG:
            if configuration.network != network:
                continue
            reason = why_not(configuration)
            name = configuration.label.split(" · ", 1)[-1]
            choices.append(
                Choice(
                    name,
                    value=configuration.key,
                    disabled=reason,
                    description=_description(configuration, reason),
                )
            )
    key = questionary.select(
        "Which configuration?", choices=choices, use_shortcuts=False, **prompt
    ).ask()
    return None if key is None else find(key)


def ask_videos(
    names: dict[int, str], sources: list[VideoSource] | None = None, **prompt
) -> Selection | None:
    """The videos to run it on: the real test set, a part of it, or any file.

    Args:
        names: Label to gesture name.
        sources: The real test videos; read from the dataset when not given.
        prompt: Passed to every prompt, which is how a test types into them.
    """
    import questionary
    from questionary import Choice

    sources = test_videos() if sources is None else sources
    kind = questionary.select(
        "Which videos?",
        choices=[
            Choice(f"All {len(sources)} real test videos", "all"),
            Choice("The test videos of one gesture…", "gesture"),
            Choice("The test videos of one subject…", "subject"),
            Choice("Pick test videos one by one…", "pick"),
            Choice("A video file from this computer…", "file"),
        ],
        **prompt,
    ).ask()
    if kind is None:
        return None
    if kind == "all":
        return all_tested(sources)
    if kind == "gesture":
        label = questionary.select(
            "Which gesture?",
            choices=[Choice(name, label) for label, name in sorted(names.items())],
            **prompt,
        ).ask()
        return None if label is None else of_gesture(sources, label, names[label])
    if kind == "subject":
        subjects = sorted({subject_of(source) for source in sources})
        subject = questionary.select("Which subject?", choices=subjects, **prompt).ask()
        return None if subject is None else of_subject(sources, subject)
    if kind == "pick":
        chosen = questionary.checkbox(
            "Which videos? (space to mark, enter to run)",
            choices=[
                Choice(f"{source.path.stem}  ·  {names.get(source.label, '?')}", source)
                for source in sources
            ],
            use_search_filter=True,
            use_jk_keys=False,
            validate=lambda marked: bool(marked) or "mark at least one video",
            **prompt,
        ).ask()
        return None if not chosen else picked(chosen)
    path = questionary.path(
        "Which file?",
        validate=lambda text: Path(text).expanduser().is_file(),
        **prompt,
    ).ask()
    return None if path is None else from_file(Path(path).expanduser())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line; with no configuration named, the menu asks."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--configuration",
        choices=[configuration.key for configuration in CATALOG],
        help="run this one without the menu",
    )
    parser.add_argument(
        "--videos",
        default="test",
        help="test, gesture:NAME, subject:ID, or file:PATH (default: test)",
    )
    parser.add_argument(
        "--limit", type=int, help="only the first this many videos, for a quick look"
    )
    parser.add_argument(
        "--one-length",
        action="store_true",
        help="score a shifted network at its own clip length only",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    parser.add_argument(
        "--no-save", action="store_true", help="print the report and keep nothing"
    )
    return parser.parse_args(argv)


def selection_from(
    text: str, names: dict[int, str], sources: list[VideoSource] | None = None
) -> Selection:
    """The videos a ``--videos`` value names, among the real test videos given."""
    kind, _, value = text.partition(":")
    if kind == "file":
        return from_file(Path(value).expanduser())
    sources = test_videos() if sources is None else sources
    if kind == "test":
        return all_tested(sources)
    if kind == "gesture":
        labels = {name.lower(): label for label, name in names.items()}
        if value.lower() not in labels:
            raise SystemExit(
                f"no gesture {value!r}; the gestures are {list(names.values())}"
            )
        label = labels[value.lower()]
        return of_gesture(sources, label, names[label])
    if kind == "subject":
        return of_subject(sources, value)
    raise SystemExit(f"--videos {text!r}: expected test, gesture:, subject: or file:")


def main(argv: list[str] | None = None) -> None:
    """Ask, score, read, show, keep."""
    import torch
    from rich.console import Console
    from rich.progress import Progress

    import scoring
    from device import describe, pick_device

    args = parse_args(argv)
    console = Console(record=True)
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    class_names = load_class_names(DATA / "class_dict.json")
    names = [class_names[label] for label in sorted(class_names)][:GESTURES]
    console.print(
        f"[bold]RoCoG-v2 gesture recognition[/bold] · device: {describe(device)}"
    )

    if args.configuration:
        configuration = find(args.configuration)
        reason = scoring.why_not(configuration)
        if reason:
            raise SystemExit(f"{configuration.label}: {reason}")
        selection = selection_from(args.videos, class_names)
        lengths = () if args.one_length else CLIP_LENGTHS
    else:
        configuration = ask_configuration(scoring.why_not)
        if configuration is None:
            return
        console.print(_description(configuration, None))
        selection = ask_videos(class_names)
        if selection is None:
            return
        lengths = CLIP_LENGTHS
        if configuration.clip:
            import questionary

            again = questionary.confirm(
                "Also score it at the other clip lengths, for the table by K?"
                f" (about {len(CLIP_LENGTHS)} times as long)",
                default=True,
            ).ask()
            lengths = CLIP_LENGTHS if again else ()

    if args.limit:
        selection = Selection(
            f"the first {args.limit} of {selection.label}",
            selection.slug,
            selection.sources[: args.limit],
            selection.labelled,
            selection.tested,
        )
    if not selection.sources:
        raise SystemExit("no video matches that choice")
    if configuration.real_videos != NO_REAL_VIDEOS and not selection.tested:
        console.print(
            "note: this configuration used the real training videos"
            f" ({configuration.real_videos}); one of those is no test of it"
        )

    started = time.perf_counter()
    with Progress(console=console, transient=True) as progress:
        frames = progress.add_task("preparing frames", total=len(selection.sources))
        runs = progress.add_task("scoring", total=None, visible=False)

        def told(stage: str, done: int, total: int) -> None:
            if stage == "frames":
                progress.update(frames, completed=done, total=total)
            else:
                progress.update(
                    runs,
                    completed=done,
                    total=total,
                    visible=True,
                    description=f"scoring · {stage}",
                )

        rows, dense = scoring.prepare_videos(
            selection.sources, class_names, bool(configuration.spacings), told
        )
        scored = scoring.score(
            configuration, rows, dense, device, class_names, lengths, told
        )
    outcome = evaluate(
        configuration,
        selection,
        scored.tables,
        scored.lengths,
        scored.marginals,
    )
    outcome.seconds = time.perf_counter() - started
    report(console, outcome, names)
    if not args.no_save:
        folder = save(outcome, console.export_text())
        console.print(f"saved to {folder.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
