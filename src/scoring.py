"""Score chosen videos with every run of a configuration, as the thesis scored them.

The thesis's figures were computed from probability tables that a script wrote,
one checkpoint and one set of frames at a time. The program asks the same thing
of videos someone chooses, so it takes the same path: the frames are prepared
the way the dataset's were, each run's checkpoint is rebuilt from what the file
says about it, and the scores come back as a table with the columns the stored
ones have. Nothing about the network is decided here. What the checkpoint does
not say, the catalog does.

A shifted network is scored once for each clip length it is read at, because a
frame's scores depend on the frames that shared its clip. A corrected one
carries the marginal its correction divides by, read from the scores stored for
the real training videos, so that it is the estimate the thesis used and not a
second one made on whatever was scored today.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from catalog import CHECKPOINTS, SEEDS, Configuration, available_seeds
from evaluation import (
    NUM_WORKERS,
    load_scorer,
    probability_table,
    score_frames,
    with_neighbour_frames,
)
from frame_cache import VideoSource, prepare
from manifest import with_idle_class
from readings import PREDICTIONS, TRAIN_SPLIT, stored_marginal, stored_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Told what has been done and out of how much, after each video and each run.
Progress = Callable[[str, int, int], None]


def _quiet(stage: str, done: int, total: int) -> None:
    """The progress report of a caller that wants none."""


def why_not(
    configuration: Configuration,
    checkpoints: Path = CHECKPOINTS,
    predictions: Path = PREDICTIONS,
) -> str | None:
    """Why a configuration cannot be run on this computer, or ``None`` if it can.

    Asked before the menu offers it, so that a missing file is a sentence
    beside its name rather than a traceback after the videos were prepared.
    """
    if configuration.network is None:
        return "its detector runs in an environment of its own, not wired in yet"
    seeds = available_seeds(configuration, checkpoints)
    if not seeds:
        return "none of its checkpoints is on this computer"
    if configuration.prior and not all(
        stored_table(configuration.stem, seed, TRAIN_SPLIT, root=predictions).is_file()
        for seed in seeds
    ):
        return "its correction needs the scores stored for the real training videos"
    return None


def prepare_videos(
    sources: list[VideoSource],
    class_names: dict[int, str],
    dense: bool,
    progress: Progress = _quiet,
    root: Path = PROJECT_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """The frames of every chosen video, from the cache or extracted into it.

    Args:
        sources: The videos, in the order their rows will be scored.
        class_names: Label to gesture name, for the rows' class column.
        dense: Also every frame of each gesture window, which a network reading
            differences takes its neighbours from.
        progress: Told after each video.
        root: Where the cache lives, and what the rows' paths are relative to.

    Returns:
        The sampled rows, and the dense ones when asked for.
    """
    sampled, windows = [], []
    for index, source in enumerate(sources, 1):
        rows, window = prepare(source, class_names, dense=dense, root=root)
        sampled.append(rows)
        windows.append(window)
        progress("frames", index, len(sources))
    dense_rows = pd.concat(windows, ignore_index=True) if dense else None
    return pd.concat(sampled, ignore_index=True), dense_rows


@dataclass
class Scored:
    """Every table one configuration produced on the chosen videos.

    Attributes:
        configuration: What was run.
        tables: Each run's table, by seed. For a shifted network, the one at
            the clip length it trained on.
        lengths: For a shifted network, each run's table at every clip length
            it was scored at, by seed and then length; empty otherwise.
        marginals: For a corrected configuration, each run's frozen marginal,
            by seed; empty otherwise.
    """

    configuration: Configuration
    tables: dict[int, pd.DataFrame] = field(default_factory=dict)
    lengths: dict[int, dict[int, pd.DataFrame]] = field(default_factory=dict)
    marginals: dict[int, np.ndarray] = field(default_factory=dict)


def _release(device: torch.device) -> None:
    """Hand a run's memory back before the next run's network is built."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _table(
    checkpoint: Path,
    rows: pd.DataFrame,
    dense: pd.DataFrame | None,
    configuration: Configuration,
    device: torch.device,
    class_names: dict[int, str],
    clips: int | None,
    root: Path,
    num_workers: int,
) -> pd.DataFrame:
    """One checkpoint's table on the prepared rows, at one clip length."""
    scorer = load_scorer(checkpoint, device, clips)
    names = (
        with_idle_class(class_names)
        if scorer.num_classes > len(class_names)
        else class_names
    )
    served = with_neighbour_frames(
        rows, scorer.in_channels, configuration.spacings or None, dense=dense
    )
    logits, _, video_ids = score_frames(
        scorer.model,
        served,
        device,
        root=root,
        segments=scorer.segments,
        num_workers=num_workers,
    )
    table = probability_table(logits, video_ids, served, names)
    del scorer
    _release(device)
    return table


def score(
    configuration: Configuration,
    rows: pd.DataFrame,
    dense: pd.DataFrame | None,
    device: torch.device,
    class_names: dict[int, str],
    lengths: tuple[int, ...] = (),
    progress: Progress = _quiet,
    root: Path = PROJECT_ROOT,
    checkpoints: Path = CHECKPOINTS,
    predictions: Path = PREDICTIONS,
    num_workers: int = NUM_WORKERS,
) -> Scored:
    """Score the prepared rows with every run of a configuration on this computer.

    Args:
        configuration: What to run.
        rows: The sampled frames, as ``prepare_videos`` returns them.
        dense: The windows, for a configuration reading differences.
        device: Where the networks run.
        class_names: Label to gesture name.
        lengths: For a shifted network, the clip lengths to score at besides
            the one it trained on. Ignored for any other.
        progress: Told after each scoring.
        root: What the rows' paths are relative to.
        checkpoints: Where the runs' files are.
        predictions: Where the stored tables are, for a corrected
            configuration's marginal.
        num_workers: Processes preparing frames for the network.

    Returns:
        Each run's tables, and each corrected run's marginal.
    """
    seeds = available_seeds(configuration, checkpoints)
    extra = (
        [length for length in lengths if length != configuration.clip]
        if configuration.clip
        else []
    )
    total = len(seeds) * (1 + len(extra))
    scored = Scored(configuration)
    done = 0
    for seed in seeds:
        checkpoint = checkpoints / f"{configuration.stem}_s{seed}.pt"
        scored.tables[seed] = _table(
            checkpoint,
            rows,
            dense,
            configuration,
            device,
            class_names,
            None,
            root,
            num_workers,
        )
        done += 1
        progress(f"run {seed}", done, total)
        if configuration.clip:
            by_length = {configuration.clip: scored.tables[seed]}
            for length in extra:
                by_length[length] = _table(
                    checkpoint,
                    rows,
                    dense,
                    configuration,
                    device,
                    class_names,
                    length,
                    root,
                    num_workers,
                )
                done += 1
                progress(f"run {seed}, clips of {length}", done, total)
            scored.lengths[seed] = dict(sorted(by_length.items()))
        if configuration.prior:
            scored.marginals[seed] = stored_marginal(configuration, seed, predictions)
    return scored


def seeds_scored(scored: Scored) -> list[int]:
    """The runs that produced a table, in seed order."""
    return [seed for seed in SEEDS if seed in scored.tables]
