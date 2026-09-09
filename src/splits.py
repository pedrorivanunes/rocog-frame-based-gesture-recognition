"""Choose the manifest rows a run trains and evaluates on.

Every experiment in this project is a selection: a camera viewpoint, a fraction
of real data, a validation split, a stretch of the gesture. The manifest is the
table those selections are made on, and keeping them here means the data a run
saw can be read from the code that built it, rather than inferred from a file
name.

Two kinds live here. A split cuts the rows in two and hands back both sides, so
that nothing straddles the boundary between training and validation. A selection
narrows one side and hands back what is left. They are kept together because
both answer the same question about a result — which frames produced it.
"""

import numpy as np
import pandas as pd

from manifest import IDLE_CLASS_NAME, IDLE_LABEL

SPLIT_SEED = 7
WINDOW_SEED = 23

WINDOWS = ("full", "middle", "scattered")

# Frames dropped from each end of a video by the narrower windows. Extraction
# stores twenty-four per video, one per segment of the gesture, so four is the
# outer sixth at each end and sixteen survive. Sixteen is also a count the
# sampler can cut into the eight blocks an epoch draws from, which a threshold on
# ``position`` would not guarantee: it leaves a different number of frames in
# every video.
EDGE_FRAMES = 4


def split_by_group(
    manifest: pd.DataFrame,
    held_out: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole groups, named rather than drawn.

    The group is the unit that must not straddle the boundary: the scene for
    synthetic videos, which fixes terrain, lighting, camera and avatar, and the
    recorded subject for real ones, since a person who appears on both sides
    turns a measure of generalisation into a measure of memorisation.

    Which groups are held out is named by the caller rather than drawn, because
    here it is a control and not a source of variation. The real subjects differ
    in size by a factor of three — one of the seven carries a third of the
    videos on its own — so drawing one would change how much data a run trains
    on, and two runs meant to differ only in their seed would differ in that
    too. Naming it also puts the choice in the command that produced a result.

    Args:
        manifest: Rows to split. Requires the ``group_id`` column.
        held_out: Groups whose rows become the validation side.

    Returns:
        The training rows and the validation rows, in that order. Both carry
        every column of the input.

    Raises:
        ValueError: If nothing is held out, if a named group is absent from the
            manifest, or if holding these out would leave nothing to train on.
            An absent name is worth stopping for rather than ignoring: it
            otherwise yields an empty validation set, which fails much later and
            says nothing about what caused it.
    """
    if not held_out:
        raise ValueError("no groups named to hold out")

    present = set(manifest["group_id"].unique())
    missing = sorted(set(held_out) - present)
    if missing:
        raise ValueError(f"groups not in the manifest: {missing}")
    if not present - set(held_out):
        raise ValueError("holding out every group would leave nothing to train on")

    is_validation = manifest["group_id"].isin(held_out)
    train = manifest[~is_validation].reset_index(drop=True)
    validation = manifest[is_validation].reset_index(drop=True)

    return train, validation


def split_by_scene(
    manifest: pd.DataFrame,
    scenes_per_view: int = 1,
    seed: int = SPLIT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole synthetic scenes for validation, drawn one per viewpoint.

    Videos from a single scene share terrain, lighting, camera position and
    avatar, so splitting by video would put near-duplicates on both sides and
    report a validation accuracy that measures memorisation rather than
    generalisation. The scene is the unit that must not straddle the boundary.

    Drawing within each viewpoint, rather than across all scenes at once, keeps
    all six represented on both sides. The scenes are spread unevenly: view 3
    has five and view 5 has eight, so a blind draw could take two of those five,
    or none of some other view — leaving validation blind to an angle the model
    trains on, and the viewpoint experiment without a control.

    Args:
        manifest: Synthetic manifest, one row per frame. Requires the ``view``
            and ``group_id`` columns. The real manifests carry no viewpoint and
            are split by subject instead.
        scenes_per_view: How many scenes each viewpoint contributes to
            validation. One of the forty scenes per view is roughly 15% of the
            videos, which varies with the draw because scenes hold between 186
            and 364 videos each.
        seed: Draws the held-out scenes. Deliberately its own generator rather
            than one shared with the rest of a run: a shared generator would
            move this boundary whenever anything else consumed a draw, and two
            runs meant to be comparable would have trained on different data.

    Returns:
        The training rows and the validation rows, in that order. Both carry
        every column of the input.

    Raises:
        RuntimeError: If any viewpoint holds fewer scenes than requested, which
            would leave that viewpoint absent from one side of the split.
    """
    rng = np.random.default_rng(seed)

    per_video = manifest.drop_duplicates("video_id")
    scenes_by_view = per_video.groupby("view")["group_id"].unique()

    held_out = []
    for view, scenes in scenes_by_view.items():
        if len(scenes) < scenes_per_view:
            raise RuntimeError(
                f"view {view} has {len(scenes)} scenes, need {scenes_per_view}"
            )

        held_out.extend(rng.choice(scenes, size=scenes_per_view, replace=False))

    return split_by_group(manifest, held_out)


def select_frames(
    manifest: pd.DataFrame,
    window: str = "full",
    seed: int = 0,
) -> pd.DataFrame:
    """Choose which of each video's stored frames a run may train on.

    The annotated window spans the whole movement, its rise and its fall
    included, so the frames nearest each end show a body close to neutral while
    still carrying the gesture's label. Training on them asks the model to read
    a gesture out of a body at rest — a demand no single frame can meet, and one
    that leaves the neutral pose with no label of its own to fall into.

    ``middle`` drops that material from training. ``scattered`` is its control
    and exists only for it: cutting the ends also cuts a third of the frames, so
    a difference between ``middle`` and ``full`` confounds *which* frames with
    *how many*. Keeping the same reduced count, drawn across the whole window
    instead of taken from its centre, separates the two — whatever ``middle``
    buys over ``scattered`` is the position of the frames and nothing else.

    Trimming by rank rather than by a threshold on ``position`` is what keeps
    the count identical across videos, which the sampler requires; on the
    twenty-four frames extraction stores, one per segment, dropping four from
    each end leaves the window from 0.154 to 0.846.

    This is a training-side selection. Applying it to evaluation would decide
    which frames to score using where they sit in the clip, and a model in the
    field does not know where a frame sits — that is temporal information, and
    reading it at decision time makes the selection a treatment rather than a
    control.

    Args:
        manifest: Rows to select from, one per frame. Requires the ``video_id``
            column, and expects each video's frames to be consecutive and in
            ascending order, which is how extraction writes them.
        window: ``full`` keeps every stored frame, ``middle`` keeps all but the
            outermost of each video, and ``scattered`` keeps as many as
            ``middle`` does, drawn at random across the whole window.
        seed: Draws the ``scattered`` frames, and is ignored by the other two.
            Moving with the run's seed is deliberate: the control asks what an
            arbitrary subset of this size does, so three repetitions should meet
            three arbitrary subsets rather than agree on one that might happen
            to be lucky. It costs the control some variance and buys it freedom
            from a single draw.

    Returns:
        The rows the window keeps, in the order the manifest holds them. The
        input frame itself under ``full``, since nothing is dropped.

    Raises:
        ValueError: If the window is not one this knows, or if a video holds too
            few frames to trim.
    """
    if window not in WINDOWS:
        raise ValueError(f"window {window!r} is not one of {', '.join(WINDOWS)}")

    if window == "full":
        return manifest

    videos = manifest.groupby("video_id", sort=False).indices

    smallest = min(len(frames) for frames in videos.values())
    if smallest <= 2 * EDGE_FRAMES:
        raise ValueError(
            f"a video holds {smallest} frames, too few to drop {EDGE_FRAMES} "
            "from each end"
        )

    if window == "middle":
        kept = [frames[EDGE_FRAMES:-EDGE_FRAMES] for frames in videos.values()]
    else:
        rng = np.random.default_rng(WINDOW_SEED + seed)
        kept = [
            rng.choice(frames, size=len(frames) - 2 * EDGE_FRAMES, replace=False)
            for frames in videos.values()
        ]

    return manifest.iloc[np.sort(np.concatenate(kept))]


def add_idle_rows(gesture: pd.DataFrame, idle: pd.DataFrame) -> pd.DataFrame:
    """Put a video's idle frames behind its gesture frames, under a label of their own.

    The two are stored in separate manifests because a video contributes a
    different number of each, and a table where every video holds the same
    number of rows is what both resuming an extraction and selecting a window
    rest on. Training is where the two meet.

    The idle rows arrive carrying the gesture of the clip they came from, which
    is what extraction recorded and what leaves a question like "whose idle
    frames are hardest" answerable later. Here that label is replaced. What the
    model is asked is which of eight things a frame shows, and a body at rest is
    the same answer whichever gesture it is about to perform.

    Args:
        gesture: Rows sampled across the gesture, already narrowed to whatever
            the run trains on.
        idle: Rows sampled before it, for the same videos.

    Returns:
        The two sets in one frame, gesture rows first, numbered from zero. The
        numbering matters: the dataset serves rows by position, so a sampler
        over this frame names positions in it.
    """
    return pd.concat(
        [gesture, idle.assign(label=IDLE_LABEL, class_name=IDLE_CLASS_NAME)],
        ignore_index=True,
    )
