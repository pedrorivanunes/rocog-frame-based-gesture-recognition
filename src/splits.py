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

from collections.abc import Sequence

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


def keep_views(manifest: pd.DataFrame, views: list[int]) -> pd.DataFrame:
    """Keep only the rows shot from the named camera positions.

    The synthetic videos are shot from six positions around the subject and the
    real ones from a single frontal camera, so two thirds of the synthetic
    material is filmed from an angle the target domain never shows. Whether
    those two thirds help — more data, and variation that could push the network
    towards features that do not depend on one camera — or hurt — capacity spent
    on a condition that is never scored — has not been measured here or in the
    published baselines, which train on all six.

    Applied before the split rather than after it. A run trained on one set of
    viewpoints has to be model-selected on the same ones; validating on angles
    it never trains on would pick a checkpoint by performance on a question the
    run was never asked.

    Viewpoints are named rather than reduced to a frontal flag because the two
    frontal positions do not behave alike: a pose detector recovers every arm
    joint in 98.8% of frames from one of them and 66.7% from the other, so a
    later run may want one alone.

    Args:
        manifest: Rows to filter. Requires the ``view`` column, which only the
            synthetic manifests carry.
        views: Camera positions to keep.

    Returns:
        The rows shot from those positions, carrying every column of the input.

    Raises:
        ValueError: If no viewpoint is named, or if a named one is absent from
            the manifest. An absent name is worth stopping for: on a real
            manifest, whose viewpoint column is empty, it would otherwise hand
            back nothing and fail much later without saying why.
    """
    if not views:
        raise ValueError("no viewpoints named to keep")

    present = set(manifest["view"].dropna().astype(int).unique())
    missing = sorted(set(views) - present)
    if missing:
        raise ValueError(f"viewpoints not in the manifest: {missing}")

    return manifest[manifest["view"].isin(views)].reset_index(drop=True)


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


# What the column holding a frame's neighbour is called. A run reading several
# spacings numbers the rest after it.
NEIGHBOUR_COLUMN = "neighbour_path"


def with_neighbour(
    manifest: pd.DataFrame, stride: int, dense: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Name, for every frame, the frame it should be differenced against.

    A difference between neighbouring frames says where the person moved and
    where nothing did. Which frame counts as the neighbour is a choice, and
    keeping it here rather than in the extraction is what lets one dense pass
    serve every spacing: the column is recomputed in seconds, the frames are
    not re-read.

    The neighbour is looked up somewhere else than it is attached, and that
    separation is what keeps a comparison honest. The rows a run trains on are
    two dozen a video, spread apart; the neighbour of one of them is almost
    never another of them. So the lookup happens in the dense manifest while
    the column lands on the rows being served — which means a run reading
    differences trains on exactly the frames the run without them trains on,
    and the two differ in the extra channels rather than in which frames they
    saw.

    Frames whose neighbour falls before the window fall back to the earliest
    frame the video has. The first frame of all then points at itself and its
    difference is exactly zero, which is the honest reading: no earlier frame
    exists, so no movement was measured.

    Args:
        manifest: Dense manifest, one row per frame of each gesture window.
        stride: How many frames back the neighbour sits. Small values measure
            local movement; large ones measure the gesture changing shape.
        dense: Where to look the neighbour up, every frame of every window.
            Defaults to ``manifest`` itself, which is right only when that is
            already dense.

    Returns:
        The manifest with a ``neighbour_path`` column added.

    Raises:
        ValueError: If the stride is not positive, or if a video carries the
            same frame number twice — which a dense pass never writes, and
            which would make the neighbour of that frame ambiguous.
    """
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")

    source = manifest if dense is None else dense
    keys = ["video_id", "frame_number"]
    if source.duplicated(keys).any():
        raise ValueError("a video carries the same frame number twice")

    lookup = source.set_index(keys)["path"]
    starts = source.groupby("video_id")["frame_number"].min()
    earliest = manifest["video_id"].map(starts)
    wanted = (manifest["frame_number"] - stride).clip(lower=earliest)

    annotated = manifest.copy()
    annotated["neighbour_path"] = lookup.reindex(
        pd.MultiIndex.from_arrays([manifest["video_id"], wanted])
    ).to_numpy()

    if annotated["neighbour_path"].isna().any():
        missing = annotated.loc[annotated["neighbour_path"].isna(), "video_id"]
        raise ValueError(
            f"no neighbour for {missing.nunique()} video(s), starting with "
            f"{missing.iloc[0]}: is this the dense manifest for these rows?"
        )

    return annotated


def neighbour_columns(count: int) -> list[str]:
    """Name the columns that hold one neighbour per spacing.

    The first keeps the name a single spacing has always written, so every
    table, checkpoint and reader produced before spacings could be plural goes
    on meaning what it meant.

    Args:
        count: How many spacings the run reads.

    Returns:
        One column name per spacing, in the order the spacings were given.
    """
    return [NEIGHBOUR_COLUMN] + [
        f"{NEIGHBOUR_COLUMN}_{index}" for index in range(1, count)
    ]


def neighbours_in(manifest: pd.DataFrame) -> list[str]:
    """The neighbour columns a manifest carries, in the order to stack them.

    Asked of the table rather than passed alongside it, because the number of
    spacings is a property of the rows being served and everything downstream
    would otherwise have to be told it twice and could be told two different
    things.

    Args:
        manifest: Rows to inspect.

    Returns:
        The names present, first spacing first. Empty when the manifest names
        no neighbour, which is the plain three-channel case.
    """
    present: list[str] = []
    while neighbour_columns(len(present) + 1)[-1] in manifest.columns:
        present.append(neighbour_columns(len(present) + 1)[-1])

    return present


def with_neighbours(
    manifest: pd.DataFrame, strides: Sequence[int], dense: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Name one neighbour per spacing, so a frame can be differenced at several.

    One spacing has to choose a scale, and the two domains do not agree on
    which. Measured over eight spacings, the source keeps improving as the
    neighbour moves further back while the target turns around between five
    and eight frames: the movement that survives the crossing is narrow, and
    the movement the source rewards is not. Giving the network more than one
    spacing at once is what removes the choice, and it is the second level of
    the difference design this project took the first level from.

    Each spacing is looked up by the same routine that serves a single one, so
    a run asking for one gets exactly the table it always got — the column
    names, their order and their contents are unchanged. That is deliberate:
    every measured cell of the difference family would otherwise stop being
    reproducible.

    A repeated spacing is allowed, and it is not a mistake waiting to happen:
    it is the control this design needs. Adding spacings widens the stem as
    well as adding scales, so a gain confounds the two; the same spacing
    stacked as many times carries no second signal and holds the width fixed,
    which separates them. The spacings a run used are printed in its header,
    so a repeat that was a typo is visible where the result is.

    Args:
        manifest: Rows to annotate, as ``with_neighbour`` takes them.
        strides: How many frames back each neighbour sits.
        dense: Where to look the neighbours up.

    Returns:
        The manifest with one neighbour column per spacing.

    Raises:
        ValueError: If no spacing is named.
    """
    if not strides:
        raise ValueError("no spacings named")

    annotated = manifest
    for column, stride in zip(neighbour_columns(len(strides)), strides, strict=True):
        named = with_neighbour(manifest, stride, dense)
        annotated = annotated.assign(**{column: named[NEIGHBOUR_COLUMN].to_numpy()})

    return annotated


def with_previous_anchor(manifest: pd.DataFrame) -> pd.DataFrame:
    """Name, for every frame, the frame served before it in the same video.

    The cheaper arrangement of the same idea. ``with_neighbour`` reaches into
    the dense pass for a frame at a fixed distance, which is a frame the run
    reads and never trains on; this one differences a frame against the one the
    run already holds, so a decision over K frames reads K frames rather than
    2K. No second pass over storage, and one parameter instead of two: the
    spacing stops being chosen and becomes a consequence of how many frames
    were extracted.

    What it gives up is that the spacing is no longer fixed. It is roughly the
    window divided by the frames stored, so it varies with the length of the
    video and differs between the two domains — measured at a median of four
    frames in the source against six in the real clips. Since the stride was
    measured to matter, that variation is the cost, and comparing the two
    arrangements is the only way to price it.

    A repeated frame number is skipped rather than differenced against itself.
    Extraction writes the same frame twice for a small share of rows, and a
    difference of exactly zero is not a weak signal but a distinctive one: the
    network could read the class off the arithmetic. The first frame of each
    video does point at itself, which is the same convention
    ``with_neighbour`` uses and the honest reading — no earlier frame was
    served, so no movement was seen.

    Args:
        manifest: The rows to be served, after every selection. Requires the
            ``video_id``, ``frame_number`` and ``path`` columns.

    Returns:
        The manifest with a ``neighbour_path`` column added, in its original
        row order.
    """
    ordered = manifest.sort_values(["video_id", "frame_number"], kind="stable")

    # Shifting within a video names the row before; masking the repeats first
    # and filling forward is what makes that row the previous *distinct* one,
    # however many copies sit between them.
    repeated = ordered["frame_number"].eq(
        ordered.groupby("video_id")["frame_number"].shift()
    )
    distinct = ordered["path"].mask(repeated)
    within = distinct.groupby(ordered["video_id"])
    previous = within.shift().groupby(ordered["video_id"]).ffill()

    annotated = manifest.copy()
    annotated["neighbour_path"] = previous.reindex(manifest.index).fillna(
        manifest["path"]
    )
    return annotated


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


def sample_videos(
    manifest: pd.DataFrame,
    fraction: float,
    seed: int = 0,
) -> pd.DataFrame:
    """Keep a fraction of the videos, drawn so the fractions nest.

    Whole videos, never a fraction of each one's frames. Dropping frames leaves
    every video still represented and measures something else entirely — how
    densely a gesture is sampled, not how many gestures were collected. The
    question a fraction is asked here is what a smaller collection effort would
    have bought, and a collection effort yields videos.

    Drawn per class, because the fractions that matter are small: a twentieth of
    two hundred videos is ten, and ten drawn without regard to class can leave a
    gesture with no examples at all, which turns a data-quantity curve into a
    curve about which classes survived the draw. Subject is deliberately not
    stratified on — ten videos cannot cover seven classes across six people, and
    a missing class is fatal to the measurement where a missing person is only
    less variety.

    The draw nests: each class is shuffled once and the fractions take prefixes
    of that order, so everything a run at five percent sees, the same run at ten
    percent sees too. Without that, two points on the curve could differ by
    which videos they happened to get rather than by how many.

    The order depends on the seed, which is this project's label for a
    repetition. So three seeds of one fraction see three different draws, and
    the spread across them includes the luck of the draw — which is the honest
    error bar for a curve about data quantity, and larger than the spread of
    three seeds that differ only in initialisation.

    Every class keeps at least one video however small the fraction, so the
    smallest points hold slightly more data than their name suggests. Callers
    should report the video count they actually got, not the fraction they
    asked for.

    Args:
        manifest: Rows to sample from, one per frame.
        fraction: Share of each class's videos to keep, 0 to 1.
        seed: Chooses the shuffle, so a fraction is reproducible.

    Returns:
        The rows of the videos that were kept, in the manifest's own order.

    Raises:
        ValueError: If the fraction is not in ``(0, 1]``. Zero would hand back
            an empty manifest, which fails later and further from the cause.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1:
        return manifest

    generator = np.random.default_rng(seed)
    kept: list[str] = []
    for label in sorted(manifest["label"].unique()):
        videos = manifest.loc[manifest["label"] == label, "video_id"].unique()
        order = generator.permutation(videos)
        kept.extend(order[: max(1, round(fraction * len(videos)))])

    return manifest[manifest["video_id"].isin(kept)]
