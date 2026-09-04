"""Partition a manifest into the subsets a run trains and evaluates on.

Every experiment in this project is a selection: a camera viewpoint, a fraction
of real data, a validation split. The manifest is the table those selections are
made on, and keeping them here means the data a run saw can be read from the
code that built it, rather than inferred from a file name.
"""

import numpy as np
import pandas as pd

SPLIT_SEED = 7


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
