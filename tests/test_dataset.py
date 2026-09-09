import numpy as np
import pandas as pd
import pytest
import torch

from dataset import (
    TEXTURE_FILLS,
    Augmentation,
    BackgroundRandomiser,
    CombinedSampler,
    RandomGamma,
    SegmentSampler,
    TextureRandomiser,
    noise_background,
    solid_background,
    train_transform,
)
from manifest import IDLE_LABEL

FRAMES_STORED = 24


def frame_manifest(videos, frames_per_video=FRAMES_STORED):
    """Build a manifest with consecutive rows per video, as extraction writes it.

    Only ``video_id`` is read by the sampler, and no file is opened: what it
    yields are row positions, so the frames behind them never have to exist.

    Args:
        videos: How many videos the manifest holds.
        frames_per_video: Rows each video contributes, in frame order.

    Returns:
        One row per frame, videos in contiguous blocks.
    """
    return pd.DataFrame(
        {
            "video_id": [
                f"video{v}" for v in range(videos) for _ in range(frames_per_video)
            ],
            "frame_number": list(range(frames_per_video)) * videos,
        }
    )


def blocks_of(manifest, video, frames_per_video):
    """Return the row positions of one video, grouped as the sampler groups them."""
    positions = manifest.reset_index(drop=True).groupby("video_id", sort=False).indices
    return positions[video].reshape(frames_per_video, -1)


def test_one_frame_is_drawn_from_every_block():
    """Frames clustering in one stretch would leave the rest of the gesture unseen."""
    manifest = frame_manifest(videos=5)

    drawn = set(SegmentSampler(manifest, frames_per_video=8))

    for video in range(5):
        for block in blocks_of(manifest, f"video{video}", 8):
            assert len(drawn & set(block)) == 1


def test_an_epoch_draws_the_requested_count():
    manifest = frame_manifest(videos=5)

    sampler = SegmentSampler(manifest, frames_per_video=8)

    assert len(sampler) == 40
    assert len(list(sampler)) == 40


def test_frames_never_repeat_within_an_epoch():
    manifest = frame_manifest(videos=5)

    drawn = list(SegmentSampler(manifest, frames_per_video=8))

    assert len(set(drawn)) == len(drawn)


def test_successive_epochs_draw_differently():
    """Redrawing every epoch is what keeps the subset from losing information."""
    manifest = frame_manifest(videos=50)

    sampler = SegmentSampler(manifest, frames_per_video=8)

    assert set(sampler) != set(sampler)


def test_a_count_that_does_not_divide_the_stored_frames_is_rejected():
    """Uneven blocks would silently weight one part of the gesture over another."""
    manifest = frame_manifest(videos=5)

    with pytest.raises(RuntimeError):
        SegmentSampler(manifest, frames_per_video=7)


def test_videos_of_differing_length_are_rejected():
    manifest = pd.concat(
        [frame_manifest(videos=1), frame_manifest(videos=1, frames_per_video=12)]
    )

    with pytest.raises(RuntimeError):
        SegmentSampler(manifest, frames_per_video=8)


def silhouette_and_frame(size: int = 8):
    """A frame of one colour with a square person in the middle."""
    frame = np.full((size, size, 3), 200, dtype=np.uint8)
    silhouette = np.zeros((size, size), dtype=bool)
    silhouette[2:6, 2:6] = True
    frame[silhouette] = 50
    return frame, silhouette


def test_background_randomiser_leaves_the_frame_alone_at_probability_zero():
    """Zero is how a run turns this off without a second code path."""
    frame, silhouette = silhouette_and_frame()

    composited = BackgroundRandomiser(probability=0.0)(frame, silhouette)

    assert np.array_equal(composited, frame)


def test_background_randomiser_keeps_the_person_pixel_for_pixel():
    """Everything inside the silhouette has to survive untouched."""
    frame, silhouette = silhouette_and_frame()

    composited = BackgroundRandomiser(probability=1.0)(frame, silhouette)

    assert np.array_equal(composited[silhouette], frame[silhouette])


def test_background_randomiser_replaces_everything_outside_the_person():
    frame, silhouette = silhouette_and_frame()

    composited = BackgroundRandomiser(probability=1.0, kinds=("solid",))(
        frame, silhouette
    )

    assert not np.array_equal(composited[~silhouette], frame[~silhouette])
    assert len(np.unique(composited[~silhouette].reshape(-1, 3), axis=0)) == 1


def test_background_randomiser_draws_a_new_background_each_call():
    """A background fixed across calls would be a texture to memorise."""
    frame, silhouette = silhouette_and_frame()
    randomiser = BackgroundRandomiser(probability=1.0, kinds=("solid",))

    backgrounds = {randomiser(frame, silhouette)[0, 0].tobytes() for _ in range(20)}

    assert len(backgrounds) > 1


def test_background_randomiser_honours_its_probability():
    frame, silhouette = silhouette_and_frame()
    randomiser = BackgroundRandomiser(probability=0.5, kinds=("solid",))
    generator = torch.Generator().manual_seed(0)

    swapped = sum(
        not np.array_equal(randomiser(frame, silhouette, generator), frame)
        for _ in range(400)
    )

    assert 150 < swapped < 250


def test_background_randomiser_reproduces_from_a_generator():
    frame, silhouette = silhouette_and_frame()
    randomiser = BackgroundRandomiser(probability=1.0)

    first = randomiser(frame, silhouette, torch.Generator().manual_seed(7))
    second = randomiser(frame, silhouette, torch.Generator().manual_seed(7))

    assert np.array_equal(first, second)


def test_background_randomiser_rejects_a_mismatched_silhouette():
    """A silhouette of the wrong size would cut the person somewhere else."""
    frame, _ = silhouette_and_frame(size=8)

    with pytest.raises(ValueError, match="silhouette is"):
        BackgroundRandomiser(probability=1.0)(frame, np.zeros((4, 4), dtype=bool))


def test_background_randomiser_rejects_an_impossible_probability():
    with pytest.raises(ValueError, match="between 0 and 1"):
        BackgroundRandomiser(probability=1.5)


def test_background_randomiser_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown background kinds"):
        BackgroundRandomiser(kinds=("solid", "chequerboard"))


def test_noise_background_is_not_one_colour():
    assert len(np.unique(noise_background((16, 16)).reshape(-1, 3), axis=0)) > 1


def test_solid_background_is_one_colour():
    assert len(np.unique(solid_background((16, 16)).reshape(-1, 3), axis=0)) == 1


def test_backgrounds_match_the_frame_they_replace():
    for build in (solid_background, noise_background):
        assert build((11, 13)).shape == (11, 13, 3)
        assert build((11, 13)).dtype == np.uint8


def flat_gradient():
    """A frame with a known, modest spread, standing in for a rendered one."""
    ramp = torch.linspace(96, 160, 64, dtype=torch.float32)

    return ramp.expand(3, 64, 64).round().to(torch.uint8)


def dark_share(image):
    """The measured gap between the domains: how much of the frame is in shadow."""
    return (image.to(torch.float32).mean(dim=0) < 96).float().mean().item()


def test_gamma_creates_the_dark_population_the_renders_lack():
    """The whole point: renders carry almost no shadow, and gamma supplies it."""
    frame = flat_gradient()

    darkened = RandomGamma((1.6, 2.4))(frame)

    assert dark_share(darkened) > dark_share(frame)


def test_gamma_darkens_on_every_draw():
    """A symmetric jitter would brighten about half the time; this must not."""
    frame = flat_gradient()
    shift = RandomGamma((1.6, 2.4))

    assert all(
        shift(frame).to(torch.float32).mean() < frame.to(torch.float32).mean()
        for _ in range(50)
    )


def test_gamma_never_pushes_a_pixel_past_white():
    """Why gamma and not a contrast stretch: the top of the scale cannot clip."""
    bright = torch.full((3, 32, 32), 250, dtype=torch.uint8)

    assert RandomGamma((1.0, 3.0))(bright).max().item() <= 250


def test_a_gamma_range_of_one_leaves_the_frame_alone():
    frame = flat_gradient()

    assert torch.equal(RandomGamma((1.0, 1.0))(frame), frame)


def test_gamma_draws_a_new_exponent_each_call():
    """One value reused for the whole epoch would be a fixed edit, not augmentation."""
    frame = flat_gradient()
    shift = RandomGamma((1.0, 3.0))

    means = {round(shift(frame).to(torch.float32).mean().item(), 4) for _ in range(20)}

    assert len(means) > 1


def test_gamma_follows_torch_seeding():
    """The loader seeds torch per worker; drawing elsewhere would ignore that."""
    frame = flat_gradient()
    shift = RandomGamma((1.0, 3.0))

    torch.manual_seed(0)
    first = [shift(frame).to(torch.float32).mean().item() for _ in range(5)]
    torch.manual_seed(0)
    second = [shift(frame).to(torch.float32).mean().item() for _ in range(5)]

    assert first == second


@pytest.mark.parametrize("bad", [(0.0, 1.5), (-1.0, 1.5), (2.2, 1.5)])
def test_gamma_rejects_an_impossible_range(bad):
    with pytest.raises(ValueError):
        RandomGamma(bad)


def test_train_transform_adds_the_shift_only_when_asked():
    """Omitting it has to reproduce the pipeline every earlier run used."""
    without = train_transform(Augmentation(photometric=True, geometric=True))
    with_shift = train_transform(
        Augmentation(photometric=True, geometric=True, gamma_shift=(1.6, 2.4))
    )

    kinds = [type(step).__name__ for step in with_shift.transforms]

    assert "RandomGamma" not in [type(step).__name__ for step in without.transforms]
    assert kinds.count("RandomGamma") == 1
    assert kinds.index("RandomGamma") > kinds.index("ColorJitter")


def test_the_shift_runs_before_normalising():
    """Gamma after normalisation would act on a signed scale, not on grey levels."""
    kinds = [
        type(step).__name__
        for step in train_transform(Augmentation(gamma_shift=(1.6, 2.4))).transforms
    ]

    assert kinds.index("RandomGamma") < kinds.index("Normalize")


def test_texture_randomiser_leaves_the_frame_alone_at_probability_zero():
    """Zero is how a run turns this off without a second code path."""
    frame, silhouette = silhouette_and_frame()

    touched = TextureRandomiser("blend", probability=0.0)(frame, silhouette)

    assert np.array_equal(touched, frame)


def test_texture_randomiser_keeps_the_scene_pixel_for_pixel():
    """The person's modes must not touch the background — that is the other knob."""
    frame, silhouette = silhouette_and_frame()

    touched = TextureRandomiser("replace", probability=1.0)(frame, silhouette)

    assert np.array_equal(touched[~silhouette], frame[~silhouette])


def test_replace_leaves_the_person_one_flat_colour():
    """Going all the way is what makes this cell shape and nothing else."""
    frame, silhouette = silhouette_and_frame()

    touched = TextureRandomiser("replace", probability=1.0, kinds=("solid",))(
        frame, silhouette
    )

    assert len(np.unique(touched[silhouette].reshape(-1, 3), axis=0)) == 1
    assert not np.array_equal(touched[silhouette], frame[silhouette])


def test_blend_keeps_the_person_between_the_frame_and_the_fill():
    """A partial mixture has to stay a mixture, not jump to either end."""
    frame, silhouette = silhouette_and_frame()
    generator = torch.Generator().manual_seed(4)

    touched = TextureRandomiser("blend", probability=1.0, kinds=("solid",))(
        frame, silhouette, generator
    )
    person = touched[silhouette].reshape(-1, 3)

    # One fill colour and one original colour, so every mixed pixel is equal and
    # sits on the segment between them.
    assert len(np.unique(person, axis=0)) == 1
    assert not np.array_equal(person[0], frame[silhouette][0])


def test_everywhere_touches_the_scene_as_well_as_the_person():
    """The control has to be unrestricted, or it is not controlling for anything."""
    frame, silhouette = silhouette_and_frame()

    touched = TextureRandomiser("everywhere", probability=1.0, kinds=("solid",))(
        frame, silhouette
    )

    assert not np.array_equal(touched[~silhouette], frame[~silhouette])
    assert not np.array_equal(touched[silhouette], frame[silhouette])


def test_texture_randomiser_draws_a_new_fill_each_call():
    """A fill reused across frames would be a constant, not a randomisation."""
    frame, silhouette = silhouette_and_frame()
    randomiser = TextureRandomiser("replace", probability=1.0, kinds=("solid",))

    first = randomiser(frame, silhouette)
    second = randomiser(frame, silhouette)

    assert not np.array_equal(first, second)


def test_texture_randomiser_reproduces_from_a_generator():
    """Two workers handed the same generator must produce the same frame."""
    frame, silhouette = silhouette_and_frame()
    randomiser = TextureRandomiser("blend", probability=1.0)

    first = randomiser(frame, silhouette, torch.Generator().manual_seed(11))
    second = randomiser(frame, silhouette, torch.Generator().manual_seed(11))

    assert np.array_equal(first, second)


def test_texture_randomiser_honours_its_probability():
    """Half means half; a mode that always fired would be a different cell."""
    frame, silhouette = silhouette_and_frame()
    randomiser = TextureRandomiser("replace", probability=0.5, kinds=("solid",))
    generator = torch.Generator().manual_seed(0)

    touched = sum(
        not np.array_equal(randomiser(frame, silhouette, generator), frame)
        for _ in range(200)
    )

    assert 70 < touched < 130


def test_texture_randomiser_rejects_a_mismatched_silhouette():
    """A silhouette of the wrong size would mask the wrong pixels in silence."""
    frame, _ = silhouette_and_frame()

    with pytest.raises(ValueError, match="silhouette is"):
        TextureRandomiser("blend", probability=1.0)(frame, np.zeros((4, 4), dtype=bool))


def test_texture_randomiser_rejects_a_mode_it_does_not_know():
    """A typo would land in the sweep as a cell that trained on plain frames."""
    with pytest.raises(ValueError, match="is not one of"):
        TextureRandomiser("stylise")


def test_none_is_not_a_mode_this_builds():
    """Off is the absence of a randomiser, not one that passes frames through."""
    with pytest.raises(ValueError, match="is not one of"):
        TextureRandomiser("none")


def test_every_fill_names_known_generators():
    """A fill naming a generator that does not exist would fail mid-epoch."""
    for kinds in TEXTURE_FILLS.values():
        assert TextureRandomiser("blend", kinds=kinds).kinds == kinds


def test_a_single_fill_uses_only_that_generator():
    """Splitting the two is the whole point of naming one."""
    frame, silhouette = silhouette_and_frame()
    randomiser = TextureRandomiser(
        "replace", probability=1.0, kinds=TEXTURE_FILLS["solid"]
    )

    for _ in range(20):
        person = randomiser(frame, silhouette)[silhouette].reshape(-1, 3)
        assert len(np.unique(person, axis=0)) == 1


def mixed_manifest(videos, gesture_frames=16, idle_frames=4):
    """Build a manifest holding both kinds of row, as training combines them.

    Gesture rows come first for every video, then the idle rows, which is the
    order the two manifests are concatenated in.
    """
    gesture = frame_manifest(videos, gesture_frames)
    gesture["label"] = 0
    idle = frame_manifest(videos, idle_frames)
    idle["label"] = IDLE_LABEL

    return pd.concat([gesture, idle], ignore_index=True)


def test_a_sampler_over_part_of_a_frame_names_positions_in_the_whole_frame():
    """The dataset serves the whole frame, so a slice's own numbering is wrong."""
    manifest = mixed_manifest(videos=3)

    drawn = list(SegmentSampler(manifest, 1, rows=manifest["label"] == IDLE_LABEL))

    assert all(manifest.iloc[position]["label"] == IDLE_LABEL for position in drawn)
    assert min(drawn) >= 3 * 16


def test_a_sampler_over_part_of_a_frame_draws_one_per_block():
    manifest = mixed_manifest(videos=3)

    drawn = list(SegmentSampler(manifest, 2, rows=manifest["label"] == IDLE_LABEL))

    assert len(drawn) == 6
    assert len(set(drawn)) == 6


def test_naming_every_row_draws_what_naming_none_draws():
    """The selection has to be an addition, not a change to what was there."""
    manifest = frame_manifest(videos=3)

    without = list(SegmentSampler(manifest, 8, seed=1))
    with_all = list(
        SegmentSampler(manifest, 8, seed=1, rows=np.ones(len(manifest), bool))
    )

    assert without == with_all


def test_the_two_kinds_of_row_are_counted_apart():
    """Counted together, twenty rows a video would refuse to split into eight."""
    manifest = mixed_manifest(videos=3)
    is_gesture = manifest["label"] != IDLE_LABEL

    gestures = SegmentSampler(manifest, 8, rows=is_gesture)
    idle = SegmentSampler(manifest, 1, rows=~is_gesture)

    assert len(gestures) == 24
    assert len(idle) == 3


def test_a_combined_epoch_holds_every_sampler_it_was_given():
    manifest = mixed_manifest(videos=3)
    is_gesture = manifest["label"] != IDLE_LABEL

    combined = CombinedSampler(
        [
            SegmentSampler(manifest, 8, rows=is_gesture),
            SegmentSampler(manifest, 1, rows=~is_gesture),
        ]
    )

    drawn = list(combined)
    assert len(combined) == 27
    assert len(drawn) == 27
    assert (
        sum(manifest.iloc[position]["label"] == IDLE_LABEL for position in drawn) == 3
    )


def test_a_combined_epoch_mixes_the_kinds_rather_than_stacking_them():
    """Stacked, the last batches would hold nothing but bodies at rest."""
    manifest = mixed_manifest(videos=20)
    is_gesture = manifest["label"] != IDLE_LABEL

    drawn = list(
        CombinedSampler(
            [
                SegmentSampler(manifest, 8, rows=is_gesture),
                SegmentSampler(manifest, 1, rows=~is_gesture),
            ]
        )
    )

    kinds = [manifest.iloc[position]["label"] == IDLE_LABEL for position in drawn]
    assert any(kinds[: len(kinds) // 2])


def test_a_combined_epoch_draws_again_every_time():
    """Extraction stored more than an epoch uses so the draw could move."""
    manifest = mixed_manifest(videos=20)
    is_gesture = manifest["label"] != IDLE_LABEL

    combined = CombinedSampler(
        [
            SegmentSampler(manifest, 8, rows=is_gesture),
            SegmentSampler(manifest, 1, rows=~is_gesture),
        ]
    )

    assert list(combined) != list(combined)


def test_a_treatment_is_off_until_a_run_asks_for_it():
    """An option that arrived on would make every run before it incomparable."""
    standard = Augmentation()

    assert standard.background == 0.0
    assert standard.texture == "none"
    assert standard.gamma_shift is None
