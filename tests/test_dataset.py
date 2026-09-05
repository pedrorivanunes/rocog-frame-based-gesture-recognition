import numpy as np
import pandas as pd
import pytest
import torch

from dataset import (
    BackgroundRandomiser,
    RandomGamma,
    SegmentSampler,
    noise_background,
    solid_background,
    train_transform,
)

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
    without = train_transform(photometric=True, geometric=True)
    with_shift = train_transform(
        photometric=True, geometric=True, gamma_shift=(1.6, 2.4)
    )

    kinds = [type(step).__name__ for step in with_shift.transforms]

    assert "RandomGamma" not in [type(step).__name__ for step in without.transforms]
    assert kinds.count("RandomGamma") == 1
    assert kinds.index("RandomGamma") > kinds.index("ColorJitter")


def test_the_shift_runs_before_normalising():
    """Gamma after normalisation would act on a signed scale, not on grey levels."""
    kinds = [
        type(step).__name__
        for step in train_transform(gamma_shift=(1.6, 2.4)).transforms
    ]

    assert kinds.index("RandomGamma") < kinds.index("Normalize")
