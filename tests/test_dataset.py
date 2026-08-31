import pandas as pd
import pytest

from dataset import SegmentSampler

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
