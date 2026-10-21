import numpy as np
import pandas as pd
import pytest

from flow_benchmark import available, centre_crop, flow_call, frame_pairs, main


def _manifests(stored: dict[str, list[int]], every: int) -> tuple:
    """A stored manifest and its dense twin, every frame of each video in the latter."""
    sparse = pd.DataFrame(
        [
            {"video_id": video, "frame_number": number, "path": f"{video}_{number:03d}"}
            for video, numbers in stored.items()
            for number in numbers
        ]
    )
    dense = pd.DataFrame(
        [
            {"video_id": video, "frame_number": number, "path": f"{video}_{number:03d}"}
            for video in stored
            for number in range(every)
        ]
    )
    return sparse, dense


def test_a_frame_is_paired_with_the_one_five_frames_before_it():
    """The spacing the frame difference uses, so the flow spans the same gap."""
    sparse, dense = _manifests({"a": [10, 20, 30]}, every=40)

    assert frame_pairs(sparse, dense) == [("a_015", "a_020")]


def test_a_frame_near_the_start_is_paired_with_one_ahead_instead():
    sparse, dense = _manifests({"a": [1, 2, 3]}, every=40)

    assert frame_pairs(sparse, dense) == [("a_002", "a_007")]


def test_the_crop_is_the_square_at_the_centre():
    frame = np.arange(8 * 8, dtype=np.uint8).reshape(8, 8)

    cropped = centre_crop(frame, side=4)

    assert cropped.shape == (4, 4)
    assert cropped[0, 0] == frame[2, 2]


def test_the_classical_method_shipped_with_opencv_is_always_there():
    assert available("farneback")


def test_a_classical_method_runs_on_a_pair_and_carries_no_parameters():
    pairs = [
        tuple(
            np.random.default_rng(seed).integers(0, 256, (32, 32, 3), np.uint8)
            for seed in (2 * index, 2 * index + 1)
        )
        for index in range(2)
    ]

    call, parameters = flow_call("farneback", pairs, "cpu")
    call()
    call()

    assert parameters == ""


def test_a_classical_method_is_not_timed_on_a_gpu():
    """It would run on the CPU and be labelled as a GPU row."""
    with pytest.raises(SystemExit, match="CPU only"):
        main(["--method", "farneback", "--device", "cuda", "--output", "x.csv"])
