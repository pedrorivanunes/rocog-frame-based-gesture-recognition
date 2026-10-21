import numpy as np
import pandas as pd
import pytest

from pose_benchmark import classifier_call, detector_call, middle_frames, parse_args


def _manifest(videos: dict[str, int]) -> pd.DataFrame:
    """Stored frames numbered within each video, in the order a manifest lists them."""
    rows = [
        {"video_id": video, "path": f"{video}_f{index:02d}.jpg"}
        for video, count in videos.items()
        for index in range(count)
    ]
    return pd.DataFrame(rows)


class _Detector:
    """Records the frames it was handed, as the detector's process would be."""

    def __init__(self) -> None:
        self.seen: list[int] = []

    def process(self, frame: np.ndarray) -> None:
        self.seen.append(int(frame[0, 0, 0]))


def test_the_sample_takes_the_middle_frame_of_each_video():
    """The ends of a window are where the body rests, not where a signal is read."""
    picked = middle_frames(_manifest({"a": 24, "b": 24}), count=2)

    assert picked == ["a_f12.jpg", "b_f12.jpg"]


def test_the_sample_keeps_manifest_order_and_stops_at_the_count():
    picked = middle_frames(_manifest({"c": 4, "a": 4, "b": 4}), count=2)

    assert picked == ["c_f02.jpg", "a_f02.jpg"]


def test_a_decision_runs_the_detector_once_per_frame_and_moves_on():
    """K frames are K different frames, not the same one K times."""
    frames = [np.full((4, 4, 3), value, dtype=np.uint8) for value in range(3)]
    detector = _Detector()

    call = detector_call(detector, frames, 2)
    call()
    call()

    # Converted to RGB on the way in, which leaves a grey value where it was.
    assert detector.seen == [0, 1, 2, 0]


def test_the_classifier_decides_and_reports_its_size():
    call, parameters = classifier_call(8)

    call()

    assert parameters > 0


def test_only_one_core_or_all_of_them_can_be_asked_for():
    """Anything between is not what the image rows were measured at."""
    with pytest.raises(SystemExit):
        parse_args(["--part", "detector", "--threads", "2", "--output", "x.csv"])
