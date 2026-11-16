from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

import scoring
from catalog import find
from evaluation import load_scorer
from frame_cache import VideoSource
from model import build_model, insert_temporal_shift
from readings import stored_table
from scoring import prepare_videos, score, seeds_scored, why_not

CPU = torch.device("cpu")
NAMES = {
    0: "Advance",
    1: "Attention",
    2: "Rally",
    3: "MoveForward",
    4: "Halt",
    5: "FollowMe",
    6: "MoveInReverse",
}


def write_video(path: Path, frames: int = 60, size: int = 32) -> Path:
    """A short lossless video whose frames brighten one step at a time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (size, size)
    )
    for index in range(frames):
        writer.write(np.full((size, size, 3), 4 * index, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def videos(tmp_path):
    """Two clips named the way the dataset names real ones, never seen before."""
    return [
        VideoSource(
            write_video(
                tmp_path / "videos" / f"S0{subject}_10m_ground_label1_start0.avi"
            ),
            "real",
            0,
        )
        for subject in (1, 2)
    ]


def save_network(path: Path, backbone: str, channels: int = 3, shift: bool = False):
    """A network as training leaves it, with weights nobody trained."""
    network = build_model(8, backbone, channels, pretrained=False)
    if shift:
        insert_temporal_shift(network, backbone, segments=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), path)


def test_a_configuration_that_cannot_run_here_says_why(tmp_path):
    checkpoints, predictions = tmp_path / "checkpoints", tmp_path / "predictions"
    frames = find("mobilenet-v3-small-frames")
    prior = find("mobilenet-v3-small-prior")

    assert "environment of its own" in why_not(find("pose"), checkpoints, predictions)
    assert "none of its checkpoints" in why_not(frames, checkpoints, predictions)
    checkpoints.mkdir()
    (checkpoints / f"{frames.stem}_s1.pt").touch()
    (checkpoints / f"{prior.stem}_s1.pt").touch()
    assert why_not(frames, checkpoints, predictions) is None
    assert "real training videos" in why_not(prior, checkpoints, predictions)
    predictions.mkdir()
    stored_table(prior.stem, 1, "real_ground_train", root=predictions).touch()
    assert why_not(prior, checkpoints, predictions) is None


def test_the_frames_of_every_chosen_video_come_back_in_order(tmp_path, videos):
    told = []

    rows, dense = prepare_videos(
        videos,
        NAMES,
        dense=True,
        progress=lambda *step: told.append(step),
        root=tmp_path,
    )

    assert rows["video_id"].unique().tolist() == [source.path.stem for source in videos]
    assert len(rows) == 48
    assert set(dense["video_id"]) == set(rows["video_id"])
    assert told == [("frames", 1, 2), ("frames", 2, 2)]
    assert prepare_videos(videos, NAMES, dense=False, root=tmp_path)[1] is None


def test_every_run_on_this_computer_is_scored_into_a_table(tmp_path, videos):
    entry = find("mobilenet-v3-small-frames")
    for seed in (0, 2):
        save_network(
            tmp_path / "checkpoints" / f"{entry.stem}_s{seed}.pt", entry.network
        )
    rows, _ = prepare_videos(videos, NAMES, dense=False, root=tmp_path)
    told = []

    scored = score(
        entry,
        rows,
        None,
        CPU,
        NAMES,
        progress=lambda *step: told.append(step),
        root=tmp_path,
        checkpoints=tmp_path / "checkpoints",
        num_workers=0,
    )

    assert seeds_scored(scored) == [0, 2]
    table = scored.tables[0]
    assert len(table) == 48
    assert [column for column in table.columns if column.startswith("p_")][
        -1
    ] == "p_Idle"
    assert np.allclose(table.filter(like="p_").sum(axis=1), 1, atol=1e-5)
    assert told == [("run 0", 1, 2), ("run 2", 2, 2)]
    assert not scored.lengths and not scored.marginals


def test_a_shifted_network_is_scored_again_at_each_clip_length(
    tmp_path, videos, monkeypatch
):
    """Rebuilt for every other length, because a frame's scores depend on its clip.

    The trained length is scored once and reused: rebuilding at the length the
    file records would score the same network twice.
    """
    entry = find("mobilenet-v3-small-shift")
    save_network(
        tmp_path / "checkpoints" / f"{entry.stem}_s1.pt", entry.network, shift=True
    )
    rows, _ = prepare_videos(videos, NAMES, dense=False, root=tmp_path)
    rebuilt = []

    def recording(checkpoint, device, clips=None):
        rebuilt.append(clips)
        return load_scorer(checkpoint, device, clips)

    monkeypatch.setattr(scoring, "load_scorer", recording)

    scored = score(
        entry,
        rows,
        None,
        CPU,
        NAMES,
        lengths=(1, 8, 24),
        root=tmp_path,
        checkpoints=tmp_path / "checkpoints",
        num_workers=0,
    )

    assert rebuilt == [None, 1, 24]
    by_length = scored.lengths[1]
    assert sorted(by_length) == [1, 8, 24]
    assert by_length[8] is scored.tables[1]
    assert all(len(table) == 48 for table in by_length.values())


def test_a_corrected_configuration_carries_the_marginal_stored_for_it(tmp_path, videos):
    entry = find("mobilenet-v3-small-prior")
    save_network(
        tmp_path / "checkpoints" / f"{entry.stem}_s0.pt", entry.network, channels=12
    )
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    stored = pd.DataFrame(
        {
            "video_id": ["v"] * 2,
            "frame_number": [1, 2],
            "position": [0.25, 0.75],
            "label": [0, 0],
            **{f"p_{name}": [1 / 7, 1 / 7] for name in NAMES.values()},
        }
    )
    stored["p_Advance"] = [0.4, 0.2]
    stored.to_csv(
        stored_table(entry.stem, 0, "real_ground_train", root=predictions), index=False
    )
    rows, dense = prepare_videos(videos, NAMES, dense=True, root=tmp_path)

    scored = score(
        entry,
        rows,
        dense,
        CPU,
        NAMES,
        root=tmp_path,
        checkpoints=tmp_path / "checkpoints",
        predictions=predictions,
        num_workers=0,
    )

    assert scored.marginals[0][0] == pytest.approx(0.3)
    assert len(scored.tables[0]) == 48
