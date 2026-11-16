from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from catalog import find
from frame_cache import EXTERNAL, VideoSource
from readings import GESTURES, stored_table
from rocog import (
    Selection,
    all_tested,
    ask_configuration,
    ask_videos,
    decisions_by_video,
    evaluate,
    figures_rows,
    from_file,
    k_rows,
    of_gesture,
    of_subject,
    parse_args,
    picked,
    report,
    save,
    selection_from,
)

NAMES = {
    0: "Advance",
    1: "Attention",
    2: "Rally",
    3: "MoveForward",
    4: "Halt",
    5: "FollowMe",
    6: "MoveInReverse",
}
ORDERED = [NAMES[label] for label in sorted(NAMES)]
DOWN, ENTER = "\x1b[B", "\r"


def source(subject: str, label: int, start: int) -> VideoSource:
    """A real test video as the dataset names it; it is never opened."""
    name = f"{subject}_10m_ground_label{label + 1}_start{start}.mp4"
    return VideoSource(Path("data/real/ground") / name, "real", label)


SOURCES = [
    source("S04", 4, 1),
    source("S04", 0, 2),
    source("S06", 4, 3),
    source("S09", 1, 4),
]


def table(
    videos: list[str],
    labels: list[int],
    answers: list[int],
    frames: int = 4,
    idle: bool = True,
):
    """A table as scoring writes it, every frame of a video naming its answer."""
    rows = []
    for video, label, answer in zip(videos, labels, answers, strict=True):
        for place in range(frames):
            probabilities = np.full(GESTURES + int(idle), 0.02)
            probabilities[answer] = 1 - probabilities.sum() + 0.02
            rows.append(
                {
                    "video_id": video,
                    "frame_number": 3 * place,
                    "position": (place + 0.5) / frames,
                    "label": label,
                    **{
                        f"p_{name}": probabilities[index]
                        for index, name in enumerate(
                            [*ORDERED, "Idle"][: len(probabilities)]
                        )
                    },
                }
            )
    return pd.DataFrame(rows)


def test_the_test_videos_can_be_cut_by_gesture_and_by_subject():
    halt = of_gesture(SOURCES, 4, "Halt")
    subject = of_subject(SOURCES, "S04")

    assert [chosen.label for chosen in halt.sources] == [4, 4]
    assert halt.label == "the 2 real test videos of Halt"
    assert halt.slug == "real_ground_test_Halt"
    assert len(subject.sources) == 2
    assert subject.slug == "real_ground_test_S04"
    assert all_tested(SOURCES).label == "all 4 real test videos"
    assert picked(SOURCES[:1]).label == "1 real test video"


def test_a_file_from_outside_the_dataset_has_no_label_and_no_thesis(tmp_path):
    clip = tmp_path / "mine.mov"
    clip.touch()

    chosen = from_file(clip)

    assert chosen.sources == [VideoSource(clip, EXTERNAL)]
    assert not chosen.labelled and not chosen.tested


def test_the_videos_named_on_the_command_line(tmp_path):
    assert (
        selection_from("gesture:halt", NAMES, SOURCES).slug == "real_ground_test_Halt"
    )
    assert selection_from("subject:S09", NAMES, SOURCES).sources == [SOURCES[3]]
    assert selection_from("test", NAMES, SOURCES).sources == SOURCES
    assert (
        selection_from(f"file:{tmp_path / 'a.mp4'}", NAMES, SOURCES).slug == "external"
    )
    with pytest.raises(SystemExit, match="no gesture"):
        selection_from("gesture:wave", NAMES, SOURCES)
    with pytest.raises(SystemExit, match="expected test"):
        selection_from("everything", NAMES, SOURCES)


def test_with_no_configuration_named_the_menu_asks():
    assert parse_args([]).configuration is None
    assert parse_args(["--configuration", "resnet18-shift"]).configuration == (
        "resnet18-shift"
    )


def write_stored(root: Path, stem: str, seed: int, frame: pd.DataFrame, **name):
    path = stored_table(stem, seed, root=root, **name)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_the_thesis_is_read_on_the_same_videos_as_the_run(tmp_path):
    entry = find("resnet18-frames")
    chosen = Selection("two", "slug", [source("S04", 0, 1), source("S04", 1, 2)])
    videos = [item.path.stem for item in chosen.sources]
    now = table(videos, [0, 1], [0, 1])
    # The stored table holds a third video the run did not choose, decided wrong.
    stored = table([*videos, "S06_10m_ground_label1_start9"], [0, 1, 0], [0, 1, 3])
    for seed in (0, 1):
        write_stored(tmp_path, entry.stem, seed, stored)

    outcome = evaluate(entry, chosen, {0: now, 1: now}, predictions=tmp_path)

    assert outcome.now.video == (100.0, 0.0)
    assert outcome.thesis.video == (100.0, 0.0)
    assert outcome.agreement == (16, 16, pytest.approx(0.0))
    assert [row[0] for row in figures_rows(outcome)] == [
        "per frame (%)",
        "per video (%)",
        "true gesture in the top 3 (%)",
    ]


def test_a_run_with_no_stored_tables_has_no_thesis_to_compare(tmp_path):
    entry = find("resnet18-frames")
    chosen = Selection("one", "slug", [source("S04", 0, 1)])
    now = table([chosen.sources[0].path.stem], [0], [0])

    outcome = evaluate(entry, chosen, {0: now}, predictions=tmp_path)

    assert outcome.now is not None
    assert outcome.thesis is None and outcome.agreement is None
    assert figures_rows(outcome)[0] == ("per frame (%)", "100.00", "—")


def test_a_shifted_network_is_shown_per_decision_and_by_clip_length(tmp_path):
    entry = find("resnet18-shift")
    chosen = Selection("one", "slug", [source("S04", 2, 1)])
    video = chosen.sources[0].path.stem
    lengths = {length: table([video], [2], [2], frames=8) for length in (1, 2, 4, 8)}
    lengths[1] = table([video], [2], [5], frames=8)

    outcome = evaluate(
        entry, chosen, {0: lengths[8]}, lengths={0: lengths}, predictions=tmp_path
    )

    labels = [row[0] for row in figures_rows(outcome)]
    assert "per decision on 8 frames (%)" in labels
    header, rows = k_rows(outcome)
    assert header[:3] == ["K", "accuracy now (%)", "in the thesis (%)"]
    assert [row[:2] for row in rows] == [
        ["1", "0.00"],
        ["2", "100.00"],
        ["4", "100.00"],
        ["8", "100.00"],
    ]


def test_a_corrected_configuration_shows_the_figure_before_the_correction(tmp_path):
    entry = find("resnet18-prior")
    chosen = Selection("one", "slug", [source("S04", 1, 1)])
    now = table([chosen.sources[0].path.stem], [1], [1])
    marginal = np.full(GESTURES, 1 / GESTURES)

    outcome = evaluate(
        entry, chosen, {0: now}, marginals={0: marginal}, predictions=tmp_path
    )

    labels = [row[0] for row in figures_rows(outcome)]
    assert labels[1:3] == [
        "per video, corrected (%)",
        "per video, before the correction (%)",
    ]


def test_the_cost_by_k_comes_from_the_session_and_says_when_it_is_missing(tmp_path):
    entry = find("resnet18-frames")
    chosen = Selection("one", "slug", [source("S04", 0, 1)])
    now = table([chosen.sources[0].path.stem], [0], [0])
    pd.DataFrame(
        [
            {"model": "resnet18", "level": "decision", "frames": 1, "median_ms": 30.0},
            {"model": "resnet18", "level": "decision", "frames": 4, "median_ms": 120.0},
        ]
    ).to_csv(tmp_path / "latency_catalog_cpu_1t_isolated.csv", index=False)

    outcome = evaluate(
        entry, chosen, {0: now}, predictions=tmp_path, benchmarks=tmp_path
    )

    header, rows = k_rows(outcome)
    assert header[-1] == "1 CPU thread (ms)"
    assert rows[0][-1] == "30.0"
    assert rows[1][-1] == "—"  # K = 2 was never timed
    assert rows[2][-1] == "120.0"


def test_an_unlabelled_video_gets_each_run_s_decision_and_how_sure_it_was(tmp_path):
    entry = find("resnet18-frames")
    clip = tmp_path / "mine.mp4"
    clip.touch()
    chosen = from_file(clip)
    now = table(["mine"], [-1], [4])

    outcome = evaluate(entry, chosen, {0: now, 2: now}, predictions=tmp_path)

    assert outcome.now is None
    ((video, truth, said, sure),) = decisions_by_video(outcome, ORDERED)
    assert (video, truth, said) == ("mine", "?", ["Halt", "Halt"])
    # 0.86 on the gesture named, 0.02 on each of the other six.
    assert sure[0] == pytest.approx(0.86 / 0.98)


def test_the_report_prints_the_comparison_with_the_thesis(tmp_path):
    entry = find("resnet18-frames")
    chosen = Selection("one", "slug", [source("S04", 0, 1)])
    now = table([chosen.sources[0].path.stem], [0], [0])
    write_stored(tmp_path, entry.stem, 0, now)
    outcome = evaluate(
        entry, chosen, {0: now}, predictions=tmp_path, benchmarks=tmp_path
    )
    console = Console(record=True, width=120)

    report(console, outcome, ORDERED)

    text = console.export_text()
    assert "How it did" in text
    assert "same answer as the thesis on 4 of 4 frames" in text
    assert "latency: not measured yet" in text


def test_a_run_is_kept_under_a_folder_of_its_own(tmp_path):
    entry = find("resnet18-shift")
    chosen = Selection("one", "real_ground_test_S04", [source("S04", 0, 1)])
    video = chosen.sources[0].path.stem
    by_length = {length: table([video], [0], [0], frames=8) for length in (1, 8)}
    outcome = evaluate(
        entry, chosen, {0: by_length[8]}, lengths={0: by_length}, predictions=tmp_path
    )

    folder = save(outcome, "the report", tmp_path, when=datetime(2026, 10, 23, 9, 5, 7))

    assert folder.name == "2026-10-23_090507_resnet18-shift"
    assert sorted(path.name for path in folder.iterdir()) == [
        "report.txt",
        "tsm_ordered_rerun_s0__real_ground_test_S04.csv",
        "tsm_ordered_rerun_s0__real_ground_test_S04_clips1.csv",
    ]
    with pytest.raises(FileExistsError):
        save(outcome, "again", tmp_path, when=datetime(2026, 10, 23, 9, 5, 7))


def test_enter_runs_the_first_configuration_this_computer_can_run():
    with create_pipe_input() as typed:
        typed.send_text(ENTER)
        chosen = ask_configuration(lambda _: None, input=typed, output=DummyOutput())
    assert chosen.key == "resnet18-frames"

    with create_pipe_input() as typed:
        typed.send_text(ENTER)
        chosen = ask_configuration(
            lambda entry: "absent" if entry.key == "resnet18-frames" else None,
            input=typed,
            output=DummyOutput(),
        )
    assert chosen.key == "resnet18-difference"


def test_the_videos_of_one_gesture_are_two_choices_away():
    with create_pipe_input() as typed:
        typed.send_text(DOWN + ENTER + DOWN * 4 + ENTER)
        chosen = ask_videos(NAMES, SOURCES, input=typed, output=DummyOutput())

    assert chosen.slug == "real_ground_test_Halt"
    assert len(chosen.sources) == 2
