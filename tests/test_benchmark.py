from pathlib import Path

import pytest
import torch
from torch import nn

from benchmark import (
    NUM_CLASSES,
    SHIFT_SEGMENTS,
    STORED_SIZE,
    Entry,
    decision_call,
    forward_input,
    parameters_of,
    parse_args,
    rows_for,
    shorten,
    stored_frames,
    time_calls,
)

CPU = torch.device("cpu")


class _Counter(nn.Module):
    """Records the shape it was handed, and answers with fixed logits."""

    def __init__(self, classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.classes = classes
        self.seen: list[tuple[int, ...]] = []
        self.weight = nn.Parameter(torch.zeros(3))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        self.seen.append(tuple(batch.shape))
        return torch.zeros(batch.shape[0], self.classes)


def _frame_entry() -> Entry:
    return Entry("stub_frame", _Counter, 224, None, "test")


def _clip_entry() -> Entry:
    return Entry("stub_clip", _Counter, 224, 16, "test")


def _difference_entry() -> Entry:
    return Entry("stub_diff", _Counter, 224, None, "test", channels=6)


def _offline_shift_entry() -> Entry:
    return Entry("stub_shift", _Counter, 224, None, "test", shift="offline")


def _online_shift_entry() -> Entry:
    return Entry("stub_stream", _Counter, 224, None, "test", shift="online")


def test_a_frame_model_is_fed_one_frame_and_a_clip_model_a_clip():
    assert forward_input(_frame_entry(), CPU).shape == (1, 3, 224, 224)
    assert forward_input(_clip_entry(), CPU).shape == (1, 3, 16, 224, 224)


def test_frames_start_at_the_size_they_are_stored_at():
    """Timing begins from the stored frame, not from an already cropped one."""
    frames = stored_frames(4)

    assert frames.shape == (4, 3, STORED_SIZE, STORED_SIZE)
    assert frames.dtype == torch.uint8


def test_a_decision_aggregates_the_frames_into_one_answer():
    entry, model = _frame_entry(), _Counter()

    answer = decision_call(entry, model, 8, CPU)()

    assert answer.shape == (1,)
    assert model.seen == [(8, 3, 224, 224)]


def test_a_difference_model_is_fed_the_channels_it_reads():
    assert forward_input(_difference_entry(), CPU).shape == (1, 6, 224, 224)


def test_a_difference_decision_pays_for_two_frames_per_frame_aggregated():
    """The neighbour is a second frame through the transform, not a free one.

    This is the half of the cost the widened stem does not show. The arithmetic
    of reading six channels instead of three is a few per cent; putting 2K
    frames through the evaluation pipeline to aggregate K is not, and it is the
    part a reader would otherwise have to take on trust.
    """
    entry, model = _difference_entry(), _Counter()

    answer = decision_call(entry, model, 8, CPU)()

    assert answer.shape == (1,)
    assert model.seen == [(8, 6, 224, 224)]


def test_a_difference_decision_subtracts_after_the_transform():
    """Normalised space, matching the dataset: a still background cancels to zero.

    Subtracting the stored frames first would be cheaper and would measure a
    different arrangement than the one the accuracy was produced under.
    """
    entry = _difference_entry()

    class _Keeper(_Counter):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            self.batch = batch
            return super().forward(batch)

    model = _Keeper()
    decision_call(entry, model, 2, CPU)()

    frame, difference = model.batch[:, :3], model.batch[:, 3:]
    assert not torch.allclose(frame, difference)
    # Normalised pixels leave zero; raw uint8 ones could not reach it.
    assert frame.min() < 0


def test_a_clip_model_receives_time_as_its_own_axis():
    """A clip network reads (batch, channels, time, height, width), not a stack."""
    entry, model = _clip_entry(), _Counter()

    decision_call(entry, model, 8, CPU)()

    assert model.seen == [(1, 3, 16, 224, 224)]


def test_a_clip_model_ignores_the_requested_frame_count():
    """Its clip length fixes what one decision costs, so --frames cannot move it."""
    entry, model = _clip_entry(), _Counter()

    decision_call(entry, model, 24, CPU)()

    assert model.seen[0][2] == 16


def test_an_offline_shift_reads_its_clip_as_one_batch_of_frames():
    """The frames share a batch so that the shift can reach across them."""
    assert forward_input(_offline_shift_entry(), CPU).shape == (
        SHIFT_SEGMENTS,
        3,
        224,
        224,
    )


def test_an_offline_shift_decides_on_the_clip_it_trained_on_whatever_was_asked():
    """Its decision cannot be made before the clip is complete, at any K."""
    entry, model = _offline_shift_entry(), _Counter()

    decision_call(entry, model, 24, CPU)()

    assert model.seen == [(SHIFT_SEGMENTS, 3, 224, 224)]


def test_an_online_shift_runs_one_frame_per_decision_whatever_k_is():
    """The decision averages recent answers; it does not rerun recent frames."""
    entry, model = _online_shift_entry(), _Counter()
    decide = decision_call(entry, model, 16, CPU)

    decide()
    decide()

    assert model.seen == [(1, 3, 224, 224), (1, 3, 224, 224)]


def test_an_online_decision_averages_the_last_k_answers():
    """Each answer joins the window and the oldest one leaves it."""
    entry = _online_shift_entry()

    class _Votes(_Counter):
        """Answers class 2 on the first call and class 5 on every later one."""

        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros(1, self.classes)
            logits[0, 2 if not self.seen else 5] = 50.0
            super().forward(batch)
            return logits

    model = _Votes()
    decide = decision_call(entry, model, 2, CPU)

    first, second, third = decide(), decide(), decide()

    # One confident answer outweighs the uniform start, so the first decision
    # is class 2; by the third, both answers in a window of two are class 5.
    assert int(first) == 2
    assert int(third) == 5
    assert int(second) in (2, 5)


def test_time_calls_runs_the_warmup_outside_the_measured_repetitions():
    calls = []

    summary = time_calls(lambda: calls.append(1), CPU, warmup=3, repeats=4)

    assert summary.samples == 4
    assert len(calls) == 7


def test_a_frame_model_is_timed_once_per_requested_frame_count():
    records = rows_for(
        _frame_entry(), CPU, [1, 8], parse_args(["--warmup", "0", "--repeats", "1"])
    )

    levels = [(record["level"], record["frames"]) for record in records]
    assert levels == [("forward", 1), ("decision", 1), ("decision", 8)]


def test_a_clip_model_is_timed_once_whatever_was_requested():
    records = rows_for(
        _clip_entry(),
        CPU,
        [1, 8, 24],
        parse_args(["--warmup", "0", "--repeats", "1"]),
    )

    levels = [(record["level"], record["frames"]) for record in records]
    assert levels == [("forward", 16), ("decision", 16)]


def test_an_offline_shift_is_timed_once_at_its_clip_length():
    records = rows_for(
        _offline_shift_entry(),
        CPU,
        [1, 8, 24],
        parse_args(["--warmup", "0", "--repeats", "1"]),
    )

    levels = [(record["level"], record["frames"]) for record in records]
    assert levels == [("forward", SHIFT_SEGMENTS), ("decision", SHIFT_SEGMENTS)]


def test_an_online_shift_is_timed_at_every_requested_window():
    records = rows_for(
        _online_shift_entry(),
        CPU,
        [1, 8],
        parse_args(["--warmup", "0", "--repeats", "1"]),
    )

    levels = [(record["level"], record["frames"]) for record in records]
    assert levels == [("forward", 1), ("decision", 1), ("decision", 8)]


def test_parameters_are_reported_in_millions():
    assert parameters_of(_Counter()) == pytest.approx(3 / 1e6)


def test_a_path_inside_the_project_is_named_relative_to_it():
    assert not Path(shorten(Path(__file__))).is_absolute()


def test_a_path_outside_the_project_keeps_its_full_name():
    assert shorten(Path("/somewhere/else/latency.csv")) == "/somewhere/else/latency.csv"


def test_the_rows_run_in_the_order_they_were_asked_for():
    """Order is a variable here, not a detail.

    A pass runs its rows back to back, so a row near the end meets a machine
    that has been at full load for minutes. Repeating a pass in one fixed
    order would reproduce that bias three times rather than average it away,
    so reversing the order between passes has to actually reverse it.
    """
    forwards = parse_args(["--models", "resnet18", "mobilenet_v3_small"]).models
    backwards = parse_args(["--models", "mobilenet_v3_small", "resnet18"]).models

    assert forwards == ["resnet18", "mobilenet_v3_small"]
    assert backwards == ["mobilenet_v3_small", "resnet18"]


def test_the_default_frame_counts_match_the_accuracy_curve():
    """Both readings share an axis only if they share these values."""
    assert parse_args([]).frames == [1, 4, 8, 16, 24]


def test_the_default_device_is_the_constrained_one():
    """The cost argument is about a small machine, so cpu is what runs unasked."""
    arguments = parse_args([])

    assert arguments.device == "cpu"
    assert arguments.threads == 1
