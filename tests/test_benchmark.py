from pathlib import Path

import pytest
import torch
from torch import nn

from benchmark import (
    NUM_CLASSES,
    STORED_SIZE,
    Entry,
    decision_call,
    forward_input,
    parameters_of,
    parse_args,
    rows_for,
    shorten,
    stored_frames,
    summarise,
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


def test_summarise_reports_the_median_not_the_mean():
    """One descheduled run must not move the number the table reports."""
    summary = summarise([10.0, 10.0, 10.0, 10.0, 500.0])

    assert summary.median_ms == 10.0
    assert summary.min_ms == 10.0
    assert summary.samples == 5


def test_summarise_spreads_the_quartiles_over_four_or_more_samples():
    summary = summarise([1.0, 2.0, 3.0, 4.0, 5.0])

    assert summary.iqr_ms == pytest.approx(2.0)


def test_summarise_falls_back_to_the_full_range_below_four_samples():
    """Quantiles need four points; fewer still deserve some measure of spread."""
    summary = summarise([2.0, 5.0])

    assert summary.iqr_ms == pytest.approx(3.0)


def test_summarise_refuses_an_empty_run():
    """An empty set means the timing loop never ran, which is a bug not a result."""
    with pytest.raises(ValueError):
        summarise([])


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
