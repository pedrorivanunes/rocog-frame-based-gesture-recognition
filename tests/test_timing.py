import pytest

from timing import summarise, time_repeatedly


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


def test_the_warmup_runs_but_is_not_timed():
    """The first calls pay for what is initialised once, not for each decision."""
    calls = []

    summary = time_repeatedly(lambda: calls.append(None), warmup=3, repeats=7)

    assert len(calls) == 10
    assert summary.samples == 7
