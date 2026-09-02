from types import SimpleNamespace

import torch

from device import describe, pick_device


def _props(total_gb: float, *, integrated: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        total_memory=int(total_gb * 1024**3), is_integrated=int(integrated)
    )


def _fake_cuda(monkeypatch, props: list[SimpleNamespace]) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(props))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: props[index])


def test_a_cuda_device_wins_over_metal_and_cpu(monkeypatch):
    _fake_cuda(monkeypatch, [_props(24)])

    assert pick_device() == torch.device("cuda", 0)


def test_the_integrated_gpu_is_skipped_even_though_it_reports_more_memory(monkeypatch):
    """An APU's integrated GPU borrows system RAM and outweighs the discrete card."""
    _fake_cuda(monkeypatch, [_props(24), _props(31, integrated=True)])

    assert pick_device() == torch.device("cuda", 0)


def test_memory_breaks_the_tie_between_two_discrete_gpus(monkeypatch):
    """With no integrated device to exclude, the larger card wins whatever its index."""
    _fake_cuda(monkeypatch, [_props(16), _props(24)])

    assert pick_device() == torch.device("cuda", 1)


def test_a_build_that_omits_is_integrated_falls_back_to_memory(monkeypatch):
    """Not every build reports the field; a missing one must not end the run."""
    _fake_cuda(
        monkeypatch,
        [
            SimpleNamespace(total_memory=8 * 1024**3),
            SimpleNamespace(total_memory=24 * 1024**3),
        ],
    )

    assert pick_device() == torch.device("cuda", 1)


def test_an_integrated_gpu_is_still_used_when_it_is_the_only_one(monkeypatch):
    _fake_cuda(monkeypatch, [_props(31, integrated=True)])

    assert pick_device() == torch.device("cuda", 0)


def test_metal_is_used_when_no_cuda_device_is_present(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert pick_device() == torch.device("mps")


def test_cpu_is_the_fallback_when_no_accelerator_is_present(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert pick_device() == torch.device("cpu")


def test_describe_names_the_gpu_model_for_a_cuda_device(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda device: "AMD Radeon RX 7900 XTX"
    )

    assert describe(torch.device("cuda", 0)) == "cuda:0 — AMD Radeon RX 7900 XTX"


def test_describe_is_just_the_device_for_cpu_and_metal():
    assert describe(torch.device("cpu")) == "cpu"
    assert describe(torch.device("mps")) == "mps"
