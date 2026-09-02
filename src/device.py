"""Choose the accelerator a run should use.

Its own module because the choice happens in more than one place — a training
run and an inference pass — and it differs by machine. An Apple laptop exposes
Metal, which PyTorch calls mps; the desktop's Radeon exposes ROCm, which PyTorch
reaches through the same cuda name it uses for every GPU backend; CI has
neither. Each run prints the device it picked, so the machine a result came
from can be read back from its log — and results from two machines must not
share a table.
"""

import torch


def pick_device() -> torch.device:
    """Return the fastest device on this host: discrete GPU, then Metal, then CPU.

    A desktop APU exposes its integrated GPU as a cuda device too, and it
    reports more memory than the discrete card because it borrows system RAM.
    So the integrated devices are dropped first, and memory only breaks a tie
    between two real GPUs.

    Returns:
        The device to move the model and every batch onto.
    """
    if torch.cuda.is_available():
        indices = list(range(torch.cuda.device_count()))
        properties = [torch.cuda.get_device_properties(index) for index in indices]
        # Not every build reports is_integrated. Reading a missing field as
        # discrete leaves memory to decide, which is the behaviour on a host
        # with one GPU anyway, rather than failing before training starts.
        discrete = [
            index
            for index in indices
            if not getattr(properties[index], "is_integrated", 0)
        ]
        best = max(
            discrete or indices, key=lambda index: properties[index].total_memory
        )
        return torch.device("cuda", best)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe(device: torch.device) -> str:
    """Render a device for a run's log, naming the GPU model when there is one."""
    if device.type == "cuda":
        return f"{device} — {torch.cuda.get_device_name(device)}"
    return str(device)
