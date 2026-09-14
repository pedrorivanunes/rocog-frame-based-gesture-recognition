"""Measure how long a decision takes, for this model and for the ones it argues against.

Why this file exists. The case for classifying frames rather than clips is a
cost argument, and until now the project could only state it in arithmetic:
multiply-accumulate counts for each network. Arithmetic is a poor proxy for
time. Two networks measured here differ by thirty-one times in multiply-adds
and by less than ten per cent in wall clock, because memory traffic, kernel
launches and cache behaviour decide as much as the arithmetic does. A cost
claim that never leaves the units of arithmetic is half a claim.

What latency depends on, and what it does not. It depends on the architecture,
the shape of the input, the batch size, the hardware, the numeric precision and
the software stack. It does not depend on the values of the weights: a multiply
by 0.3 costs exactly what a multiply by 0.7 costs. Nothing about how a network
was trained — the data, the augmentation, the schedule, the accuracy it reached
— is visible here. That is what makes it legitimate to time an untrained copy
of a published architecture and read its accuracy from the paper that trained
it, as long as each half is labelled for what it is.

The one thing that escapes that rule is the evaluation protocol. How many
frames a clip holds, at what resolution, and how many views of a video are
averaged into one decision are not training choices, and they multiply the cost
of a decision. Views are therefore left as an explicit multiplier here rather
than assumed: this file reports the cost of one view.

Two levels are measured, because they answer different questions.

``forward``
    One call, batch of one: one frame for a network that reads frames, one clip
    for a network that reads clips. The unit that compares architectures with
    nothing else mixed in.

``decision``
    What it costs to answer "which gesture is this video". For a frame-based
    model that is K frames through the evaluation transform, one batched
    forward, and the mean of the per-frame probabilities. For a clip-based
    model it is one clip through the same steps. This is the number the cost
    argument is actually about.

A remark that neither level captures, and that belongs in prose rather than in
a column: a frame-based model can emit a decision as each frame arrives, while
a clip-based model cannot answer until its sixteenth frame exists. The two are
equal in arithmetic and unequal in when an answer is available.

Run from the project root::

    python src/benchmark.py --device cpu --threads 1
    python src/benchmark.py --device cpu --threads 0      # every core
    python src/benchmark.py --device auto
    python src/benchmark.py --models resnet18 i3d_r50 --frames 1 8 16

The clip-based models come from ``pytorchvideo``, which is not needed by
anything else here; see ``requirements-benchmark.txt``. Without it the frame
based rows still run and the others are skipped with a note.
"""

from __future__ import annotations

import argparse
import csv
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import eval_transform  # noqa: E402
from device import pick_device  # noqa: E402
from model import build_model  # noqa: E402

# Frames the stored dataset holds per side, before the evaluation crop. Timing
# starts from this shape because it is the shape the project's own evaluation
# starts from; a camera would add a decode and a resize, equally for every row.
STORED_SIZE = 256

NUM_CLASSES = 8
WARMUP = 5
REPEATS = 30
FRAME_COUNTS = (1, 4, 8, 16, 24)


@dataclass(frozen=True)
class Entry:
    """One network, and the shape of the input its own protocol gives it.

    Attributes:
        name: How the row is labelled.
        build: Makes the network. Weights are never loaded — see the module
            docstring for why untrained copies time the same as trained ones.
        crop: Side of the square the network is fed, after cropping.
        clip: Frames consumed by one forward, or ``None`` for a network that
            reads one frame at a time. A clip length also fixes how many frames
            one decision costs, which is why those rows ignore ``--frames``.
        source: Where the architecture comes from, for the printed provenance.
    """

    name: str
    build: Callable[[], nn.Module]
    crop: int
    clip: int | None
    source: str


def _pytorchvideo(factory: str):
    """Defer importing pytorchvideo until a clip-based row is actually built."""

    def build() -> nn.Module:
        from pytorchvideo.models import hub

        return getattr(hub, factory)(pretrained=False)

    return build


# The three frame-based networks are the project's own. The two clip-based ones
# are the baselines the dataset's paper reports, at the input sizes it states;
# their parameter counts match the published figures, which is the check that
# they are the right architectures and not merely similar ones.
ENTRIES: tuple[Entry, ...] = (
    Entry("resnet18", lambda: build_model(NUM_CLASSES, "resnet18"), 224, None, "ours"),
    Entry(
        "efficientnet_b0",
        lambda: build_model(NUM_CLASSES, "efficientnet_b0"),
        224,
        None,
        "ours",
    ),
    Entry(
        "mobilenet_v3_small",
        lambda: build_model(NUM_CLASSES, "mobilenet_v3_small"),
        224,
        None,
        "ours",
    ),
    Entry("i3d_r50", _pytorchvideo("i3d_r50"), 256, 16, "baseline"),
    Entry("x3d_m", _pytorchvideo("x3d_m"), 224, 16, "baseline"),
)


@dataclass(frozen=True)
class Summary:
    """What a set of timings is reported as.

    The median and the interquartile range rather than a mean and a standard
    deviation, because wall clock on a general purpose machine is bounded below
    and has a long right tail: one descheduled run drags a mean and leaves a
    median alone. The minimum is kept as the closest thing to an uninterrupted
    run.
    """

    median_ms: float
    iqr_ms: float
    min_ms: float
    samples: int


def summarise(samples: Sequence[float]) -> Summary:
    """Reduce raw timings, in milliseconds, to what gets reported.

    Args:
        samples: One duration per timed repetition, in milliseconds.

    Returns:
        The summary described by ``Summary``.

    Raises:
        ValueError: If there is nothing to summarise. An empty set here means
            the timing loop never ran, which is a bug rather than a result.
    """
    if not samples:
        raise ValueError("no timings to summarise")

    ordered = sorted(samples)
    if len(ordered) >= 4:
        quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
        iqr = quartiles[2] - quartiles[0]
    else:
        iqr = ordered[-1] - ordered[0]

    return Summary(statistics.median(ordered), iqr, ordered[0], len(ordered))


def synchronize(device: torch.device) -> None:
    """Wait for the device to finish, so the clock measures work and not queuing.

    Accelerator calls are asynchronous: the Python line returns as soon as the
    work is enqueued. Timing without this measures how fast a queue accepts
    work, which is the same for every network.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_calls(
    call: Callable[[], object],
    device: torch.device,
    warmup: int = WARMUP,
    repeats: int = REPEATS,
) -> Summary:
    """Run something repeatedly and report how long it took.

    The warmup is not padding. The first calls pay for lazy allocator growth,
    kernel selection and, on an accelerator, compilation of the kernels the
    shapes resolve to; including them would report a cost that is paid once as
    though it were paid every time.
    """
    with torch.inference_mode():
        for _ in range(warmup):
            call()
        synchronize(device)

        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            call()
            synchronize(device)
            samples.append((time.perf_counter() - started) * 1000)

    return summarise(samples)


def forward_input(entry: Entry, device: torch.device) -> torch.Tensor:
    """The batch-of-one tensor one forward consumes, already on the device."""
    if entry.clip is None:
        shape = (1, 3, entry.crop, entry.crop)
    else:
        shape = (1, 3, entry.clip, entry.crop, entry.crop)

    return torch.randn(*shape, device=device)


def stored_frames(count: int) -> torch.Tensor:
    """Frames as they sit on disk: ``count`` of them, uint8, square, unnormalised.

    Decoding is left out on purpose and the reason is comparability, not
    convenience: every row would pay the same JPEG cost, so including it would
    add a constant to each and shrink every ratio the table is read for.
    """
    return torch.randint(
        0, 256, (count, 3, STORED_SIZE, STORED_SIZE), dtype=torch.uint8
    )


def decision_call(
    entry: Entry, model: nn.Module, frames: int, device: torch.device
) -> Callable[[], object]:
    """Build the closure that answers "which gesture is this", once.

    Everything a decision pays for is inside: the evaluation transform, the
    move to the device, the forward, and — for a frame-based model — averaging
    the per-frame probabilities into one answer. The frames themselves are
    prepared outside, since reading them from disk is storage, not inference.

    Args:
        entry: The network's row.
        model: Its built and evaluated instance.
        frames: Frames the decision aggregates. Ignored by a clip-based row,
            whose clip length already fixes it.
        device: Where the forward runs.

    Returns:
        A callable taking no arguments and returning the chosen class.
    """
    transform = eval_transform(entry.crop)
    count = entry.clip if entry.clip is not None else frames
    raw = stored_frames(count)

    def decide() -> torch.Tensor:
        batch = transform(raw).to(device)
        if entry.clip is not None:
            # A clip-based network reads (batch, channels, time, height, width),
            # so the frame axis moves from the front to position two.
            batch = batch.permute(1, 0, 2, 3).unsqueeze(0)
        logits = model(batch)
        probabilities = logits.softmax(dim=1)
        if entry.clip is None:
            probabilities = probabilities.mean(dim=0, keepdim=True)
        return probabilities.argmax(dim=1)

    return decide


def parameters_of(model: nn.Module) -> float:
    """Parameter count in millions, to label a row with what it is carrying."""
    return sum(parameter.numel() for parameter in model.parameters()) / 1e6


def describe(device: torch.device, threads: int) -> str:
    """One line naming what the numbers were measured on.

    Latency without its machine is not a measurement, so this is written into
    the output rather than left to whoever reads the table to remember.
    """
    if device.type == "cuda":
        where = torch.cuda.get_device_name(0)
    elif device.type == "mps":
        where = "Apple MPS"
    else:
        where = f"{platform.processor() or platform.machine()}, {threads} thread(s)"

    return f"{where} | torch {torch.__version__} | {platform.system()}"


def rows_for(
    entry: Entry, device: torch.device, frame_counts: Sequence[int], args
) -> list[dict]:
    """Time one network at both levels, returning one record per measurement."""
    model = entry.build().to(device).eval()
    records = []

    level = time_calls(
        lambda: model(forward_input(entry, device)), device, args.warmup, args.repeats
    )
    records.append(
        {
            "model": entry.name,
            "source": entry.source,
            "level": "forward",
            "frames": entry.clip or 1,
            "median_ms": level.median_ms,
            "iqr_ms": level.iqr_ms,
            "min_ms": level.min_ms,
            "samples": level.samples,
            "parameters_m": parameters_of(model),
        }
    )

    # A clip-based row has one decision size, so asking for five frame counts
    # would time the same thing five times under different labels.
    counts = [entry.clip] if entry.clip is not None else list(frame_counts)
    for count in counts:
        decision = time_calls(
            decision_call(entry, model, count, device),
            device,
            args.warmup,
            args.repeats,
        )
        records.append(
            {
                "model": entry.name,
                "source": entry.source,
                "level": "decision",
                "frames": count,
                "median_ms": decision.median_ms,
                "iqr_ms": decision.iqr_ms,
                "min_ms": decision.min_ms,
                "samples": decision.samples,
                "parameters_m": parameters_of(model),
            }
        )

    return records


def shorten(path: Path) -> str:
    """Name a path relative to the project when it lives there, absolute otherwise."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu, cuda, mps, or auto. Defaults to cpu, which is the "
        "constrained-machine reading the cost argument is about.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="CPU threads. 1 is the closest stand-in available here for a "
        "device with few cores; 0 leaves torch's own default alone.",
    )
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=list(FRAME_COUNTS),
        help="Frames a decision aggregates, one run per value. Matching the "
        "accuracy curve's own values puts both on one axis.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[entry.name for entry in ENTRIES],
        help="Subset of rows to time.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where the CSV goes. Defaults to data/benchmarks/ named by device.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Time every requested network and write the table."""
    args = parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    device = pick_device() if args.device == "auto" else torch.device(args.device)

    print(describe(device, args.threads))
    print(f"warmup {args.warmup}, {args.repeats} timed repetitions\n")

    wanted = [entry for entry in ENTRIES if entry.name in args.models]
    missing = set(args.models) - {entry.name for entry in wanted}
    if missing:
        raise SystemExit(f"unknown model(s): {', '.join(sorted(missing))}")

    records = []
    for entry in wanted:
        try:
            measured = rows_for(entry, device, args.frames, args)
        except ImportError:
            print(
                f"{entry.name:20s} skipped: needs pytorchvideo "
                "(pip install -r requirements-benchmark.txt)"
            )
            continue

        records.extend(measured)
        for record in measured:
            print(
                f"{record['model']:20s} {record['level']:9s}"
                f" K={record['frames']:<3d} {record['median_ms']:9.2f} ms"
                f"  IQR {record['iqr_ms']:6.2f}  min {record['min_ms']:8.2f}"
            )

    if not records:
        return

    destination = args.output or (
        PROJECT_ROOT / "data/benchmarks" / f"latency_{device.type}_{args.threads}t.csv"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nwritten to {shorten(destination)}")


if __name__ == "__main__":
    main()
