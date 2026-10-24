"""The configurations the program offers, and what each needs beyond its checkpoint.

A checkpoint holds weights and, for a network that exchanges channels between
frames, how its clips were arranged. It does not hold how many frames back each
difference was taken, whether the decisions are corrected for how often the
network picks each gesture, or what the network learned from. Those are facts
about the experiment rather than about the file, and scoring the file without
them answers a different question from the one the thesis reports. So they live
here, one record per configuration, and nothing else in the program decides
them.

The catalog is not every configuration that was trained. It is the best version
of each method on each network: the one the thesis reports for that method,
chosen without reading the test videos. Six methods on three networks, and the
pose configuration, which has no image network and so stands apart.

Nothing here reads a result. What the thesis measured for each configuration is
read from the stored prediction tables when it is shown, so a number cannot be
left behind in this file when the tables change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = PROJECT_ROOT / "checkpoints"

SEEDS = (0, 1, 2)

# The networks in the order the menu lists them, with the names the thesis uses.
NETWORKS = {
    "resnet18": "ResNet18",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenet_v3_small": "MobileNetV3-Small",
}

# How much of the real training videos a configuration used. A video it used is
# no test of it: one it was fine-tuned on, it has seen with its label, and one
# its correction was estimated on has already shaped its decisions.
NO_REAL_VIDEOS = "none"
UNLABELLED_REAL_VIDEOS = "unlabelled"
LABELLED_REAL_VIDEOS = "labelled"

SYNTHETIC_ONLY = "synthetic videos only (never saw a real one)"


@dataclass(frozen=True)
class Method:
    """What was added to a network, the same on every network it was tried on.

    Attributes:
        label: How the menu names it.
        sees: What reaches the network for each frame, in words, for the card.
        learned_from: The data it was trained on, in words, for the card.
        real_videos: How much of the real training videos it used.
        spacings: How many frames back each difference is taken, counted in
            the video's own frames. Empty for a network that reads frames alone.
        clip: Frames a shifted network reads together, or ``None`` for one that
            reads every frame alone. Its decisions are made per clip, because
            once frames share channels the mean over a video stops being what
            the network was trained to answer.
        prior: Whether each decision is divided by how often the network picked
            each gesture on the real training videos, the correction that needs
            no labels.
        latency_suffix: Appended to the network's name to find the benchmark
            row its decisions cost. A correction or a fine-tuning leaves the
            network as it was, so they share their base method's row.
    """

    label: str
    sees: str
    learned_from: str
    real_videos: str = NO_REAL_VIDEOS
    spacings: tuple[int, ...] = ()
    clip: int | None = None
    prior: bool = False
    latency_suffix: str = ""


ONE_SPACING = "each frame, plus how it changed since the frame 5 before"
THREE_SPACINGS = "each frame, plus how it changed since 2, 5 and 10 frames before"

# In the order the menu lists them: the base, the three temporal treatments that
# need no real video, then the two ways of using the real training videos.
METHODS = {
    "frames": Method("isolated frames", "each frame on its own", SYNTHETIC_ONLY),
    "difference": Method(
        "frame difference, 1 spacing",
        ONE_SPACING,
        SYNTHETIC_ONLY,
        spacings=(5,),
        latency_suffix="_diff",
    ),
    "spacings": Method(
        "frame difference, 3 spacings",
        THREE_SPACINGS,
        SYNTHETIC_ONLY,
        spacings=(2, 5, 10),
        latency_suffix="_multi",
    ),
    "shift": Method(
        "temporal shift (TSM)",
        "clips of 8 frames spread over the gesture, each frame reading part of"
        " its neighbours inside the network",
        SYNTHETIC_ONLY,
        clip=8,
        latency_suffix="_tsm",
    ),
    "prior": Method(
        "3 spacings, prior correction",
        THREE_SPACINGS,
        "synthetic videos; its answers are then rebalanced by how often it named"
        " each gesture on the real training videos, without their labels",
        real_videos=UNLABELLED_REAL_VIDEOS,
        spacings=(2, 5, 10),
        prior=True,
        latency_suffix="_multi",
    ),
    "fine_tuned": Method(
        "frame difference, fine-tuned",
        ONE_SPACING,
        "synthetic videos, then the 204 real training videos with their labels",
        real_videos=LABELLED_REAL_VIDEOS,
        spacings=(5,),
        latency_suffix="_diff",
    ),
}

# The checkpoints of the best version of each method on each network. The
# prior correction is applied to the three-spacing network, so the two share a
# file; everything else was trained on its own.
STEMS = {
    ("resnet18", "frames"): "idle",
    ("resnet18", "difference"): "diff_k5",
    ("resnet18", "spacings"): "diff_multi",
    ("resnet18", "shift"): "tsm_ordered_rerun",
    ("resnet18", "prior"): "diff_multi",
    ("resnet18", "fine_tuned"): "diff_ft100_lr3e6",
    ("efficientnet_b0", "frames"): "efficientnet",
    ("efficientnet_b0", "difference"): "efficientnet_diff",
    ("efficientnet_b0", "spacings"): "efficientnet_diff_multi",
    ("efficientnet_b0", "shift"): "efficientnet_tsm_ordered",
    ("efficientnet_b0", "prior"): "efficientnet_diff_multi",
    ("efficientnet_b0", "fine_tuned"): "efficientnet_diff_ft_lr3e5",
    ("mobilenet_v3_small", "frames"): "mobilenet",
    ("mobilenet_v3_small", "difference"): "mobilenet_diff",
    ("mobilenet_v3_small", "spacings"): "mobilenet_diff_multi",
    ("mobilenet_v3_small", "shift"): "mobilenet_tsm_ordered",
    ("mobilenet_v3_small", "prior"): "mobilenet_diff_multi",
    ("mobilenet_v3_small", "fine_tuned"): "mobilenet_diff_ft_lr3e5",
}


@dataclass(frozen=True)
class Configuration:
    """One entry of the menu: a method on a network, and its checkpoints.

    Attributes:
        key: Short and unique, for the names of saved results.
        network: The backbone in torchvision's spelling, or ``None`` for the
            pose configuration, whose classifier reads joints.
        method: Key of ``METHODS``, or ``"pose"``.
        stem: The checkpoints are ``{stem}_s{seed}.pt``.
        label: How the menu names it, the network first.
        sees: What reaches the classifier for each frame, for the card.
        learned_from: The data it was trained on, for the card.
        real_videos: How much of the real training videos it used.
        spacings: Frames back of each difference; see ``Method``.
        clip: Frames read together by a shifted network; see ``Method``.
        prior: Whether decisions are corrected; see ``Method``.
        latency_rows: The benchmark rows one decision costs. Two for the pose
            configuration, whose decision pays for the detector and then the
            classifier.
    """

    key: str
    network: str | None
    method: str
    stem: str
    label: str
    sees: str
    learned_from: str
    real_videos: str
    spacings: tuple[int, ...]
    clip: int | None
    prior: bool
    latency_rows: tuple[str, ...]

    @property
    def channels(self) -> int:
        """What an image network's first convolution reads: three per frame given.

        The frame itself, and one difference per spacing, each in colour. It is
        what a checkpoint's stem has to match, which is how a wrong spacing in
        this file is caught before a single video is scored.
        """
        return 3 * (1 + len(self.spacings))


def _configuration(network: str, method: str) -> Configuration:
    """Put a method on a network, with the checkpoints that pair trained."""
    spec = METHODS[method]
    return Configuration(
        key=f"{network}-{method}".replace("_", "-"),
        network=network,
        method=method,
        stem=STEMS[(network, method)],
        label=f"{NETWORKS[network]} · {spec.label}",
        sees=spec.sees,
        learned_from=spec.learned_from,
        real_videos=spec.real_videos,
        spacings=spec.spacings,
        clip=spec.clip,
        prior=spec.prior,
        latency_rows=(f"{network}{spec.latency_suffix}",),
    )


# The joints come from a detector that runs in an environment of its own, and
# the classifier reading them was trained on the synthetic videos' joints. Of
# the three inputs tried, positions and motion is the one the thesis reports.
POSE = Configuration(
    key="pose",
    network=None,
    method="pose",
    stem="pose_motion",
    label="Body pose (MediaPipe) · joints and their motion",
    sees="where the arm joints are in each frame, as MediaPipe finds them, and"
    " how far each moved since the frame before",
    learned_from=SYNTHETIC_ONLY,
    real_videos=NO_REAL_VIDEOS,
    spacings=(),
    clip=None,
    prior=False,
    latency_rows=("pose_detector", "pose_classifier"),
)

CATALOG: tuple[Configuration, ...] = (
    *(_configuration(network, method) for network in NETWORKS for method in METHODS),
    POSE,
)


def find(key: str) -> Configuration:
    """The configuration a key names.

    Raises:
        KeyError: If no configuration has that key, naming the ones that do.
    """
    for configuration in CATALOG:
        if configuration.key == key:
            return configuration
    known = ", ".join(configuration.key for configuration in CATALOG)
    raise KeyError(f"no configuration {key!r}; the catalog has {known}")


def checkpoint_paths(
    configuration: Configuration, root: Path = CHECKPOINTS
) -> list[Path]:
    """Where each seed's checkpoint would be, whether or not it is there."""
    return [root / f"{configuration.stem}_s{seed}.pt" for seed in SEEDS]


def available_seeds(
    configuration: Configuration, root: Path = CHECKPOINTS
) -> list[int]:
    """The seeds whose checkpoints this computer has.

    The thesis reports the mean of three trainings, so the card says how many
    of the three are here: a copy with fewer runs answers with the mean of the
    ones it has, which is not the number the thesis reports.
    """
    paths = checkpoint_paths(configuration, root)
    return [seed for seed, path in zip(SEEDS, paths, strict=True) if path.is_file()]


def card(
    configuration: Configuration,
    root: Path = CHECKPOINTS,
    in_the_thesis: str | None = None,
) -> list[tuple[str, str]]:
    """The lines that describe a configuration before it is chosen.

    Args:
        configuration: The one being described.
        root: Where its checkpoints are looked for.
        in_the_thesis: What the thesis measured for it, already worded. Left
            to the caller because it is read from the stored prediction tables,
            which this module does not touch.

    Returns:
        Pairs of label and text, in the order they are shown.
    """
    found = len(available_seeds(configuration, root))
    if found == len(SEEDS):
        runs = f"{found} of {len(SEEDS)} on this computer (results average the three)"
    elif found:
        runs = (
            f"{found} of {len(SEEDS)} on this computer (results average the ones"
            " here, so they are not the thesis's figures)"
        )
    else:
        runs = f"none of the {len(SEEDS)} on this computer"
    if configuration.network is None:
        network = "none: a small classifier reads the joints"
    else:
        network = NETWORKS[configuration.network]
    lines = [
        ("network", network),
        ("sees", configuration.sees),
        ("learned from", configuration.learned_from),
        ("training runs", runs),
    ]
    if in_the_thesis is not None:
        lines.append(("in the thesis", in_the_thesis))
    return lines
