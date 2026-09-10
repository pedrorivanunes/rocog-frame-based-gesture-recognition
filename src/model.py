"""Build the network a run trains and evaluates.

Kept apart from the training loop because more than one caller needs it: a
training run, and an inference pass that scores a manifest with weights trained
earlier. Keeping it here also gives the cost study its home — comparing this
backbone against cheaper ones is a choice about the model, not about the loop
that consumes it.
"""

from collections.abc import Iterable, Mapping
from contextlib import contextmanager

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

NUM_CLASSES = 7

# The backbones a run may train. Both are ResNet18: the same stage names, the
# same arithmetic per frame, and — once the head is resized to seven or eight
# outputs — the same 11.18 M parameters. They differ in one thing, which is why
# the second is here at all: it normalises half of each shallow stage's channels
# by the instance rather than the batch, which removes appearance statistics a
# batch would have preserved. The cost argument the project rests on survives
# the swap untouched.
DEFAULT_BACKBONE = "resnet18"
BACKBONES = ("resnet18", "resnet18_ibn_a")

# Where the second one comes from, and what its weights carry that torchvision's
# do not. Reading the marker back off a file is how a checkpoint says which
# backbone wrote it, the same way ``fc.bias`` says how wide its head is — a
# caller that had to remember would eventually pair the wrong two.
IBN_HUB_REPOSITORY = "XingangPan/IBN-Net"
IBN_MARKER = ".IN."

# The layers that carry a running mean and variance. Listed by their public
# classes rather than by the private base they share, because the list is also
# the claim: these are the only places a domain's statistics are stored, so
# these are the only places re-estimating them can reach.
NORMALISATION_LAYERS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


@contextmanager
def _weights_landing_on_the_cpu():
    """Force every ``torch.load`` inside the block onto the CPU.

    Needed for one backbone and worth the intrusion. The IBN authors' entry
    point fetches its own weights without naming a map location, and the file
    they published was saved from a CUDA device — so on a host with no GPU it
    raises rather than loading. This module promises a network on the CPU
    whatever the host has, and the two places that promise is about to be
    collected on are a laptop with no CUDA and a prototype expected to run on
    one.

    Narrower than it looks: the patch lives only for the call it wraps, and it
    overrides the argument rather than filling it in, because the caller being
    corrected passes it explicitly as ``None``.
    """
    original = torch.load

    def on_the_cpu(*args, **keywords):
        return original(*args, **{**keywords, "map_location": "cpu"})

    torch.load = on_the_cpu
    try:
        yield
    finally:
        torch.load = original


def build_model(
    num_classes: int = NUM_CLASSES, backbone: str = DEFAULT_BACKBONE
) -> nn.Module:
    """Build a ResNet18 with ImageNet weights and a fresh classification head.

    The convolutional layers keep what they learned on ImageNet — edges, textures,
    shapes. Only the final layer is replaced, mapping the 512 features it produces
    to the gesture classes instead of ImageNet's 1000 categories. That mapping is
    what training has to learn.

    ⚠️ The IBN variant is fetched from its authors' repository through
    ``torch.hub`` and cached under ``~/.cache/torch``, so the first build of it
    needs a network and later ones do not. Its weights are ImageNet's, trained by
    them; nothing here retrains it.

    Args:
        num_classes: Outputs the head produces, one per gesture.
        backbone: Which of ``BACKBONES`` to build.

    Returns:
        The network, on the CPU. Moving it to a device is the caller's job.

    Raises:
        ValueError: If the backbone is not one this project knows how to build.
    """
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}; expected one of {BACKBONES}")

    if backbone == DEFAULT_BACKBONE:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        with _weights_landing_on_the_cpu():
            model = torch.hub.load(
                IBN_HUB_REPOSITORY,
                backbone,
                pretrained=True,
                trust_repo=True,
                verbose=False,
            )

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def backbone_of(weights: Mapping[str, object]) -> str:
    """Read which backbone wrote a checkpoint, off the checkpoint itself.

    Scoring a file with the wrong architecture fails on a shape mismatch if the
    project is lucky and on a wrong number if it is not, and a caller asked to
    remember which run wrote which file will eventually pair the wrong two. The
    file already says: an instance-normalised stage carries parameters a
    torchvision ResNet18 has no name for.

    Args:
        weights: A checkpoint's state dictionary, or anything with its keys.

    Returns:
        The name of the backbone to hand ``build_model``.
    """
    return (
        "resnet18_ibn_a"
        if any(IBN_MARKER in key for key in weights)
        else DEFAULT_BACKBONE
    )


# How far into the network a run holds the pretrained weights. Named in stages
# rather than in a layer count because that is the unit torchvision exposes, and
# because the boundary that matters is one of them: "backbone" is every stage
# before the head, so the head is all that is left to train.
FROZEN_STAGES = {
    "none": (),
    "early": ("conv1", "bn1", "layer1", "layer2"),
    "backbone": ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4"),
}


def freeze(model: nn.Module, depth: str) -> list[nn.Module]:
    """Hold the first stages of a network at the values they arrived with.

    Training every layer is not free. The gradient that fits the source domain
    moves the features themselves, not only the mapping from them to the
    classes, and features reshaped to fit rendered frames are not necessarily
    the ones that survive a photographed one. Freezing turns that off for part
    of the network, which is what separates two questions a single fine-tuned
    number cannot: how much of a result comes from the pretrained representation
    and how much from having rebuilt it.

    Two things are frozen here, not one. Clearing ``requires_grad`` stops the
    optimizer from writing to the weights, but batch normalization also carries
    a running mean and variance, and those are re-estimated from every batch the
    layer sees in training mode whether a gradient reaches it or not. A stage
    left that way would keep drifting onto the training domain's statistics —
    the drift being measured. So the stages are put in evaluation mode too, and
    the caller has to keep them there: ``Module.train()`` is recursive and
    switches them back at the top of the next epoch, which is why the frozen
    stages come back as a return value rather than being applied and forgotten.

    Args:
        model: The network to freeze, modified in place.
        depth: How far to freeze, a key of ``FROZEN_STAGES``. ``"none"`` touches
            nothing, so a run that does not ask for this is the run as it was
            before the option existed.

    Returns:
        The frozen stages, for the caller to put back into evaluation mode after
        each ``model.train()``. Empty for ``"none"``.
    """
    stages = [getattr(model, name) for name in FROZEN_STAGES[depth]]
    for stage in stages:
        stage.eval()
        for parameter in stage.parameters():
            parameter.requires_grad_(False)
    return stages


def adapt_batchnorm(model: nn.Module, batches: Iterable, device: torch.device) -> int:
    """Re-estimate the normalisation statistics on frames from another domain.

    A batch normalisation layer standardises its input by a mean and variance it
    accumulated while training. Those describe the domain it was trained on, and
    a model carried to a different one keeps applying them to features that no
    longer have those moments. Re-estimating them is the cheapest correction
    there is: no gradient, no optimiser, no parameter added, and no label read —
    the frames are pushed through and the layers write down what they see.

    ``freeze`` is the same mechanism seen from the other side. There the running
    statistics are held still so they cannot drift onto the training domain;
    here they are deliberately moved onto another one.

    The momentum is set aside for the duration, not merely reset. A layer with a
    momentum weights recent batches more heavily than early ones, which would
    make the result depend on the order the loader happened to serve — with it
    cleared, PyTorch averages every batch equally instead. It is put back
    afterwards so a model adapted here can still be trained later.

    ⚠️ Equal weight per *batch*, not per frame: a short final batch counts as
    much as a full one. With frames in the thousands and batches in the dozens
    the difference is far below the noise of any comparison this feeds, and
    dropping the short batch would discard target frames to fix it.

    ⚠️ This reads data from the domain being scored, so a result it produces is
    not *source-only* and cannot be set against a source-only baseline. Which
    frames it saw is the whole protocol question, and it belongs beside the
    number.

    Args:
        model: The network to adapt, modified in place. Left in evaluation mode,
            ready to score.
        batches: Anything yielding batches whose first element is a tensor of
            frames — a ``DataLoader`` over the target's frames in practice, and
            the labels it also serves are ignored on purpose.
        device: Where the forward pass runs.

    Returns:
        How many frames the statistics were estimated from, so a caller can
        report the size of what it adapted on.

    Raises:
        ValueError: If no frames arrived. The layers are reset before the pass,
            so returning quietly would hand back a model normalising by a mean
            of zero and a variance of one — worse than the one that came in, and
            wrong in a way no later number would reveal.
    """
    layers = [
        module for module in model.modules() if isinstance(module, NORMALISATION_LAYERS)
    ]
    momenta = [layer.momentum for layer in layers]
    for layer in layers:
        layer.reset_running_stats()
        layer.momentum = None

    model.train()
    frames_seen = 0
    with torch.no_grad():
        for batch in batches:
            frames = batch[0]
            model(frames.to(device))
            frames_seen += len(frames)

    for layer, momentum in zip(layers, momenta, strict=True):
        layer.momentum = momentum
    model.eval()

    if frames_seen == 0:
        raise ValueError("no frames to adapt on; the statistics would stay reset")

    return frames_seen
