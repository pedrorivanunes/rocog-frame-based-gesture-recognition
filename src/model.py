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
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
)

NUM_CLASSES = 7

# The backbones a run may train, in the order the project acquired them. The
# first two are ResNet18 twice over: the same stage names, the same arithmetic
# per frame, and — once the head is resized — the same 11.18 M parameters. They
# differ in one thing, which is why the second is here at all: it normalises
# half of each shallow stage's channels by the instance rather than the batch,
# which removes appearance statistics a batch would have preserved.
#
# The last two are here for the opposite reason: they change the arithmetic and
# little else. A cost figure measured on one network is a claim about that
# network, and the two below were designed later against a different constraint
# — arithmetic per image — so putting them beside it turns a single point into a
# comparison. Which way that comparison runs is a question for measurement
# rather than for this comment.
DEFAULT_BACKBONE = "resnet18"
BACKBONES = (
    "resnet18",
    "resnet18_ibn_a",
    "mobilenet_v3_small",
    "efficientnet_b0",
)

# Where each backbone keeps the linear layer that names the classes, as a dotted
# path. One constant rather than two because the path serves both readings of
# the same fact: it reaches the module when a head is being replaced, and naming
# its bias gives the state dictionary key that says how wide a checkpoint's head
# is. A second constant would be a second place for them to disagree.
# Where each backbone takes its input. Named for the same reason HEAD_PATHS is:
# a run that feeds the network more than three channels has to reach the layer
# that reads them, and the four architectures spell its location differently.
STEM_PATHS = {
    "resnet18": "conv1",
    "resnet18_ibn_a": "conv1",
    "mobilenet_v3_small": "features.0.0",
    "efficientnet_b0": "features.0.0",
}

HEAD_PATHS = {
    "resnet18": "fc",
    "resnet18_ibn_a": "fc",
    "mobilenet_v3_small": "classifier.3",
    "efficientnet_b0": "classifier.1",
}

# How each torchvision backbone is built with the weights it was published with.
# The IBN variant is absent on purpose: it does not come from torchvision, and
# the exception it needs is spelled out in ``build_model``.
TORCHVISION_BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "mobilenet_v3_small": (
        mobilenet_v3_small,
        MobileNet_V3_Small_Weights.IMAGENET1K_V1,
    ),
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
}

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
    num_classes: int = NUM_CLASSES,
    backbone: str = DEFAULT_BACKBONE,
    in_channels: int = 3,
    pretrained: bool = True,
) -> nn.Module:
    """Build a ResNet18 with ImageNet weights and a fresh classification head.

    The convolutional layers keep what they learned on ImageNet — edges, textures,
    shapes. Only the final layer is replaced, mapping the 512 features it produces
    to the gesture classes instead of ImageNet's 1000 categories. That mapping is
    what training has to learn.

    The IBN variant is fetched from its authors' repository through
    ``torch.hub`` and cached under ``~/.cache/torch``, so the first build of it
    needs a network and later ones do not. Its weights are ImageNet's, trained by
    them; nothing here retrains it.

    Args:
        num_classes: Outputs the head produces, one per gesture.
        backbone: Which of ``BACKBONES`` to build.
        in_channels: Channels the first convolution reads. Three is the frame
            as published; more is a run that hands the network something
            alongside it, and the extra weights start at zero so that the model
            begins as the unmodified one. See ``widen_stem``.
        pretrained: Start from ImageNet's weights. Training needs them. A
            network about to be overwritten by a checkpoint does not, and
            asking for them anyway reaches the network on any machine that has
            not cached them, which a program meant to score models offline
            cannot afford. Every weight is replaced on loading either way, so
            the two builds score identically.

    Returns:
        The network, on the CPU. Moving it to a device is the caller's job.

    Raises:
        ValueError: If the backbone is not one this project knows how to build.
    """
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}; expected one of {BACKBONES}")

    if backbone in TORCHVISION_BACKBONES:
        builder, published = TORCHVISION_BACKBONES[backbone]
        model = builder(weights=published if pretrained else None)
    else:
        with _weights_landing_on_the_cpu():
            model = torch.hub.load(
                IBN_HUB_REPOSITORY,
                backbone,
                pretrained=pretrained,
                trust_repo=True,
                verbose=False,
            )

    path = HEAD_PATHS[backbone]
    head = _module_at(model, path)
    _replace_module_at(model, path, nn.Linear(head.in_features, num_classes))

    # After the head, not before, so that widening consumes no draw the head
    # would otherwise have taken. A run with extra channels and a run without
    # then start from the same head under the same seed, and differ in the one
    # thing the experiment is about.
    widen_stem(model, backbone, in_channels)
    return model


def widen_stem(model: nn.Module, backbone: str, in_channels: int) -> None:
    """Let the first convolution read more channels than the three it was given.

    A run that hands the network a frame and what changed since its neighbour
    is handing it six channels, and the layer that meets them was published
    expecting three. Rebuilding it wholesale would throw away what ImageNet
    taught the only layer that looks at raw pixels.

    The extra channels start at zero, and that is the experiment's control.
    A model whose new weights are zero computes, on its first step, exactly
    what the unmodified model computes: the extra input is multiplied away.
    So the run begins at the base rather than somewhere near it, and whatever
    it gains it gained from the signal rather than from a different starting
    point. Zero does not stop it learning — a gradient depends on the input,
    which is not zero — it only stops it from starting anywhere else.

    Copying the published weights into the new channels instead would seed the
    difference as though it were a photograph, which it is not: it is signed,
    centred near zero, and dark where nothing happened.

    Args:
        model: Network to modify in place.
        backbone: Which one it is, to find where its first convolution sits.
        in_channels: How many channels the run will feed it. Three leaves the
            model exactly as it was.

    Raises:
        ValueError: If asked for fewer channels than the published layer reads,
            which would silently discard part of a pretrained kernel.
    """
    stem = _module_at(model, STEM_PATHS[backbone])
    if in_channels == stem.in_channels:
        return
    if in_channels < stem.in_channels:
        raise ValueError(
            f"{backbone} reads {stem.in_channels} channels; "
            f"{in_channels} would drop part of its pretrained kernel"
        )

    widened = nn.Conv2d(
        in_channels,
        stem.out_channels,
        kernel_size=stem.kernel_size,
        stride=stem.stride,
        padding=stem.padding,
        bias=stem.bias is not None,
    )
    with torch.no_grad():
        widened.weight.zero_()
        widened.weight[:, : stem.in_channels] = stem.weight
        if stem.bias is not None:
            widened.bias.copy_(stem.bias)

    _replace_module_at(model, STEM_PATHS[backbone], widened)


def stem_width(weights: Mapping[str, object]) -> int:
    """Read how many channels a checkpoint's first convolution expects.

    Read off the file for the same reason the head's width is: a run scoring a
    checkpoint should not have to be told how it was built, and being told
    wrongly fails on a shape mismatch at best.
    """
    return weights[f"{STEM_PATHS[backbone_of(weights)]}.weight"].shape[1]


def strip_head(model: nn.Module, backbone: str = DEFAULT_BACKBONE) -> int:
    """Replace the classifier with a pass-through, so the model emits features.

    What comes out is the vector the classifier would have read: the pooled
    output of the last stage, before anything maps it onto gesture names. A
    head that reads a sequence of those is reading what the network saw, rather
    than what it concluded, and that is a wider signal — seven probabilities
    carry only the conclusion.

    The model is modified in place, which is why nothing here loads weights:
    the caller loads the trained network first and strips it afterwards, so the
    features are the ones that checkpoint produces.

    Args:
        model: A network built by ``build_model``, modified in place.
        backbone: Which one it is, to find where its head lives.

    Returns:
        How wide the feature vector is.
    """
    head = _module_at(model, HEAD_PATHS[backbone])
    width = head.in_features
    _replace_module_at(model, HEAD_PATHS[backbone], nn.Identity())

    return width


def _module_at(model: nn.Module, path: str) -> nn.Module:
    """Follow a dotted path of attribute names and sequence positions.

    Two of the four backbones keep their classifier inside a ``Sequential``, so
    reaching it means indexing as well as attribute lookup. Spelling both in one
    string is what lets ``HEAD_PATHS`` hold a single entry per backbone rather
    than a name and a position that could drift apart.
    """
    module = model
    for step in path.split("."):
        module = module[int(step)] if step.isdigit() else getattr(module, step)
    return module


def _replace_module_at(model: nn.Module, path: str, layer: nn.Module) -> None:
    """Put a layer where a dotted path points, inside whatever holds it."""
    *parents, last = path.split(".")
    parent = _module_at(model, ".".join(parents)) if parents else model
    if last.isdigit():
        parent[int(last)] = layer
    else:
        setattr(parent, last, layer)


def head_width(weights: Mapping[str, object]) -> int:
    """How many classes a checkpoint's head names, whichever backbone wrote it.

    The width belongs to the file, not to the run reading it: a checkpoint
    trained with the idle class carries an eighth output, and a caller that
    assumed seven would build a network the weights do not fit. Asking the file
    keeps that from being something anyone has to remember.

    Args:
        weights: A checkpoint's state dictionary.

    Returns:
        The number of outputs its classification layer produces.
    """
    return len(weights[f"{HEAD_PATHS[backbone_of(weights)]}.bias"])


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
    if any(IBN_MARKER in key for key in weights):
        return "resnet18_ibn_a"

    # Every remaining backbone is told apart by where it keeps its head, and the
    # ResNet the IBN variant shares ``fc`` with has already been ruled out above.
    for name, path in HEAD_PATHS.items():
        if name != "resnet18_ibn_a" and f"{path}.bias" in weights:
            return name

    raise ValueError(
        "the checkpoint names no head this project knows how to build; "
        f"expected one of {sorted(set(HEAD_PATHS.values()))}"
    )


# Where a shift can be inserted, and the omissions are part of the claim. The
# shift has to sit inside a residual branch so that the identity path still
# carries the frame's own activations forward: moving channels on the trunk
# takes them away from the current frame, and the published ablation measures
# that as worse at every proportion. A block with no identity path therefore
# gets no shift at all.
#
# The two ResNets add the input back in every block of their four stages, and
# name the branch's first convolution the same way in each.
RESIDUAL_STAGE_BACKBONES = ("resnet18", "resnet18_ibn_a")
SHIFT_STAGES = ("layer1", "layer2", "layer3", "layer4")

# The first convolution of a block's residual branch. Shifting its input is what
# "inside the residual branch" means in practice: everything the branch computes
# afterwards reads the moved channels, and the skip connection does not.
SHIFT_BRANCH = "conv1"

# The two cheaper networks are built from inverted residual blocks, which keep
# the whole branch as one module and add the input back only when the block
# keeps its width and resolution: nine of EfficientNet-B0's sixteen blocks and
# six of MobileNetV3-Small's eleven. Which ones is read off each block, where
# torchvision records it, rather than listed here. The published online form
# caches "the first 1/8 feature maps of each residual block", and its edge
# deployment runs on MobileNet-V2, a network of this kind.
INVERTED_RESIDUAL_BACKBONES = ("mobilenet_v3_small", "efficientnet_b0")
INVERTED_BRANCH = "block"

# One eighth of the channels travel in each direction, a quarter in total, which
# is where the published ablation puts the peak. The divisor rather than a
# fraction because the fold has to be a whole number of channels, and a float
# would round differently at each width.
FOLD_DIVISOR = 8

# Where a checkpoint records how a shift was configured while it trained. They
# exist so that scoring reads the arrangement off the file instead of being told
# it: a model whose features were learned with neighbours visible produces
# different predictions without them, and a flag that has to be remembered
# eventually is not. Same reasoning as the head width and the backbone name.
SEGMENTS_KEY = "shift_segments"
SHUFFLE_KEY = "shift_shuffled"
CAUSAL_KEY = "shift_causal"
# The share of channels moved joined them late. It could be changed from the
# first shifted run, and until it was recorded a run that changed it was
# rebuilt with the default.
FOLD_KEY = "shift_fold_divisor"

# Draws the control's permutations. Fixed rather than tied to the run's seed,
# and reset before every scoring pass, because successive epochs have to be
# measured against the same thing: a permutation that moved between epochs would
# put the validation curve's noise inside the measurement.
SHUFFLE_SEED = 31


class TemporalShift:
    """Move a fraction of the channels one frame forward and one frame back.

    The classifier this project reports is blind to order by construction: a
    frame is scored on its own and the decision averages those scores, so no
    permutation of the frames changes the answer. This is the cheapest way to
    cross that line without changing what the network costs. A slice of the
    channels of frame t is replaced by the same slice from frame t+1, another
    slice by the one from t-1, and the rest is left alone. No multiply happens
    and no weight is added; the arithmetic of the forward pass is untouched and
    only the contents of a tensor move.

    The frames at the ends of a clip have no neighbour on one side, and what
    arrives there is zero rather than a wrap-around. Wrapping would hand the
    first frame the last one as its past, which is a sequence the data never
    contains.

    It is a hook rather than a module on purpose. Wrapping a convolution would
    rename every weight below it, and the checkpoints this project already has
    would stop loading into a network that carries the shift. As a hook the
    parameter names are the ones every other run wrote, and what marks a
    checkpoint as having trained with a shift is a recorded number instead of a
    renamed tree.

    How many frames a clip holds is set per batch rather than fixed at
    construction, because a run trains on gesture clips and clips of the
    material before the gesture in the same epoch, and fewer of the latter are
    stored.

    The control arm permutes the clip before shifting and puts the order back
    afterwards. That is what makes it a control rather than a different
    experiment: each frame still receives a slice of some other frame of the
    same clip, so the capacity, the parameter count and the arithmetic are
    identical, and the only thing destroyed is which frames are adjacent.
    Restoring the order matters as much as the permutation, because the rows
    have to come back in the order the manifest lists them or every probability
    is attached to the wrong frame.

    The causal arm takes channels from the past only, and it exists for a reason
    that is about the argument rather than the accuracy. Reading a slice of the
    next frame means a decision cannot be made until that frame exists — which
    is the property this project holds against models that consume a clip, and
    the shift as published for offline use has it too. One direction removes it:
    a frame then needs only frames already seen, so the model can answer as each
    one arrives. The published online variant is exactly this, and verified in
    the source: *"in the online setting, we cannot access future frames,
    therefore, we only shift from past frames to future frames in a
    uni-directional fashion"*, saving an eighth of each block's feature maps to
    reuse on the next frame.

    ``fold_divisor`` is the share taken from each direction, so the causal arm
    at the default moves half of what the two-directional one moves. That is
    the published design and the deployable one; holding the moved share equal
    instead is the same flag at a divisor of four, and it is the control that
    separates losing a direction from moving fewer channels.
    """

    def __init__(self, fold_divisor: int = FOLD_DIVISOR, causal: bool = False):
        """Hold the proportion to move, with no clip length yet.

        Args:
            fold_divisor: One over the share of channels travelling in each
                direction. Eight moves a quarter of them in total, or an eighth
                when only one direction is used.
            causal: Take channels from the past only, so that a frame needs no
                frame after it and the model can answer as each one arrives.
        """
        self.fold_divisor = fold_divisor
        self.causal = causal
        self.segments: int | None = None
        self.order: torch.Tensor | None = None

    def __call__(self, module: nn.Module, inputs: tuple) -> tuple:
        """Shift the tensor about to enter a residual branch.

        Args:
            module: Where the branch begins. Unused, and present because a
                forward pre-hook is called with it.
            inputs: What the convolution was going to read, as a one-tuple.

        Returns:
            The same tuple with the channels moved along the clip.

        Raises:
            RuntimeError: If no clip length has been set, if the batch does not
                divide into clips of that length, or if the proportion rounds
                down to no channels at this width. All three would otherwise
                read as a model that trains and means nothing.
        """
        (frames,) = inputs

        if self.segments is None:
            raise RuntimeError(
                "no clip length set; call set_segments before the forward pass"
            )
        if frames.shape[0] % self.segments:
            raise RuntimeError(
                f"{frames.shape[0]} frames do not divide into clips of "
                f"{self.segments}; the batch is not clip-aligned"
            )

        channels = frames.shape[1]
        fold = channels // self.fold_divisor
        if not fold:
            raise RuntimeError(
                f"{channels} channels over a divisor of {self.fold_divisor} "
                "leaves nothing to shift"
            )

        clips = frames.shape[0] // self.segments
        ordered = frames.view(clips, self.segments, *frames.shape[1:])
        if self.order is not None:
            ordered = ordered[:, self.order.to(frames.device)]

        shifted = torch.zeros_like(ordered)
        if self.causal:
            # One direction: the fold reads the previous frame and nothing reads
            # ahead, so a decision never waits on a frame that has not arrived.
            shifted[:, 1:, :fold] = ordered[:, :-1, :fold]
            shifted[:, :, fold:] = ordered[:, :, fold:]
        else:
            # The first fold reads the next frame, the second the previous one,
            # and the remainder stays where it is. Assigning into zeros is what
            # leaves the ends without a neighbour at zero.
            shifted[:, :-1, :fold] = ordered[:, 1:, :fold]
            shifted[:, 1:, fold : 2 * fold] = ordered[:, :-1, fold : 2 * fold]
            shifted[:, :, 2 * fold :] = ordered[:, :, 2 * fold :]

        if self.order is not None:
            shifted = shifted[:, self.order.argsort().to(frames.device)]

        return (shifted.reshape(frames.shape),)


def residual_branches(model: nn.Module, backbone: str) -> list[nn.Module]:
    """Find where each residual branch of a network begins.

    A shift is hooked onto the input of each module returned, so that the
    branch reads the moved channels and the identity path around it does not.

    Args:
        model: The network, as ``build_model`` returns it.
        backbone: Which architecture it is, which says how its blocks are laid
            out.

    Returns:
        One module per block that adds its input back, shallowest first.

    Raises:
        ValueError: If the backbone is not one this project builds.
    """
    if backbone in RESIDUAL_STAGE_BACKBONES:
        return [
            _module_at(block, SHIFT_BRANCH)
            for stage in SHIFT_STAGES
            for block in _module_at(model, stage)
        ]
    if backbone in INVERTED_RESIDUAL_BACKBONES:
        return [
            _module_at(block, INVERTED_BRANCH)
            for block in model.modules()
            if getattr(block, "use_res_connect", False)
        ]

    raise ValueError(f"cannot shift inside {backbone!r}; expected one of {BACKBONES}")


def insert_temporal_shift(
    model: nn.Module,
    backbone: str = DEFAULT_BACKBONE,
    segments: int = 8,
    fold_divisor: int = FOLD_DIVISOR,
    shuffle: bool = False,
    causal: bool = False,
) -> list[TemporalShift]:
    """Give every residual branch a shift, and record the arrangement on the model.

    The shifts are returned and also kept on the model, because two callers need
    them for different reasons: a test wants to inspect them, and the training
    loop has to tell them how long the batch's clips are before each forward.

    What was recorded travels in the checkpoint, as buffers rather than
    attributes. That is what lets scoring rebuild the same network from the file
    alone, including whether this was the ordered arm or the control.

    Args:
        model: The network to modify, already carrying its final head.
        backbone: Which architecture it is. Read rather than guessed, for the
            same reason every other edit here takes it.
        segments: Frames per clip the run trains on. Recorded, not enforced:
            the shift reads the length of the batch in front of it.
        fold_divisor: Passed to each shift, and recorded with the rest: a
            network that learned to read a quarter of its channels from its
            neighbours answers differently when it is handed an eighth.
        shuffle: Destroys which frames are adjacent, leaving everything else
            in place. This is the control that isolates order.
        causal: Takes channels from the past only, so that no decision waits on
            a frame that has not arrived. Recorded like the rest, because a
            model that learned to read ahead scores differently when it cannot.

    Returns:
        The shifts, one per residual branch, in the order they were inserted.

    Raises:
        ValueError: If the backbone is not one this project builds.
    """
    shifts = []
    for branch in residual_branches(model, backbone):
        shift = TemporalShift(fold_divisor, causal)
        branch.register_forward_pre_hook(shift)
        shifts.append(shift)

    model.register_buffer(SEGMENTS_KEY, torch.tensor(segments))
    model.register_buffer(SHUFFLE_KEY, torch.tensor(shuffle))
    model.register_buffer(CAUSAL_KEY, torch.tensor(causal))
    model.register_buffer(FOLD_KEY, torch.tensor(fold_divisor))
    model.temporal_shifts = shifts
    model.shift_generator = torch.Generator().manual_seed(SHUFFLE_SEED)
    return shifts


def set_segments(model: nn.Module, segments: int) -> None:
    """Tell every shift how many frames the next batch's clips hold.

    The control's permutation is drawn here, once, and handed to every shift.
    One draw rather than one per shift is the point: a permutation that differed
    by depth would leave each layer reading a different sequence, which is noise
    injected eight times over rather than a clip whose order has been destroyed.

    A no-op on a network without shifts, so that the training and scoring loops
    can call it unconditionally and a run without the treatment behaves exactly
    as the runs recorded before it.

    Args:
        model: The network about to run.
        segments: Frames per clip in the batch it is about to read.
    """
    shifts = getattr(model, "temporal_shifts", ())
    if not shifts:
        return

    order = None
    if bool(model.get_buffer(SHUFFLE_KEY)):
        order = torch.randperm(segments, generator=model.shift_generator)

    for shift in shifts:
        shift.segments = segments
        shift.order = order


def reset_shuffle(model: nn.Module, seed: int = SHUFFLE_SEED) -> None:
    """Put the control's permutations back to the start of their sequence.

    Called before a scoring pass so that successive passes over the same frames
    return the same numbers. Without it the permutations a validation pass draws
    would depend on how many batches training happened to consume beforehand,
    and the epoch-to-epoch curve would carry that as noise.

    A no-op on a network without shifts, and on the ordered arm it changes
    nothing that is read.

    Args:
        model: The network about to be scored.
        seed: Where the sequence restarts.
    """
    if hasattr(model, "shift_generator"):
        model.shift_generator.manual_seed(seed)


class StreamingShift:
    """The causal shift for frames that arrive one at a time.

    Offline a whole clip is in memory, and the causal shift hands each frame a
    slice of the channels of the frame before it. Online there is no clip: a
    frame arrives, the model answers, and the next frame has not been filmed
    yet. What that next frame will need from this one is exactly the slice, so
    it is kept, one per residual branch, and handed over when the next frame
    arrives. That is the published online form: the slice a frame passes on is
    cached rather than recomputed.

    The first frame of a stream finds nothing kept and reads zeros, which is
    what the offline shift gives the first frame of a clip, so over one clip the
    two produce the same outputs frame for frame. After that they part: a
    stream never stops passing slices along, while the scored clips start
    again from zero every eight frames. That changes what a frame reads, not
    what it costs to read it.

    One frame per call. Several streams at once would each need their own
    cache, and nothing here asks for that.
    """

    def __init__(self, fold_divisor: int = FOLD_DIVISOR):
        """Start with nothing kept, as at the start of a stream.

        Args:
            fold_divisor: One over the share of channels handed to the next
                frame, as in the offline causal shift.
        """
        self.fold_divisor = fold_divisor
        self.kept: torch.Tensor | None = None

    def __call__(self, module: nn.Module, inputs: tuple) -> tuple:
        """Hand this frame the slice the previous one kept, and keep its own.

        Args:
            module: Where the branch begins. Unused, and present because a
                forward pre-hook is called with it.
            inputs: The frame about to enter the residual branch, as a one-tuple.

        Returns:
            The same tuple with the slice replaced by the previous frame's.

        Raises:
            RuntimeError: If more than one frame arrives at once, or if the
                proportion rounds down to no channels at this width.
        """
        (frame,) = inputs

        if frame.shape[0] != 1:
            raise RuntimeError(
                f"a stream carries one frame per call, not {frame.shape[0]}"
            )
        fold = frame.shape[1] // self.fold_divisor
        if not fold:
            raise RuntimeError(
                f"{frame.shape[1]} channels over a divisor of {self.fold_divisor} "
                "leaves nothing to shift"
            )

        shifted = frame.clone()
        if self.kept is None:
            shifted[:, :fold] = 0
        else:
            shifted[:, :fold] = self.kept
        self.kept = frame[:, :fold].clone()

        return (shifted,)


def stream_temporal_shift(
    model: nn.Module,
    backbone: str = DEFAULT_BACKBONE,
    fold_divisor: int = FOLD_DIVISOR,
) -> list[StreamingShift]:
    """Give every residual branch the online causal shift.

    The branches are the ones ``insert_temporal_shift`` uses, so a network
    built this way runs the arrangement the causal arm trained with, one frame
    at a time.

    Args:
        model: The network to modify, already carrying its final head.
        backbone: Which architecture it is.
        fold_divisor: Passed to each shift.

    Returns:
        The shifts, one per residual branch, in the order they were inserted.

    Raises:
        ValueError: If the backbone is not one this project builds.
    """
    shifts = []
    for branch in residual_branches(model, backbone):
        shift = StreamingShift(fold_divisor)
        branch.register_forward_pre_hook(shift)
        shifts.append(shift)

    model.streaming_shifts = shifts
    return shifts


def reset_stream(model: nn.Module) -> None:
    """Forget what a stream kept, so that the next frame starts a new one.

    A no-op on a network without a streaming shift.

    Args:
        model: The network about to read a new stream.
    """
    for shift in getattr(model, "streaming_shifts", ()):
        shift.kept = None


def shift_segments(weights: Mapping[str, object]) -> int | None:
    """How many frames per clip a checkpoint trained with, or None if it had no shift.

    Args:
        weights: A state dictionary, as ``torch.load`` returns it.

    Returns:
        The clip length the run recorded, or ``None`` for every checkpoint
        written before the shift existed and every run without it.
    """
    recorded = weights.get(SEGMENTS_KEY)
    return None if recorded is None else int(recorded)


# The markers, as one list. Named once because two things read it: what gets
# registered on a model, and what a load is allowed to fill in.
MARKER_KEYS = (SEGMENTS_KEY, SHUFFLE_KEY, CAUSAL_KEY, FOLD_KEY)


def load_shifted(model: nn.Module, weights: Mapping[str, object]) -> None:
    """Load weights into a shifted network, tolerating a marker it predates.

    The markers are the one part of a shifted network's state that this project
    adds to over time: the clip length came first, then whether the order was
    destroyed, then whether only the past was read, then what share of the
    channels moved. A checkpoint written before one of them existed is not
    wrong — it is a run that did not have the choice — but a strict load
    refuses it for a missing key, and that refusal arrives after the manifest
    and the model are already in memory.

    Filling a missing marker from the network's own buffer is correct rather
    than permissive, because the network was built from this same file: the
    readers return the default for a marker that is absent, and for the three
    that describe the arrangement the default is what the run actually did,
    since each was recorded from the day its option existed. The share of
    channels is the exception. It could be changed before it was recorded, so
    a file without it reads as the default, and a run that changed it is
    misread unless the marker is written into its file.

    Everything that is not a marker stays strict, so a real mismatch — a head
    of the wrong width, a stem of the wrong depth — still fails as loudly as
    before.

    Args:
        model: The network to load into, with its shift already inserted.
        weights: A state dictionary, as ``torch.load`` returns it.
    """
    state = dict(weights)
    for key in MARKER_KEYS:
        if key not in state:
            state[key] = model.get_buffer(key)

    model.load_state_dict(state)


def shift_causal(weights: Mapping[str, object]) -> bool:
    """Whether a checkpoint reads only the past, and can therefore run online.

    Args:
        weights: A state dictionary, as ``torch.load`` returns it.

    Returns:
        True if the run that wrote it shifted in one direction. False for the
        two-directional arm and for every checkpoint written without a shift.
    """
    recorded = weights.get(CAUSAL_KEY)
    return False if recorded is None else bool(recorded)


def shift_shuffled(weights: Mapping[str, object]) -> bool:
    """Whether a checkpoint is the control arm rather than the ordered one.

    Args:
        weights: A state dictionary, as ``torch.load`` returns it.

    Returns:
        True if the run that wrote it destroyed the order of each clip. False
        for the ordered arm and for every checkpoint written without a shift.
    """
    recorded = weights.get(SHUFFLE_KEY)
    return False if recorded is None else bool(recorded)


def shift_fold_divisor(weights: Mapping[str, object]) -> int:
    """What share of its channels a checkpoint's shift moved, as the divisor.

    Args:
        weights: A state dictionary, as ``torch.load`` returns it.

    Returns:
        The divisor the run recorded, or the default for a file written before
        it was recorded. The default is right for every such file except one
        whose run changed it, which is why the marker now travels with the rest.
    """
    recorded = weights.get(FOLD_KEY)
    return FOLD_DIVISOR if recorded is None else int(recorded)


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
    # Only the ResNets name their stages this way. Refusing a backbone that does
    # not is deliberate: the alternative is inventing a mapping for a network
    # whose blocks divide differently, on an axis the project measured and
    # closed as negative. Failing here is cheaper than a silently different
    # experiment.
    missing = [name for name in FROZEN_STAGES[depth] if not hasattr(model, name)]
    if missing:
        raise ValueError(
            f"this backbone has no stages named {missing}; --freeze is only "
            "defined for the ResNet backbones"
        )

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

    Equal weight per *batch*, not per frame: a short final batch counts as
    much as a full one. With frames in the thousands and batches in the dozens
    the difference is far below the noise of any comparison this feeds, and
    dropping the short batch would discard target frames to fix it.

    This reads data from the domain being scored, so a result it produces is
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
