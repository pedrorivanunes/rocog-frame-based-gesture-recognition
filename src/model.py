"""Build the network a run trains and evaluates.

Kept apart from the training loop because more than one caller needs it: a
training run, and an inference pass that scores a manifest with weights trained
earlier. Keeping it here also gives the cost study its home — comparing this
backbone against cheaper ones is a choice about the model, not about the loop
that consumes it.
"""

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

NUM_CLASSES = 7


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Build a ResNet18 with ImageNet weights and a fresh classification head.

    The convolutional layers keep what they learned on ImageNet — edges, textures,
    shapes. Only the final layer is replaced, mapping the 512 features it produces
    to the gesture classes instead of ImageNet's 1000 categories. That mapping is
    what training has to learn.

    Args:
        num_classes: Outputs the head produces, one per gesture.

    Returns:
        The network, on the CPU. Moving it to a device is the caller's job.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


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
