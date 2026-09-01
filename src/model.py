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
