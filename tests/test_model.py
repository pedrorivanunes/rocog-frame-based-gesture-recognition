import pytest
import torch
from torchvision.models import resnet18

from model import FROZEN_STAGES, freeze

# Weights are irrelevant to what these tests check — which parameters carry a
# gradient and which statistics move — and skipping the download keeps the suite
# runnable offline.
BATCH = torch.randn(4, 3, 224, 224)


@pytest.fixture
def model():
    return resnet18(weights=None)


def test_freezing_nothing_leaves_every_parameter_trainable(model):
    """The default has to reproduce the runs recorded before the option."""
    assert freeze(model, "none") == []
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_freezing_the_backbone_leaves_only_the_head(model):
    freeze(model, "backbone")

    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]

    assert trainable == ["fc.weight", "fc.bias"]


def test_freezing_the_early_stages_leaves_the_late_ones(model):
    freeze(model, "early")

    trainable = {
        name.split(".")[0]
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert trainable == {"layer3", "layer4", "fc"}


def test_a_frozen_stage_keeps_its_statistics_through_a_training_pass(model):
    """The point of the freeze: the normalisation must not drift onto the data.

    Clearing ``requires_grad`` alone would not do this — a batch normalization
    layer updates its running mean from what it sees, gradient or not.
    """
    frozen = freeze(model, "backbone")
    before = model.bn1.running_mean.clone()

    model.train()
    for stage in frozen:
        stage.eval()
    model(BATCH)

    assert torch.equal(model.bn1.running_mean, before)


def test_training_mode_alone_would_have_moved_them(model):
    """Why the frozen stages come back as a return value rather than a side effect."""
    freeze(model, "backbone")
    before = model.bn1.running_mean.clone()

    model.train()
    model(BATCH)

    assert not torch.equal(model.bn1.running_mean, before)


def test_the_head_is_the_only_stage_no_depth_freezes(model):
    """A structural check: 'backbone' has to stay 'everything but the head'.

    Named stages can fall out of step with the architecture — a torchvision that
    added a parametrised stage would leave it silently trainable under a depth
    that promises to freeze the lot.
    """
    parametrised = {name.split(".")[0] for name, _ in model.named_parameters()}

    assert parametrised - set(FROZEN_STAGES["backbone"]) == {"fc"}
