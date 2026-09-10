import pytest
import torch
from torch import nn
from torchvision.models import resnet18

from model import (
    BACKBONES,
    DEFAULT_BACKBONE,
    FROZEN_STAGES,
    NORMALISATION_LAYERS,
    adapt_batchnorm,
    backbone_of,
    build_model,
    freeze,
)

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


CPU = torch.device("cpu")
CHANNELS = 3


@pytest.fixture
def normaliser():
    """A network that is nothing but the layer under test."""
    return nn.BatchNorm2d(CHANNELS)


def batch(*, scale: float = 1.0, shift: float = 0.0, size: int = 8):
    """One batch shaped the way a loader serves them: frames first, rest ignored."""
    return (torch.randn(size, CHANNELS, 4, 4) * scale + shift, torch.zeros(size))


def test_the_statistics_land_on_the_frames_it_saw(normaliser):
    """The whole point: the layer stops describing the domain it was trained on."""
    frames, _ = batch(scale=3.0, shift=5.0)

    assert adapt_batchnorm(normaliser, [(frames,)], CPU) == len(frames)
    assert torch.allclose(
        normaliser.running_mean, frames.mean(dim=(0, 2, 3)), atol=1e-5
    )


def test_the_order_the_batches_arrive_in_does_not_matter():
    """Why the momentum is cleared rather than left at its default.

    A layer keeping its momentum weights the last batches most, which would make
    the adapted model depend on how the loader happened to shuffle.
    """
    batches = [batch(shift=-4.0), batch(shift=0.0), batch(shift=4.0)]
    forwards, backwards = nn.BatchNorm2d(CHANNELS), nn.BatchNorm2d(CHANNELS)

    adapt_batchnorm(forwards, batches, CPU)
    adapt_batchnorm(backwards, list(reversed(batches)), CPU)

    assert torch.allclose(forwards.running_mean, backwards.running_mean, atol=1e-6)


def test_the_momentum_comes_back(normaliser):
    """It is borrowed for the pass, not spent: the model can still be trained."""
    before = normaliser.momentum

    adapt_batchnorm(normaliser, [batch()], CPU)

    assert normaliser.momentum == before


def test_the_model_comes_back_ready_to_score(normaliser):
    adapt_batchnorm(normaliser, [batch()], CPU)

    assert not normaliser.training


def test_no_weight_moves_and_no_gradient_is_built(normaliser):
    """It is a re-estimation, not a training step, and nothing should suggest one."""
    weight = normaliser.weight.clone()

    adapt_batchnorm(normaliser, [batch()], CPU)

    assert torch.equal(normaliser.weight, weight)
    assert all(parameter.grad is None for parameter in normaliser.parameters())


def test_adapting_on_nothing_is_refused(normaliser):
    """Silence here would hand back a layer normalising by a mean of zero."""
    with pytest.raises(ValueError, match="no frames"):
        adapt_batchnorm(normaliser, [], CPU)


def test_every_normalisation_layer_in_the_network_is_reached(model):
    """A structural check: reaching only the top-level ones would be invisible."""
    before = {
        name: module.running_mean.clone()
        for name, module in model.named_modules()
        if isinstance(module, NORMALISATION_LAYERS)
    }

    adapt_batchnorm(model, [(BATCH,)], CPU)

    assert len(before) == 20
    assert not any(
        torch.equal(dict(model.named_modules())[name].running_mean, running_mean)
        for name, running_mean in before.items()
    )


def test_a_torchvision_checkpoint_reads_back_as_the_default_backbone(model):
    assert backbone_of(model.state_dict()) == DEFAULT_BACKBONE


def test_an_instance_normalised_checkpoint_reads_back_as_the_ibn_one(model):
    """The marker is a parameter torchvision's ResNet18 has no name for."""
    weights = dict(model.state_dict())
    weights["layer1.0.bn1.IN.weight"] = torch.ones(64)

    assert backbone_of(weights) == "resnet18_ibn_a"


def test_an_unknown_backbone_is_refused_before_anything_is_downloaded():
    """Named wrong, it must fail here rather than fetching some other network."""
    with pytest.raises(ValueError, match="unknown backbone"):
        build_model(7, "resnet50")


def test_the_default_backbone_is_one_of_the_offered_ones():
    """A structural check: the two constants must not drift apart."""
    assert DEFAULT_BACKBONE in BACKBONES


def test_the_cpu_loading_patch_is_put_back():
    """A leaked patch would silently redirect every later load in the process."""
    from model import _weights_landing_on_the_cpu

    original = torch.load
    with _weights_landing_on_the_cpu():
        assert torch.load is not original
    assert torch.load is original


def test_the_cpu_loading_patch_is_put_back_even_after_a_failure():
    """The backbone it exists for is fetched over a network, which can fail."""
    from model import _weights_landing_on_the_cpu

    original = torch.load
    with pytest.raises(RuntimeError), _weights_landing_on_the_cpu():
        raise RuntimeError("as a download would")
    assert torch.load is original
