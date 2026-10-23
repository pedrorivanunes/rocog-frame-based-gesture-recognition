import pytest
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, resnet18

import model as model_module
from model import (
    BACKBONES,
    CAUSAL_KEY,
    DEFAULT_BACKBONE,
    FOLD_DIVISOR,
    FOLD_KEY,
    FROZEN_STAGES,
    HEAD_PATHS,
    NORMALISATION_LAYERS,
    SEGMENTS_KEY,
    SHUFFLE_KEY,
    TORCHVISION_BACKBONES,
    TemporalShift,
    adapt_batchnorm,
    backbone_of,
    build_model,
    freeze,
    head_width,
    insert_temporal_shift,
    load_shifted,
    reset_shuffle,
    reset_stream,
    set_segments,
    shift_causal,
    shift_fold_divisor,
    shift_segments,
    shift_shuffled,
    stem_width,
    stream_temporal_shift,
    strip_head,
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


# --- the cost curve's backbones ------------------------------------------------
#
# A ResNet18 is a 2015 design, so every cost figure the project quotes is about
# that network rather than about frame classification. These three answer the
# same task at 11.18, 4.02 and 1.53 M parameters, which is what makes the
# comparison a curve. They differ in where they keep the layer that names the
# classes, and that difference is the whole of what the code had to learn.


@pytest.fixture
def offline_backbones(monkeypatch):
    """Build the torchvision backbones unweighted, so the suite stays offline.

    ``build_model`` asks for published weights, which would reach the network on
    a machine that has not cached them. What these tests check — where the head
    lands and how wide it is — does not depend on a single weight.
    """
    monkeypatch.setattr(
        model_module,
        "TORCHVISION_BACKBONES",
        {name: (builder, None) for name, (builder, _) in TORCHVISION_BACKBONES.items()},
    )


@pytest.mark.parametrize("backbone", sorted(TORCHVISION_BACKBONES))
def test_every_backbone_gets_a_head_of_the_width_it_was_asked_for(
    offline_backbones, backbone
):
    """Two of them keep a second Linear earlier in the classifier.

    MobileNetV3 runs its features through a Linear, an activation and a dropout
    before the layer that names the classes, so replacing the first one found
    would leave a network still answering ImageNet's thousand categories.
    """
    built = build_model(8, backbone)

    assert built(torch.randn(2, 3, 224, 224)).shape == (2, 8)


@pytest.mark.parametrize("backbone", sorted(TORCHVISION_BACKBONES))
def test_each_backbone_reads_back_off_its_own_checkpoint(offline_backbones, backbone):
    """Which architecture wrote a file is the file's to say, not the caller's."""
    weights = build_model(8, backbone).state_dict()

    assert backbone_of(weights) == backbone
    assert head_width(weights) == 8


def test_the_head_width_comes_from_the_file_rather_than_a_default(offline_backbones):
    """A run trained with the idle class carries an output the dataset lacks."""
    seven = build_model(7, "efficientnet_b0").state_dict()
    eight = build_model(8, "efficientnet_b0").state_dict()

    assert (head_width(seven), head_width(eight)) == (7, 8)


@pytest.fixture
def requested_weights(monkeypatch):
    """Record which weights each torchvision builder is asked for, fetching none."""
    requested = []

    def recording(builder):
        def build(weights):
            requested.append(weights)
            return builder(weights=None)

        return build

    monkeypatch.setattr(
        model_module,
        "TORCHVISION_BACKBONES",
        {
            name: (recording(builder), published)
            for name, (builder, published) in TORCHVISION_BACKBONES.items()
        },
    )
    return requested


@pytest.mark.parametrize("backbone", sorted(TORCHVISION_BACKBONES))
def test_a_network_about_to_be_overwritten_asks_for_no_weights(
    requested_weights, backbone
):
    """Loading a checkpoint replaces every weight, so fetching first is a download."""
    build_model(8, backbone, pretrained=False)

    assert requested_weights == [None]


@pytest.mark.parametrize("backbone", sorted(TORCHVISION_BACKBONES))
def test_training_still_starts_from_the_published_weights(requested_weights, backbone):
    """The default has to reproduce every run trained before the option existed."""
    build_model(8, backbone)

    assert requested_weights == [TORCHVISION_BACKBONES[backbone][1]]


def test_a_checkpoint_naming_no_head_this_project_builds_is_refused():
    """Silence here would build the wrong network and score it anyway."""
    with pytest.raises(ValueError, match="names no head"):
        backbone_of({"features.0.weight": torch.zeros(3)})


def test_freezing_is_refused_on_a_backbone_with_no_stages_to_freeze():
    """The stage names are the ResNets'. Inventing a mapping would be a guess."""
    with pytest.raises(ValueError, match="--freeze is only"):
        freeze(mobilenet_v3_small(weights=None), "early")


def test_every_backbone_offered_knows_where_it_keeps_its_head():
    """``BACKBONES`` is what the command line accepts; a gap there is a crash."""
    assert set(HEAD_PATHS) == set(BACKBONES)


def test_stripping_the_head_leaves_the_features_the_classifier_would_have_read():
    """A sequence head reads what the network saw, not what it concluded."""
    model = build_model(8, "resnet18")

    width = strip_head(model, "resnet18")

    assert width == 512
    assert model(torch.zeros(2, 3, 224, 224)).shape == (2, 512)


def test_every_backbone_can_be_stripped():
    """The option has to survive the cost curve's networks, not only the base."""
    for backbone in ("mobilenet_v3_small", "efficientnet_b0"):
        model = build_model(8, backbone)
        width = strip_head(model, backbone)
        assert model(torch.zeros(1, 3, 224, 224)).shape == (1, width)


def test_a_widened_stem_ignores_its_new_channels_until_it_learns_to():
    """The control the experiment rests on: the run starts at the base.

    Weights of zero multiply the extra input away, so whatever the run gains
    it gained from the signal and not from having started somewhere else.
    """
    model = build_model(8, "resnet18", in_channels=6).eval()
    frame = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        with_noise = model(torch.cat([frame, torch.randn(2, 3, 224, 224) * 5], 1))
        with_nothing = model(torch.cat([frame, torch.zeros(2, 3, 224, 224)], 1))

    assert torch.equal(with_noise, with_nothing)


def test_a_widened_model_answers_exactly_what_the_base_answers():
    """Same seed, same head, same numbers — they differ in one thing only.

    Widening happens after the head is replaced so that it consumes no draw
    the head would have taken; otherwise the two would start from different
    classifiers and the comparison would carry that difference too.
    """
    torch.manual_seed(0)
    base = build_model(8, "resnet18").eval()
    torch.manual_seed(0)
    widened = build_model(8, "resnet18", in_channels=6).eval()
    frame = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        plain = base(frame)
        paired = widened(torch.cat([frame, torch.randn(2, 3, 224, 224)], 1))

    assert torch.equal(plain, paired)


def test_a_widened_stem_keeps_the_kernel_imagenet_taught_it():
    base = build_model(8, "resnet18")
    widened = build_model(8, "resnet18", in_channels=6)
    assert torch.equal(widened.conv1.weight[:, :3], base.conv1.weight)
    assert torch.equal(widened.conv1.weight[:, 3:], torch.zeros(64, 3, 7, 7))


def test_the_new_channels_can_still_be_learned():
    """Zero weights do not stop a gradient: it depends on the input, not them."""
    model = build_model(8, "resnet18", in_channels=6)

    model(torch.randn(1, 6, 224, 224)).sum().backward()

    assert model.conv1.weight.grad[:, 3:].abs().sum() > 0


def test_asking_for_fewer_channels_than_published_is_refused():
    """It would quietly discard part of a pretrained kernel."""
    with pytest.raises(ValueError, match="drop part of its pretrained kernel"):
        build_model(8, "resnet18", in_channels=1)


@pytest.mark.parametrize("backbone", ["mobilenet_v3_small", "efficientnet_b0"])
def test_every_backbone_can_read_a_pair(backbone):
    """The cost curve's networks have to survive the option too."""
    model = build_model(8, backbone, in_channels=6).eval()

    assert model(torch.zeros(1, 6, 224, 224)).shape == (1, 8)


def test_how_many_channels_a_checkpoint_expects_is_read_off_the_file():
    widened = build_model(8, "resnet18", in_channels=6)

    assert stem_width(widened.state_dict()) == 6
    assert stem_width(build_model(8, "resnet18").state_dict()) == 3


def shifted_network(backbone="resnet18", **options):
    """A network carrying a shift, with weights that are irrelevant here."""
    builder, _ = TORCHVISION_BACKBONES[backbone]
    network = builder(weights=None)
    insert_temporal_shift(network, backbone, **options)

    return network


def same(first: torch.Tensor, second: torch.Tensor) -> bool:
    """Equal up to rounding, at whatever scale the network answers in.

    Untrained, the two inverted residual networks answer in the order of 1e-10,
    so a fixed tolerance would call any two of their outputs equal.
    """
    return bool((first - second).norm() <= 1e-5 * first.norm())


def indexed_clip(frames: int, channels: int) -> torch.Tensor:
    """A clip where every value is the index of the frame it belongs to.

    A value therefore says which frame it travelled from, which is what makes
    the movement checkable rather than merely non-crashing.
    """
    return (
        torch.arange(frames, dtype=torch.float32)
        .view(frames, 1, 1, 1)
        .expand(frames, channels, 1, 1)
        .contiguous()
    )


def test_the_shift_takes_a_fold_from_each_neighbour_and_leaves_the_rest():
    shift = TemporalShift(fold_divisor=4)
    shift.segments = 4

    (moved,) = shift(nn.Identity(), (indexed_clip(4, 8),))

    values = moved[:, :, 0, 0]
    assert values[:, 0].tolist() == [1.0, 2.0, 3.0, 0.0]
    assert values[:, 2].tolist() == [0.0, 0.0, 1.0, 2.0]
    assert values[:, 4].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_the_ends_of_a_clip_receive_zero_rather_than_the_other_end():
    """Wrapping would hand the first frame the last one as its past."""
    shift = TemporalShift(fold_divisor=4)
    shift.segments = 4

    (moved,) = shift(nn.Identity(), (indexed_clip(4, 8),))

    assert moved[3, 0, 0, 0] == 0.0
    assert moved[0, 2, 0, 0] == 0.0


def test_a_permuted_clip_comes_back_in_the_order_it_arrived():
    """The rows are matched to a manifest positionally, so they cannot move."""
    shift = TemporalShift(fold_divisor=4)
    shift.segments = 4
    shift.order = torch.tensor([1, 0, 3, 2])

    (moved,) = shift(nn.Identity(), (indexed_clip(4, 8),))

    # Whatever the permutation did, a row still carries its own frame in the
    # channels the shift does not touch.
    assert moved[:, 4, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    # And the neighbours it received are the ones the permuted sequence puts
    # beside it, not the ones in filmed order.
    assert moved[:, 0, 0, 0].tolist() == [3.0, 0.0, 0.0, 2.0]


@pytest.mark.parametrize(
    ("backbone", "blocks"),
    [("resnet18", 8), ("efficientnet_b0", 9), ("mobilenet_v3_small", 6)],
)
def test_every_block_that_adds_its_input_back_gets_one_shift(backbone, blocks):
    """Eight basic blocks; nine and six inverted ones that keep width and size.

    The inverted residual blocks that change width or resolution have no
    identity path, and a shift there would take channels away from the frame.
    """
    assert len(shifted_network(backbone, segments=8).temporal_shifts) == blocks


@pytest.mark.parametrize("backbone", sorted(TORCHVISION_BACKBONES))
def test_the_shift_costs_no_parameter_and_renames_no_weight(backbone):
    """The published claim, and the reason a hook was used instead of a wrapper."""
    plain = TORCHVISION_BACKBONES[backbone][0](weights=None)
    shifted = shifted_network(backbone, segments=8)

    assert sum(p.numel() for p in plain.parameters()) == sum(
        p.numel() for p in shifted.parameters()
    )
    # The four recorded markers, and nothing else: how long a clip was, whether
    # the order was destroyed, whether it read only the past, and what share of
    # the channels moved. Each is read back by scoring, which is why they travel
    # in the file at all.
    assert set(shifted.state_dict()) - set(plain.state_dict()) == {
        SEGMENTS_KEY,
        SHUFFLE_KEY,
        CAUSAL_KEY,
        FOLD_KEY,
    }
    assert not set(plain.state_dict()) - set(shifted.state_dict())


def test_a_checkpoint_says_how_it_was_shifted():
    """Scoring reads it off the file, so the flag cannot be forgotten."""
    ordered = shifted_network(segments=8).state_dict()
    control = shifted_network(segments=4, shuffle=True).state_dict()

    assert shift_segments(ordered) == 8
    assert shift_shuffled(ordered) is False
    assert shift_segments(control) == 4
    assert shift_shuffled(control) is True


def test_a_checkpoint_says_what_share_of_its_channels_moved():
    wide = shifted_network(segments=8, fold_divisor=4, causal=True).state_dict()

    assert shift_fold_divisor(wide) == 4
    assert shift_fold_divisor(shifted_network(segments=8).state_dict()) == 8


def test_a_checkpoint_written_before_the_share_was_recorded_reads_the_default():
    """Right for every such run but one that changed it, hence the marker."""
    model = shifted_network(segments=8)
    weights = {
        key: value for key, value in model.state_dict().items() if key != FOLD_KEY
    }

    assert shift_fold_divisor(weights) == FOLD_DIVISOR
    load_shifted(model, weights)
    assert shift_fold_divisor(model.state_dict()) == FOLD_DIVISOR


def test_a_checkpoint_written_before_the_shift_says_so_rather_than_failing():
    plain = resnet18(weights=None).state_dict()

    assert shift_segments(plain) is None
    assert shift_shuffled(plain) is False


def test_every_depth_shares_one_permutation():
    """A permutation per depth is noise at each layer, not a destroyed sequence."""
    control = shifted_network(segments=8, shuffle=True)

    set_segments(control, 8)

    orders = {tuple(shift.order.tolist()) for shift in control.temporal_shifts}
    assert len(control.temporal_shifts) == 8
    assert len(orders) == 1


def test_the_ordered_arm_draws_no_permutation():
    shifted = shifted_network(segments=8)

    set_segments(shifted, 8)

    assert all(shift.order is None for shift in shifted.temporal_shifts)
    assert all(shift.segments == 8 for shift in shifted.temporal_shifts)


def test_setting_the_clip_length_on_a_network_without_a_shift_does_nothing():
    """Both loops call it unconditionally, so a run without the shift is unchanged."""
    plain = resnet18(weights=None)

    set_segments(plain, 8)
    reset_shuffle(plain)


def test_a_reset_scoring_pass_repeats_itself():
    """Successive epochs have to be measured against the same thing."""
    control = shifted_network(segments=4, shuffle=True).eval()
    clip = torch.randn(4, 3, 64, 64)

    with torch.no_grad():
        reset_shuffle(control)
        set_segments(control, 4)
        first = control(clip)
        reset_shuffle(control)
        set_segments(control, 4)
        again = control(clip)

    assert torch.allclose(first, again)


@pytest.mark.parametrize("backbone", sorted(TORCHVISION_BACKBONES))
def test_the_shift_makes_a_frame_depend_on_the_frames_beside_it(backbone):
    """The line the order-blind baseline cannot cross, by construction."""
    shifted = shifted_network(backbone, segments=4).eval()
    clip = torch.randn(4, 3, 64, 64)

    with torch.no_grad():
        set_segments(shifted, 4)
        scored = shifted(clip)
        reordered = shifted(clip[torch.tensor([2, 1, 3, 0])])

    assert not same(scored[2], reordered[0])


def streaming_network(backbone="resnet18", weights=None):
    """A network carrying the online shift, optionally with given weights."""
    builder, _ = TORCHVISION_BACKBONES[backbone]
    network = builder(weights=None)
    if weights is not None:
        # The offline network records how it was shifted; a stream has no clip
        # length or permutation to record, so those three entries are left out.
        network.load_state_dict(
            {k: v for k, v in weights.items() if k not in model_module.MARKER_KEYS}
        )
    stream_temporal_shift(network, backbone)

    return network


@pytest.mark.parametrize("backbone", sorted(TORCHVISION_BACKBONES))
def test_a_stream_reproduces_the_causal_clip_frame_for_frame(backbone):
    """Online keeps what offline has in memory, so over one clip the two agree."""
    offline = shifted_network(backbone, segments=4, causal=True).eval()
    online = streaming_network(backbone, offline.state_dict()).eval()
    clip = torch.randn(4, 3, 64, 64)

    with torch.no_grad():
        set_segments(offline, 4)
        whole = offline(clip)
        streamed = torch.cat([online(clip[index : index + 1]) for index in range(4)])

    assert same(whole, streamed)


@pytest.mark.parametrize("backbone", sorted(TORCHVISION_BACKBONES))
def test_a_stream_reads_the_past_and_not_only_the_present(backbone):
    """Without the kept slice the second frame would score as if it came first."""
    online = streaming_network(backbone).eval()
    first, second = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        alone = online(second)
        reset_stream(online)
        online(first)
        after_first = online(second)

    assert not same(alone, after_first)


def test_a_reset_stream_starts_over():
    online = streaming_network().eval()
    frame = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        first = online(frame)
        online(torch.randn(1, 3, 64, 64))
        reset_stream(online)
        again = online(frame)

    assert torch.allclose(first, again)


def test_a_stream_refuses_several_frames_at_once():
    online = streaming_network().eval()

    with torch.no_grad(), pytest.raises(RuntimeError, match="one frame per call"):
        online(torch.randn(2, 3, 64, 64))


def test_a_batch_that_is_not_whole_clips_is_refused():
    shifted = shifted_network(segments=8).eval()

    set_segments(shifted, 8)
    with pytest.raises(RuntimeError, match="do not divide into clips"):
        shifted(torch.randn(12, 3, 64, 64))


def test_a_shift_with_no_clip_length_refuses_rather_than_guesses():
    with pytest.raises(RuntimeError, match="no clip length set"):
        TemporalShift()(nn.Identity(), (torch.randn(8, 64, 4, 4),))


def test_a_fold_that_rounds_down_to_nothing_is_refused():
    """A divisor wider than the layer would shift nothing and report success."""
    shift = TemporalShift(fold_divisor=64)
    shift.segments = 2

    with pytest.raises(RuntimeError, match="leaves nothing to shift"):
        shift(nn.Identity(), (torch.randn(2, 8, 4, 4),))


def test_a_backbone_this_project_does_not_build_is_refused():
    """Where its residual branches begin is not something the shift can guess."""
    with pytest.raises(ValueError, match="cannot shift inside"):
        insert_temporal_shift(resnet18(weights=None), "vgg11", segments=8)


def test_the_causal_shift_takes_from_the_past_and_nothing_from_ahead():
    shift = TemporalShift(fold_divisor=4, causal=True)
    shift.segments = 4

    (moved,) = shift(nn.Identity(), (indexed_clip(4, 8),))

    values = moved[:, :, 0, 0]
    assert values[:, 0].tolist() == [0.0, 0.0, 1.0, 2.0]
    # The group that reads ahead in the two-directional arm is left alone here.
    assert values[:, 2].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_one_direction_moves_the_share_of_one_direction():
    """The divisor is per direction, so causal moves half of what both move."""
    for divisor, share in ((8, 1 / 8), (4, 1 / 4)):
        shift = TemporalShift(fold_divisor=divisor, causal=True)
        shift.segments = 2
        frames = torch.zeros(2, 64, 1, 1)
        frames[0] = 1.0

        (moved,) = shift(nn.Identity(), (frames,))

        assert int((moved[1, :, 0, 0] == 1.0).sum()) == int(64 * share)


@pytest.mark.parametrize("backbone", sorted(TORCHVISION_BACKBONES))
def test_a_causal_frame_does_not_depend_on_any_frame_after_it(backbone):
    """The property the arm exists for: it is what restores the latency claim."""
    model = shifted_network(backbone, segments=4, causal=True).eval()
    clip = torch.randn(4, 3, 64, 64)
    altered = clip.clone()
    altered[3] = torch.randn(3, 64, 64)

    with torch.no_grad():
        set_segments(model, 4)
        before = model(clip)
        set_segments(model, 4)
        after = model(altered)

    for frame in range(3):
        assert same(before[frame], after[frame])


def test_the_two_directional_shift_reaches_back_through_every_depth():
    """A late frame reaches an early one, through the depth of the network.

    Eight stacked shifts carry it, so the whole clip has to exist before any of
    it can be decided. That is the cost the causal arm removes, and it is larger
    than one frame of lookahead.
    """
    model = shifted_network(segments=4).eval()
    clip = torch.randn(4, 3, 64, 64)
    altered = clip.clone()
    altered[3] = torch.randn(3, 64, 64)

    with torch.no_grad():
        set_segments(model, 4)
        before = model(clip)
        set_segments(model, 4)
        after = model(altered)

    assert not torch.allclose(before[0], after[0], atol=1e-6)


def test_a_checkpoint_says_whether_it_can_run_online():
    assert shift_causal(shifted_network(segments=8, causal=True).state_dict()) is True
    assert shift_causal(shifted_network(segments=8).state_dict()) is False
    assert shift_causal(resnet18(weights=None).state_dict()) is False
    assert CAUSAL_KEY in shifted_network(segments=8).state_dict()


def test_the_causal_shift_costs_no_parameter_either():
    plain = resnet18(weights=None)
    causal = shifted_network(segments=8, causal=True)

    assert sum(p.numel() for p in plain.parameters()) == sum(
        p.numel() for p in causal.parameters()
    )


def test_a_checkpoint_written_before_a_marker_existed_still_loads():
    """The markers grew over time; a run that predates one is not wrong."""
    model = shifted_network(segments=8)
    weights = {
        key: value
        for key, value in model.state_dict().items()
        if key != CAUSAL_KEY  # as every checkpoint written before it looked
    }

    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(weights)

    load_shifted(model, weights)
    assert shift_causal(model.state_dict()) is False


def test_a_real_mismatch_still_fails_loudly():
    """Filling a marker is not permission to absorb a wrong architecture."""
    model = shifted_network(segments=8)
    weights = dict(model.state_dict())
    weights["fc.weight"] = torch.randn(3, 512)
    weights["fc.bias"] = torch.randn(3)

    with pytest.raises(RuntimeError, match="size mismatch"):
        load_shifted(model, weights)
