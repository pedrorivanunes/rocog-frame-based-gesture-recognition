import pandas as pd
import pytest
import torch
from torch import nn
from torchvision.models import resnet18

from manifest import IDLE_LABEL
from model import freeze
from train import (
    EarlyStopping,
    build_criteria,
    build_loaders,
    checkpoint_name,
    parse_args,
    train_one_epoch,
)

# The run without photometric jitter, whose curve is not monotonic.
BASELINE_CURVE = [0.7773, 0.6929, 0.6992, 0.6294, 0.7510]


def test_parse_args_with_no_arguments_is_the_standard_run():
    args = parse_args([])

    assert args.seed == 0
    assert args.photometric is True
    assert args.geometric is True
    assert args.max_epochs == 15
    assert args.patience == 3
    assert args.checkpoint_name == "syn_ground_train.pt"
    assert args.num_workers == 12
    assert args.save_every_epoch is False
    assert args.label_smoothing == 0.0
    assert args.freeze == "none"
    assert args.texture == "none"
    assert args.texture_fill == "both"
    assert args.window == "full"
    assert args.lr == 1e-4


def test_parse_args_keeps_every_epoch_when_asked():
    assert parse_args(["--save-every-epoch"]).save_every_epoch is True


def test_parse_args_takes_a_repetition_seed():
    assert parse_args(["--seed", "2"]).seed == 2


def test_parse_args_takes_a_gamma_range():
    args = parse_args(["--gamma-shift", "1.0", "3.0"])

    assert args.gamma_shift == [1.0, 3.0]


def test_the_standard_run_shifts_no_gamma():
    """Every run before this option left tone to the jitter; the default keeps that."""
    assert parse_args([]).gamma_shift is None


def test_parse_args_takes_a_learning_rate():
    assert parse_args(["--lr", "1e-3"]).lr == 1e-3


def test_the_standard_run_freezes_nothing():
    """Every run before this option trained the whole net; the default keeps that."""
    assert parse_args([]).freeze == "none"


def test_parse_args_takes_a_freeze_depth():
    assert parse_args(["--freeze", "backbone"]).freeze == "backbone"


def test_parse_args_rejects_a_depth_that_is_not_defined():
    """A typo has to fail loudly.

    A run that silently trained everything would land in the sweep as if it had
    been frozen.
    """
    with pytest.raises(SystemExit):
        parse_args(["--freeze", "everything"])


def test_the_standard_run_meets_the_person_as_rendered():
    """Every run before this option trained on the rendered body."""
    assert parse_args([]).texture == "none"


def test_parse_args_takes_a_texture_mode():
    assert parse_args(["--texture", "everywhere"]).texture == "everywhere"


def test_parse_args_rejects_a_texture_mode_that_is_not_defined():
    """A typo would land in the sweep as a cell that trained on plain frames."""
    with pytest.raises(SystemExit):
        parse_args(["--texture", "stylise"])


def test_parse_args_takes_a_single_fill_kind():
    """Naming one is how the sweep asks which of the two carries the result."""
    assert parse_args(["--texture-fill", "noise"]).texture_fill == "noise"


def test_parse_args_rejects_a_fill_that_is_not_defined():
    with pytest.raises(SystemExit):
        parse_args(["--texture-fill", "stripes"])


def test_the_standard_run_trains_on_the_whole_window():
    """Every run before this option saw the annotated window entire."""
    assert parse_args([]).window == "full"


def test_parse_args_takes_a_window_extent():
    assert parse_args(["--window", "middle"]).window == "middle"


def test_parse_args_rejects_an_extent_that_is_not_defined():
    """A typo would land in the sweep as a narrowed run that trained on everything."""
    with pytest.raises(SystemExit):
        parse_args(["--window", "centre"])


def test_parse_args_takes_a_smoothing_fraction():
    assert parse_args(["--label-smoothing", "0.1"]).label_smoothing == 0.1


def test_the_validation_criterion_is_never_smoothed():
    """The treatment must not reach the number that measures it."""
    training, validation = build_criteria(0.1)

    assert training.label_smoothing == 0.1
    assert validation.label_smoothing == 0.0


def test_smoothing_lifts_the_loss_that_a_perfect_answer_still_carries():
    """Why the two criteria stay separate: the floor moves with the treatment."""
    certain = torch.tensor([[20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    truth = torch.tensor([0])
    training, validation = build_criteria(0.1)

    assert validation(certain, truth).item() == pytest.approx(0.0, abs=1e-6)
    assert training(certain, truth).item() > 1.0


def test_no_smoothing_leaves_the_two_criteria_agreeing():
    """The default has to reproduce every run recorded before the option."""
    certain = torch.tensor([[20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    truth = torch.tensor([0])
    training, validation = build_criteria(0.0)

    assert training(certain, truth).item() == pytest.approx(
        validation(certain, truth).item()
    )


def test_checkpoint_name_marks_the_seed():
    """A grid runs each configuration once per seed; unmarked they overwrite."""
    assert checkpoint_name("es_none.pt", 0) == "es_none_s0.pt"
    assert checkpoint_name("es_none.pt", 2) == "es_none_s2.pt"


def test_checkpoint_name_adds_a_padded_epoch_when_given_one():
    assert checkpoint_name("es_none.pt", 2, 3) == "es_none_s2_e03.pt"
    assert checkpoint_name("es_none.pt", 2, 12) == "es_none_s2_e12.pt"


def test_checkpoint_names_of_one_grid_are_all_distinct():
    names = {
        checkpoint_name(base, seed)
        for base in ("none.pt", "photometric.pt")
        for seed in range(3)
    }

    assert len(names) == 6


def test_parse_args_turns_each_augmentation_off_on_its_own():
    assert parse_args(["--no-photometric"]).photometric is False
    assert parse_args(["--no-photometric"]).geometric is True
    assert parse_args(["--no-geometric"]).geometric is False


def test_parse_args_overrides_the_epoch_budget_and_the_checkpoint():
    args = parse_args(["--max-epochs", "5", "--checkpoint-name", "es_none.pt"])

    assert args.max_epochs == 5
    assert args.checkpoint_name == "es_none.pt"


def test_the_first_epoch_is_always_the_best_so_far():
    stopper = EarlyStopping(patience=3)

    assert stopper.improved(9.0)


def test_a_worse_epoch_is_not_a_best():
    stopper = EarlyStopping(patience=3)
    stopper.improved(1.0)

    assert not stopper.improved(2.0)


def test_patience_runs_out_after_enough_epochs_without_a_best():
    stopper = EarlyStopping(patience=3)
    stopper.improved(1.0)

    for _ in range(2):
        stopper.improved(2.0)
    assert not stopper.exhausted

    stopper.improved(2.0)
    assert stopper.exhausted


def test_a_new_best_restores_the_patience():
    """The curve dips and recovers, and stopping on the first dip loses the best."""
    stopper = EarlyStopping(patience=3)
    stopper.improved(1.0)
    stopper.improved(2.0)
    stopper.improved(2.0)

    assert stopper.improved(0.5)
    assert not stopper.exhausted


def test_the_measured_baseline_curve_would_not_stop_early():
    """Its best epoch is the fourth, after a worse third — patience must survive it."""
    stopper = EarlyStopping(patience=3)

    best = [stopper.improved(loss) for loss in BASELINE_CURVE]

    assert best == [True, True, False, True, False]
    assert not stopper.exhausted
    assert stopper.best_loss == 0.6294


def test_an_epoch_leaves_a_frozen_stage_exactly_as_it_found_it():
    """The integration the sweep depends on: freeze applied, epoch run, nothing moved.

    Both halves are checked, because they fail separately. Weights move when the
    optimizer is handed a parameter it should not have; statistics move when the
    epoch's ``model.train()`` is allowed to stand.
    """
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 7)
    frozen = freeze(model, "backbone")
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )
    batches = [(torch.randn(4, 3, 64, 64), torch.tensor([0, 1, 2, 3]), ["v"] * 4)]
    weights = model.layer4[0].conv1.weight.clone()
    statistics = model.bn1.running_mean.clone()
    head = model.fc.weight.clone()

    train_one_epoch(
        model, batches, nn.CrossEntropyLoss(), optimizer, torch.device("cpu"), frozen
    )

    assert torch.equal(model.layer4[0].conv1.weight, weights)
    assert torch.equal(model.bn1.running_mean, statistics)
    assert not torch.equal(model.fc.weight, head), "the head should still be learning"


def test_a_run_trains_on_the_gestures_alone_by_default():
    """Every run recorded so far was seven classes, and stays comparable to itself."""
    args = parse_args([])

    assert args.idle_manifest is None


def test_a_run_can_name_the_frames_that_carry_no_gesture():
    args = parse_args(["--idle-manifest", "syn_ground_train_idle.csv"])

    assert args.idle_manifest == "syn_ground_train_idle.csv"


def rows_for(video, label, count, first=0):
    """Manifest rows for one video, carrying the columns the loaders read.

    No image has to exist: building a loader reads the table and nothing else,
    and these tests never draw a batch from it.
    """
    return pd.DataFrame(
        {
            "video_id": [video] * count,
            "label": [label] * count,
            "path": [f"frames/{video}_f{first + i}.jpg" for i in range(count)],
        }
    )


def mixed_rows(videos=3, gesture_frames=16, idle_frames=4):
    """Training rows as a run with the idle class assembles them."""
    return pd.concat(
        [rows_for(f"v{v}", 0, gesture_frames) for v in range(videos)]
        + [
            rows_for(f"v{v}", IDLE_LABEL, idle_frames, first=gesture_frames)
            for v in range(videos)
        ],
        ignore_index=True,
    )


def test_the_loaders_draw_each_kind_of_row_at_its_own_rate(tmp_path):
    """Counted as one kind, twenty rows a video refuse to split into eight blocks."""
    train_rows = mixed_rows(videos=3)

    train_loader, _ = build_loaders(
        train_rows, rows_for("v0", 0, 16), tmp_path, num_workers=0
    )

    assert len(train_loader.sampler) == 3 * (8 + 1)


def test_rows_of_one_kind_are_drawn_by_one_sampler(tmp_path):
    """A run without the idle class has to stay the run it was."""
    train_rows = pd.concat(
        [rows_for(f"v{v}", 0, 16) for v in range(3)], ignore_index=True
    )

    train_loader, _ = build_loaders(
        train_rows, rows_for("v0", 0, 16), tmp_path, num_workers=0
    )

    assert len(train_loader.sampler) == 3 * 8
