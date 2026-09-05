import pytest
import torch

from train import EarlyStopping, build_criteria, checkpoint_name, parse_args

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


def test_parse_args_keeps_every_epoch_when_asked():
    assert parse_args(["--save-every-epoch"]).save_every_epoch is True


def test_parse_args_takes_a_repetition_seed():
    assert parse_args(["--seed", "2"]).seed == 2


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
