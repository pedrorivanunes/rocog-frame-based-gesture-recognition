from train import EarlyStopping, parse_args

# The run without photometric jitter, whose curve is not monotonic.
BASELINE_CURVE = [0.7773, 0.6929, 0.6992, 0.6294, 0.7510]


def test_parse_args_with_no_arguments_is_the_standard_run():
    args = parse_args([])

    assert args.photometric is True
    assert args.geometric is True
    assert args.max_epochs == 15
    assert args.patience == 3
    assert args.checkpoint_name == "syn_ground_train.pt"
    assert args.num_workers == 12


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
