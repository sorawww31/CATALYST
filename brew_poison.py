"""General interface script to launch poisoning jobs."""

import datetime
import os
import time
from datetime import date

import torch

import forest
import wandb

torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)

# Parse input arguments
args = forest.options().parse_args()
# 100% reproducibility?
if args.deterministic:
    forest.utils.set_deterministic()


if __name__ == "__main__":
    if args.wandb:
        os.environ["WANDB_API_KEY"] = "b89e9995f493fd65200bf57ec07b503531990699"
        print("Logging to wandb...")
        wandb.init(
            project=args.name, name=f"{args.name}_{args.poisonkey}_{date.today()}"
        )
        wandb.config.conservative = f"{args.name}"

    setup = forest.utils.system_startup(args)

    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(
        args,
        model.defs.batch_size,
        model.defs.augmentations,
        mixing_method=model.defs.mixing_method,
        setup=setup,
    )
    witch = forest.Witch(args, setup=setup)

    start_time = time.time()
    if args.pretrained:
        print("Loading pretrained model...")
        stats_clean = None
    else:
        stats_clean = model.train(data, max_epoch=args.max_epoch)
    train_time = time.time()

    poison_delta = witch.brew(model, data)
    brew_time = time.time()

    if not args.pretrained and args.retrain_from_init:
        stats_rerun = model.retrain(data, poison_delta)
    else:
        stats_rerun = None  # we dont know the initial seed for a pretrained model so retraining makes no sense

    if args.vnet is not None:  # Validate the transfer model given by args.vnet
        train_net = args.net
        args.net = args.vnet
        if args.vruns > 0:
            model = forest.Victim(args, setup=setup)
            stats_results = model.validate(data, poison_delta)
        else:
            stats_results = None
        args.net = train_net
    else:  # Validate the main model
        if args.vruns > 0:
            stats_results = model.validate(data, poison_delta)
        else:
            stats_results = None
    test_time = time.time()

    timestamps = dict(
        train_time=str(datetime.timedelta(seconds=train_time - start_time)).replace(
            ",", ""
        ),
        brew_time=str(datetime.timedelta(seconds=brew_time - train_time)).replace(
            ",", ""
        ),
        test_time=str(datetime.timedelta(seconds=test_time - brew_time)).replace(
            ",", ""
        ),
    )
    # Save run to table
    results = (stats_clean, stats_rerun, stats_results)
    if args.save_stats_to_pickle:
        import pickle

        with open("stats_rerun.pkl", "wb") as f:
            pickle.dump(results, f)

    forest.utils.record_results(
        data,
        witch.stat_optimal_loss,
        results,
        args,
        model.defs,
        model.model_init_seed,
        extra_stats=timestamps,
    )

    if args.wandb:
        wandb.finish()
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print("---------------------------------------------------")
    print(
        f"Finished computations with train time: {str(datetime.timedelta(seconds=train_time - start_time))}"
    )
    print(
        f"--------------------------- brew time: {str(datetime.timedelta(seconds=brew_time - train_time))}"
    )
    print(
        f"--------------------------- test time: {str(datetime.timedelta(seconds=test_time - brew_time))}"
    )
    print("-------------Job finished.-------------------------")
