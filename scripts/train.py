"""Training script.

1. Training on a combination of agents.
=======================================
$ agent=gripper
$ python scripts/train.py \
    --experiment_name comb_$agent \
    --config_path configs/magical_config/$agent.yml
"""

import numpy as np
import argparse
import logging

import os.path as osp

from torch.optim.lr_scheduler import ReduceLROnPlateau

from kronos.config import CONFIG
from kronos.utils import experiment_utils, checkpoint
from kronos.utils.logger import Logger


def main(args):
    log_dir = osp.join(CONFIG.DIRS.LOG_DIR, args.experiment_name)
    checkpoint_dir = osp.join(log_dir, 'checkpoints')

    # Initialize experiment
    opts = []
    config, device = experiment_utils.init_experiment(
        log_dir, CONFIG, args.config_path, opts)

    # Setup logger.
    logger = Logger(log_dir, args.resume)

    # Load factories.
    (
        model,
        optimizer,
        loaders,
        trainer,
        evaluators,
    ) = experiment_utils.get_factories(config, device)

    # Create lr scheduler.
    scheduler = ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10, verbose=True)

    # Create checkpoint manager.
    checkpoint_manager = checkpoint.CheckpointManager(
        checkpoint.Checkpoint(model, optimizer),
        checkpoint_dir, device)

    global_step = checkpoint_manager.restore_or_initialize()
    epoch = int(global_step / len(loaders["pretrain_train"]))
    num_valid = loaders["pretrain_valid"].dataset.total_vids
    valid_iters = config.EVAL.VAL_ITERS * config.BATCH_SIZE
    complete = False
    stopwatch = experiment_utils.Stopwatch()
    try:
        while not complete:
            logger.log_learning_rate(optimizer, global_step)
            for batch_idx, batch in enumerate(loaders["pretrain_train"]):
                # Train one iteration.
                loss = trainer.train_one_iter(batch, global_step)

                # Save model checkpoint.
                if not global_step % config.CHECKPOINT.SAVE_INTERVAL:
                    checkpoint_manager.save(global_step)

                if not global_step % config.LOGGING.REPORT_INTERVAL:
                    train_eval_loss = trainer.eval_num_iters(
                        loaders["pretrain_train"], 20
                    )
                    valid_loss = trainer.eval_num_iters(
                        loaders["pretrain_valid"], 20
                    )
                    losses = {
                        "train": loss,
                        "valid": valid_loss,
                        "train_eval": train_eval_loss,
                    }
                    logger.log_loss(losses, global_step)

                if not global_step % config.LOGGING.EVAL_INTERVAL:
                    for eval_name, evaluator in evaluators.items():
                        eval_metric_train = evaluator.evaluate(
                            global_step,
                            model,
                            loaders["downstream_train"],
                            device,
                            valid_iters,
                            msg="train",
                        )
                        eval_metric_valid = evaluator.evaluate(
                            global_step,
                            model,
                            loaders["downstream_valid"],
                            device,
                            valid_iters,
                            msg="valid",
                        )
                        metrics = {
                            "train": eval_metric_train,
                            "valid": eval_metric_valid,
                        }
                        logger.log_metric(metrics, global_step, eval_name)

                # Exit if complete.
                global_step += 1
                if global_step > config.TRAIN_MAX_ITERS:
                    complete = True
                    break

                time_per_iter = stopwatch.elapsed()
                logging.info(
                    "Iter[{}/{}] (Epoch {}), {:.1f}s/iter, Loss: {:.3f}".format(
                        global_step,
                        config.TRAIN_MAX_ITERS,
                        epoch,
                        time_per_iter,
                        loss.item(),
                    )
                )
                stopwatch.reset()
            epoch += 1

            # At end of epoch, evaluate validation loss on a fraction of the
            # validation data, then step the scheduler.
            v_loss = trainer.eval_num_iters(
                loaders["pretrain_valid"],
                int(1.0 * num_valid / config.BATCH_SIZE),
            )
            scheduler.step(v_loss)

    except KeyboardInterrupt:
        logging.info(
            "Caught keyboard interrupt. Saving model before quitting.")

    finally:
        checkpoint_manager.save(global_step)
        logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument(
        "--resume", type=lambda s: s.lower() in ["true", "1"], default=False
    )
    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
