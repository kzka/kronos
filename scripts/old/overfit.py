"""Overfitting script for sanity checking.
"""

import argparse
import logging

import os.path as osp

from kronos.config import CONFIG
from kronos.utils import experiment_utils, checkpoint
from kronos.utils.logger import Logger


def main(args):
    log_dir = osp.join(CONFIG.DIRS.LOG_DIR, args.experiment_name)

    # initialize experiment
    config, device = experiment_utils.init_experiment(
        log_dir, CONFIG, args.config_path
    )

    # setup logger
    logger = Logger(log_dir, args.resume)

    # load model, data loaders and trainer
    debug = {"sample_sequential": True, "num_workers": 4}
    (
        model,
        optimizer,
        loaders,
        trainer,
        evaluators,
    ) = experiment_utils.get_factories(config, device, debug=debug)

    # create checkpoint manager
    checkpoint_manager = checkpoint.CheckpointManager(
        checkpoint.Checkpoint(model, optimizer),
        osp.join(config.DIRS.CKPT_DIR, args.experiment_name),
        device,
    )

    global_step = checkpoint_manager.restore_or_initialize()
    epoch = int(global_step / len(loaders["pretrain_train"]))
    complete = False
    stopwatch = experiment_utils.Stopwatch()
    try:
        while not complete:
            logger.log_learning_rate(optimizer, global_step)
            for batch_idx, batch in enumerate(loaders["pretrain_train"]):
                if batch_idx >= args.num_samples:
                    break

                # train one iteration
                loss = trainer.train_one_iter(batch, global_step)

                # save model checkpoint
                if not global_step % config.CHECKPOINT.SAVE_INTERVAL:
                    checkpoint_manager.save(global_step)

                if not global_step % config.LOGGING.REPORT_INTERVAL:
                    # train loss with model in eval mode
                    train_eval_loss = trainer.eval_num_iters(
                        loaders["pretrain_train"],
                        (args.num_samples // config.BATCH_SIZE),
                    )
                    losses = {"train": loss, "train_eval": train_eval_loss}
                    logger.log_loss(losses, global_step)
                    logging.info(
                        "Iter[{}/{}] (Epoch {}), Train (eval) Loss: {:.3f}".format(
                            global_step,
                            config.TRAIN.MAX_ITERS,
                            epoch,
                            train_eval_loss.item(),
                        )
                    )

                if not global_step % config.LOGGING.EVAL_INTERVAL:
                    for eval_name, evaluator in evaluators.items():
                        eval_metric_train = evaluator.evaluate(
                            global_step,
                            model,
                            loaders["downstream_train"],
                            device,
                            args.num_samples,
                            msg="train",
                        )
                        metrics = {"train": eval_metric_train}
                        logger.log_metric(metrics, global_step, eval_name)
                        logging.info(
                            "Iter[{}/{}] (Epoch {}) - {} train: {:.4f}".format(
                                global_step,
                                config.TRAIN.MAX_ITERS,
                                epoch,
                                eval_name,
                                eval_metric_train["scalar"],
                            )
                        )

                # exit if complete
                global_step += 1
                if global_step >= config.TRAIN.MAX_ITERS:
                    complete = True
                    break

                time_per_iter = stopwatch.elapsed()
                logging.info(
                    "Iter[{}/{}] (Epoch {}), {:.1f}s/iter, Loss: {:.3f}".format(
                        global_step,
                        config.TRAIN.MAX_ITERS,
                        epoch,
                        time_per_iter,
                        loss.item(),
                    )
                )
                stopwatch.reset()
            epoch += 1

    except KeyboardInterrupt:
        logging.info(
            "Caught keyboard interrupt. Saving model before quitting."
        )

    finally:
        checkpoint_manager.save(global_step)
        logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument(
        "--resume", type=lambda s: s.lower() in ["true", "1"], default=False
    )
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
