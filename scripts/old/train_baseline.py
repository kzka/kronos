"""
Train last-frame classifier as a baseline.
"""

import argparse
import logging
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ipdb import set_trace
from sklearn import metrics
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from kronos.config import CONFIG
from kronos.datasets.transforms import UnNormalize
from kronos.utils.torch_utils import freeze_model
from kronos.utils import experiment_utils, file_utils, checkpoint


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def plot_loss(loss_list, savepath, window=1):
    loss_arr = np.array(loss_list)
    window = rolling_window(loss_arr, window)
    rolling_mean = np.mean(window, 1)
    rolling_std = np.std(window, 1)
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel("Iteration #")
    ax.set_ylabel("Loss")
    ax.plot(range(len(rolling_mean)), rolling_mean, alpha=0.98, linewidth=0.9)
    ax.fill_between(
        range(len(rolling_std)),
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        alpha=0.5,
    )
    ax.grid(True)
    plt.savefig(savepath, format="png", dpi=200)
    plt.close()


def plot_acc(expert_acc, learner_acc, savepath):
    plt.figure()
    plt.plot(expert_acc, label="train")
    plt.plot(learner_acc, label="valid")
    plt.xlabel("# of epochs")
    plt.ylabel("classification accuracy")
    plt.grid()
    plt.legend()
    plt.savefig(savepath, format="png", dpi=200)
    plt.close()


def plot_auc(query_metric, savename):
    plt.plot(
        query_metric["fpr"],
        query_metric["tpr"],
        label="auc={}".format(query_metric["auc"]),
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUC Curve")
    plt.legend()
    plt.savefig(savename, format="png", dpi=200)
    plt.close()


def test_query(device, model, loaders, batch_size, num_ctx_frames):
    model.eval()
    with torch.no_grad():
        batch_embs, batch_labels, batch_names = [], [], []
        y_pred, y_true, probas, mistakes, mistake_pred = [], [], [], [], []
        for loader in loaders:
            for batch_idx, batch in enumerate(loader):
                if len(batch_embs) < batch_size:
                    frames = batch["frames"]
                    b, t, c, h, w = frames.shape
                    frames = frames.view(
                        b, t // num_ctx_frames, num_ctx_frames, c, h, w
                    )
                    frame = frames[:, -1, -1]
                    batch_names.extend(batch["video_name"][0])
                    batch_embs.append(frame)
                    batch_labels.extend(batch["success"])
                if len(batch_embs) == batch_size or batch_idx == (
                    len(loader) - 1
                ):
                    batch_embs = torch.cat(batch_embs, dim=0).to(device)
                    batch_labels = (
                        torch.stack(batch_labels)
                        .unsqueeze(1)
                        .float()
                        .to(device)
                    )
                    out = model(batch_embs)
                    prob = torch.sigmoid(out)
                    pred = prob > 0.5
                    corr = pred.eq(batch_labels)
                    pred_np = corr.flatten().cpu().numpy()
                    mistakes.extend(
                        [
                            batch_names[i]
                            for i in np.argwhere(pred_np == False)
                            .flatten()
                            .tolist()
                        ]
                    )
                    for jj in np.argwhere(pred_np == False).flatten().tolist():
                        mistake_pred.append(
                            [
                                pred.bool()
                                .flatten()
                                .cpu()
                                .numpy()
                                .tolist()[jj],
                                batch_labels.bool()
                                .flatten()
                                .cpu()
                                .numpy()
                                .tolist()[jj],
                            ]
                        )
                    probas.extend(prob.flatten().cpu().numpy().tolist())
                    y_pred.extend(pred.flatten().cpu().numpy().tolist())
                    y_true.extend(
                        batch_labels.bool().flatten().cpu().numpy().tolist()
                    )
                    batch_embs, batch_labels, batch_names = [], [], []
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
        auc = metrics.roc_auc_score(y_true, y_pred)
        return {
            "accuracy": metrics.accuracy_score(y_true, y_pred),
            "confusion_matrix": metrics.confusion_matrix(y_true, y_pred),
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc,
            "mistakes": mistakes,
            "mistakes_pred": mistake_pred,
        }


class Model(nn.Module):
    """A resnet18 based convnet.
    """

    def __init__(self, in_dim, out_dim, pretrained=True, freeze=False):
        super().__init__()

        self.model = torchvision.models.resnet18(pretrained=pretrained)
        for param in self.model.parameters():
            if freeze:
                param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, out_dim)

    def forward(self, x):
        return self.model(x)


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(
            "Using GPU {}.".format(torch.cuda.get_device_name(device))
        )
    else:
        logging.info("No GPU found. Falling back to CPU.")
        device = torch.device("cpu")

    # initialize experiment
    opts = [
        "SAMPLING.STRIDE_ALL_SAMPLER",
        args.stride,
        "ACTION_CLASS",
        [],
        "DATASET",
        "embodied_glasses",
    ]
    config_path = osp.join(args.logdir, "config.yml")
    config, device = experiment_utils.init_experiment(
        args.logdir, CONFIG, config_path, opts,
    )

    # determine the number of context frames
    num_ctx_frames = config.SAMPLING.NUM_CONTEXT_FRAMES
    print(f"ctx frames: {num_ctx_frames}")

    # load TCC model and checkpoint
    debug = {
        "sample_sequential": True,
        "augment": False,
        "labeled": "both",
    }
    _, _, loaders, _, _ = experiment_utils.get_factories(
        config, device, debug=debug
    )

    # create model
    model = Model(in_dim=3, out_dim=1)
    model.to(device).train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 250], gamma=0.1
    )

    query_metric = test_query(
        device,
        model,
        (
            loaders["downstream_train"][args.query],
            loaders["downstream_valid"][args.query],
        ),
        args.batch_size,
        num_ctx_frames,
    )
    conf_matrix = query_metric["confusion_matrix"]
    num_correct = conf_matrix[0, 0] + conf_matrix[1, 1]
    num_total = conf_matrix.ravel().sum()
    print(
        f"Initial pretrained model accuracy: {query_metric['accuracy']} ({num_correct}/{num_total})"
    )

    global_step = 0
    max_acc = 0
    complete = False
    losses, expert_acc, learner_acc = [], [], []
    try:
        while not complete:
            accs = []
            model.train()
            for action_name, loader in loaders["downstream_train"].items():
                if action_name != args.query:
                    batch_embs, batch_labels, batch_names = [], [], []
                    correct, num_x = 0, 0
                    for batch_idx, batch in enumerate(loader):
                        if len(batch_embs) < args.batch_size:
                            frames = batch["frames"]
                            b, t, c, h, w = frames.shape
                            frames = frames.view(
                                b, t // num_ctx_frames, num_ctx_frames, c, h, w
                            )
                            frame = frames[:, -1, -1]
                            # img = UnNormalize()(frame)[0].permute(1, 2, 0).cpu().numpy()
                            # plt.imshow(img)
                            # plt.show()
                            # set_trace()
                            batch_names.extend(batch["video_name"][0])
                            batch_embs.append(frame)
                            batch_labels.extend(batch["success"])
                        if len(batch_embs) == args.batch_size or batch_idx == (
                            len(loader) - 1
                        ):
                            batch_embs = torch.cat(batch_embs, dim=0).to(
                                device
                            )
                            batch_labels = (
                                torch.stack(batch_labels)
                                .unsqueeze(1)
                                .float()
                                .to(device)
                            )

                            # forward through model and compute loss
                            out = model(batch_embs)
                            loss = F.binary_cross_entropy_with_logits(
                                out, batch_labels
                            )
                            losses.append(loss.item())

                            # backprop
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                            for param_group in optimizer.param_groups:
                                print(global_step, param_group["lr"])

                            # compute accuracy
                            with torch.no_grad():
                                probs = torch.sigmoid(out)
                                pred = probs > 0.5
                                correct += (
                                    pred.eq(batch_labels.view_as(pred))
                                    .sum()
                                    .item()
                                )
                                num_x += len(pred)
                                logging.info(
                                    "{}: Loss: {:.6f}".format(
                                        global_step, loss.item()
                                    )
                                )

                            global_step += 1
                            batch_embs, batch_labels, batch_names = [], [], []

                        # exit if complete
                        if global_step >= args.max_iters:
                            complete = True
                            break

                    if num_x > 0:
                        acc = correct / num_x
                        logging.info(
                            "{} accuracy: {:.2f} ({}/{})".format(
                                action_name, acc, correct, num_x
                            )
                        )
                        accs.append(acc)
            mean_acc = np.mean(accs)
            expert_acc.append(mean_acc)
            print("Mean embodiment accuracy: {}".format(mean_acc))

            plot_loss(losses, osp.join(args.logdir, "conv_loss.png"))
            query_metric = test_query(
                device,
                model,
                (
                    loaders["downstream_train"][args.query],
                    loaders["downstream_valid"][args.query],
                ),
                args.batch_size,
                num_ctx_frames,
            )
            learner_acc.append(query_metric["accuracy"])
            plot_acc(
                expert_acc, learner_acc, osp.join(args.logdir, "conv_acc.png")
            )
            if query_metric["accuracy"] > max_acc:
                max_acc = query_metric["accuracy"]
            conf_matrix = query_metric["confusion_matrix"]
            num_correct = conf_matrix[0, 0] + conf_matrix[1, 1]
            num_total = conf_matrix.ravel().sum()
            print(
                f"Learner accuracy: {query_metric['accuracy']} ({num_correct}/{num_total})"
            )

    except KeyboardInterrupt:
        logging.info(
            "Caught keyboard interrupt. Saving model before quitting."
        )
    finally:
        conf_matrix = query_metric["confusion_matrix"]
        np.savetxt(
            osp.join(args.logdir, "confusion_matrix_conv.txt"), conf_matrix
        )
        plot_auc(query_metric, osp.join(args.logdir, "auc_curve_conv.png"))
        learner_acc.append(query_metric["accuracy"])
        plot_acc(
            expert_acc, learner_acc, osp.join(args.logdir, "conv_acc.png")
        )
        plot_loss(losses, osp.join(args.logdir, "conv_loss.png"))
        print("best acc achieved: {}".format(max_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train ConvNet on last frame.")
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument(
        "--batch_size", type=int, default=5, help="The batch size.",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=200,
        help="The number of training iterations.",
    )
    parser.add_argument(
        "--stride", type=int, default=5, help="The spacing between the frames."
    )
    args = parser.parse_args()
    main(args)
