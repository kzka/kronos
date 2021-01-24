"""
Add dropout
Add regularization
Add early stopping
Add data augmentation
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


def test_query(
    device, lstm, tcc_model, max_batch_size, loaders, normalize, batch_size,
):
    tcc_model.encoder_net.eval()
    lstm.eval()
    with torch.no_grad():
        batch_embs, batch_labels, batch_names = [], [], []
        y_pred, y_true, probas, mistakes, mistake_pred = [], [], [], [], []
        for loader in loaders:
            for batch_idx, batch in enumerate(loader):
                if len(batch_embs) < batch_size:
                    frames = batch["frames"]
                    embs = embed(
                        tcc_model, frames, device, max_batch_size, normalize
                    )
                    batch_embs.append(embs[0])
                    batch_labels.extend(batch["success"])
                    batch_names.extend(batch["video_name"][0])
                if len(batch_embs) == batch_size or batch_idx == (
                    len(loader) - 1
                ):
                    idxs_sorted = np.argsort([-len(x) for x in batch_embs])
                    batch_embs = [batch_embs[i] for i in idxs_sorted]
                    batch_labels = torch.stack(
                        [batch_labels[i] for i in idxs_sorted]
                    )
                    batch_names = [batch_names[i] for i in idxs_sorted]
                    batch_labels = batch_labels.unsqueeze(1).float().to(device)
                    out = lstm(batch_embs)
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


def embed(model, frames, device, max_batch_size, normalize):
    # with torch.no_grad():
    if frames.shape[1] > max_batch_size:
        embs = []
        for i in range(math.ceil(frames.shape[1] / max_batch_size)):
            sub_frames = frames[
                :, i * max_batch_size : (i + 1) * max_batch_size
            ].to(device)
            sub_embs = model(sub_frames)["embs"]
            embs.append(sub_embs.cpu())
        embs = torch.cat(embs, dim=1).to(device)
    else:
        embs = model(frames.to(device))["embs"]
    if normalize:
        embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-7)
    return embs


class LSTMModel(nn.Module):
    """An RNN with a classification head.
    """

    def __init__(self, in_dim, lstm_dim, out_dim, num_layers):
        super().__init__()

        self.norm = nn.LayerNorm(in_dim)
        self.rnn = nn.GRU(
            in_dim, lstm_dim, num_layers=num_layers, batch_first=True
        )
        self.head = nn.Linear(lstm_dim, out_dim)

    def forward(self, batch_embs):
        seq_lens = [len(b) for b in batch_embs]
        out = pad_sequence(batch_embs, batch_first=True)
        out = pack_padded_sequence(
            self.norm(out), lengths=seq_lens, batch_first=True
        )
        out, ht = self.rnn(out)
        out = self.head(ht[-1])
        return out


class MLPModel(nn.Module):
    """A multi-layer perceptron.
    """

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, batch_embs):
        return self.mlp(batch_embs)


def augment_batch(batch_embs, batch_labels):
    """Augment a batch of embeddings with positive / negative permutations.
    """
    num_frames = 10
    augmented = []
    for i, emb in enumerate(batch_embs):
        # decide if we do an augmentation or not
        do = np.random.random() > 0.5
        if do:
            vid_len = emb.shape[0]
            # perform a positive data augmentation
            if np.random.random() > 0.5:
                # a positive data augmentation is going to be
                # going backwards in time for a few frames
                # then going forwards again for a few frames
                range_min = random.randrange(vid_len - num_frames)
                range_max = range_min + num_frames
                idxs = list(range(range_min, range_max))
                first_part = emb[0 : idxs[0], :]
                end_part = emb[idxs[-1] + 1 :, :]
                frames = emb[idxs, :]
                frames_reverse = (
                    torch.from_numpy(frames.cpu().numpy()[::-1, :].copy())
                    .float()
                    .cuda()
                )
                emb_aug = torch.cat(
                    [first_part, frames, frames_reverse, end_part]
                )
                augmented.append(emb_aug)
            # perform a negative data augmentation
            else:
                # a negative data augmentation is going to be
                # randomly chopping off from a starting sequence
                # until the end of the sequence
                # let's say we chop off from anywhere in the middle to the end
                rand_idx = random.randrange(vid_len // 2, vid_len)
                emb_aug = emb[:rand_idx, :]
                augmented.append(emb_aug)
                # if the label is positive, make it
                # negative.
                # else, leave as is.
                if batch_labels[i] == 1:
                    batch_labels[i] == 0
        else:
            augmented.append(emb)
    return augmented, batch_labels


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
        # "IMAGE_SIZE",
        # (216, 224),
        "DATASET",
        "embodied_glasses",
    ]
    config_path = osp.join(args.logdir, "config.yml")
    config, device = experiment_utils.init_experiment(
        args.logdir, CONFIG, config_path, opts,
    )

    # load TCC model and checkpoint
    debug = {
        "sample_sequential": True,
        "augment": False,
        "labeled": "both",
    }
    tcc_model, _, loaders, _, _ = experiment_utils.get_factories(
        config, device, debug=debug
    )
    checkpoint.CheckpointManager.load_latest_checkpoint(
        checkpoint.Checkpoint(tcc_model),
        osp.join(
            config.DIRS.CKPT_DIR, osp.basename(osp.normpath(args.logdir))
        ),
        device,
    )
    checkpoint_manager = checkpoint.CheckpointManager(
        checkpoint.Checkpoint(tcc_model),
        osp.join(
            config.DIRS.CKPT_DIR, osp.basename(osp.normpath(args.logdir))
        ),
        device,
    )
    init_step = checkpoint_manager.restore_or_initialize()
    # tcc_model.to(device).eval()
    tcc_model.to(device)
    tcc_model.featurizer_net.eval()
    tcc_model.encoder_net.train()
    freeze_model(tcc_model.featurizer_net, True, True)

    # create LSTM model
    lstm = LSTMModel(
        in_dim=config.MODEL.EMBEDDER.EMBEDDING_SIZE,
        lstm_dim=512,
        out_dim=1,
        num_layers=1,
    )
    # lstm = MLPModel(
    #     in_dim=config.MODEL.EMBEDDER.EMBEDDING_SIZE,
    #     hidden_dim=128,
    #     out_dim=1,
    # )
    lstm.to(device).train()

    # optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-3, weight_decay=0)
    optimizer = torch.optim.Adam(
        [
            {"params": lstm.parameters(), "weight_decay": 1e-5},
            {"params": tcc_model.parameters(), "lr": 1e-4},
        ],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 250], gamma=0.1
    )

    # figure out max batch size that's
    # a multiple of the number of context
    # frames.
    # this is so we can support large videos
    # with many frames.
    lcm = tcc_model.num_ctx_frames
    max_batch_size = math.floor(128 / lcm) * lcm

    query_metric = test_query(
        device,
        lstm,
        tcc_model,
        max_batch_size,
        (
            loaders["downstream_train"][args.query],
            loaders["downstream_valid"][args.query],
        ),
        args.l2_normalize,
        args.batch_size,
    )
    conf_matrix = query_metric["confusion_matrix"]
    num_correct = conf_matrix[0, 0] + conf_matrix[1, 1]
    num_total = conf_matrix.ravel().sum()
    print(
        f"Random LSTM accuracy: {query_metric['accuracy']} ({num_correct}/{num_total})"
    )

    global_step = 0
    max_acc = 0
    complete = False
    losses, expert_acc, learner_acc = [], [], []
    try:
        while not complete:
            accs = []
            lstm.train()
            tcc_model.encoder_net.train()
            for action_name, loader in loaders["downstream_train"].items():
                if action_name != args.query:
                    batch_embs, batch_labels, batch_names = [], [], []
                    correct, num_x = 0, 0
                    for batch_idx, batch in enumerate(loader):
                        if len(batch_embs) < args.batch_size:
                            frames = batch["frames"]
                            # b, t, c, h, w = frames.shape
                            # frames = frames.view(b, t // tcc_model.num_ctx_frames, tcc_model.num_ctx_frames, c, h, w)
                            # img = UnNormalize()(frames[:, -1, -1])[0].permute(1, 2, 0).cpu().numpy()
                            # plt.imshow(img)
                            # plt.show()
                            # set_trace()
                            batch_names.extend(batch["video_name"][0])
                            embs = embed(
                                tcc_model,
                                frames,
                                device,
                                max_batch_size,
                                args.l2_normalize,
                            )
                            batch_embs.append(embs[0])
                            batch_labels.extend(batch["success"])

                        if len(batch_embs) == args.batch_size or batch_idx == (
                            len(loader) - 1
                        ):
                            # batch_embs, batch_labels = augment_batch(batch_embs, batch_labels)

                            # sort list of embeddings by sequence length
                            # in descending order
                            idxs_sorted = np.argsort(
                                [-len(x) for x in batch_embs]
                            )
                            batch_embs = [batch_embs[i] for i in idxs_sorted]
                            batch_labels = torch.stack(
                                [batch_labels[i] for i in idxs_sorted]
                            )
                            batch_labels = (
                                batch_labels.unsqueeze(1).float().to(device)
                            )
                            batch_names = [batch_names[i] for i in idxs_sorted]

                            # forward through lstm and compute loss
                            out = lstm(batch_embs)
                            loss = F.binary_cross_entropy_with_logits(
                                out, batch_labels
                            )
                            losses.append(loss.item())

                            # backprop
                            optimizer.zero_grad()
                            loss.backward()
                            # torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1.0)
                            optimizer.step()
                            scheduler.step()
                            for param_group in optimizer.param_groups:
                                print(global_step, param_group["lr"])

                            # compute accuracy
                            with torch.no_grad():
                                probs = torch.sigmoid(out)
                                pred = probs > 0.5
                                # corr = pred.eq(batch_labels)
                                # pred_np = corr.flatten().cpu().numpy()
                                # mistakes = [batch_names[i] for i in np.argwhere(pred_np == False).flatten().tolist()]
                                # if global_step > 100:
                                #     for m in mistakes:
                                #         print(m)
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

            plot_loss(losses, osp.join(args.logdir, "lstm_loss.png"))
            query_metric = test_query(
                device,
                lstm,
                tcc_model,
                max_batch_size,
                (
                    loaders["downstream_train"][args.query],
                    loaders["downstream_valid"][args.query],
                ),
                args.l2_normalize,
                args.batch_size,
            )
            learner_acc.append(query_metric["accuracy"])
            plot_acc(
                expert_acc, learner_acc, osp.join(args.logdir, "lstm_acc.png")
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
            osp.join(args.logdir, "confusion_matrix_lstm.txt"), conf_matrix
        )
        plot_auc(query_metric, osp.join(args.logdir, "auc_curve_lstm.png"))
        learner_acc.append(query_metric["accuracy"])
        plot_acc(
            expert_acc, learner_acc, osp.join(args.logdir, "lstm_acc.png")
        )
        plot_loss(losses, osp.join(args.logdir, "lstm_loss.png"))
        checkpoint_manager.save(init_step + global_step)
        print("best acc achieved: {}".format(max_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train LSTM on embedding sequences")
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument(
        "--l2_normalize",
        type=lambda s: s.lower() in ["true", "1"],
        default=False,
        help="Whether to l2 normalize embeddings before feeding to LSTM.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="The batch size for the LSTM.",
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
