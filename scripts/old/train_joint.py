import argparse
import logging
import math

import numpy as np
import os.path as osp
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from kronos.config import CONFIG
from kronos.utils.torch_utils import freeze_model
from kronos.utils import experiment_utils, file_utils, checkpoint
from kronos.datasets.transforms import UnNormalize
from ipdb import set_trace


def test_query(
    device, lstm, tcc_model, max_batch_size, loaders, normalize, batch_size
):
    tcc_model.encoder_net.eval()
    lstm.eval()
    with torch.no_grad():
        batch_embs, batch_labels = [], []
        correct, num_x = 0, 0
        for loader in loaders:
            for batch_idx, batch in enumerate(loader):
                if len(batch_embs) < batch_size:
                    frames = batch["frames"]
                    embs = embed(
                        tcc_model, frames, device, max_batch_size, normalize
                    )
                    batch_embs.append(embs[0])
                    batch_labels.extend(batch["success"])
                if len(batch_embs) == batch_size or batch_idx == (
                    len(loader) - 1
                ):
                    idxs_sorted = np.argsort([-len(x) for x in batch_embs])
                    batch_embs = [batch_embs[i] for i in idxs_sorted]
                    batch_labels = torch.stack(
                        [batch_labels[i] for i in idxs_sorted]
                    )
                    batch_labels = batch_labels.unsqueeze(1).float().to(device)
                    out = lstm(batch_embs)
                    pred = torch.sigmoid(out) > 0.5
                    correct += pred.eq(batch_labels.view_as(pred)).sum().item()
                    num_x += len(pred)
                    batch_embs, batch_labels = [], []
        return correct, num_x


def embed(model, frames, device, max_batch_size, normalize):
    if frames.shape[1] > max_batch_size:
        embs = []
        for i in range(math.ceil(frames.shape[1] / max_batch_size)):
            sub_frames = frames[
                :, i * max_batch_size : (i + 1) * max_batch_size
            ].to(device)
            sub_embs = model(sub_frames)["embs"]
            del sub_frames
            embs.append(sub_embs.cpu())
        embs = torch.cat(embs, dim=1).to(device)
    else:
        embs = model(frames.to(device))["embs"]
    if normalize:
        embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-7)
    return embs


class LSTMModel(nn.Module):
    def __init__(self, in_dim, lstm_dim, num_layers):
        super().__init__()

        self.drop = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(in_dim)
        self.lstm = nn.GRU(
            in_dim, lstm_dim, num_layers=num_layers, batch_first=True,
        )
        self.head = nn.Linear(lstm_dim, 1)

    def forward(self, batch_embs):
        seq_lens = [len(b) for b in batch_embs]
        seq_pad = pad_sequence(batch_embs, batch_first=True)
        seq_pad = self.norm(self.drop(seq_pad))
        packed_seq_pad = pack_padded_sequence(
            seq_pad, lengths=seq_lens, batch_first=True
        )
        # out, (ht, ct) = self.lstm(packed_seq_pad)
        out, ht = self.lstm(packed_seq_pad)
        out = self.head(ht[0])
        return out


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
    ]
    config_path = osp.join(args.logdir, "config.yml")
    config, device = experiment_utils.init_experiment(
        args.logdir, CONFIG, config_path, opts,
    )

    # load model and data loaders
    debug = {"sample_sequential": False, "augment": False, "labeled": "both"}
    tcc_model, _, loaders, _, _ = experiment_utils.get_factories(
        config, device, debug=debug
    )

    # load model checkpoint
    checkpoint.CheckpointManager.load_latest_checkpoint(
        checkpoint.Checkpoint(tcc_model),
        osp.join(
            config.DIRS.CKPT_DIR, osp.basename(osp.normpath(args.logdir))
        ),
        device,
    )
    tcc_model.to(device)
    tcc_model.featurizer_net.eval()
    tcc_model.encoder_net.train()
    freeze_model(tcc_model.featurizer_net, True, True)

    # for name, param in tcc_model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    # create LSTM model
    lstm = LSTMModel(
        in_dim=config.MODEL.EMBEDDER.EMBEDDING_SIZE, lstm_dim=32, num_layers=1,
    )
    lstm.to(device).train()

    optimizer = torch.optim.Adam(
        [
            {"params": lstm.parameters()},
            {"params": tcc_model.parameters(), "lr": 1e-4},
        ],
        lr=1e-3,
    )

    # figure out max batch size that's
    # a multiple of the number of context
    # frames.
    # this is so we can support large videos
    # with many frames.
    lcm = tcc_model.num_ctx_frames
    max_batch_size = math.floor(128 / lcm) * lcm

    # test on query
    rand_corr, rand_num = test_query(
        device,
        lstm,
        tcc_model,
        max_batch_size,
        (
            loaders["downstream_train"]["rms"],
            loaders["downstream_valid"]["rms"],
        ),
        args.l2_normalize,
        args.batch_size,
    )
    rand_acc = rand_corr / rand_num
    print(
        "Randomly initialized LSTM accuracy: {} ({}/{})".format(
            rand_acc, rand_corr, rand_num
        )
    )

    global_step = 0
    complete = False
    try:
        while not complete:
            accs = []
            for action_name, loader in loaders["downstream_train"].items():
                tcc_model.encoder_net.train()
                lstm.train()
                if action_name != "rms":
                    batch_embs, batch_labels = [], []
                    correct, num_x = 0, 0
                    set_trace()
                    for batch_idx, batch in enumerate(loader):
                        if batch_idx > 2:
                            break
                        if len(batch_embs) < args.batch_size:
                            frames = batch["frames"]
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
                            print(len(batch_embs))
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

                            # forward through lstm and compute loss
                            out = lstm(batch_embs)
                            loss = F.binary_cross_entropy_with_logits(
                                out, batch_labels
                            )

                            # backprop
                            optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                lstm.parameters(), 1.0
                            )
                            optimizer.step()

                            # compute accuracy
                            pred = torch.sigmoid(out) > 0.5
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
                            batch_embs, batch_labels = [], []

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
            print("Mean embodiment accuracy: {}".format(np.mean(accs)))

            query_corr, query_num = test_query(
                device,
                lstm,
                tcc_model,
                max_batch_size,
                (
                    loaders["downstream_train"]["rms"],
                    loaders["downstream_valid"]["rms"],
                ),
                args.l2_normalize,
                args.batch_size,
            )
            query_acc = query_corr / query_num
            print(
                "Learner accuracy: {} ({}/{})".format(
                    query_acc, query_corr, query_num
                )
            )

    except KeyboardInterrupt:
        logging.info(
            "Caught keyboard interrupt. Saving model before quitting."
        )
    finally:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train LSTM on embedding sequences")
    parser.add_argument("--logdir", type=str, required=True)
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
        default=100,
        help="The number of training iterations.",
    )
    parser.add_argument(
        "--stride", type=int, default=5, help="The spacing between the frames."
    )
    args = parser.parse_args()
    main(args)
