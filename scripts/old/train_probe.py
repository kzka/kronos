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

import matplotlib

matplotlib.use("TkAgg")  # noqa: E402
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


class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.probe = nn.Linear(in_dim, out_dim)

    def forward(self, batch_embs):
        out = self.probe(F.relu(batch_embs))
        return F.log_softmax(out, dim=1)


class Net(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

        import torchvision.models as models

        self.model = models.resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 128)
        self.head = nn.Linear(128, out_dim)

    def forward(self, x):
        x = F.relu(self.model(x))
        x = self.head(x)
        output = F.log_softmax(x, dim=1)
        return output


def _embed(model, frames, device, max_batch_size, normalize):
    if frames.shape[1] > max_batch_size:
        embs = []
        for i in range(math.ceil(frames.shape[1] / max_batch_size)):
            sub_frames = frames[
                :, i * max_batch_size : (i + 1) * max_batch_size
            ].to(device)
            embs.append(model(sub_frames)["embs"].cpu())
        embs = torch.cat(embs, dim=1).to(device)
    else:
        embs = model(frames.to(device))["embs"]
    if normalize:
        embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-7)
    return embs


def embed_no_grad(model, frames, device, max_batch_size, normalize):
    with torch.no_grad():
        return _embed(model, frames, device, max_batch_size, normalize)


def embed_with_grad(
    model, probe, frames, target, device, max_batch_size, normalize, optimizer
):
    lcm = 2
    b, s, c, h, w = frames.shape
    frames = frames.view(b, s // lcm, lcm, c, h, w)
    frames = frames[:, :, -1].to(device)
    target = target.flatten()
    frames = frames.view(b * (s // lcm), c, h, w)
    optimizer.zero_grad()
    output = probe(frames)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = (
            pred.eq(target.view_as(pred)).sum().true_divide(pred.shape[0])
        )
    return loss.item(), accuracy.item()


def test_probe(
    test_loader,
    tcc_model,
    probe,
    device,
    max_batch_size,
    l2_normalize,
    global_step,
):
    # tcc_model.encoder_net.eval()
    probe.eval()
    test_loss = 0
    correct = 0
    num_pts = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            frames = batch["frames"].to(device)
            target = batch["debris_nums"].to(device)
            # embs = embed_no_grad(
            #     tcc_model,
            #     frames,
            #     device,
            #     max_batch_size,
            #     args.l2_normalize,
            # )[0]
            lcm = 2
            b, s, c, h, w = frames.shape
            frames = frames.view(b, s // lcm, lcm, c, h, w)
            frames = frames[:, :, -1].to(device)
            target = target.flatten()
            frames = frames.view(b * (s // lcm), c, h, w)
            output = probe(frames)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            set_trace()
            plt.imshow(frames[0].permute(1, 2, 0).detach().cpu().numpy())
            plt.show()
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_pts += output.shape[0]
    test_loss /= num_pts
    test_acc = 100.0 * correct / num_pts
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, num_pts, test_acc
        )
    )


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(
            "Using GPU {}.".format(torch.cuda.get_device_name(device))
        )
    else:
        logging.info("No GPU found. Falling back to CPU.")
        device = torch.device("cpu")

    # Initialize experiment.
    opts = [
        # "SAMPLING.STRIDE_ALL_SAMPLER",
        # args.stride,
        "IMAGE_SIZE",
        (64, 64),
        "ACTION_CLASS",
        [args.query],
        "DATASET",
        "demo",
    ]
    config_path = osp.join(args.logdir, "config.yml")
    config, device = experiment_utils.init_experiment(
        args.logdir, CONFIG, config_path, opts,
    )

    # Load TCC model and checkpoint.
    debug = {
        "sample_sequential": False,
        "augment": False,
        "labeled": None,
    }
    _, _, loaders, _, _ = experiment_utils.get_factories(
        config, device, debug=debug
    )
    # checkpoint.CheckpointManager.load_latest_checkpoint(
    #     checkpoint.Checkpoint(tcc_model),
    #     osp.join(
    #         config.DIRS.CKPT_DIR, osp.basename(osp.normpath(args.logdir))
    #     ),
    #     device,
    # )

    # Freeze TCC weights.
    # tcc_model.to(device)
    # tcc_model.featurizer_net.eval()
    # freeze_model(tcc_model.featurizer_net, False, False)
    # tcc_model.encoder_net.train()

    probe = Net(
        out_dim=5
    )  # LinearProbe(in_dim=config.MODEL.EMBEDDER.EMBEDDING_SIZE, out_dim=5)
    probe.to(device).train()

    optimizer = torch.optim.Adam(
        [
            {"params": probe.parameters(), "weight_decay": 0},
            # {"params": tcc_model.parameters()},
        ],
        lr=1e-3,
    )

    # Figure out max batch size that's a multiple of the number of context
    # frames. This is so we can support large videos with many frames.
    lcm = 1  # tcc_model.num_ctx_frames
    max_batch_size = math.floor(32 / lcm) * lcm

    # Grab dataloader.
    train_loader = loaders["pretrain_train"]  # [args.query]
    test_loader = loaders["pretrain_valid"]  # [args.query]

    global_step = 0
    complete = False
    try:
        while not complete:
            probe.train()
            # tcc_model.encoder_net.train()
            for batch_idx, batch in enumerate(train_loader):
                frames = batch["frames"].to(device)
                target = batch["debris_nums"].to(device)

                # Forward pass.
                loss, acc = embed_with_grad(
                    None,  # tcc_model,
                    probe,
                    frames,
                    target,
                    device,
                    max_batch_size,
                    args.l2_normalize,
                    optimizer,
                )

                print(f"{global_step}: Loss: {loss} - Acc: {100 * acc}")
                global_step += 1

            # Get test accuracy.
            test_probe(
                test_loader,
                None,  # tcc_model,
                probe,
                device,
                max_batch_size,
                args.l2_normalize,
                global_step,
            )

            # Exit if complete.
            if global_step >= args.max_iters:
                complete = True
                break

    except KeyboardInterrupt:
        logging.info(
            "Caught keyboard interrupt. Saving model before quitting."
        )
    finally:
        print("Exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
