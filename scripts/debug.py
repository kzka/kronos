"""Debugging script."""

import argparse
import matplotlib

matplotlib.use("TkAgg")  # noqa: E402
import matplotlib.pyplot as plt
import os.path as osp
import sys
import torchvision

from pdb import set_trace

from kronos.config import CONFIG
from kronos.datasets.transforms import UnNormalize
from kronos.utils import experiment_utils


def main(args):
    log_dir = osp.join(CONFIG.DIRS.LOG_DIR, "debug")

    # Initialize experiment
    config, device = experiment_utils.init_experiment(
        log_dir, CONFIG, args.config_path, transient=True)

    # Load model, data loaders and trainer.
    debug = {
        "sample_sequential": not args.shuffle,
        "augment": args.augment,
        "num_workers": 0,
    }
    model, optimizer, loaders, trainer, _ = experiment_utils.get_factories(
        config, device, debug=debug)

    num_ctx_frames = config.SAMPLING.NUM_CONTEXT_FRAMES
    num_frames = config.SAMPLING.NUM_FRAMES_PER_SEQUENCE
    try:
        loader = loaders["pretrain_train"]
        for batch_idx, batch in enumerate(loader):
            if batch_idx > 4:
                break
            print(f"Batch #{batch_idx}")
            frames = batch["frames"]
            b, _, c, h, w = frames.shape
            frames = frames.view(b, num_frames, num_ctx_frames, c, h, w)

            # # To visualize baseline batch.
            # for i in range(frames.shape[1]):
            #     grid_img = torchvision.utils.make_grid(frames[:, i, -1], nrow=5)
            #     plt.imshow(grid_img.permute(1, 2, 0))
            #     plt.show()

            for b in range(frames.shape[0]):
                print(f"\tBatch Item {b}")
                if args.viz_context:
                    fig, axes = plt.subplots(
                        num_ctx_frames, num_frames, constrained_layout=True
                    )
                    for i in range(num_frames):
                        for j in range(num_ctx_frames):
                            axes[j, i].imshow(
                                UnNormalize()(frames[b, i, j]).permute(1, 2, 0)
                            )
                    for ax in axes.flatten():
                        ax.axis("off")
                    plt.show()
                else:
                    grid_img = torchvision.utils.make_grid(
                        frames[b, :, -1], nrow=5)
                    # grid_img = UnNormalize()(grid_img)
                    plt.imshow(grid_img.permute(1, 2, 0))
                    plt.show()
    except KeyboardInterrupt:
        sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--viz_context", action='store_true')
    parser.add_argument("--shuffle", action='store_true')
    parser.add_argument("--augment", action='store_true')
    args = parser.parse_args()
    main(args)
