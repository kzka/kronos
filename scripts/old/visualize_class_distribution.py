"""Visualize the class action distribution of a dataset.

TODO: add coloring by actor_type once labeling is complete.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp

from kronos.factories import DATASET_NAME_TO_PATH
from kronos.utils import file_utils


def plot_histo(counts, dataset):
    """Bar chart, adapted from [1].

    References:
     - [1]: https://matplotlib.org/3.2.1/gallery/lines_bars_and_markers/barchart.html
    """
    labels = counts["train"].keys()
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(15, 8))
    rects1 = ax.bar(
        x - width / 2, counts["train"].values(), width, label="train"
    )
    rects2 = ax.bar(
        x + width / 2, counts["valid"].values(), width, label="valid"
    )
    ax.set_ylabel("# of videos")
    ax.set_title("Action Class Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for tick in ax.get_xticklabels():
        tick.set_rotation(76)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.savefig(
        "../tmp/{}_class_dist.png".format(dataset), format="png", dpi=300
    )
    plt.show()


def count_vids(dir):
    """Counts the number of subdirs in a directory.
    """
    counts = {}
    subdirs = file_utils.get_subdirs(dir)
    for dirname in subdirs:
        class_name = osp.basename(dirname)
        counts[class_name] = len(file_utils.get_subdirs(dirname))
    return counts


def main(args):
    paths = DATASET_NAME_TO_PATH[args.dataset]
    counts = {}
    for split, path in paths.items():
        counts[split] = count_vids(path)
    plot_histo(counts, args.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, choices=["penn_action", "mime"], default="mime"
    )
    args, unparsed = parser.parse_known_args()
    main(args)
