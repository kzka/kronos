import argparse
import os.path as osp
import subprocess


if __name__ == "__main__":
    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type=str)
    args = parser.parse_args()

    logdir = osp.join("kronos/logs", args.experiment_name)
    embs_path = osp.join("kronos/tmp", args.experiment_name)

    # embed trajectories using trained TCC model
    subprocess.call(
        [
            "python",
            "scripts/embed.py",
            "--logdir",
            logdir,
            "--split",
            "train",
            "--max_embs",
            "9",
            "--shuffle",
            "True",
            "--negative",
            "False",
            "--stride",
            "1",
        ]
    )

    # visualize nearest neighbor alignment
    subprocess.call(
        [
            "python",
            "scripts/visualize_alignment.py",
            "--embs_path",
            embs_path,
            "--align",
            "True",
            "--use_dtw",
            "True",
        ]
    )
