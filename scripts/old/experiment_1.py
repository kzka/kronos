import json
import numpy as np
import os.path as osp
import subprocess

from itertools import combinations

from kronos.utils import file_utils


pens_to_two_cups_action_classes = [
    "crab",
    "one_hand_two_fingers",
    "one_hand_five_fingers",
    "ski_gloves",
    "quick_grasp",
    "rms",
    "tongs",
    "tweezers",
    "two_hands_two_fingers",
]
eyeglasses_in_case_action_classes = [
    "crab",
    "double_quick_grip",
    "one_hand_five_fingers",
    "one_hand_two_fingers",
    "quick_grip",
    "rms",
    "ski_gloves",
    "tongs",
    "tweezers",
    "two_hands_two_fingers",
]
cup_to_plate_action_classes = [
    "crab",
    "one_hand_five_fingers",
    "one_hand_two_fingers",
    "quick_grip",
    "rms",
    "ski_gloves",
    "tongs",
    "tweezers",
    "two_hands_two_fingers",
    "wooden_spoon",
]
flip_book_action_classes = [
    "crab",
    "double_quick_grip",
    "one_hand_five_fingers",
    "one_hand_two_fingers",
    "quick_grip",
    "rms",
    "ski_gloves",
    "tongs",
    "two_hands_two_fingers",
    "wooden_spoon",
]
pens_to_one_cup_action_classes = [
    "",
]

TASK_TO_EMBODIMENTS = {
    "pens_to_one_cup": pens_to_one_cup_action_classes,
    "pens_to_two_cups": pens_to_two_cups_action_classes,
    "eyeglasses_in_case": eyeglasses_in_case_action_classes,
    "cup_in_plate": cup_to_plate_action_classes,
    "flip_book": flip_book_action_classes,
}


if __name__ == "__main__":
    task = "eyeglasses_in_case"
    action_classes = TASK_TO_EMBODIMENTS[task]
    N = len(action_classes)
    combos = combinations(action_classes, N - 1)
    num_combos = N

    best_accs = {}
    for i, combo in enumerate(combos):
        query_action = list(set(action_classes) - set(combo))[0]
        combo = list(combo)
        print("{}/{}: {}".format(i + 1, num_combos, query_action))

        # experiment_name = f"{task}_exp_1_{i}"
        # experiment_name = f"exp_1_{i}"
        experiment_name = f"eye_exp_1_{i}"

        # # train tcc model
        # subprocess.call(
        #     [
        #         "python",
        #         "scripts/train.py",
        #         "--experiment_name",
        #         experiment_name,
        #         "--config_path",
        #         "configs/tcc_config.yml",
        #         "--action_class",
        #         *combo,
        #         "--resume",
        #         "True",
        #     ]
        # )

        # train lstm model
        out = subprocess.check_output(
            [
                "python",
                "scripts/train_lstm.py",
                "--logdir",
                "kronos/logs/{}".format(experiment_name),
                "--query",
                query_action,
                "--l2_normalize",
                "False",
                "--batch_size",
                "20",
                "--stride",
                "20",
                "--max_iters",
                "300",
            ]
        )

        best_acc = float(out.decode().split("\n")[-2].split(":")[-1].lstrip())
        best_accs[query_action] = best_acc
        print("{}: {}".format(query_action, best_acc))

    save_dir = f"tmp/exp_1/{task}"
    file_utils.mkdir(save_dir)
    file_utils.write_json(osp.join(save_dir, "result.json"), best_accs)
