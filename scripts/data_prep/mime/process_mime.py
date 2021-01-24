"""Process MIME videos into frames.
"""

import os.path as osp

from multiprocessing import cpu_count, Pool, set_start_method
from tqdm import tqdm

from kronos.utils import file_utils
from kronos.utils.video_utils import video_to_frames


def process_video(args):
    file, out_action_video, video = args
    out_action_video_file = osp.join(
        out_action_video, osp.basename(file).split(".")[0]
    )
    file_utils.mkdir(out_action_video_file)
    tstamps = []
    for i, (frame, tstamp) in enumerate(video_to_frames(file, fps=0)):
        file_utils.write_image_to_jpeg(
            frame, osp.join(out_action_video_file, "{}.jpg".format(i))
        )
        tstamps.append(tstamp)
    file_utils.write_timestamps_to_txt(
        tstamps, osp.join(out_action_video_file, "img_timestamps.txt")
    )


def videos_to_images(in_dir, out_dir, num_cores):
    set_start_method("spawn")
    with Pool(num_cores) as pool:
        action_dirs = file_utils.get_subdirs(in_dir)
        for action_dir in action_dirs:
            print("Processing ", osp.basename(action_dir))
            out_action = osp.join(out_dir, osp.basename(action_dir))
            file_utils.mkdir(out_action)
            videos = file_utils.get_subdirs(action_dir)
            for video in videos:
                out_action_video = osp.join(out_action, osp.basename(video))
                file_utils.mkdir(out_action_video)
                files = file_utils.get_files(video, "*.mp4", False)
                func_args = [[f, out_action_video, video] for f in files]
                for _ in tqdm(
                    pool.imap_unordered(process_video, func_args),
                    total=len(files),
                ):
                    pass
                file_utils.copy_file(
                    osp.join(video, "joint_angles.txt"),
                    osp.join(out_action_video, "joint_angles.txt"),
                )


if __name__ == "__main__":
    in_dir = "/home/kevin/repos/mime/data/"
    out_dir = "/home/kevin/repos/kronos/kronos/data/mime/"
    videos_to_images(in_dir, out_dir, cpu_count())
