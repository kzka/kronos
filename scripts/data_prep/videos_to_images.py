"""Convert folders of videos into folders of frames and audio samples.

The expected structure of the input directory is:

└── input_dir
    ├── class1
    │   ├── vid1.mp4
    |   ...
    │   ├── vid4.mp4
    ├── class2
    │   ├── vid1.mp4
    |   ...
    │   └── vid10.mp4
    ...

The above is an example of the Kinetics dataset
training directory, which contains ~700 subfolders
corresponding to different types of actions.
In each action class folder, there is a list of videos
in a certain format (e.g. mp4). The set of videos can
be filtered using a glob pattern specified with the
`file_pattern` arg.

This script will create a folder for every video in each
subdirectory. In this folder, it will store the extracted
frames as JPG images and the extracted audio samples as
binary numpy files. You can also choose to save the
processed dataset to a new `output_dir`.

└── output_dir
    ├── class1
    │   ├── vid1
    │   |   ├── 0.jpg
    │   |   ├── 0.npy
    │   |   ├── 1.jpg
    │   |   ├── 1.npy
    |   |   ...
    |   |   |── audio_timestamps.txt
    |   |   |── video_timestamps.txt
    |   ...
    │   ├── vid4
    │   |   ├── 0.jpg
    │   |   ├── 0.npy
    │   |   ├── 1.jpg
    │   |   ├── 1.npy
    |   |   ...
    |   |   |── audio_timestamps.txt
    |   |   |── video_timestamps.txt
    ├── class2
    │   ├── vid1
    │   |   ├── 0.jpg
    │   |   ├── 0.npy
    │   |   ├── 1.jpg
    │   |   ├── 1.npy
    |   |   ...
    |   |   |── audio_timestamps.txt
    |   |   |── video_timestamps.txt
    |   ...
    │   ├── vid10
    │   |   ├── 0.jpg
    │   |   ├── 0.npy
    │   |   ├── 1.jpg
    │   |   ├── 1.npy
    |   |   ...
    |   |   |── audio_timestamps.txt
    |   |   |── video_timestamps.txt
    ...
"""

import argparse
import logging
import os
import pdb

from glob import glob
from multiprocessing import cpu_count, Pool, set_start_method
from tqdm import tqdm

from kronos.utils import file_utils
from kronos.utils.video_utils import video_to_frames, video_to_audio


def unpack_video(args):
    # unpack args
    video_dir, video_filename, fps, resize, extract_audio, sampling_rate = args

    file_utils.mkdir(video_dir)

    # unpack video into frames
    tstamps = []
    for i, (frame, tstamp) in enumerate(
        video_to_frames(video_filename, fps, resize)
    ):
        # save frame as jpg
        # file_utils.write_image_to_jpeg(
        #     frame, os.path.join(video_dir, "{}.jpg".format(i))
        # )
        file_utils.write_image_to_png(
            frame, os.path.join(video_dir, "{}.png".format(i))
        )
        tstamps.append(tstamp)
    # save frame timestamps as single txt file
    file_utils.write_timestamps_to_txt(
        tstamps, os.path.join(video_dir, "img_timestamps.txt")
    )

    # unpack audio into snippets
    if extract_audio:
        # some files do not have audio data so we skip them
        try:
            tstamps = []
            for i, (audio, tstamp) in enumerate(
                video_to_audio(video_filename, sampling_rate)
            ):
                file_utils.write_audio_to_binary(
                    audio, os.path.join(video_dir, "{}.npy".format(i))
                )
                tstamps.append(tstamp)
            # save audio timestamps as single txt file
            file_utils.write_timestamps_to_txt(
                tstamps, os.path.join(video_dir, "audio_timestamps.txt")
            )
        except (FileNotFoundError, OSError) as err:
            logging.exception("{}: ".format(video_filename), err)
    logging.debug("Unpack of {} complete.".format(video_dir))


if __name__ == "__main__":

    def str2bool(s):
        return s.lower() in ["true", "1"]

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Path to videos.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory where images will be stored."
        + " By default, stored in the input directory.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=0,
        help="The fps of the video. Set to 0 to read the video's fps from metadata.",
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.mp4",
        help="Pattern used to search for files in the directory.",
    )
    parser.add_argument(
        "--resize",
        type=str2bool,
        default=True,
        help="Resize the video frames to a specified size.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="The height of processed video frames if resize is set to True.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="The width of processed video frames if resize is set to True.",
    )
    parser.add_argument(
        "--extract_audio",
        type=str2bool,
        default=False,
        help="Whether to extract audio snippers at every second.",
    )
    parser.add_argument(
        "--sampling_rate",
        type=float,
        default=44100,
        help="The sampling rate of the audio signal.",
    )
    parser.add_argument(
        "--num_cores",
        type=int,
        default=cpu_count(),
        help="The number of cores to use in parallel.",
    )
    args, unparsed = parser.parse_known_args()

    # get list of subdirectories in the input dir
    # each subdir in this list corresponds to a different embodiment
    input_subdirs = file_utils.get_subdirs(args.input_dir)

    # create output dir if it does not exist
    if args.output_dir is not None:
        file_utils.mkdir(args.output_dir)
        output_subdirs = [
            os.path.join(args.output_dir, os.path.basename(x))
            for x in input_subdirs
        ]
    else:
        output_subdirs = input_subdirs

    if args.resize:
        args.resize = (args.height, args.width)
    else:
        args.resize = None

    args.num_cores = min(args.num_cores, cpu_count())
    print("Using {} cores.".format(args.num_cores))

    # video processing params
    params = (
        args.fps,
        args.resize,
        args.extract_audio,
        args.sampling_rate,
    )

    set_start_method("spawn")
    for input_subdir, output_subdir in zip(input_subdirs, output_subdirs):
        logging.debug("Processing {}...".format(input_subdir))

        # Create the output subdir if it doesn't exist.
        file_utils.mkdir(output_subdir)

        # For each input_subdir, we want to get all the subfolders which
        # correspond to a rollout trajectory.
        input_demos = file_utils.get_subdirs(input_subdir, nonempty=True)
        logging.debug("Contains {} vidoes.".format(len(input_demos)))

        # Get the video in each demo.
        input_videos = []
        for input_demo in input_demos:
            vid_file = file_utils.get_files(
                input_demo, args.file_pattern, sort=False)
            if len(vid_file) > 0:
                input_videos.extend(vid_file)

        out_video_dirs = []
        for video_filename in input_videos:
            video_name = os.path.dirname(video_filename).split('/')[-1]
            out_video_dirs.append(os.path.join(output_subdir, video_name))

        # Unpack videos into frames.
        with Pool(args.num_cores) as pool:
            func_args = [
                [vd, vf, *params]
                for vd, vf in zip(out_video_dirs, input_videos)
            ]
            for _ in tqdm(
                pool.imap_unordered(unpack_video, func_args),
                total=len(out_video_dirs),
            ):
                pass
