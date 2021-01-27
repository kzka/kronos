import json
import os
import os.path as osp
import shutil

import numpy as np

from glob import glob
from PIL import Image


def mkdir(path):
    """Create a directory if it doesn't already exist.
    """
    if not osp.exists(path):
        os.makedirs(path)


def rm(path):
    """Remove a file or folder.
    """
    if osp.exists(path):
        if osp.isfile(path):
            os.remove(path)
        elif osp.isdir(path):
            shutil.rmtree(path)
        else:
            raise IOError("{} is not a file or directory.".format(path))


def get_subdirs(d, nonempty=False, sort=True, basename=False, sortfunc=None):
    """Return a list of subdirectories in a given directory.

    Args:
        d (str): The path to the directory.
        nonempty (bool): Only return non-empty subdirs.
        sort (bool): Whether to sort by alphabetical order.
        basename (bool): Only return the tail of the subdir paths.
        sortfunc (func): An optional sort function to use if
            sort is set to `True`.
    """
    subdirs = [f.path for f in os.scandir(d) \
               if f.is_dir() \
               and not f.name.startswith('.')]
    if nonempty:
        subdirs = [f for f in subdirs if not is_folder_empty(f)]
    if sort:
        if sortfunc is None:
            subdirs.sort(key=lambda x: x.split("/")[-1])
        else:
            subdirs.sort(key=sortfunc)
    if basename:
        return [osp.basename(x) for x in subdirs]
    return subdirs


def get_files(d, pattern, sort=True):
    """Return a list of files in a given directory.

    Args:
        d (str): The path to the directory.
        pattern (str): The wildcard to filter files with.
        sort (bool): Whether to sort the returned list.
    """
    files = glob(osp.join(d, pattern))
    files = [f for f in files if osp.isfile(f)]
    if sort:
        files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    return files


def is_folder_empty(d):
    """A folder is not empty if it contains >1 non hidden file.
    """
    return len(glob(osp.join(d, "*"))) == 0


def load_jpeg(filename, resize=None):
    """Loads a JPEG image as a numpy array.

    Args:
        filename (str): The name of the image file.
        resize (tuple): The height and width of the
            loaded image. Set to None to keep original
            image dims.

    Returns:
        A numpy uint8 array.
    """
    img = Image.open(filename)
    if resize is not None:
        # PIL expects a (width, height) tuple
        img = img.resize((resize[1], resize[0]))
    return np.asarray(img)


def write_image_to_jpeg(arr, filename, jpeg_quality=95):
    """Save a numpy uint8 image array as a JPEG image to disk.

    Args:
        arr (ndarray): A numpy RGB uint8 image.
        filename (str): The filename of the saved image.
        jpeg_quality (int): The quality of the jpeg image.

    Raises:
        ValueError: If the numpy dtype is not uint8.
    """
    if arr.dtype is not np.dtype(np.uint8):
        raise ValueError("Image must be uint8.")
    jpeg_quality = np.clip(jpeg_quality, a_min=1, a_max=95)
    image = Image.fromarray(arr)
    image.save(filename, jpeg_quality=jpeg_quality)


def write_image_to_png(arr, filename):
    """Save a numpy uint8 image array as a PNG image to disk.

    Args:
        arr (ndarray): A numpy RGB uint8 image.
        filename (str): The filename of the saved image.

    Raises:
        ValueError: If the numpy dtype is not uint8.
    """
    if arr.dtype is not np.dtype(np.uint8):
        raise ValueError("Image must be uint8.")
    image = Image.fromarray(arr)
    image.save(filename, format='png')


def write_audio_to_binary(arr, filename):
    """Save a numpy float64 audio array as a binary npy file.

    Args:
        arr (ndarray): A numpy float64 array.
        filename (str): The filename of the saved audio file.

    Raises:
        ValueError: If the numpy dtype is not float64.
    """
    if arr.dtype is not np.dtype(np.float64):
        raise ValueError("Audio must be float64.")
    np.save(filename, arr, allow_pickle=True)


def write_timestamps_to_txt(arr, filename):
    """Save a list of timestamps as a txt file.

    Args:
        arr (ndarray): A numpy float64 array.
        filename (str): The filename of the saved txt file.
    """
    np.savetxt(filename, arr)


def flatten(list_of_list):
    """Flattens a list of lists.
    """
    return [item for sublist in list_of_list for item in sublist]


def copy_folder(src, dst):
    """Copies a folder to a new location.
    """
    shutil.copytree(src, dst)


def move_folder(src, dst):
    """Copies a folder to a new location.
    """
    shutil.move(src, dst)


def copy_file(src, dst):
    """Copies a file to a new location.
    """
    shutil.copyfile(src, dst)


def write_json(filename, d):
    """Write a dict to a json file.
    """
    with open(filename, "w") as f:
        json.dump(d, f, indent=2)


def load_json(filename):
    """Load a json file to a dict.
    """
    with open(filename, "r") as f:
        d = json.load(f)
    return d


def dict_from_list(x):
    """Creates a dictionary from a list.

    Args:
        x (list): A flat list.

    Returns:
        A dict where the keys are range(0, len(x)) and
        the values are the list contents.
    """
    return {k: v for k, v in zip(range(len(x)), x)}
