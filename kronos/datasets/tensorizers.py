"""Tensorizers convert a packet of video data into a packet of video tensors.
"""

import abc

import numpy as np
import torch
import torchvision.transforms.functional as TF

from enum import Enum, unique


@unique
class SequenceType(Enum):
    """Sequence data types we know how to preprocess.

    If you need to load and tensorize additional video
    data, you must add them to `SequenceType`.
    """

    FRAMES = "frames"
    AUDIOS = "audios"
    FRAME_IDXS = "frame_idxs"
    VIDEO_NAME = "video_name"
    VIDEO_LEN = "video_len"
    PHASE_IDXS = "phase_idxs"
    ACTOR_TYPE = "actor_type"
    SUCCESS = "success"
    PHASE_LABELS = "phase_labels"
    DEBRIS_NUMS = "debris_nums"

    def __str__(self):
        return self.value


class Tensorizer(abc.ABC):
    """Base tensorizer class.

    Custom tensorizers must subclass this class.
    """

    @abc.abstractmethod
    def __call__(self, x):
        pass


class IdentityTensorizer(Tensorizer):
    """Outputs the input as is.
    """

    def __call__(self, x):
        return x


class LongTensorizer(Tensorizer):
    """Converts the input to a long tensor.
    """

    def __call__(self, x):
        return torch.from_numpy(np.asarray(x)).long()


class FramesTensorizer(Tensorizer):
    """Converts a sequence of video frames to a float tensor.
    """

    def __call__(self, x):
        if x.ndim != 4:
            raise ValueError("Input must be a 4D sequence of frames.")
        frames = []
        for frame in x:
            frames.append(TF.to_tensor(frame))
        return torch.stack(frames, dim=0)


class ActorTypeTensorizer(Tensorizer):
    """Actor type tensorizer.
    """

    def __call__(self, x):
        x = 1 if x == "human" else 0
        return LongTensorizer()(x)


class ToTensor:
    """Convert video data to video tensors.

    Example::

        >>> data_np = {'frames': np.random.randn(10, 224, 224, 3)}
        >>> data_torch = ToTensor()(data_np)
    """

    MAP = {
        str(SequenceType.FRAMES): FramesTensorizer,
        str(SequenceType.AUDIOS): IdentityTensorizer,
        str(SequenceType.FRAME_IDXS): LongTensorizer,
        str(SequenceType.VIDEO_NAME): IdentityTensorizer,
        str(SequenceType.VIDEO_LEN): LongTensorizer,
        str(SequenceType.PHASE_IDXS): IdentityTensorizer,
        str(SequenceType.ACTOR_TYPE): ActorTypeTensorizer,
        str(SequenceType.SUCCESS): LongTensorizer,
        str(SequenceType.PHASE_LABELS): LongTensorizer,
        str(SequenceType.DEBRIS_NUMS): LongTensorizer,
    }

    def __call__(self, data):
        """Iterate and transform the data values.

        Args:
            data (dict): A dictionary containing key, value pairs
                where the key is an enum member of `SequenceType`
                and the value is a list of numpy arrays
                respecting the key type.

        Raises:
            ValueError: If the input is not a dictionary or
                one of its keys is not a supported sequence
                type.
        """
        if not isinstance(data, dict):
            raise ValueError("Expecting a dictionary.")
        res = {}
        for key, v_np in data.items():
            res[key] = ToTensor.MAP[key]()(v_np)
        return res
