"""Transformations for video data.
"""

import albumentations as alb
import numpy as np
import random
import torch

from functools import partial
from enum import Enum, unique

from .tensorizers import SequenceType


# since we're using a resnet backbone pretrained
# on imagenet, we should normalize all images
# using the same normalization that was used
# during pretraining
IMAGENET_MEANS = (0.485, 0.456, 0.406)
IMAGENET_STDS = (0.229, 0.224, 0.225)


@unique
class TransformationType(Enum):
    """Transformations we know how to run.

    If you want to add additional data augmentation
    techniques, you must add them to `TransformationType`.
    """

    RANDOM_RESIZED_CROP = "random_resized_crop"
    CENTER_CROP = "center_crop"
    GLOBAL_RESIZE = "global_resize"
    VERTICAL_FLIP = "vertical_flip"
    HORIZONTAL_FLIP = "horizontal_flip"
    COLOR_JITTER = "color_jitter"
    ROTATE = "rotate"
    NORMALIZE = "normalize"

    def __str__(self):
        return self.value


class UnNormalize:
    """Unnormalize a batch of images that have been normalized.

    Speficially, re-multiply by the standard deviation and
    shift by the mean. Default values are taken from [1].

    References:
        [1]: GitHub discussion,
        https://github.com/pytorch/vision/issues/1439
    """

    def __init__(self, mean=IMAGENET_MEANS, std=IMAGENET_STDS):
        """Constructor.

        Args:
            mean (tuple): The color channel means. By default, ImageNet
                channel means are used.
            std (tuple): The color channel standard deviation. By default,
                ImageNet channel standard deviations are used.
        """
        if np.asarray(mean).shape:
            self.mean = torch.tensor(mean)[..., :, None, None]
        if np.asarray(std).shape:
            self.std = torch.tensor(std)[..., :, None, None]

    def __call__(self, tensor):
        return (tensor * self.std) + self.mean


class ColorJitter(alb.ImageOnlyTransform):
    """Randomly change the brightness, contrast, hue and saturation of an image [1].

    References:
        [1]: kdexd's virtex implementation,
        https://github.com/kdexd/virtex
    """

    def __init__(
        self, brightness, contrast, hue, saturation, always_apply=False, p=0.5,
    ):
        """Constructor.
        """
        super().__init__(always_apply, p)

        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue
        self.saturation = saturation

    def get_params(self):
        return {
            "brightness_factor": random.uniform(
                1 - self.brightness, 1 + self.brightness
            ),
            "contrast_factor": random.uniform(
                1 - self.contrast, 1 + self.contrast
            ),
            "saturation_factor": random.uniform(
                -self.saturation, self.saturation
            ),
            "hue_factor": random.uniform(-self.hue, self.hue),
        }

    def apply(self, img, **params):
        original_dtype = img.dtype
        img = alb.augmentations.functional.brightness_contrast_adjust(
            img,
            alpha=params["contrast_factor"],
            beta=params["brightness_factor"] - 1,
        )
        img = alb.augmentations.functional.shift_hsv(
            img,
            hue_shift=int(params["hue_factor"] * 255),
            sat_shift=int(params["saturation_factor"] * 255),
            val_shift=0,
        )
        return img.astype(original_dtype)

    def get_transform_init_args_names(self):
        return ("brightness", "contrast", "hue", "saturation")


class Augmentor:
    """Data augmentation for videos.

    Augmentor consistently augments data across the time
    dimension (i.e. dim 0). In other words, the same
    transformation is applied to every single frame
    in a video sequence.

    Example::

        >>> data_np = {'frames': np.random.randn(10, 224, 224, 3)}
        >>> augmentor = Augmentor('frames', ['color_jitter', 'normalize'])
        >>> data_aug = augmentor(data_np)
    """

    MAP = {
        str(TransformationType.RANDOM_RESIZED_CROP): partial(
            alb.RandomResizedCrop, scale=(0.2, 1.0), ratio=(0.75, 1.333), p=1.0
        ),
        str(TransformationType.CENTER_CROP): partial(alb.CenterCrop, p=1.0),
        str(TransformationType.GLOBAL_RESIZE): partial(alb.Resize, p=1.0),
        str(TransformationType.COLOR_JITTER): partial(
            ColorJitter,
            brightness=0.1,
            contrast=0.1,
            hue=0.08,
            saturation=0.08,
            p=0.8,
        ),
        str(TransformationType.HORIZONTAL_FLIP): partial(
            alb.HorizontalFlip, p=0.5
        ),
        str(TransformationType.VERTICAL_FLIP): partial(
            alb.VerticalFlip, p=0.5
        ),
        str(TransformationType.ROTATE): partial(
            alb.Rotate, limit=(-45, 45), border_mode=0, p=0.5
        ),
        str(TransformationType.NORMALIZE): partial(
            alb.Normalize, mean=IMAGENET_MEANS, std=IMAGENET_STDS, p=1.0
        ),
    }

    def __init__(self, key, params, image_size=None):
        """Constructor.

        Args:
            key (SequenceType): Which key of the dataset dict to
                transform.
            params (list): A list of strings specifying an ordered
                set of data transformations to apply to the video
                images.
            image_size (tuple): What size to reshape or crop the
                video images if a resize or crop transform is
                specified.

        Raises:
            ValueError: If params contains an unsupported data augmentation.
        """
        if key not in SequenceType._value2member_map_:
            raise ValueError(f"{key} is not a supported SequenceType.")
        self._key = key

        # make sure transformations we receive
        # are supported
        for p_name in params:
            if p_name not in Augmentor.MAP:
                raise ValueError(
                    f"{p_name} is not a supported data transformation."
                )

        img_transform_list = []
        for name in params:
            if "resize" in name or "crop" in name:
                img_transform_list.append(Augmentor.MAP[name](*image_size))
            else:
                img_transform_list.append(Augmentor.MAP[name]())
        self._pipeline = alb.Compose(img_transform_list)

    @staticmethod
    def augment_video(frames, pipeline):
        """Apply the same augmentation pipeline to all frames in a video.

        Args:
            frames (ndarray): A numpy array of shape (T, H, W, 3),
                where `T` is the number of frames in the video.
            pipeline (list): A list containing albumentation
                augmentations.

        Returns:
            The augmented frames of shape (T, H, W, 3).

        Raises:
            ValueError: If the input video doesn't have the correct shape.
        """
        if frames.ndim != 4:
            raise ValueError("Input video must be a 4D sequence of frames.")

        transform = alb.ReplayCompose(pipeline, p=1.0)

        # apply a transformation to the first frame and record
        # the parameters that were sampled in a replay, then
        # use the parameters stored in the replay to
        # apply an identical transform to the remaining
        # frames in the sequence.
        replay, frames_aug = None, []
        for frame in frames:
            if replay is None:
                aug = transform(image=frame)
                replay = aug.pop("replay")
            else:
                aug = transform.replay(replay, image=frame)
            frames_aug.append(aug["image"])

        return np.stack(frames_aug, axis=0)

    def __call__(self, data):
        """Iterate and transform the data values.

        Currently, data augmentation is only applied
        to video frames, i.e. the value of the data
        dict associated with the `SequenceType.IMAGE`
        key.
        """
        data[self._key] = Augmentor.augment_video(
            data[self._key], self._pipeline
        )
        return data
