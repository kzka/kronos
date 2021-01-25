"""Video frame samplers.
"""

import abc
import random
import numpy as np

from kronos.utils.file_utils import get_files


class FrameSampler(abc.ABC):
    """Video frame sampler base abstraction.
    """

    def __init__(
        self,
        num_frames,
        num_ctx_frames=1,
        ctx_stride=1,
        pattern="*.jpg",
        seed=None,
    ):
        """Constructor.

        Args:
            num_frames (int): How many frames to sample
                in each video.
            num_ctx_frames (int): How many context frames
                to sample for each sampled frame. A value
                of 1 is equivalent to not sampling any
                context frames.
            ctx_stride (int): The spacing between sampled
                context frames.
            pattern (str): The widlcard pattern for the
                video frames.
            seed (int): The seed for the rng.
        """
        assert num_ctx_frames > 0, "num_ctx_frames must be >= 1."

        self._num_frames = int(num_frames)
        self._num_ctx_frames = int(num_ctx_frames)
        self._ctx_stride = int(ctx_stride)
        self._pattern = pattern
        self._seed = seed

        self.seed_rng()

    def seed_rng(self):
        """Reseed the RNG.
        """
        if self._seed is not None:
            random.seed(self._seed)

    def _get_context_steps(self, frame_idxs, vid_len):
        """Generate context frame indices for each sampled frame.

        Currently, context idxs are sampled up to the
        current step, i.e. we don't want to encode
        information from future timesteps.
        """
        ctx_idxs = []
        for idx in frame_idxs:
            idxs = list(
                range(
                    idx - (self._num_ctx_frames - 1) * self._ctx_stride,
                    idx + self._ctx_stride,
                    self._ctx_stride,
                )
            )
            idxs = np.clip(idxs, a_min=0, a_max=vid_len - 1)
            ctx_idxs.append(idxs)
        return ctx_idxs

    @abc.abstractmethod
    def _sample(self, frames, num_frames, phase_indices=None):
        """Subclasses should override this method.

        Args:
            frames (list): A list where each element if
                a list of strings containing the absolute
                path to all the frames in a video.
            num_frames (int): The number of frames we wish
                to sample.

        Returns:
            The indices of the `frames` list to sample.

        :meta public:
        """
        pass

    @abc.abstractmethod
    def _load_frames(self, vid_dirs):
        pass

    def sample(self, vid_dirs, phase_indices=None):
        """Sample the frames in a video directory.

        Args:
            vid_dirs (str): A list of video folder paths from which
                to sample frames.
            num_frames (int): The number of frames we wish to sample.
                This should be a strict subset of the sampled video
                frames.

        Returns:
            A dict containing a list with the sampled frame indices,
            a list of all frame paths in the video directory and
            optionally, a list with indices of the context frames
            for each sampled frame.
        """
        frames = self._load_frames(vid_dirs)
        ret = self._sample(frames, self._num_frames, phase_indices)
        if isinstance(ret, tuple):
            frame_idxs, frame_labels = ret
        else:
            frame_idxs = ret
            frame_labels = None
        ret = {
            "frames": frames,
            "frame_idxs": frame_idxs,
            "vid_len": len(frames),
            "frame_labels": frame_labels,
        }
        ret["ctx_idxs"] = self._get_context_steps(frame_idxs, len(frames))
        return ret

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def num_ctx_frames(self):
        return self._num_ctx_frames


class SingleVideoFrameSampler(FrameSampler):
    """Frame samplers that operate on a single video at a time.

    Subclasses should implemented the `_sample` method.
    """

    def _load_frames(self, vid_dirs):
        assert (
            len(vid_dirs) == 1
        ), f"{self.__class__.__name__} can only operate on a single video at a time."
        return get_files(vid_dirs[0], self._pattern)


class MultiVideoFrameSampler(FrameSampler):
    """Frame samplers that operate on multiple videos at a time.

    Subclasses should implemented the `_sample` method.
    """

    def _load_frames(self, vid_dirs):
        assert (
            len(vid_dirs) > 1
        ), f"{self.__class__.__name__} can only operate on multiple videos at a time."
        return [get_files(vd, self._pattern) for vd in vid_dirs]


class StridedSampler(SingleVideoFrameSampler):
    """Sample every n'th frame of a video.
    """

    def __init__(
        self,
        stride,
        num_frames,
        offset=True,
        num_ctx_frames=1,
        ctx_stride=1,
        pattern="*.jpg",
        seed=None,
    ):
        """Constructor.

        Args:
            stride (int): The spacing between consecutively sampled
                frames. A stride of 1 is equivalent to `AllSampler`.
            offset (bool): If set to `True`, a random starting
                point is chosen along the length of the video. Else,
                the sampling starts at the 0th frame.
        """
        super().__init__(num_frames, num_ctx_frames, ctx_stride, pattern, seed)

        assert stride >= 1, "[!] The stride must be greater or equal to 1."

        self._offset = offset
        self._stride = int(stride)

    def _sample(self, frames, num_frames=None, phase_indices=None):
        if phase_indices is not None:
            idxs, classes = [], []
            for i, exists_idxs in enumerate(phase_indices.values()):
                if exists_idxs[0]:
                    idxs.append(exists_idxs[1])
                    classes.append(i)
            frame_idxs, frame_labels = [], []
            for i, (start, end) in enumerate(idxs):
                cc_idxs = list(range(start, end))
                frame_labels.extend([classes[i]] * len(cc_idxs))
                frame_idxs.extend(cc_idxs)
            return (frame_idxs, frame_labels)
        else:
            vid_len = len(frames)
            poop = False
            if self._offset:
                # the offset can be set between 0 and the maximum location
                # from which we can get total coverage of the video without
                # having to pad.
                offset = random.randint(
                    0, max(1, vid_len - self._stride * num_frames)
                )
            else:
                offset = 0
            if num_frames is None:
                num_frames = int(np.ceil(vid_len / self._stride))
                poop = True
            cc_idxs = list(
                range(
                    offset,
                    offset + num_frames * self._stride + 1,
                    self._stride,
                )
            )
            cc_idxs = np.clip(cc_idxs, a_min=0, a_max=vid_len - 1)
            if poop:
                cc_idxs = cc_idxs[-num_frames:]
            else:
                cc_idxs = cc_idxs[:num_frames]
            return cc_idxs


class AllSampler(StridedSampler):
    """Sample all the frames of a video.

    This should really only be used for evaluation, i.e.
    when embedding all frames of a video, since sampling
    all frames of a video, especially long ones, dramatically
    increases compute and memory requirements.
    """

    def __init__(
        self,
        stride=1,
        num_ctx_frames=1,
        ctx_stride=1,
        offset=False,
        pattern="*.jpg",
        seed=None,
    ):
        super().__init__(
            stride,
            1,
            offset,
            num_ctx_frames,
            ctx_stride,
            pattern,
            seed,  # dummy
        )

    def _sample(self, frames, num_frames=None, phase_indices=None):
        num_frames = None
        return super()._sample(frames, num_frames, phase_indices)


class OffsetUniformSampler(SingleVideoFrameSampler):
    """Uniformly sample video frames starting from an offset.
    """

    def __init__(
        self,
        offset,
        num_frames,
        num_ctx_frames=1,
        ctx_stride=1,
        pattern="*.jpg",
        seed=None,
    ):
        """Constructor.

        Args:
            offset (int): An offset from which to start the
                uniform random sampling.
        """
        super().__init__(num_frames, num_ctx_frames, ctx_stride, pattern, seed)

        self._offset = int(offset)

    def _sample(self, frames, num_frames, phase_indices=None):
        vid_len = len(frames)
        cond1 = vid_len >= self._offset
        cond2 = num_frames < (vid_len - self._offset)
        if cond1 and cond2:
            cc_idxs = list(range(self._offset, vid_len))
            random.shuffle(cc_idxs)
            cc_idxs = cc_idxs[:num_frames]
            return sorted(cc_idxs)
        return list(range(0, num_frames))


class WindowSampler(SingleVideoFrameSampler):
    """Samples a contiguous window of frames.
    """

    def _sample(self, frames, num_frames, phase_indices=None):
        vid_len = len(frames)
        if vid_len > num_frames:
            range_min = random.randrange(vid_len - num_frames)
            range_max = range_min + num_frames
            return list(range(range_min, range_max))
        return list(range(vid_len))


class PhaseSampler(SingleVideoFrameSampler):
    """Uniformly sample video frames starting from an offset.
    """

    def __init__(
        self,
        num_frames,
        num_ctx_frames=1,
        ctx_stride=1,
        pattern="*.png",
        seed=None,
    ):
        """Constructor.

        Args:
            offset (int): An offset from which to start the
                uniform random sampling.
        """
        super().__init__(num_frames, num_ctx_frames, ctx_stride, pattern, seed)

    def _sample(self, frames, num_frames, phase_indices):
        # only extract the phases that exist in the video
        idxs, classes = [], []
        for i, exists_idxs in enumerate(phase_indices.values()):
            if exists_idxs[0]:
                idxs.append(exists_idxs[1])
                classes.append(i)
        # sample an equal proportion of frames for each phase
        num_frames_per_phase = num_frames // len(idxs)
        frame_idxs, frame_labels = [], []
        for i, (start, end) in enumerate(idxs):
            cc_idxs = list(range(start + 10, end))
            random.shuffle(cc_idxs)
            # cc_idxs = cc_idxs[:min(num_frames_per_phase, len(cc_idxs))]
            cc_idxs = cc_idxs[:num_frames_per_phase]
            if len(cc_idxs) < num_frames_per_phase:
                cc_idxs = cc_idxs + [cc_idxs[-1]] * (
                    num_frames_per_phase - len(cc_idxs)
                )
            assert len(cc_idxs) == num_frames_per_phase
            frame_labels.extend([classes[i]] * len(cc_idxs))
            frame_idxs.extend(sorted(cc_idxs))
        return (frame_idxs, frame_labels)
