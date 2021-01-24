import unittest

import numpy as np

from kronos.datasets import frame_samplers


class SamplersTest(unittest.TestCase):
    def setUp(self):
        self.num_frames = 50
        self.vid_dir = ["{}.jpg".format(i) for i in range(self.num_frames)]

    def test_all_sampler(self):
        actual = frame_samplers.AllSampler()._sample(self.vid_dir)
        expected = list(range(self.num_frames))
        self.assertTrue(np.allclose(actual, expected))

    def test_all_sampler_stride_2(self):
        sampler = frame_samplers.AllSampler(stride=2)
        actual = sampler._sample(self.vid_dir)
        expected = list(range(0, self.num_frames, 2))
        self.assertTrue(np.allclose(actual, expected))

    def test_strided_sampler_stride_1(self):
        sampler = frame_samplers.StridedSampler(
            1, self.num_frames, offset=False
        )
        actual = sampler._sample(self.vid_dir)
        expected = frame_samplers.AllSampler()._sample(self.vid_dir)
        self.assertTrue(np.allclose(actual, expected))

    def test_strided_sampler_stride_2(self):
        sampler = frame_samplers.StridedSampler(
            2, self.num_frames, offset=False
        )
        actual = sampler._sample(self.vid_dir)
        expected = list(range(0, self.num_frames, 2))
        self.assertTrue(np.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
