import unittest

import torch

from kronos.models import layers


class LayersTest(unittest.TestCase):
    def test_spatial_soft_argmax(self):
        b, c, h, w = 32, 64, 16, 16
        x = torch.zeros(b, c, h, w)
        true_max = torch.randint(0, 10, size=(b, c, 2))
        for i in range(b):
            for j in range(c):
                x[i, j, true_max[i, j, 0], true_max[i, j, 1]] = 1000
        soft_max = layers.SpatialSoftArgmax(normalize=False)(x).reshape(
            b, c, 2
        )
        self.assertTrue(torch.allclose(true_max.float(), soft_max))


if __name__ == "__main__":
    unittest.main()
