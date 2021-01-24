import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(
    dim,
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    dilation=1,
    bias=True,
):
    """`same` convolution, i.e. output shape equals input shape.

    Args:
        dim (int): The dimension of the convolution. 2 is conv2d,
            3 is conv3d.
        in_planes (int): The number of input feature maps.
        out_planes (int): The number of output feature maps.
        kernel_size (int): The filter size.
        dilation (int): The filter dilation factor.
        stride (int): The filter stride.
    """
    assert dim in [2, 3], "[!] Only 2D and 3D convolution supported."
    conv = nn.Conv2d if dim == 2 else nn.Conv3d

    # compute new filter size after dilation
    # and necessary padding for `same` output size
    dilated_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size
    same_padding = (dilated_kernel_size - 1) // 2

    return conv(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=same_padding,
        dilation=dilation,
        bias=bias,
    )


def conv2d(*args, **kwargs):
    """`same` 2D convolution, i.e. output shape equals input shape.

    Args:
        in_planes (int): The number of input feature maps.
        out_planes (int): The number of output feature maps.
        kernel_size (int): The filter size.
        dilation (int): The filter dilation factor.
        stride (int): The filter stride.
    """
    return _conv(2, *args, **kwargs)


def conv3d(*args, **kwargs):
    """`same` 3D convolution, i.e. output shape equals input shape.

    Args:
        in_planes (int): The number of input feature maps.
        out_planes (int): The number of output feature maps.
        kernel_size (int): The filter size.
        dilation (int): The filter dilation factor.
        stride (int): The filter stride.
    """
    return _conv(3, *args, **kwargs)


class Flatten(nn.Module):
    """Flattens convolutional feature maps for fc layers.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class SpatialSoftArgmax(nn.Module):
    """Spatial softmax as defined in [1].

   Concretely, the spatial softmax of each feature
   map is used to compute a weighted mean of the pixel
   locations, effectively performing a soft arg-max
   over the feature dimension.

    References:
        [1]: End-to-End Training of Deep Visuomotor Policies,
        https://arxiv.org/abs/1504.00702
    """

    def __init__(self, normalize=False):
        """Constructor.

        Args:
            normalize (bool): Whether to use normalized
                image coordinates, i.e. coordinates in
                the range `[-1, 1]`.
        """
        super().__init__()

        self.normalize = normalize

    def _coord_grid(self, h, w, device):
        if self.normalize:
            return torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, w, device=device),
                    torch.linspace(-1, 1, h, device=device),
                )
            )
        return torch.stack(
            torch.meshgrid(
                torch.arange(0, w, device=device),
                torch.arange(0, h, device=device),
            )
        )

    def forward(self, x):
        assert x.ndim == 4, "Expecting a tensor of shape (B, C, H, W)."

        # compute a spatial softmax over the input:
        # given an input of shape (B, C, H, W),
        # reshape it to (B*C, H*W) then apply
        # the softmax operator over the last dimension
        b, c, h, w = x.shape
        softmax = F.softmax(x.view(-1, h * w), dim=-1)

        # create a meshgrid of normalized pixel
        # coordinates both in the x and y axes
        xc, yc = self._coord_grid(h, w, x.device)

        # element-wise multiply the x and y coordinates
        # with the softmax, then sum over the h*w dimension
        # this effectively computes the weighted mean x and
        # y locations
        x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)

        # concatenate and reshape the result
        # to (B, C*2) where for every feature
        # we have the expected x and y pixel
        # locations
        return torch.cat([x_mean, y_mean], dim=1).view(-1, c * 2)


class _GlobalMaxPool(nn.Module):
    """Global max pooling layer.
    """

    def __init__(self, dim):
        super().__init__()

        if dim == 1:
            self._pool = F.max_pool1d
        elif dim == 2:
            self._pool = F.max_pool2d
        elif dim == 3:
            self._pool = F.max_pool3d
        else:
            raise ValueError("{}D is not supported.")

    def forward(self, x):
        return self._pool(x, kernel_size=x.size()[2:]).squeeze()


class GlobalMaxPool1d(_GlobalMaxPool):
    """Global max pooling operation for temporal data.
    """

    def __init__(self):
        super().__init__(dim=1)


class GlobalMaxPool2d(_GlobalMaxPool):
    """Global max pooling operation for spatial data.
    """

    def __init__(self):
        super().__init__(dim=2)


class GlobalMaxPool3d(_GlobalMaxPool):
    """Global max pooling operation for 3D data.
    """

    def __init__(self):
        super().__init__(dim=3)


class _GlobalAvgPool(nn.Module):
    """Global average pooling layer.
    """

    def __init__(self, dim):
        super().__init__()

        if dim == 1:
            self._pool = F.avg_pool1d
        elif dim == 2:
            self._pool = F.avg_pool2d
        elif dim == 3:
            self._pool = F.avg_pool3d
        else:
            raise ValueError("{}D is not supported.")

    def forward(self, x):
        return self._pool(x, kernel_size=x.size()[2:]).squeeze()


class GlobalAvgPool1d(_GlobalAvgPool):
    """Global average pooling operation for temporal data.
    """

    def __init__(self):
        super().__init__(dim=1)


class GlobalAvgPool2d(_GlobalAvgPool):
    """Global average pooling operation for spatial data.
    """

    def __init__(self):
        super().__init__(dim=2)


class GlobalAvgPool3d(_GlobalAvgPool):
    """Global average pooling operation for 3D data.
    """

    def __init__(self):
        super().__init__(dim=3)


class CausalConv1d(nn.Conv1d):
    """A causal 1D convolution.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The filter size.
        stride (int): The filter stride.
        dilation (int): The filter dilation factor.
        bias (bool): Whether to add the bias term or not.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
    ):
        self.__padding = (kernel_size - 1) * dilation

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        res = super().forward(x)
        if self.__padding != 0:
            return res[:, :, : -self.__padding]
        return res
