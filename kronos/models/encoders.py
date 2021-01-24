import torch.nn as nn

from kronos.models import layers
from kronos.models.base import BaseEncoder


class Conv3DFullyConnectedEncoder(BaseEncoder):
    """An encoder with 3D convolutions followed by fully-connected layers.
    """

    def __init__(
        self,
        in_channels,
        num_channels,
        embedding_size,
        num_conv_layers,
        num_fc_layers,
        widen_factor,
        dropout_spatial,
        spatial_dropout_rate,
        fc_dropout_rate,
        flatten_method,
    ):
        """Constructor.

        Args:
            in_channels (int): The number of channels in the input
                feature map.
            num_channels (int): The base number of channels to preserve
                in the 3D convs and fc layers. This number is multiplied
                by the `widen_factor` to increase channel capacity.
            embedding_size (int): The dimension of the embedding space.
            num_conv_layers (int): The number of 3D conv layers.
            num_fc_layers (int): The number of fully-connected layers
                after the 3D convolutions.
            widen_factor (int): A factor that controls the width or
                number of output feature maps of the encoder network.
            dropout_spatial (bool): Whether to use spatial dropout or
                fully independent dropout.
            spatial_dropout_rate (float): The spatial dropout rate.
            fc_dropout_rate (float): The fc dropout rate.
            flatten_method (str): What flattening layer to use when
                transitioning from convolutional to linear layers.
        """
        self._in_channels = in_channels
        self._num_channels = num_channels
        self._embedding_size = int(embedding_size)
        self._num_conv_layers = int(num_conv_layers)
        self._num_fc_layers = int(num_fc_layers)
        self._widen_factor = int(widen_factor)
        self._dropout_spatial = dropout_spatial
        self._spatial_dropout_rate = spatial_dropout_rate
        self._fc_dropout_rate = fc_dropout_rate
        self._flatten_method = flatten_method

        # compute network width, i.e. number of feature maps
        self._num_fm = self._num_channels * self._widen_factor

        super().__init__()

    def _sanity_check_input(self, x):
        assert x.ndim == 5

    def _build_encoder(self):
        layers_ = []
        # dropout layer
        if self._dropout_spatial:
            layers_.append(nn.Dropout3d(self._spatial_dropout_rate))
        else:
            layers_.append(nn.Dropout(self._spatial_dropout_rate))
        # conv - bn - relu layers
        for i in range(self._num_conv_layers):
            in_chs = self._in_channels if i == 0 else self._num_fm
            layers_.append(layers.conv3d(in_chs, self._num_fm, bias=False))
            layers_.append(nn.BatchNorm3d(self._num_fm))
            layers_.append(nn.ReLU())
        # flatten layer
        if self._flatten_method == "max_pool":
            layers_.append(layers.GlobalMaxPool3d())
        elif self._flatten_method == "avg_pool":
            layers_.append(layers.GlobalAvgPool3d())
        else:
            raise ValueError(
                "{} is not supported.".format(self._flatten_method)
            )
        # dropout - fc layers
        for _ in range(self._num_fc_layers):
            layers_.append(nn.Dropout(self._fc_dropout_rate))
            layers_.append(nn.Linear(self._num_fm, self._num_fm))
            layers_.append(nn.ReLU())
        # linear projection
        layers_.append(nn.Linear(self._num_fm, self._embedding_size))
        return nn.Sequential(*layers_)

    def _init_weights(self):
        """Initialize the weights of the model.
        """
        for m in self.modules():
            if isinstance(m, nn.modules.conv._ConvNd):
                nn.init.kaiming_normal_(m.weight, mode="fan_in")


class Conv2DFullyConnectedEncoder(BaseEncoder):
    """An encoder with 2D convolutions followed by fully-connected layers.
    """

    def __init__(
        self,
        in_channels,
        num_channels,
        embedding_size,
        num_conv_layers,
        num_fc_layers,
        widen_factor,
        dropout_spatial,
        spatial_dropout_rate,
        fc_dropout_rate,
        flatten_method,
    ):
        """Constructor.

        Args:
            in_channels (int): The number of channels in the input
                feature map.
            num_channels (int): The base number of channels to preserve
                in the 2D convs and fc layers. This number is multiplied
                by the `widen_factor` to increase channel capacity.
            embedding_size (int): The dimension of the embedding space.
            num_conv_layers (int): The number of 2D conv layers.
            num_fc_layers (int): The number of fully-connected layers
                after the 2D convolutions.
            widen_factor (int): A factor that controls the width or
                number of output feature maps of the encoder network.
            dropout_spatial (bool): Whether to use spatial dropout or
                fully independent dropout.
            spatial_dropout_rate (float): The spatial dropout rate.
            fc_dropout_rate (float): The fc dropout rate.
            flatten_method (str): What flattening layer to use when
                transitioning from convolutional to linear layers.
        """
        self._in_channels = in_channels
        self._num_channels = num_channels
        self._embedding_size = int(embedding_size)
        self._num_conv_layers = int(num_conv_layers)
        self._num_fc_layers = int(num_fc_layers)
        self._widen_factor = int(widen_factor)
        self._dropout_spatial = dropout_spatial
        self._spatial_dropout_rate = spatial_dropout_rate
        self._fc_dropout_rate = fc_dropout_rate
        self._flatten_method = flatten_method

        # compute network width, i.e. number of feature maps
        self._num_fm = self._num_channels * self._widen_factor

        super().__init__()

    def _sanity_check_input(self, x):
        assert x.ndim == 4

    def _build_encoder(self):
        layers_ = []
        # dropout layer
        if self._dropout_spatial:
            layers_.append(nn.Dropout2d(self._spatial_dropout_rate))
        else:
            layers_.append(nn.Dropout(self._spatial_dropout_rate))
        # conv - bn - relu layers
        for i in range(self._num_conv_layers):
            in_chs = self._in_channels if i == 0 else self._num_fm
            layers_.append(layers.conv2d(in_chs, self._num_fm, bias=False))
            layers_.append(nn.BatchNorm2d(self._num_fm))
            layers_.append(nn.ReLU())
        # flatten layer
        if self._flatten_method == "max_pool":
            layers_.append(layers.GlobalMaxPool2d())
        elif self._flatten_method == "avg_pool":
            layers_.append(layers.GlobalAvgPool2d())
        elif self._flatten_method == "spatial_softmax":
            layers_.append(layers.SpatialSoftArgmax())
            self._num_fm *= 2
        else:
            raise ValueError(
                "{} is not supported.".format(self._flatten_method)
            )
        # dropout - fc layers
        for _ in range(self._num_fc_layers):
            layers_.append(nn.Dropout(self._fc_dropout_rate))
            layers_.append(nn.Linear(self._num_fm, self._num_fm))
            layers_.append(nn.ReLU())
        # linear projection
        layers_.append(nn.Linear(self._num_fm, self._embedding_size))
        return nn.Sequential(*layers_)

    def _init_weights(self):
        """Initialize the weights of the model.
        """
        for m in self.modules():
            if isinstance(m, nn.modules.conv._ConvNd):
                nn.init.kaiming_normal_(m.weight, mode="fan_in")
