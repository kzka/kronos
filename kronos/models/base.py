import abc
import logging
import torch
import torch.nn as nn

from kronos.utils.torch_utils import freeze_model


class BaseFeaturizer(abc.ABC, nn.Module):
    """Abstract base class for featurizers.

    Subclasses must implement the `_build_featurizer` method.
    """

    def __init__(
        self,
        pretrained=True,
        layers_train="bn_only",
        bn_use_running_stats=False,
    ):
        """Constructor.

        Args:
            pretrained (bool): Whether to use Imagenet pretrained weights.
            layers_train (str): Controls which layers are trained. Can be
                one of `['all', 'frozen', 'bn_only']`.
            bn_use_running_stats (bool): Set to `True` to disable batch
                statistics and use running mean and variance learned during
                training.
        """
        super().__init__()

        self._bn_use_running_stats = bn_use_running_stats
        self._layers_train = layers_train
        self._pretrained = pretrained

        # build featurizer
        self.model = self._build_featurizer()

        # figure out batch norm related freezing
        if layers_train != "all":
            if layers_train == "frozen":
                logging.info("Freezing all featurizer layers.")
                bn_freeze_affine = True
            elif layers_train == "bn_only":
                logging.info(
                    "Freezing all featurizer layers except for batch norm layers."
                )
                bn_freeze_affine = False
            else:
                raise ValueError(
                    "{} is not a valid layer selection strategy.".format(
                        layers_train
                    )
                )
            freeze_model(self.model, bn_freeze_affine, bn_use_running_stats)

        # build param to module dict
        self.param_to_module = {}
        for m in self.modules():
            for p in m.parameters(recurse=False):
                self.param_to_module[p] = type(m).__name__

    @abc.abstractmethod
    def _build_featurizer(self):
        """Build the featurizer architecture.
        """
        pass

    def forward(self, x):
        """Extract features from the video frames.

        Args:
            x (torch.FloatTensor): The video frames of shape
                `(B, T, C, H, W)`. If there are `S` video
                frames and we are using `X` context frames,
                then `T = S * X`.
        """
        assert x.ndim == 5
        batch_size, t, c, h, w = x.shape
        x = x.view((batch_size * t, c, h, w))
        feats = self.model(x)
        _, c, h, w = feats.shape
        feats = feats.view((batch_size, t, c, h, w))
        return feats

    def train(self):
        """Sets the model in `train` mode.
        """
        self.training = True
        for m in self.model.modules():
            # set everything that is NOT batchnorm to train
            if not isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.train()
            else:
                # for batch norm, we only want train mode
                # if we were not using running statistics
                if self._bn_use_running_stats:
                    m.eval()
                else:
                    m.train()

    def eval(self):
        """Sets the model in `eval` mode.
        """
        self.training = False
        for m in self.model.modules():
            m.eval()

    @property
    def pretrained(self):
        return self._pretrained


class BaseEncoder(abc.ABC, nn.Module):
    """Abstract base class for encoders.

    Subclasses must implement the `_build_featurizer` method.
    """

    def __init__(self):
        super().__init__()

        self.model = self._build_encoder()
        self._init_weights()

        # build param to module dict
        self.param_to_module = {}
        for m in self.modules():
            for p in m.parameters(recurse=False):
                self.param_to_module[p] = type(m).__name__

    @abc.abstractmethod
    def _build_encoder(self):
        """Build the encoder architecture.
        """
        pass

    @abc.abstractmethod
    def _init_weights(self):
        """Initialize the weights of the encoder.
        """
        pass

    def train(self):
        """Sets the model in `train` model.
        """
        super().train(True)

    def eval(self):
        """Sets the model in `eval` model.
        """
        super().train(False)

    def _sanity_check_input(self, x):
        """Add necessary logic to parse inputs here.
        """
        pass

    def forward(self, x):
        """Forward the video frame features through the encoder.

        Args:
            inputs (tensor): A FloatTensor of shape `(B, C, X, H, W)` where
                `X` is the number of context frames.
        """
        self._sanity_check_input(x)
        return self.model(x)


class BaseTemporalModel(abc.ABC, nn.Module):
    """Abstract base class for temporal video models.
    """

    def __init__(
        self, num_ctx_frames, model_config,
    ):
        """Constructor.

        Args:
            num_ctx_frames (int): The number of context frames stacked
                together for each individual video frame.
            model_config (edict): A dictionary containing model architecture
                hyperparameters.
        """
        super().__init__()

        self.training = True
        self._num_ctx_frames = num_ctx_frames
        self._model_config = model_config

        # initialize models
        self.featurizer_net = self._init_featurizer()
        self.encoder_net = self._init_encoder()
        self.auxiliary_net = self._init_auxiliary_net()

    @abc.abstractmethod
    def _init_featurizer(self):
        pass

    @abc.abstractmethod
    def _init_encoder(self):
        pass

    def _init_auxiliary_net(self):
        return None

    def param_groups(self):
        """Return a dict of variable optimization parameters.

        This is useful for specifying variable learning
        rates and regularizations strengths for the
        different models.
        """
        param_groups = {
            "featurizer": {"free": self.featurizer_net.parameters()},
            "encoder": {"free": self.encoder_net.parameters()},
        }
        if self.auxiliary_net is not None:
            param_groups["auxiliary"] = {
                "free": self.auxiliary_net.parameters()
            }
        return param_groups

    def _sanity_check_input(self, frames):
        """Add necessary logic to parse inputs here.
        """
        pass

    def forward(self, frames, num_ctx_frames=None):
        """Forward the video frames through the network.

        Args:
            x (torch.FloatTensor): The video frames of shape
                `(B, T, C, H, W)`. If there are `S` video
                frames and we are using `X` context frames,
                then `T = S * X`.

        Returns:
            A FloatTensor of shape `(B, S, D)` where `S`
            is the number of video frames and `D` is
            the dimension of the emebdding space.
        """
        self._sanity_check_input(frames)

        # extract frame features
        feats = self.featurizer_net(frames)

        # reshape for the 3D convs
        if num_ctx_frames is None:
            num_ctx_frames = self._num_ctx_frames
        batch_size, s, c, h, w = feats.shape
        num_cc_frames = s // num_ctx_frames
        feats = feats.view(
            (batch_size * num_cc_frames, c, num_ctx_frames, h, w)
        )

        # embed frames using encoder
        embs = self.encoder_net(feats)
        embs = embs.view((-1, num_cc_frames, embs.shape[-1]))

        return {
            "embs": embs,
            "feats": feats,
        }

    def train(self):
        """Set the model in `train` mode.
        """
        self.training = True
        self.featurizer_net.train()
        self.encoder_net.train()
        if self.auxiliary_net is not None:
            self.auxiliary_net.train()

    def eval(self):
        """Set the model in `eval` mode.
        """
        if self.training:
            logging.debug("Setting model to EVAL mode.")
            self.featurizer_net.eval()
            self.encoder_net.eval()
            if self.auxiliary_net is not None:
                self.auxiliary_net.eval()
        self.training = False

    @property
    def num_ctx_frames(self):
        return self._num_ctx_frames
