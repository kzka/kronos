import torch.nn as nn

from kronos.models.base import BaseTemporalModel
from kronos.models import featurizers, encoders


class SALClassifier(nn.Module):
    """The classification head for Shuffle and Learn.
    """

    def __init__(self, model_config):
        super().__init__()

        self.drop_prob = model_config.CLASSIFIER.FC_DROPOUT_RATE
        self.num_layers = model_config.CLASSIFIER.NUM_FC_LAYERS

        # figure out fc layer dimensions
        hidden = model_config.CLASSIFIER.HIDDEN_DIMS
        if not isinstance(hidden, list):
            hidden = [hidden]
        assert len(hidden) == self.num_layers - 1
        hidden = [3 * model_config.EMBEDDER.EMBEDDING_SIZE] + hidden + [2]

        # build classifier
        layers_ = []
        for i in range(self.num_layers):
            if self.drop_prob > 0.0:
                layers_.append(nn.Dropout2d(self.drop_prob))
            layers_.append(nn.Linear(hidden[i], hidden[i + 1]))
            if i < self.num_layers - 1:
                layers_.append(nn.ReLU())
        self.classifier = nn.Sequential(*layers_)

        # build param to module dict
        self.param_to_module = {}
        for m in self.modules():
            for p in m.parameters(recurse=False):
                self.param_to_module[p] = type(m).__name__

    def forward(self, embs):
        return self.classifier(embs)


class SALNet(BaseTemporalModel):
    """A neural net for Shuffle and Learn [1].

    References:
        [1]: Shuffle and Learn: Unsupervised Learning using Temporal
        Order Verification, https://arxiv.org/abs/1603.08561
    """

    def __init__(self, num_ctx_frames, model_config):
        super().__init__(num_ctx_frames, model_config)

    def _init_featurizer(self):
        return featurizers.ResNetFeaturizer(
            model_type=self._model_config.FEATURIZER.NETWORK_TYPE,
            pretrained=self._model_config.FEATURIZER.PRETRAINED,
            layers_train=self._model_config.FEATURIZER.TRAIN_BASE,
            bn_use_running_stats=self._model_config.FEATURIZER.BN_USE_RUNNING_STATS,
        )

    def _init_encoder(self):
        in_channels = featurizers.ResNetFeaturizer.RESNET_TO_CHANNELS[
            self._model_config.FEATURIZER.NETWORK_TYPE
        ]
        return encoders.ConvFCEncoder(
            in_channels,
            embedding_size=self._model_config.EMBEDDER.EMBEDDING_SIZE,
            num_conv_layers=self._model_config.EMBEDDER.NUM_CONV_LAYERS,
            num_fc_layers=self._model_config.EMBEDDER.NUM_FC_LAYERS,
            widen_factor=self._model_config.EMBEDDER.WIDEN_FACTOR,
            dropout_spatial=self._model_config.EMBEDDER.FEATURIZER_DROPOUT_SPATIAL,
            spatial_dropout_rate=self._model_config.EMBEDDER.FEATURIZER_DROPOUT_RATE,
            fc_dropout_rate=self._model_config.EMBEDDER.FC_DROPOUT_RATE,
            flatten_method=self._model_config.EMBEDDER.FLATTEN_METHOD,
        )

    def _init_auxiliary_net(self):
        return SALClassifier(self._model_config)

    def param_groups(self):
        params = {}
        params["featurizer"] = {"free": self.featurizer_net.parameters()}
        params["encoder"] = {"reg": [], "free": []}
        for k, v in self.encoder_net.param_to_module.items():
            if any(rl in v.lower() for rl in ["conv", "linear"]):
                params["encoder"]["reg"].append(k)
            else:
                params["encoder"]["free"].append(k)
        params["auxiliary"] = {"reg": [], "free": []}
        for k, v in self.auxiliary_net.param_to_module.items():
            if any(rl in v.lower() for rl in ["conv", "linear"]):
                params["auxiliary"]["reg"].append(k)
            else:
                params["auxiliary"]["free"].append(k)
        return params
