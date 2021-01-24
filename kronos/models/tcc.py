from kronos.models.base import BaseTemporalModel
from kronos.models import featurizers, encoders


class TCCNet(BaseTemporalModel):
    """A neural net for Temporal Cycle Consistency Learning [1].

    References:
        [1]: Temporal Cycle Consistency Learning,
        https://arxiv.org/abs/1904.07846
    """

    def __init__(self, num_ctx_frames, model_config):
        super().__init__(num_ctx_frames, model_config)

    def _init_featurizer(self):
        return featurizers.ResNetFeaturizer(
            model_type=self._model_config.FEATURIZER.NETWORK_TYPE,
            out_layer_idx=self._model_config.FEATURIZER.OUT_LAYER_IDX,
            pretrained=self._model_config.FEATURIZER.PRETRAINED,
            layers_train=self._model_config.FEATURIZER.TRAIN_BASE,
            bn_use_running_stats=self._model_config.FEATURIZER.BN_USE_RUNNING_STATS,
        )

    def _init_encoder(self):
        in_channels = featurizers.ResNetFeaturizer.RESNET_TO_CHANNELS[
            self._model_config.FEATURIZER.NETWORK_TYPE
        ]
        return encoders.Conv3DFullyConnectedEncoder(
            in_channels,
            num_channels=self._model_config.EMBEDDER.NUM_CHANNELS,
            embedding_size=self._model_config.EMBEDDER.EMBEDDING_SIZE,
            num_conv_layers=self._model_config.EMBEDDER.NUM_CONV_LAYERS,
            num_fc_layers=self._model_config.EMBEDDER.NUM_FC_LAYERS,
            widen_factor=self._model_config.EMBEDDER.WIDEN_FACTOR,
            dropout_spatial=self._model_config.EMBEDDER.FEATURIZER_DROPOUT_SPATIAL,
            spatial_dropout_rate=self._model_config.EMBEDDER.FEATURIZER_DROPOUT_RATE,
            fc_dropout_rate=self._model_config.EMBEDDER.FC_DROPOUT_RATE,
            flatten_method=self._model_config.EMBEDDER.FLATTEN_METHOD,
        )

    def param_groups(self):
        # in the TCC paper, the authors add kernel and bias
        # regularization to the convolutional and fc layers
        # of the embedding network only.
        params = {}
        params["featurizer"] = {"free": self.featurizer_net.parameters()}
        params["encoder"] = {"reg": [], "free": []}
        for k, v in self.encoder_net.param_to_module.items():
            if any(rl in v.lower() for rl in ["conv", "linear"]):
                params["encoder"]["reg"].append(k)
            else:
                params["encoder"]["free"].append(k)
        return params
