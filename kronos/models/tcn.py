from kronos.models.base import BaseTemporalModel
from kronos.models import featurizers, encoders


class TCNNet(BaseTemporalModel):
    """A neural net for Time Contrastive Networks [1].

    References:
        [1]: Time-Contrastive Networks,
        https://arxiv.org/abs/1704.06888
    """

    def __init__(self, num_ctx_frames, model_config):
        if "inception" in model_config.FEATURIZER.NETWORK_TYPE:
            self._use_resnet = False
        else:
            self._use_resnet = True

        super().__init__(num_ctx_frames, model_config)

    def _init_featurizer(self):
        if self._use_resnet:
            return featurizers.ResNetFeaturizer(
                model_type=self._model_config.FEATURIZER.NETWORK_TYPE,
                out_layer_idx=self._model_config.FEATURIZER.OUT_LAYER_IDX,
                pretrained=self._model_config.FEATURIZER.PRETRAINED,
                layers_train=self._model_config.FEATURIZER.TRAIN_BASE,
                bn_use_running_stats=self._model_config.FEATURIZER.BN_USE_RUNNING_STATS,
            )
        else:
            return featurizers.InceptionFeaturizer(
                out_layer_name=self._model_config.FEATURIZER.OUT_LAYER_NAME,
                pretrained=self._model_config.FEATURIZER.PRETRAINED,
                layers_train=self._model_config.FEATURIZER.TRAIN_BASE,
                bn_use_running_stats=self._model_config.FEATURIZER.BN_USE_RUNNING_STATS,
            )

    def _init_encoder(self):
        if self._use_resnet:
            in_channels = featurizers.ResNetFeaturizer.RESNET_TO_CHANNELS[
                self._model_config.FEATURIZER.NETWORK_TYPE
            ]
        else:
            in_channels = featurizers.InceptionFeaturizer.LAYERS_TO_CHANNELS[
                self._model_config.FEATURIZER.OUT_LAYER_NAME
            ]
        return encoders.Conv2DFullyConnectedEncoder(
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
        # in the TCN paper, the authors add kernel and bias
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

    def _sanity_check_input(self, frames):
        if not self._use_resnet:
            assert frames.shape[3:] == (
                299,
                299,
            ), "InceptionV3 needs H = W = 299."
