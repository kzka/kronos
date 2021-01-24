import torch.nn as nn
import torchvision

from kronos.models.base import BaseFeaturizer


class ResNetFeaturizer(BaseFeaturizer):
    """A ResNet-based feature extractor.
    """

    RESNETS = ["resnet18", "resnet34", "resnet50", "resnet101"]
    RESNET_TO_CHANNELS = {
        "resnet18": 256,
        "resnet34": 256,
        "resnet50": 1024,
        "resnet101": 1024,
    }

    def __init__(
        self,
        model_type="resnet18",
        out_layer_idx=7,
        pretrained=True,
        layers_train="bn_only",
        bn_use_running_stats=False,
    ):
        """Constructor.

        Args:
            model_type (str): The model type to use. Can be one of
                `['resnet18', 'resnet34', 'resnet50', 'resnet101']`.
            out_layer_idx (int): The index of the layer to use as
                output.
        """
        if model_type not in ResNetFeaturizer.RESNETS:
            raise ValueError(
                "[!] {} is not a supported resnet model.".format(model_type)
            )

        self._model_type = model_type
        self._out_layer_idx = out_layer_idx

        super().__init__(pretrained, layers_train, bn_use_running_stats)

    def _build_featurizer(self):
        resnet = getattr(torchvision.models, self._model_type)(
            pretrained=self._pretrained
        )
        layers_ = list(resnet.children())
        assert self._out_layer_idx < len(
            layers_
        ), "[!] Output layer index exceeds total layers."
        layers_ = layers_[: self._out_layer_idx]
        return nn.Sequential(*layers_)


class InceptionFeaturizer(BaseFeaturizer):
    """An InceptionV3-based feature extractor.
    """

    LAYERS = [
        "Conv2d_1a_3x3",
        "Conv2d_2a_3x3",
        "Conv2d_2b_3x3",
        "MaxPool_3a_3x3",
        "Conv2d_3b_1x1",
        "Conv2d_4a_3x3",
        "MaxPool_5a_3x3",
        "Mixed_5b",
        "Mixed_5c",
        "Mixed_5d",
        "Mixed_6a",
        "Mixed_6b",
        "Mixed_6c",
        "Mixed_6d",
        "Mixed_6e",
        "Mixed_7a",
        "Mixed_7b",
        "Mixed_7c",
    ]

    LAYERS_TO_CHANNELS = {
        "Mixed_5d": 288,
    }

    def __init__(
        self,
        out_layer_name="Mixed_5d",
        pretrained=True,
        layers_train="frozen",
        bn_use_running_stats=False,
    ):
        """Constructor.

        Args:
            out_layer_name (str): Which layer of the inception model
                to use as the output. The authors of TCN use `Mixed_5d`
                as the output.
        """
        assert (
            out_layer_name in InceptionFeaturizer.LAYERS
        ), f"{out_layer_name} is not supported."

        self._out_layer_name = out_layer_name

        super().__init__(pretrained, layers_train, bn_use_running_stats)

    def _build_featurizer(self):
        inception = torchvision.models.inception_v3(
            pretrained=self._pretrained, aux_logits=False, init_weights=False,
        )
        layers_, flag = [], False
        for name, module in inception.named_modules():
            if not name or "." in name:
                continue
            if self._out_layer_name in name:
                flag = True
            layers_.append(module)
            if flag:
                break
        return nn.Sequential(*layers_)
