import paddle
import paddle.nn as nn

from typing import Union, List, Dict, Any, cast
from ppad.utils import load_pretrained_params
from ppad.modeling.registry import BACKBONES
from ppad.modeling.param_init import xavier_uniform_


__all__ = [
    'VGG', 'VGG16'
]


class VGG(nn.Layer):

    def __init__(
        self,
        features: nn.Layer,
        num_classes: int = 1000,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2D((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        pass


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Layer] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2D(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2D(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2D(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: str, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained is not None:
        load_pretrained_params(model, pretrained)
    return model


@BACKBONES.register()
class VGG16(nn.Layer):
    def __init__(self, pretrained):
        super(VGG16, self).__init__()

        features = list(_vgg('vgg16', 'D', False, pretrained, True).features)

        self.features = nn.LayerList(features)
        self.features.eval()
        self.output = []

    def forward(self, x):
        output = []
        for i in range(31):
            x = self.features[i](x)
            if i == 1 or i == 4 or i == 6 or i == 9 or i == 11 or i == 13 or i == 16 or i == 18 or i == 20 or i == 23 or i == 25 or i == 27 or i == 30:
                output.append(x)
        return output


@BACKBONES.register()
class KDADStudentVGG(nn.Layer):
    '''
    VGG model
    '''

    def __init__(self):
        super(KDADStudentVGG, self).__init__()
        cfg = [16, 16, 'M', 16, 128, 'M', 16, 16, 256, 'M', 16, 16, 512, 'M', 16, 16, 512, 'M']
        self.features = self.make_layers(cfg, use_bias=False, batch_norm=True)

        # placeholder for the gradients
        self.gradients = None
        self.activation = None

    def make_layers(self, cfg, use_bias, batch_norm=False):
        layers = []
        in_channels = 3
        outputs = []
        for i in range(len(cfg)):
            if cfg[i] == 'O':
                outputs.append(nn.Sequential(*layers))
            elif cfg[i] == 'M':
                layers += [nn.MaxPool2D(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2D(in_channels, cfg[i], kernel_size=3, padding=1, bias_attr=use_bias)
                xavier_uniform_(conv2d.weight)
                if batch_norm and cfg[i + 1] != 'M':
                    layers += [conv2d, nn.BatchNorm2D(cfg[i]), nn.ReLU()]
                else:
                    layers += [conv2d, nn.ReLU()]
                in_channels = cfg[i]
        return nn.Sequential(*layers)

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, target_layer=11, export=False):
        result = []
        for i in range(len(nn.LayerList(self.features))):
            x = self.features[i](x)
            if i == target_layer:
                self.activation = x
                if not export:
                    h = x.register_hook(self.activations_hook)
            if i == 2 or i == 5 or i == 8 or i == 11 or i == 14 or i == 17 or i == 20 or i == 23 or i == 26 or i == 29 or i == 32 or i == 35 or i == 38:
                result.append(x)

        return result

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.activation