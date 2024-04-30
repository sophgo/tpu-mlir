import torch
import torch.nn as nn
import math
# TODO: support SyncBatchNorm for mutilGPU
BN = None

__all__ = ['resnet18', 'resnet26', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet_custom']

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_path=0.):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BN(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BN(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = DropPath(drop_path) if drop_path > 0. else None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.drop_path is not None:
            out = self.drop_path(out)

        out += residual
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_path=0.):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BN(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BN(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BN(planes * 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = DropPath(drop_path) if drop_path > 0. else None

        bypass_bn_weight_list.append(self.bn3.weight)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.drop_path is not None:
            out = self.drop_path(out)

        out += residual
        out = self.relu3(out)

        return out


class ResNet(nn.Module):
    """Redidual Networks class, based on
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_
    """
    def __init__(self,
                 block,
                 layers,
                 inplanes=64,
                 num_classes=1000,
                 deep_stem=False,
                 avg_down=False,
                 bypass_last_bn=False,
                 bn=None,
                 nnie_type=False,
                 scale=1.0,
                 final_dpr=0.):
        r"""
        Arguments:

        - layers (:obj:`list` of 4 ints): how many layers in each stage
        - num_classes (:obj:`int`): number of classification classes
        - deep_stem (:obj:`bool`): whether to use deep_stem as the first conv
        - avg_down (:obj:`bool`): whether to use avg_down when spatial downsample
        - bypass_last_bn (:obj:`bool`): whether use bypass_last_bn
        - bn (:obj:`dict`): definition of batchnorm
        """

        super(ResNet, self).__init__()

        global BN, bypass_bn_weight_list

        BN = torch.nn.BatchNorm2d
        bypass_bn_weight_list = []

        self.inplanes = int(inplanes * scale)
        self.deep_stem = deep_stem
        self.avg_down = avg_down

        self.prob_now = 0.0
        self.prob_delta = final_dpr - self.prob_now
        # self.prob_step = self.prob_delta / (sum(layers) - 1)
        self.prob_step = self.prob_delta / (sum(layers))

        if self.deep_stem:
            self.conv1 = nn.Sequential(
                        nn.Conv2d(3, self.inplanes // 2, kernel_size=3, stride=2, padding=1, bias=False),
                        BN(self.inplanes // 2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(self.inplanes // 2, self.inplanes // 2, kernel_size=3,
                                  stride=1, padding=1, bias=False),
                        BN(self.inplanes // 2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(self.inplanes // 2, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                    )
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BN(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        if nnie_type:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(64 * scale), layers[0])
        fc_in_chan = 64
        self.layer2 = None
        if layers[1] > 0:
            self.layer2 = self._make_layer(block, int(128 * scale), layers[1], stride=2)
            fc_in_chan = 128
        self.layer3 = None
        if layers[2] > 0:
            self.layer3 = self._make_layer(block, int(256 * scale), layers[2], stride=2)
            fc_in_chan = 256
        self.layer4 = None
        if layers[3] > 0:
            self.layer4 = self._make_layer(block, int(512 * scale), layers[3], stride=2)
            fc_in_chan = 512

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(fc_in_chan * scale) * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0/float(n))
                m.bias.data.zero_()

        if bypass_last_bn:
            for param in bypass_bn_weight_list:
                param.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, avg_down=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=1, bias=False),
                    BN(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    BN(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.prob_now))
        self.prob_now = self.prob_now + self.prob_step
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, drop_path=self.prob_now))
            self.prob_now = self.prob_now + self.prob_step

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if self.layer2:
            x = self.layer2(x)
        if self.layer3:
            x = self.layer3(x)
        if self.layer4:
            x = self.layer4(x)

        x = self.avgpool(x)

        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18(**kwargs):
    """
    Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet26(**kwargs):
    """
    Constructs a ResNet-26 model.
    """
    model = ResNet(Bottleneck, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """
    Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """
    Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """
    Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """
    Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet_custom(**kwargs):
    """
    Constructs a custom ResNet model with custom block and depth.
    """
    assert 'block' in kwargs and 'layers' in kwargs, 'Require block and layers'
    block = kwargs.pop('block')
    layers = kwargs.pop('layers')
    if block == 'basic':
        block = BasicBlock
    elif block == 'bottleneck':
        block = Bottleneck
    else:
        raise Exception('Unsupported block type.')
    model = ResNet(block, layers, **kwargs)
    return model
