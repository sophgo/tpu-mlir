import torch
import unittest

from sophgo_mq.prepare_by_platform import prepare_by_platform, BackendType
from sophgo_mq.convert_deploy import convert_deploy
from sophgo_mq.utils.state import enable_calibration, enable_quantization
from sophgo_mq.utils.logger import logger

from ..version import GITHUB_RES


class TestQuantizeModel(unittest.TestCase):

    def test_model_ppl(self):
        test_model_list = [
            'alexnet', 'deeplabv3_resnet50', 'densenet121', 'fcn_resnet50', 'mnasnet0_5',
            'mobilenet_v2', 'resnet18', 'resnext50_32x4d', 'shufflenet_v2_x0_5', 'squeezenet1_0',
            'vgg11', 'vgg11_bn', 'wide_resnet50_2', 'regnet_x_400mf'
        ]
        entrypoints = torch.hub.list(GITHUB_RES, force_reload=False)
        for entrypoint in entrypoints:
            if entrypoint not in test_model_list:
                continue
            logger.info(f'testing {entrypoint}')
            if 'deeplab' in entrypoint or 'fcn' in entrypoint:
                model_to_quantize = torch.hub.load(GITHUB_RES,
                                                   entrypoint,
                                                   pretrained=False,
                                                   pretrained_backbone=False)
            else:
                model_to_quantize = torch.hub.load(GITHUB_RES, entrypoint, pretrained=False)
            dummy_input = torch.randn(2, 3, 224, 224, device='cpu')
            model_to_quantize.train()
            model_prepared = prepare_by_platform(model_to_quantize, BackendType.PPLW8A16)
            enable_calibration(model_prepared)
            model_prepared(dummy_input)
            enable_quantization(model_prepared)
            output = model_prepared(dummy_input)
            try:
                loss = output.sum()
            except AttributeError:
                loss = output['out'].sum()
            loss.backward()
            model_prepared.eval()
            convert_deploy(model_prepared,
                           BackendType.PPLW8A16, {'x': [1, 3, 224, 224]},
                           model_name='{}.onnx'.format(entrypoint))
