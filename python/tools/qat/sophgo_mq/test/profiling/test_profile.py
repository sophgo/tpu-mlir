import torch
import unittest

from sophgo_mq.prepare_by_platform import prepare_by_platform, BackendType
from sophgo_mq.convert_deploy import convert_merge_bn
from sophgo_mq.utils.state import enable_calibration, enable_quantization
from sophgo_mq.utils.profiling import profiling

from ..version import GITHUB_RES


class TestProfiling(unittest.TestCase):

    def test_case_1(self):
        # pure graph model
        dummy_input = torch.randn(1, 3, 224, 224, device='cpu')
        extra_qconfig_dict = {
            'w_observer': 'MinMaxObserver',
            'a_observer': 'EMAMinMaxObserver',
            'w_fakequantize': 'LearnableFakeQuantize',
            'a_fakequantize': 'LearnableFakeQuantize',
        }
        prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
        # First model
        model_1 = torch.hub.load(GITHUB_RES, 'resnet18', pretrained=False)
        model_1 = prepare_by_platform(model_1, BackendType.Tensorrt, prepare_custom_config_dict)
        model_1.train()
        enable_calibration(model_1)
        model_1(dummy_input)
        enable_quantization(model_1)
        model_1.eval()
        model_1 = model_1
        profiling(model_1, [torch.randn(1, 3, 224, 224) for _ in range(4)], 'standalone')
        profiling(model_1, [torch.randn(1, 3, 224, 224) for _ in range(4)], 'interaction')