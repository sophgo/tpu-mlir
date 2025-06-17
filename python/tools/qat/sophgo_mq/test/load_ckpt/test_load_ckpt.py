import torch
import unittest

from sophgo_mq.prepare_by_platform import prepare_by_platform, BackendType
from sophgo_mq.convert_deploy import convert_deploy
from sophgo_mq.utils.state import enable_calibration, enable_quantization

from ..version import GITHUB_RES


class TestLoadCheckPoint(unittest.TestCase):

    def test_case_1(self):
        dummy_input = torch.randn(2, 3, 224, 224, device='cpu')
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
        prev_output = model_1(dummy_input)
        torch.save(model_1.state_dict(), 'saved_model.ckpt')
        # Second model
        model_2 = torch.hub.load(GITHUB_RES, 'resnet18', pretrained=False)
        model_2 = prepare_by_platform(model_2, BackendType.Tensorrt, prepare_custom_config_dict)
        state_dict = torch.load('saved_model.ckpt')
        model_2.load_state_dict(state_dict)
        enable_quantization(model_2)
        model_2.eval()
        new_output = model_2(dummy_input)
        # Test
        self.assertTrue((new_output - prev_output).abs().sum() < 1e-9)
