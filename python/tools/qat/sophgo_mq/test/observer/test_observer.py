import torch
import unittest

from sophgo_mq.prepare_by_platform import prepare_by_platform, BackendType
from sophgo_mq.convert_deploy import convert_deploy
from sophgo_mq.utils.state import enable_calibration, enable_quantization

from ..version import GITHUB_RES


class TestObserver(unittest.TestCase):

    def test_quantile_observer(self):
        model_to_quantize = torch.hub.load(GITHUB_RES, 'resnet18', pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224, device='cpu')
        model_to_quantize.train()
        extra_qconfig_dict = {
            'w_observer': 'MinMaxObserver',
            'a_observer': 'EMAQuantileObserver',
            'w_fakequantize': 'FixedFakeQuantize',
            'a_fakequantize': 'FixedFakeQuantize',
        }
        prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
        model_prepared = prepare_by_platform(model_to_quantize, BackendType.Tensorrt,
                                             prepare_custom_config_dict)
        enable_calibration(model_prepared)
        model_prepared(dummy_input)
        enable_quantization(model_prepared)
        loss = model_prepared(dummy_input).sum()
        loss.backward()

    def test_ema_observer(self):
        model_to_quantize = torch.hub.load(GITHUB_RES, 'resnet18', pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224, device='cpu')
        model_to_quantize.train()
        extra_qconfig_dict = {
            'w_observer': 'MinMaxObserver',
            'a_observer': 'EMAMinMaxObserver',
            'w_fakequantize': 'FixedFakeQuantize',
            'a_fakequantize': 'FixedFakeQuantize',
        }
        prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
        model_prepared = prepare_by_platform(model_to_quantize, BackendType.Tensorrt,
                                             prepare_custom_config_dict)
        enable_calibration(model_prepared)
        model_prepared(dummy_input)
        enable_quantization(model_prepared)
        loss = model_prepared(dummy_input).sum()
        loss.backward()

    def test_minmax_observer(self):
        model_to_quantize = torch.hub.load(GITHUB_RES, 'resnet18', pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224, device='cpu')
        model_to_quantize.train()
        extra_qconfig_dict = {
            'w_observer': 'MinMaxObserver',
            'a_observer': 'MinMaxObserver',
            'w_fakequantize': 'FixedFakeQuantize',
            'a_fakequantize': 'FixedFakeQuantize',
        }
        prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
        model_prepared = prepare_by_platform(model_to_quantize, BackendType.Tensorrt,
                                             prepare_custom_config_dict)
        enable_calibration(model_prepared)
        model_prepared(dummy_input)
        enable_quantization(model_prepared)
        loss = model_prepared(dummy_input).sum()
        loss.backward()

    def test_lsq_observer(self):
        model_to_quantize = torch.hub.load(GITHUB_RES, 'resnet18', pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224, device='cpu')
        model_to_quantize.train()
        extra_qconfig_dict = {
            'w_observer': 'LSQObserver',
            'a_observer': 'LSQObserver',
            'w_fakequantize': 'FixedFakeQuantize',
            'a_fakequantize': 'FixedFakeQuantize',
        }
        prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
        model_prepared = prepare_by_platform(model_to_quantize, BackendType.Tensorrt,
                                             prepare_custom_config_dict)
        enable_calibration(model_prepared)
        model_prepared(dummy_input)
        enable_quantization(model_prepared)
        loss = model_prepared(dummy_input).sum()
        loss.backward()

    def test_clip_std_observer(self):
        model_to_quantize = torch.hub.load(GITHUB_RES, 'resnet18', pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224, device='cpu')
        model_to_quantize.train()
        extra_qconfig_dict = {
            'w_observer': 'ClipStdObserver',
            'a_observer': 'ClipStdObserver',
            'w_fakequantize': 'FixedFakeQuantize',
            'a_fakequantize': 'FixedFakeQuantize',
        }
        prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
        model_prepared = prepare_by_platform(model_to_quantize, BackendType.Tensorrt,
                                             prepare_custom_config_dict)
        enable_calibration(model_prepared)
        model_prepared(dummy_input)
        enable_quantization(model_prepared)
        loss = model_prepared(dummy_input).sum()
        loss.backward()
