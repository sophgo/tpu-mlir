import torch
import torch.nn as nn
import unittest

from sophgo_mq.prepare_by_platform import prepare_by_platform, BackendType
from sophgo_mq.convert_deploy import convert_deploy
from sophgo_mq.utils.state import enable_calibration, enable_quantization
from sophgo_mq.utils.logger import logger


class TestQuantizeDetModel(unittest.TestCase):

    def test_model1(self):

        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 3, 3, 3)

            def forward(self, input):
                x = input['x']
                y = self.conv1(x)
                input.update({'y': {'z': y}})
                return input

        model_to_quantize = Model()
        dummy_input = torch.randn(1, 3, 224, 224, device='cpu')
        model_to_quantize.train()
        model_prepared = prepare_by_platform(model_to_quantize, BackendType.PPLW8A16)
        print(model_prepared.code)
        enable_calibration(model_prepared)
        model_prepared({'x': dummy_input})
        enable_quantization(model_prepared)
        output = model_prepared({'x': dummy_input})
        loss = output['y']['z'].sum()
        loss.backward()
        model_prepared.eval()
        convert_deploy(model_prepared,
                       BackendType.PPLW8A16,
                       dummy_input={'x': dummy_input},
                       model_name='test_model1')

    def test_model2(self):

        class Backbone(nn.Module):

            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 3, 3, 3)
                self.bn1 = nn.BatchNorm2d(3)
                self.relu = nn.ReLU()
                self.conv2 = nn.Conv2d(3, 3, 3, 3)
                self.bn2 = nn.BatchNorm2d(3)

            def forward(self, input):
                x = input['x']
                y = self.relu(self.bn1(self.conv1(x)))
                y = self.bn2(self.conv2(y))
                input.update({'y': y})
                return input

        class Head(nn.Module):

            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 3, 3, 3)

            def forward(self, input):
                y = input['y']
                z = self.conv1(y)
                input.update({'z': z})
                return input

        class TwoStageModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.backbone = Backbone()
                self.head = Head()

            def forward(self, input):
                input.update(self.backbone(input))
                input.update(self.head(input))
                return input

        model_to_quantize = TwoStageModel()
        dummy_input = torch.randn(1, 3, 224, 224, device='cpu')
        model_to_quantize.train()
        model_prepared = prepare_by_platform(model_to_quantize, BackendType.SNPE)
        print(model_prepared.code)
        enable_calibration(model_prepared)
        model_prepared({'x': dummy_input})
        enable_quantization(model_prepared)
        output = model_prepared({'x': dummy_input})
        loss = output['z'].sum()
        loss.backward()
        model_prepared.eval()
        convert_deploy(model_prepared,
                       BackendType.SNPE,
                       dummy_input={'x': dummy_input},
                       model_name='twostage')
