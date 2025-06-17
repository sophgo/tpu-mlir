import torch
import unittest

from sophgo_mq.prepare_by_platform import prepare_by_platform, BackendType
from sophgo_mq.convert_deploy import convert_merge_bn
from sophgo_mq.utils.state import enable_calibration, enable_quantization

from ..version import GITHUB_RES


class TestMergeBN(unittest.TestCase):

    def test_case_1(self):

        def cos(a, b):
            return (a * b).sum() / torch.sqrt((a**2).sum()) / torch.sqrt((b**2).sum())

        dummy_input = torch.randn(1, 3, 224, 224, device='cpu')
        extra_qconfig_dict = {
            'w_observer': 'MinMaxObserver',
            'a_observer': 'EMAMinMaxObserver',
            'w_fakequantize': 'LearnableFakeQuantize',
            'a_fakequantize': 'LearnableFakeQuantize',
        }
        prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
        # First model
        model_1 = torch.hub.load(GITHUB_RES, 'mobilenet_v2', pretrained=False)
        model_1 = prepare_by_platform(model_1, BackendType.Tensorrt, prepare_custom_config_dict)
        model_1.train()
        enable_calibration(model_1)
        model_1(dummy_input)
        enable_quantization(model_1)
        model_1.eval()
        prev_output = model_1(dummy_input)
        # Convert model
        convert_merge_bn(model_1)
        new_output = model_1(dummy_input)
        # Test
        # merge bn import about 1e-8 mean error.
        self.assertTrue(cos(new_output, prev_output) >= 0.999)

    def test_case_2(self):

        def cos(a, b):
            a = a.flatten()
            b = b.flatten()
            ans = (a * b).sum() / torch.sqrt((a**2).sum()) / torch.sqrt((b**2).sum())
            return ans

        import torch.nn as nn

        class TestDeConv(torch.nn.Module):
            '''
            Test code from 
                https://github.com/pgtgrly/Convolution-Deconvolution-Network-Pytorch
            '''

            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels=3,
                                       out_channels=16,
                                       kernel_size=4,
                                       stride=1,
                                       padding=0)
                nn.init.xavier_uniform(self.conv1.weight)
                self.swish1 = nn.ReLU()

                self.maxpool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)

                self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
                nn.init.xavier_uniform(self.conv2.weight)
                self.swish2 = nn.ReLU()

                self.maxpool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)

                self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
                nn.init.xavier_uniform(self.conv3.weight)
                self.swish3 = nn.ReLU()

                self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3)
                nn.init.xavier_uniform(self.deconv1.weight)
                self.bn1 = nn.BatchNorm2d(32)
                self.swish4 = nn.ReLU()

                self.maxunpool1 = nn.MaxUnpool2d(kernel_size=2)

                self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5)
                nn.init.xavier_uniform(self.deconv2.weight)
                self.swish5 = nn.ReLU()

                self.maxunpool2 = nn.MaxUnpool2d(kernel_size=2)

                self.deconv3 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=4)
                nn.init.xavier_uniform(self.deconv3.weight)
                self.bn3 = nn.BatchNorm2d(3)

            def forward(self, x):
                out = self.conv1(x)
                out = self.swish1(out)
                size1 = out.size()
                out, indices1 = self.maxpool1(out)
                out = self.conv2(out)
                out = self.swish2(out)
                size2 = out.size()
                out, indices2 = self.maxpool2(out)
                out = self.conv3(out)
                out = self.swish3(out)

                out = self.deconv1(out)
                out = self.bn1(out)
                out = self.swish4(out)
                out = self.maxunpool1(out, indices2, size2)
                out = self.deconv2(out)
                out = self.swish5(out)
                out = self.maxunpool2(out, indices1, size1)
                out = self.deconv3(out)
                out = self.bn3(out)

                return (out)

        dummy_input = torch.randn(1, 3, 224, 224, device='cpu')
        extra_qconfig_dict = {
            'w_observer': 'MinMaxObserver',
            'a_observer': 'EMAMinMaxObserver',
            'w_fakequantize': 'LearnableFakeQuantize',
            'a_fakequantize': 'LearnableFakeQuantize',
        }
        prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
        # First model
        model_1 = TestDeConv()
        model_1 = prepare_by_platform(model_1, BackendType.Tensorrt, prepare_custom_config_dict)
        model_1.train()
        enable_calibration(model_1)
        model_1(dummy_input)
        enable_quantization(model_1)
        model_1.eval()
        prev_output = model_1(dummy_input)
        # Convert model
        convert_merge_bn(model_1)
        new_output = model_1(dummy_input)
        # Test
        # merge bn import about 1e-8 mean error.
        self.assertTrue(cos(new_output, prev_output) >= 0.999)
