import torch
from torch.nn.parameter import Parameter
from torch.quantization.observer import MovingAverageMinMaxObserver

from sophgo_mq.fake_quantize.quantize_base import QuantizeBase, _version_under_1100 
from sophgo_mq.utils.hook import PerChannelLoadHook
from sophgo_mq.fake_quantize import global_var

import transformers
import math
import time
import numpy as np
import ipdb
from scipy.stats import norm

DEBUG = False 

NF4 = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]

FP4_BNB = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -0.0625, 0, 0.0625, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_E2M1 = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.0625, 0, 0.0625, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

# the order is the same as float list, bit value range is [-7, 7]
# 1111 = -1, 1110 = -2, 1101= -3, ...

NF4_BIT = [7, 1, 2, 3, 4, 5, 6, 0, -8, -7, -6, -5, -4, -3, -2, -1]
FP4_BNB_BIT = [-5, -6, -3, -4, -1, -2, -7, 0, 1, 6, 7, 4, 5, 2, 3]
FP4_E2M1_BIT = [-1, -2, -3, -4, -5, -6, -7, 0, 1, 2, 3, 4, 5, 6, 7]

FLOAT_MAPPING = {"nf4": NF4, "fp4": FP4_BNB, "fp4_e2m1_bnb": FP4_BNB, "fp4_e2m1": FP4_E2M1}
INT_MAPPING = {"nf4": NF4_BIT, "fp4": FP4_BNB_BIT, "fp4_e2m1_bnb": FP4_BNB_BIT, "fp4_e2m1": FP4_E2M1_BIT}
    
def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class GPTQFP4FakeQuantize(QuantizeBase):
    '''
    This is gptq method sophgo_mq version.
    '''

    def __init__(self, observer, **observer_kwargs):
        super(GPTQFP4FakeQuantize, self).__init__(observer, **observer_kwargs)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.is_gptq_valid = False
        self.layer_name = None
        self.rows = None
        self.columns = None
        self.H = torch.tensor([0], dtype=torch.float)
        self.nsamples = 0
        self.layer_module = None
        self.dev = torch.device('cpu')
        # self.register_buffer('H', torch.tensor([0], dtype=torch.float))
        self.is_gptq_done = False
        self.is_add_batch = False
        self.actorder = False
        self.data_type="nf4"
        self.quantile=1.0
        self.return_int=False

    def forward(self, X):
        # print('This FakeQuantizer is "', self.layer_name)
        self.dev = X.device
        x_ori = X
        # print("GPTQFakeQuantize - forward")
        if self.observer_enabled[0] == 1:
            if (self.H.shape == torch.Size([1])):
                W = X.data.clone()
                if isinstance(self.layer_module, torch.nn.Conv2d):
                    W = W.flatten(1)
                if isinstance(self.layer_module, transformers.Conv1D):
                    W = W.t()
                self.rows = W.shape[0]
                self.columns = W.shape[1]
                self.H = torch.zeros((W.shape[1], W.shape[1]), device=self.dev)

            if (self.is_gptq_valid):
                if (self.layer_name+'.inp' in global_var.get_var().keys()):
                    self.input = global_var.get_value(self.layer_name+'.inp')
                    self.add_batch()

            self.activation_post_process(X.detach())
            # All is per layer
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            _scale = _scale.to(self.scale.device)
            _zero_point = _zero_point.to(self.zero_point.device)

            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)

            self.scale.data.copy_(_scale)
            self.zero_point.data.copy_(_zero_point.float())
        if self.fake_quant_enabled[0] == 1:
            # # Use GPTQ
            if (self.is_gptq_valid):
                if (not self.is_gptq_done):
                    with torch.no_grad():
                        self.input = global_var.get_value(self.layer_name+'.inp')
                        self.output = global_var.get_value(self.layer_name+'.out')
                        if (self.input.device != self.dev or self.output.device != self.dev):
                            self.input = self.input.to(self.dev)
                            self.output = self.output.to(self.dev)
                        self.weight = X
                        # self.add_batch()
                        #ipdb.set_trace()
                        X = self.fasterquant(X)
                        #X1= self.fasterquant1(X)
                else:
                    return X
            # # Use FixedFakeQuantize per Tensor
            else:
                X = torch.fake_quantize_per_tensor_affine(
                    X, self.scale.item(), int(self.zero_point.item()),
                    self.quant_min, self.quant_max)
            # X = torch.fake_quantize_per_tensor_affine(
            #     X, self.scale.item(), int(self.zero_point.item()),
            #     self.quant_min, self.quant_max)
        return X
    
    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={}, ' \
               'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
               'scale={}, zero_point={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   self.quant_min, self.quant_max,
                   self.dtype, self.qscheme, self.ch_axis, self.scale if self.ch_axis == -1 else 'List[%s]' % str(self.scale.shape),
                   self.zero_point if self.ch_axis == -1 else 'List')
    def FP4quant(self,X,W):
        assert self.data_type in FLOAT_MAPPING, "unexpected data type."
        allow_data = FLOAT_MAPPING[self.data_type]                 #float类型
        allow_data_bit = INT_MAPPING[self.data_type]               #int类型
        _scale = W.abs().max(1)[0] * self.quantile/ max(allow_data)
        #_scale.unsqueeze_(dim=-1)
        X = X/_scale
        if self.data_type.lower=="nf4":
            cdf_values = [norm.cdf(x) for x in allow_data]
            intermediate_cdf_values = [(cdf_values[i] + cdf_values[i+1]) / 2 for i in range(len(allow_data) - 1)]
            mid_data = norm.ppf(intermediate_cdf_values)
        else:
            mid_data = [(allow_data[i] + allow_data[i + 1]) / 2 for i in range(len(allow_data) - 1)]
        #mid_data = [(allow_data[i] + allow_data[i + 1]) / 2 for i in range(len(allow_data) - 1)]
        q_X= torch.zeros_like(X)
        for i in range(len(allow_data)):
            data = allow_data_bit[i] if self.return_int else allow_data[i]
            if i == 0:
                q_X += torch.where(X <= mid_data[i], data, 0)
            elif i == len(allow_data) - 1:
                q_X += torch.where(X > mid_data[i - 1], data, 0)
            else:
                q_X += torch.where((mid_data[i - 1] < X) & (X <= mid_data[i]), data, 0)
        # if self.return_int:
        #     return q_X.type(torch.int8), _scale.type(torch.float), None
        X=q_X * _scale
        return X
    def add_batch(self):
        inp = self.input
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if isinstance(self.layer_module, torch.nn.Linear) or isinstance(self.layer_module, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer_module, torch.nn.Conv2d):
            unfold = torch.nn.Unfold(
                self.layer_module.kernel_size,
                dilation=self.layer_module.dilation,
                padding=self.layer_module.padding,
                stride=self.layer_module.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        self.H *= self.nsamples / (self.nsamples + tmp) # always zero ?
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        inp_inpt = inp.matmul(inp.t())
        self.H += inp_inpt
        # inp_inpt.to(torch.device('cpu'))
        # torch.cuda.empty_cache()
        del inp_inpt
        # print(self.H.size())

    def fasterquant(self, X, blocksize=128, percdamp=.01, groupsize=-1, actorder=False):
        # W = self.layer.weight.data.clone()
        # if isinstance(self.layer,torch.nn.Conv2d):
        # W = self.weight.data.clone()
        W = X.data.clone()

        if isinstance(self.layer_module, torch.nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer_module, transformers.Conv1D):
            W = W.t()
        W = W.float()

        # # Calculate by Observer
        # # if not self.quantizer.ready():
        # #     self.quantizer.find_params(W, weight=True)

        H = self.H
        # # del self.H

        # dead = torch.diag(H) == 0
        dead = []
        h, w = H.shape
        H_diag = torch.zeros(w)
        for i in range(h):
            if (H[i, i] == 0):
                dead.append(i)
            H_diag[i] = H[i, i]

        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        
        damp = percdamp * torch.mean(H_diag)
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp # 使 H 转变为全正数，保证正定

        H_c = H.clone()
        H_cpu = H_c.cpu()
        del H_c

        # H1 = torch.linalg.cholesky(H)
        # H2 = torch.cholesky_inverse(H1)
        # H3 = torch.linalg.cholesky(H2, upper=True)
        # Hinv = H

        H_cpu = np.linalg.inv(H_cpu)
        H_cpu = np.linalg.cholesky(H_cpu)
        Hinv = torch.tensor(H_cpu, device=self.dev)

        for i1 in range(0, self.columns, blocksize): # 以步长 blocksize (128) 循环到 columns
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # Find by observer
                # if groupsize != -1:
                #     if (i1 + i) % groupsize == 0:
                #         self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)

                # q = quantize(
                #     w, self.scale, self.zero_point, self.quant_max
                # ).flatten()
                #ipdb.set_trace()
                q = self.FP4quant(w,W)
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])


        torch.cuda.synchronize()
        # print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer_module, transformers.Conv1D):
            Q = Q.t()
        # self.weight.data = Q.reshape(self.weight.shape).to(self.weight.data.dtype)
        W = Q.reshape(self.weight.shape).to(self.weight.data.dtype) # fixed fakequantize 并没有重置 parameter weight


        # if DEBUG:
        #     print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        # self.layer_module.weight.data = W
        # if ('Conv2d' in self.layer_type):
            # print(torch.sum((self.layer_module(self.input) - self.layer_out) ** 2))
        # if ('Linear' in self.layer_type):
        # print(torch.sum((self.layer_module(self.input) - self.output) ** 2))
        # del self.layer_module
        
        return W
    
    def fasterquant1(self, X, blocksize=128, percdamp=.01, groupsize=-1, actorder=False):
        # W = self.layer.weight.data.clone()
        # if isinstance(self.layer,torch.nn.Conv2d):
        # W = self.weight.data.clone()
        W = X.data.clone()

        if isinstance(self.layer_module, torch.nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer_module, transformers.Conv1D):
            W = W.t()
        W = W.float()

        # # Calculate by Observer
        # # if not self.quantizer.ready():
        # #     self.quantizer.find_params(W, weight=True)

        H = self.H
        # # del self.H

        # dead = torch.diag(H) == 0
        dead = []
        h, w = H.shape
        H_diag = torch.zeros(w)
        for i in range(h):
            if (H[i, i] == 0):
                dead.append(i)
            H_diag[i] = H[i, i]

        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        
        damp = percdamp * torch.mean(H_diag)
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp # 使 H 转变为全正数，保证正定

        H_c = H.clone()
        H_cpu = H_c.cpu()
        del H_c

        # H1 = torch.linalg.cholesky(H)
        # H2 = torch.cholesky_inverse(H1)
        # H3 = torch.linalg.cholesky(H2, upper=True)
        # Hinv = H

        H_cpu = np.linalg.inv(H_cpu)
        H_cpu = np.linalg.cholesky(H_cpu)
        Hinv = torch.tensor(H_cpu, device=self.dev)

        for i1 in range(0, self.columns, blocksize): # 以步长 blocksize (128) 循环到 columns
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # Find by observer
                # if groupsize != -1:
                #     if (i1 + i) % groupsize == 0:
                #         self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)

                # q = quantize(
                #     w, self.scale, self.zero_point, self.quant_max
                # ).flatten()
                q = torch.fake_quantize_per_tensor_affine(
                    w, self.scale.item(), int(self.zero_point.item()),
                    self.quant_min, self.quant_max)
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])


        torch.cuda.synchronize()
        # print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer_module, transformers.Conv1D):
            Q = Q.t()
        # self.weight.data = Q.reshape(self.weight.shape).to(self.weight.data.dtype)
        W = Q.reshape(self.weight.shape).to(self.weight.data.dtype) # fixed fakequantize 并没有重置 parameter weight


        # if DEBUG:
        #     print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        # self.layer_module.weight.data = W
        # if ('Conv2d' in self.layer_type):
            # print(torch.sum((self.layer_module(self.input) - self.layer_out) ** 2))
        # if ('Linear' in self.layer_type):
        # print(torch.sum((self.layer_module(self.input) - self.output) ** 2))
        # del self.layer_module
        
        return W