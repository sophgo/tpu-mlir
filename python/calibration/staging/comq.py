# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from utils.mlir_shell import mlir_lowering
from calibration.data_selector import DataSelector
from utils.preprocess import preprocess
from utils.mlir_parser import MlirParser
import numpy as np
import torch
import os
import sys
import time
import copy
from tqdm import tqdm

from .utils import logging
from .utils import quant_requant_active
from .utils import cal_loss
from .utils import cosine_sim

from .lapq import LossAwareQuant
import pymlir
pymlir.set_mem_mode("force_value_mem")


class COMQ_Layer:
    def __init__(self, op, weight, scalar, bits=4):
        self.name = op
        self.W = torch.tensor(weight.copy()).T
        self.Q = torch.tensor(weight.copy()).T
        self.max_w = torch.ones_like(torch.max(self.Q, dim=1)[0])
        self.min_w = torch.ones_like(torch.min(self.Q, dim=1)[0])
        max_w = torch.max(self.Q)
        min_w = torch.min(self.Q)
        max_w = torch.max(max_w.abs(), min_w.abs())
        self.max_w *= max_w
        self.min_w *= -max_w
        self.bits = bits
        self.bit_code = []
        self.device = 'cpu'
        delta = (self.max_w - self.min_w)/(2**self.bits - 2)  # use -7 +7
        starts = (self.min_w[0] / delta).round()
        ends = starts + (2**self.bits - 1)
        self.bit_code = scalar * torch.stack([torch.linspace(start, end, steps=2**self.bits-1)
                                             for start, end in zip(starts, ends)]).to(self.device)
        self.bit_code = self.bit_code * delta.unsqueeze(1)
        self.greedy = True
        self.u = None

    def _quantizer(self, target_val):

        input_vector_expanded = target_val.unsqueeze(-1).expand_as(self.bit_code)

        differences = torch.abs(self.bit_code - input_vector_expanded)

        _, min_indices = torch.min(differences, dim=1)

        closest_values_efficient = self.bit_code[torch.arange(self.bit_code.size(0)), min_indices]

        return closest_values_efficient

    def _quant_layer(self, X):

        X = torch.tensor(X).to(self.device)
        print(f'op {self.name} input shape {X.shape} Q shape {self.Q.shape}')
        if self.u == None:
            self.u = torch.zeros(X.shape[0], self.W.shape[0]).to(self.device)

        if self.greedy:
            need_perm_matrix = torch.norm(X, dim=0).unsqueeze(0) * self.W.abs()
            perm = torch.argsort(need_perm_matrix, dim=1, descending=True)
            del need_perm_matrix
            self.invperm = torch.argsort(perm)
            self.W = self.W.gather(1, perm)
            self.Q = self.Q.gather(1, perm)
            for idx, w in enumerate(self.W.T):
                X_permed = torch.index_select(X.T, 0, perm[:, idx]).T
                self.u -= X_permed * (w - self.Q[:, idx]).unsqueeze(0)
                w_x = X_permed * w.unsqueeze(0)
                target_val = torch.sum((w_x + self.u) * X_permed, dim=0) / torch.sum(X_permed*X_permed, dim=0)
                q = self._quantizer(target_val)
                self.u.add_(X_permed * (w - q).unsqueeze(0))
                self.Q[:, idx] = q
                del X_permed
        else:
            for idx, w in enumerate(self.W.T):
                self.u -= torch.outer(X[:, idx], w - self.Q[:, idx])
                u_x = torch.mv(self.u.T, X[:, idx])
                w_x = torch.outer(X[:, idx], w)
                target_val = (u_x + torch.mv(w_x.T, X[:, idx])) / (torch.sum(X[:, idx]**2))
                q = self._quantizer(target_val)
                self.u.add_(torch.outer(X[:, idx], w - q))
                self.Q[:, idx] = q

    def finalize(self):
        if self.greedy:
            self.Q = self.Q.gather(1, self.invperm)
            self.W = self.W.gather(1, self.invperm)
        self.Q = self.Q.T
        self.W = self.W.T
        '''
        quantize_error_XW = torch.norm(X @ W.T - X @ Q.T, p='fro')
        quantize_error_weight = torch.norm(W - Q, p='fro')
        return Q, quantize_error_XW, quantize_error_weight
        '''


class Comq(LossAwareQuant):
    def learning(self):
        total = len(self.finetune_layers)
        quantizer = {}
        for op in self.finetune_layers:
            if op in self.finetune_layer_weights:  # only tune mm with weight
                quantizer[op] = COMQ_Layer(op, self.orig_weights[op], 1.0, 4)
        pbar_detail = tqdm(np.arange(len(self.finetune_layer_weights)))
        pbar_detail.set_description("COMQ")
        layer_cnt = 0
        for op in self.finetune_layers:
            if op in self.finetune_layer_weights:  # only tune mm with weight
                layer_cnt += 1
                pbar_detail.set_postfix_str(
                    f"Searching {layer_cnt} [Total: {len(self.finetune_layer_weights)}]")
                pbar_detail.update()
                input_name = self.parser.get_pre_op_by_op_name(op)[0]
                input_data = self.ref_tensors.get(input_name, 0)
                for idx in np.arange(1, self.num_sample):
                    input_data = np.concatenate((input_data, self.ref_tensors.get(input_name, idx)), axis=0)
                    shape = input_data.shape
                input_data = input_data.reshape(-1, shape[-1])
                quantizer[op]._quant_layer(input_data)
                del input_data
                quantizer[op].finalize()
                self.update_weight(op, quantizer[op].Q.numpy())
        # update weight and update threhold of activate
        self.save_weights()
