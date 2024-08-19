import os
import torch
import pdb
import gc
import time
import copy
import numpy as np
import logging
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func
import torchvision.models as models
#from torch._dynamo.optimizations.backends import BACKENDS, create_backend
#from torch._dynamo.optimizations.subgraph import SubGraph
from torch._functorch import compilers
from functorch.compile import min_cut_rematerialization_partition

from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition
from torch.fx.graph_module import GraphModule
from torch.fx.node import _get_qualified_name
from torch.fx.passes.operator_support import OperatorSupport
from typing import Any, List, Sequence, NamedTuple
from typing import Dict, List, Optional, Sequence, Callable
from typing import Sequence, Union
MIN_BLOCK_SIZE = 5
from datetime import datetime
from mlir.ir import *
import mlir.dialects.top as top
# import mlir.dialects.train as train
import torch._dynamo as td
from torch.cuda.amp import autocast, GradScaler
#from apex import amp
# tpu_dev = "privateuseone:0"
tpu_dev = "cpu"
device = torch.device(tpu_dev)

# td.config.log_level = logging.DEBUG
# td.config.verbose = True
# td.config.output_code = True
# os.environ["TORCHDYNAMO_PRINT_GUARDS"] = "1"

import logging
import importlib

def torch_dtype_from_tpu_mlir(dtype) -> torch.dtype:
    if dtype == 'f16':
        return torch.float16
    elif dtype == 'bf16':
        return torch.bfloat16
    elif dtype == 'f32':
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)

def get_np_type_from_torch_type2(type):
    if type == torch.float16:
        return np.float16
    elif type == torch.float32:
        return np.float32
    elif type == torch.bool:
        return np.bool_
    else:
        return np.float32

class TpuMlirModule(torch.nn.Module):
    def __init__(
        self, args, model_file, in_tensor_name_to_idx_dict, output_changed_shapes, output_tensor_names = None, output_dtypes = None, output_shapes = [], return_none_count = 0
    ):
        super(TpuMlirModule, self).__init__()
        self._register_state_dict_hook(TpuMlirModule._on_state_dict)
        self.args = args
        output_shapes = [[1] if i == [] else i for i in output_shapes]
        self.output_dtypes = output_dtypes
        self.output_shapes = output_shapes
        self.model_file = model_file
        self.initialized = False
        self.return_none_count = return_none_count
        self.output_changed_shapes = output_changed_shapes
        self.in_tensor_name_to_idx_dict = in_tensor_name_to_idx_dict
        self.output_tensor_names = output_tensor_names
        self.output_tensor_names = None
        self.skip_runtime_call = False
        if model_file and not self.skip_runtime_call:
            self._initialize()

    def _initialize(self):
        print('_initialize for', self.args.chip)
        # if self.args.chip == 'bm1690':
        #     os.system('ln -sf $TPUC_ROOT/lib/libcmodel_bm1690.so $TPUC_ROOT/lib/libcmodel.so')
        # else:
        #     os.system('ln -sf $TPUC_ROOT/lib/libcmodel_1684x.so $TPUC_ROOT/lib/libcmodel.so')
        pyruntime = importlib.import_module("pyruntime_bm")
        self.model = pyruntime.Model(self.model_file)
        self.net = self.model.Net(self.model.networks[0])
        self.initialized = True

    def engineToBmodel(self):
        with open(self.model_file, "wb") as fd:
            fd.write(self.engine)

    def _check_initialized(self):
        if not self.initialized:
            raise RuntimeError("TpuMlirModule is not initialized.")

    def _on_state_dict(self, state_dict, prefix, local_metadata):
        self._check_initialized()
        with open(self.model_file, 'rb') as fd:
            state_dict[prefix + "engine"] = bytearray(fd.read())

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.engine = state_dict[prefix + "engine"]
        self._initialize()

    def __getstate__(self):
        state = self.__dict__.copy()
        with open(self.model_file, "rb") as fd:
            state["engine"] = bytearray(fd.read())
        return state

    def __setstate__(self, state):
        self.engineToBmodel()
        self.__dict__.update(state)

    def forward(self, *inputs):
        print(f'>>>runtime call bmodel:{self.model_file}:')

        tpu_outputs: List[torch.Tensor] = []
        if self.skip_runtime_call:
            if self.output_dtypes is not None:
                for shape, dtype in zip(self.output_shapes, self.output_dtypes):
                    output = np.random.rand(*shape).astype(get_np_type_from_torch_type2(dtype))
                    tpu_outputs.append(torch.from_numpy(output).to(device))
            if self.return_none_count > 0:
                tpu_outputs.extend([None for i in range(self.return_none_count)])
            return tuple(tpu_outputs)
        print('input info:')
        new_inputs = []
        for net_input in self.net.inputs:
            torch_input = inputs[self.in_tensor_name_to_idx_dict[net_input.name]]
            if list(torch_input.shape) != list(net_input.data.shape):
                torch_input = torch_input.reshape(tuple(net_input.data.shape))
            new_inputs.append(torch_input)
            print(f' bmodel input:{net_input.name} shape:{net_input.data.shape}, torch input shape:{torch_input.shape}')

        with torch.autograd.profiler.record_function("TpuMlirModule:Forward"):
            self._check_initialized()
            input_shapes = []

            with torch.autograd.profiler.record_function("TpuMlirModule:ProcessInputs"):
                contiguous_inputs = [i.contiguous() if isinstance(i, torch.Tensor) else i for i in new_inputs]
                i = 0
                for net_input in self.net.inputs:
                    # assert contiguous_inputs[
                    #     i
                    # ].is_privateuseone, f"{i}th input({net_input.name}) is not on tpu device."

                    # dtype = torch_dtype_from_tpu_mlir(net_input.data.dtype)
                    # assert (
                    #     contiguous_inputs[i].dtype == dtype
                    # ), f"Dtype mismatch for {i}th input({net_input.name}). Expect {dtype}, got {contiguous_inputs[i].dtype}."

                    input = contiguous_inputs[i]
                    input = input if isinstance(input, np.ndarray) else input.cpu().numpy()
                    input_shapes.append(input.shape)
                    if len(input.shape) == 0 or list(input.shape) == [1]:
                        net_input.data = input
                    else:
                        net_input.data[:] = input
                    i += 1

            dyn = False
            with torch.autograd.profiler.record_function("TpuMlirModule:TpuRuntime"):
                if dyn:
                    dyn_output_shapes = self.net.forward_dynamic(input_shapes)
                else:
                    t0 = time.time()
                    dyn_output_shapes = self.net.forward()
                    elapsed_time = time.time() - t0
                    print(f'time: {elapsed_time}')
                    with open('model_runtime.txt', 'a') as f:
                        f.write(f'Model run time: {elapsed_time} seconds\n')
            with torch.autograd.profiler.record_function("TpuMlirModule:ProcessOutputs"):
                # create output tensors
                dyn_idx = 0
                for i in self.net.outputs:
                    if self.output_tensor_names is not None and i.name not in self.output_tensor_names:
                        print('skip:', i.name)
                        continue
                    output = np.array(i.data)
                    if dyn:
                        if output.shape != dyn_output_shapes[dyn_idx]:
                            dyn_len = np.prod(dyn_output_shapes[dyn_idx])
                            output = output.flatten()[:dyn_len].reshape(
                                *dyn_output_shapes[dyn_idx])
                            dyn_idx += 1
                    tmp = torch.from_numpy(output)
                    if i.name in self.output_changed_shapes:
                        tmp = tmp.reshape(self.output_changed_shapes[i.name])
                        print(f'{i.name} reshape from {i.data.shape} to {self.output_changed_shapes[i.name]}')
                    if tmp.shape == torch.Size([1]):
                        tmp = tmp[0]
                    tpu_outputs.append(tmp)
                if self.output_dtypes is not None:
                    for output, dtype in zip(tpu_outputs, self.output_dtypes):
                        if dtype == torch.int64:
                            output = output.int()
                        elif dtype == torch.float16:
                            output = output.half()
                        else:
                            output = output.float()
                print('forward output shape:', [i.shape for i in tpu_outputs])
                if self.return_none_count > 0:
                    tpu_outputs.extend([None for i in range(self.return_none_count)])
                    print('return_none_count:', self.return_none_count)
            if len(tpu_outputs) == 1:
                return tpu_outputs[0]
            # for i,t in enumerate(tpu_outputs):
                # if hasattr(t,'shape'):
                #     if t.shape == torch.Size([1]):
                #         tpu_outputs[i] = torch.tensor(t[0])
            for i in range(len(tpu_outputs)):
                if tpu_outputs[i]!=None:
                    tpu_outputs[i] = tpu_outputs[i].to(device)
            ### destory ###
            del self.model
            del self.net
            return tuple(tpu_outputs)

    def get_layer_info(self) -> str:
        """
        Get layer info of the engine. Only support for TRT > 8.2.
        """
        # inspector = self.engine.create_engine_inspector()
        # return inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        pass
