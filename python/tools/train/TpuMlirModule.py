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
from torch.fx.passes.graph_drawer import FxGraphDrawer
from mlir.ir import *
import mlir.dialects.top as top
# import mlir.dialects.train as train
import torch._dynamo as td
from torch.cuda.amp import autocast, GradScaler
#from apex import amp

# td.config.log_level = logging.DEBUG
# td.config.verbose = True
# td.config.output_code = True
# os.environ["TORCHDYNAMO_PRINT_GUARDS"] = "1"

import logging
import importlib

def torch_dtype_from_tpu_mlir(dtype) -> torch.dtype:
    if dtype == 'f16':
        return torch.float16
    elif dtype == 'f32':
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


class TpuMlirModule(torch.nn.Module):
    def __init__(
        self, model_file, output_dtypes, return_none_count = 0
    ):
        super(TpuMlirModule, self).__init__()
        print(f'TpuMlirModule __init__ output_dtypes:{output_dtypes}')
        self._register_state_dict_hook(TpuMlirModule._on_state_dict)
        self.output_dtypes = output_dtypes
        self.model_file = model_file
        self.initialized = False
        self.return_none_count = return_none_count
        if model_file:
            self._initialize()

    def _initialize(self):
        print('_initialize')
        # os.system('ln -sf $TPUC_ROOT/lib/libcmodel_1684x.so $TPUC_ROOT/lib/libcmodel.so')
        pyruntime = importlib.import_module("pyruntime_bm")
        self.model = pyruntime.Model(self.model_file)
        self.net = self.model.Net(self.model.networks[0])
        self.input_names = []
        self.output_names = []
        for i in self.net.inputs:
            self.input_names.append(i.name)
        for i in self.net.outputs:
            self.output_names.append(i.name)
        self.initialized = True

    def _check_initialized(self):
        if not self.initialized:
            raise RuntimeError("TpuMlirModule is not initialized.")

    def _on_state_dict(self, state_dict, prefix, local_metadata):
        # self._check_initialized()
        # # fd = open(self.engine, 'rb')
        # # state_dict[prefix + "engine"] = bytearray(fd.read())
        # # fd.close()
        # state_dict[prefix + "engine"] = self.engine
        # state_dict[prefix + "input_names"] = self.input_names
        # state_dict[prefix + "output_names"] = self.output_names
        pass

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
        # self.engine = state_dict[prefix + "engine"]
        # self.input_names = state_dict[prefix + "input_names"]
        # self.output_names = state_dict[prefix + "output_names"]
        # self._initialize()
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        # state["engine"] = bytearray(self.engine.serialize())
        # state["engine"] = self.engine
        # state.pop("context", None)
        return state

    def __setstate__(self, state):
        # logger = trt.Logger()
        # runtime = trt.Runtime(logger)
        # state["engine"] = runtime.deserialize_cuda_engine(state["engine"])
        # self.__dict__.update(state)
        # if self.engine:
        #     self.context = self.engine.create_execution_context()
        pass

    def forward(self, *inputs):
        print('>>>runtime call bmodel, input info:')
        for input_name, input, net_input in zip(self.input_names, inputs, self.net.inputs):
            print(f'{input_name}, pytorch input shape:{input.shape}, bmodel input shape:{net_input.data.shape}')
        with torch.autograd.profiler.record_function("TpuMlirModule:Forward"):
            self._check_initialized()
            input_shapes = []

            with torch.autograd.profiler.record_function("TpuMlirModule:ProcessInputs"):
                # assert len(inputs) == len(
                #     self.input_names
                # ), f"Wrong number of inputs, expect {len(self.input_names)} get {len(inputs)}."

                contiguous_inputs = inputs
                if isinstance(inputs[0], torch.Tensor):
                    contiguous_inputs = [i.contiguous() for i in inputs]

                i = 0
                for input_name, net_input in zip(self.input_names, self.net.inputs):
                    # assert contiguous_inputs[
                    #     i
                    # ].is_tpu, f"{i}th input({input_name}) is not on tpu device."

                    # dtype = torch_dtype_from_tpu_mlir(net_input.data.dtype)
                    # assert (
                    #     contiguous_inputs[i].dtype == dtype
                    # ), f"Dtype mismatch for {i}th input({input_name}). Expect {dtype}, got {inputs[i].dtype}."

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
                    print(f'time:{time.time()-t0}')

            with torch.autograd.profiler.record_function("TpuMlirModule:ProcessOutputs"):
                # create output tensors
                tpu_outputs: List[torch.Tensor] = []
                dyn_idx = 0
                for i in self.net.outputs:
                    output = np.array(i.data)
                    if dyn:
                        if output.shape != dyn_output_shapes[dyn_idx]:
                            dyn_len = np.prod(dyn_output_shapes[dyn_idx])
                            output = output.flatten()[:dyn_len].reshape(
                                *dyn_output_shapes[dyn_idx])
                            dyn_idx += 1
                    tpu_outputs.append(torch.from_numpy(output))
                for output, dtype in zip(tpu_outputs, self.output_dtypes):
                    if dtype == torch.int64:
                        output = output.int()
                print('forward output shape:', [i.shape for i in tpu_outputs])
                tpu_outputs.extend([None for i in range(self.return_none_count)])
                print('return_none_count:', self.return_none_count)
            if len(tpu_outputs) == 1:
                return tpu_outputs[0]
            return tuple(tpu_outputs)

    def get_layer_info(self) -> str:
        """
        Get layer info of the engine. Only support for TRT > 8.2.
        """
        # inspector = self.engine.create_engine_inspector()
        # return inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        pass
