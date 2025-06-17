import os
import torch
import pdb
import time
import numpy as np
from typing import List

MIN_BLOCK_SIZE = 5
from mlir.ir import *

tpu_dev = "cpu"
device = torch.device(tpu_dev)
from . import config

# td.config.log_level = logging.DEBUG
# td.config.verbose = True
# td.config.output_code = True
# os.environ["TORCHDYNAMO_PRINT_GUARDS"] = "1"

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

    def __init__(self,
                 model_file,
                 output_count,
                 scalar_tensor_new_fx_graph,
                 output_tensor_names=None,
                 output_dtypes=None,
                 output_shapes=[],
                 ori_output_nodes=[]):
        super(TpuMlirModule, self).__init__()
        self._register_state_dict_hook(TpuMlirModule._on_state_dict)
        output_shapes = [[1] if i == [] else i for i in output_shapes]
        self.output_dtypes = output_dtypes
        self.output_shapes = output_shapes
        self.ori_output_nodes = ori_output_nodes
        self.model_file = model_file
        self.initialized = False
        self.scalar_tensor_new_fx_graph = scalar_tensor_new_fx_graph
        self.output_count = output_count
        self.output_tensor_names = output_tensor_names
        self.output_tensor_names = None
        if model_file and 'skip_runtime_call' not in config.debug_cmd:
            self._initialize()

    def _initialize(self):
        print('_initialize for', config.chip)
        pyruntime = importlib.import_module("pyruntime_bm")
        self.model = pyruntime.Model(self.model_file)
        self.net = self.model.Net(self.model.networks[0])
        if config.run_on_cmodel:
            TPUC_ROOT = os.environ.get('TPUC_ROOT')
            print('_initialize for', config.chip, f'TPUC_ROOT: {TPUC_ROOT}')
            if config.chip == 'bm1690':
                os.system('ln -sf $TPUC_ROOT/lib/libtpuv7_emulator.so $TPUC_ROOT/lib/libcmodel.so')
            else:
                os.system('ln -sf $TPUC_ROOT/lib/libcmodel_1684x.so $TPUC_ROOT/lib/libcmodel.so')
            pyruntime = importlib.import_module("pyruntime_bm")
            self.model = pyruntime.Model(self.model_file)
            self.net = self.model.Net(self.model.networks[0])
        else:
            from torch_tpu.tpu.bmodel_runtime import BmodelRunner
            self.bmodel = BmodelRunner(self.model_file, device_id=0)
            self.bmodel_name = self.bmodel.model_info["networks"][0]
            print('bmodel_name:', self.bmodel_name)
            self.bmodel_input_info = self.bmodel.model_net_info[self.bmodel_name]["inputs"]
            print('bmodel_input_info:', self.bmodel_input_info)
            self.bmodel_output_info = self.bmodel.model_net_info[self.bmodel_name]["outputs"]
            print('bmodel_output_info:', self.bmodel_output_info)
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

    def push_res_to_output_list(self, res):
        if self.ori_output_nodes[self.out_pos] != None:
            self.ori_output_nodes[self.out_pos] = res
        self.out_pos += 1

    def forward(self, *inputs):
        self.out_pos = 0
        print(f'>>>runtime call bmodel:{self.model_file}:')
        tpu_outputs: List[torch.Tensor] = []
        if 'skip_runtime_call' in config.debug_cmd:
            assert self.output_dtypes is not None and self.output_shapes is not None
            for shape, dtype in zip(self.output_shapes, self.output_dtypes):
                output = np.random.rand(*shape).astype(get_np_type_from_torch_type2(dtype))
                self.push_res_to_output_list(torch.from_numpy(output).to(device))
            return tuple(self.ori_output_nodes)

        self._check_initialized()
        if config.run_on_cmodel:
            inputs = [i.contiguous() if isinstance(i, torch.Tensor) else i for i in inputs]
            print('inputs shape:', [i.shape for i in inputs])
            print('self.scalar_tensor_new_fx_graph:', self.scalar_tensor_new_fx_graph)
            scalar_tensor_output = {}
            idx0 = 0
            for idx, input in enumerate(inputs):
                if idx in self.scalar_tensor_new_fx_graph:
                    fx_g, out_idx = self.scalar_tensor_new_fx_graph[idx]
                    scalar_tensor_output[out_idx] = fx_g(input)
                else:
                    net_input = self.net.inputs[idx0]
                    idx0 += 1
                    # input = input.cpu().numpy()
                    if len(input.shape) == 0:
                        net_input.data[:] = input.view((1))
                    else:
                        net_input.data[:] = input
            self.net.forward()
            idx = 0
            for i in range(self.output_count):
                if i in scalar_tensor_output:
                    out = scalar_tensor_output[i]
                else:
                    out = self.net.outputs[idx]
                    #Filter out the tensor output due to debugging
                    if self.output_tensor_names is not None and out.name not in self.output_tensor_names:
                        print('skip:', out.name)
                        continue
                    out = torch.from_numpy(np.array(out.data)).reshape(*self.output_shapes[idx])
                    if out.shape == torch.Size([1]):
                        out = out[0]
                    idx += 1
                if self.output_dtypes is not None:
                    #Converts the data type output by bmodel to the type required by pytorch
                    dtype = self.output_dtypes[i]
                    if dtype == torch.int64:
                        out = out.int()
                    elif dtype == torch.float16:
                        out = out.half()
                    else:
                        out = out.float()
                out = out.to("cpu")
                self.push_res_to_output_list(out)
        else:
            self.outputs_tpu = []
            for i in range(len(self.bmodel_output_info)):
                out_v = self.bmodel.get_model_tensor(i, is_input=0)
                print(
                    f'bmodel_output device:{out_v.device}, shape:{out_v.shape}, dtype:{out_v.dtype}'
                )
                self.outputs_tpu.append(out_v)
            self.inputs_tpu = []
            scalar_tensor_output = {}
            idx0 = 0
            for idx, input in enumerate(inputs):
                print(
                    f'torch input device:{input.device}, shape:{input.shape}, dtype:{input.dtype}')
                if idx in self.scalar_tensor_new_fx_graph:
                    fx_g, out_idx = self.scalar_tensor_new_fx_graph[idx]
                    scalar_tensor_output[out_idx] = fx_g(input)
                else:
                    bmodel_in = self.bmodel.get_model_tensor(idx0)
                    idx0 += 1
                    print(
                        f'bmodel_in device:{bmodel_in.device}, shape:{bmodel_in.shape}, dtype:{bmodel_in.dtype}'
                    )
                    input = input.to(bmodel_in.dtype)
                    bmodel_in.copy_(input)
                    self.inputs_tpu.append(bmodel_in)
            print(
                f"inputs_tpu size:{len(self.inputs_tpu)}, outputs_tpu size:{len(self.outputs_tpu)}")
            self.bmodel.forward_with_outputs(self.inputs_tpu, self.outputs_tpu, with_check=False)
            print("forward_sync_with_outputs end", flush=True)
            idx = 0
            for i in range(self.output_count):
                if i in scalar_tensor_output:
                    tpu_outputs.append(scalar_tensor_output[i])
                else:
                    out = self.bmodel.get_model_tensor(idx, is_input=0)
                    out = out.reshape(*self.output_shapes[idx])
                    tpu_outputs.append(out)
                    idx += 1
            if self.output_dtypes is not None:
                #Converts the data type output by bmodel to the type required by pytorch
                for output, dtype in zip(tpu_outputs, self.output_dtypes):
                    if dtype == torch.int64:
                        output = output.int()
                    elif dtype == torch.float16:
                        output = output.half()
                    else:
                        output = output.float()

        if len(tpu_outputs) == 1:
            return tpu_outputs[0]

        for i, out in enumerate(tpu_outputs):
            print(f'out{i} shape:', out.shape)

        ### destory ###
        # del self.model
        # del self.net
        # print('wxxxx:', [type(x) for x in self.ori_output_nodes])
        return tpu_outputs

    def get_layer_info(self) -> str:
        """
        Get layer info of the engine. Only support for TRT > 8.2.
        """
        # inspector = self.engine.create_engine_inspector()
        # return inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        pass
