# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
class GlobalInfo:
    def __init__(self):
        self.subnet_list = []
        # self.arch = Arch.UNKNOWN
        self.mem_info = []
        self.archlib = None
        self.freq = None
        self.net_name = None
        self.flops = 0
        self.no_perf_data = False

    def set_arch(self, arch):
        self.arch = arch
        if self.archlib is None:
            self.archlib = load_arch_lib(arch)
        assert self.archlib is not None

class TensorInfo:
    def __init__(self):
        self.tensor_id = -1
        self.name = None
        self.shape = None
        # self.dtype = DataType.UNKNOWN
        self.address = -1
        self.in_layer = None
        self.out_layers = []


class LayerInfo:
    def __init__(self):
        self.layer_id = -1
        self.core_id = 0
        self.layer_type = ""
        self.layer_name = ""
        self.is_local = False  #TODO
        self.in_tensors = []
        self.out_tensors = []
        self.group_id = -1
        self.total_size = 0
        self.feature_size = 0
        self.weight_size = 0
        self.bd_nodes = []
        self.gdma_op = None
        self.gdma_tensor = None
        self.gdma_nodes = []
        self.engine_type = None
        self.file_line = 0

    def add_input(self, tensor):
        if tensor in self.in_tensors:
            return
        self.in_tensors.append(tensor)
        tensor.out_layers.append(self)

    def add_output(self, tensor):
        if tensor in self.out_tensors:
            return
        self.out_tensors.append(tensor)
        tensor.in_layer = self

    def set_gdma_tensor(self, tensor):
        self.gdma_tensor = tensor


class jsonObj:
    def __init__(self):
        self.file_line = -1
        self.subnet_id = 0
        self.core_id = 0
        self.opcode = None
        self.bd_ids = None # (start_bd_id, end_bd_id]
        self.dma_ids = None # (start_gdma_id, end_gdma_id]
        self.operands = []
        self.results = []

class StaticRunNode:
    __run_id = -1

    def __init__(self):
        self.__class__.__run_id += 1
        self.run_id = self.__class__.__run_id
        self.type = None
        self.core_id = -1
        self.bd_id = -1
        self.gdma_id = -1
        self.gdma_dir = None
        self.gdma_func = None
        self.bd_func = None
        self.layer = None
        self.command = None
        self.sim_info = None
        self.pmu_info = None

class SubnetInfo:
    def __init__(self):
        self.subnet_id = -1
        self.layer_list = []
        self.command_info = None
        self.gdma_nodes = []
        self.bd_nodes = []
        self.sim_info = None
