# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from enum import Enum
import numpy as np
import functools

try:
    from . import op_support
    from .disassembler import Disassembler
except:
    import op_support
    from disassembler import Disassembler


class Device(Enum):
    BM1684X = "BM1684X"
    BM1684 = "BM1684"


class Context:
    __slots__ = "opdef", "opparam", "device"

    def __init__(self, device: Device):
        self.device = Device(device)
        if self.device == Device.BM1684X:
            from . import opdef_1684x
            from . import opparam_1684x

            self.opdef = opdef_1684x
            self.opparam = opparam_1684x
        elif self.device == Device.BM1684:
            from . import opdef_1684
            from . import opparam_1684

            self.opdef = opdef_1684
            self.opparam = opparam_1684

        else:
            raise ValueError(f"Unknown device: {device}")

    def __prepare_bm1684x_resource(self, cmodel):
        # bind compute
        def call_ins(command, engine_type):
            return cmodel.lib.execute_command(
                0,
                np.packbits(
                    command.reshape(-1, 8),
                    axis=-1,
                    bitorder="little",
                ).ctypes.data,
                engine_type,
            )

        def bdc_compute(cls):
            return call_ins(cls.cmd, 0)

        def gdma_compute(cls):
            return call_ins(cls.cmd, 1)

        for _, v in self.opdef.bdc_cmd.items():
            for op in v:
                setattr(op, "compute", bdc_compute)

        for _, v in self.opdef.dma_cmd.items():
            for op in v:
                setattr(op, "compute", gdma_compute)

        # bind memory operation
        memory = self.opparam.Memory(cmodel.LMEM, cmodel.DDR)

        @property
        def data(self):
            return memory.get_data(self)

        @data.setter
        def data(self, np_data):
            memory.set_data(self, np_data)

        setattr(self.MemRef, "data", data)

    def __prepare_bm1684_resource(self, cmodel):
        pass

    def get_runner(self, memory_size):
        try:
            from . import cmodel
        except:
            import cmodel

        if self.device == Device.BM1684X:
            _cmodel = cmodel.BM1684X(memory_size)
            self.__prepare_bm1684x_resource(_cmodel)
            return _cmodel

        if self.device == Device.BM1684:
            _cmodel = cmodel.BM1684(memory_size)
            self.__prepare_bm1684_resource(_cmodel)
            return _cmodel

    @property
    def memmap(self):
        return self.opparam.memmap

    @property
    def MemRef(self):
        return self.opparam.MemRef

    @property
    @functools.lru_cache()
    def disassembler(self):
        return Disassembler(self)

    def tensor2memref(self, tensor):
        return self.MemRef(
            tensor.device_addr, tensor.shape[0], op_support.DType(tensor.dtype)
        )
