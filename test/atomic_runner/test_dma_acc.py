from debugger.target_1688.device_rt import BM1688Runner, Memory
from debugger.target_1688.regdef import DMA_tensor_0x000__reg
from debugger.target_1688.opparam import DMA_tensor_0x000__converter
from debugger.target_1688.device_rt import DmaCmd
from debugger.target_1688.context import BM1688Context

import numpy as np
import numpy as np

context = BM1688Context()
runner: BM1688Runner = context.get_runner(0)
memory: Memory = runner.memory

max_core_num = runner.lib.get_max_core_num(runner.runner_p)


def replace_local_mem(fname):
    cmodel_lmem = np.load(fname, allow_pickle=True)['mem'].view(np.uint8)
    memory._set_local_mem(cmodel_lmem, 0)
    memory._load_local_mem(0)
    assert (memory.LMEM[0] == cmodel_lmem).all()


dic = {
    'intr_en': 0,
    'stride_enable': 1,
    'nchw_copy': 1,
    'cmd_short': 0,
    'resered': 0,
    'reserved': 0,
    'Reserved': 0,
    'cmd_type': 0,
    'cmd_special_function': 0,
    'fill_constant_en': 0,
    'src_data_format': 2,
    'cmd_id_dep': 38,
    'constant_value': 0,
    'src_nstride': 2560,
    'src_cstride': 1280,
    'src_hstride': 80,
    'src_wstride': 1,
    'dst_nstride': 1638400,
    'dst_cstride': 25600,
    'dst_hstride': 160,
    'dst_wstride': 1,
    'src_nsize': 1,
    'src_csize': 64,
    'src_hsize': 16,
    'src_wsize': 80,
    'dst_nsize': 0,
    'dst_csize': 0,
    'dst_hsize': 0,
    'dst_wsize': 0,
    'src_start_addr_l32': 0,
    'src_start_addr_h8': 0,
    'dst_start_addr_l32': 0,
    'dst_start_addr_h8': 160,
    'localmem_mask_l32': 4294967295,
    'localmem_mask_h32': 4294967295
}

dma_tensor_reg = DMA_tensor_0x000__reg()
for k, v in dic.items():
    setattr(dma_tensor_reg, k, v)

# 0x7ffece384010

print(dma_tensor_reg)
cmd = DmaCmd(dma_tensor_reg,
             buf=memoryview(bytearray(dma_tensor_reg)),
             cmd_id=0,
             param_fn=lambda x: DMA_tensor_0x000__converter(context=context, reg=x))
print(f"run {cmd}")

op_data = memory.get_data(cmd.operands[0].to_ref())
runner.dma_compute(cmd)
res_data = memory.get_data(cmd.results[0].to_ref())

assert (op_data == res_data).all()
