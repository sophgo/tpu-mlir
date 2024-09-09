from tools.tdb import TdbInterface
from functools import partial
from debugger.target_1688.regdef import DMA_tensor_0x000__reg
from debugger.target_1688.opparam import DMA_tensor_0x000__converter
from debugger.target_1688.device_rt import GLOBALBUF_TAG, DType, Layout, L2S_TAG, memcpy_addr_mask, DmaCmd
from debugger.target_1688.context import BM1688Context

import ctypes
import numpy as np
import ctypes
import numpy as np

context = BM1688Context()
runner = context.get_runner(0)
memory = runner.memory

max_core_num = runner.lib.get_max_core_num(runner.runner_p)


def check_data(gd, address, tag=-1):
    actual = np.zeros_like(gd)

    runner.lib.memcpy_d2s(
        runner.runner_p,
        ctypes.c_uint64(memcpy_addr_mask(address)),
        actual.size * actual.dtype.itemsize,
        actual.ctypes.data_as(ctypes.c_void_p),
        tag,
    )
    assert (np.abs(gd - actual) < 1e-9).all()


localmem = np.arange(1 * 32 * 512 * 64, dtype=np.float32)
ddr = np.arange(1 * 32 * 512, dtype=np.float32)


def test_ddr():

    runner.lib.memcpy_s2d(
        runner.runner_p,
        ctypes.c_uint64(memcpy_addr_mask(687246991360)),
        ddr.size * ddr.dtype.itemsize,
        ddr.ctypes.data_as(ctypes.c_void_p),
        GLOBALBUF_TAG,
    )
    # p *((float*)(continuous_mem[0][0].raw_ptr + 137158656))@12
    # p *((float*)(continuous_mem[0][0].raw_ptr + 137158656))@12
    check_data(ddr, 687246991360, GLOBALBUF_TAG)
    breakpoint()


def test_ddr2():
    ddr2 = np.zeros_like(ddr)
    runner.lib.memcpy_d2s(
        runner.runner_p,
        ctypes.c_uint64(0),
        ddr2.size * ddr.dtype.itemsize,
        ddr2.ctypes.data_as(ctypes.c_void_p),
        L2S_TAG,
    )

    print((ddr2 == ddr).all())
    breakpoint()


"""
{intr_en = 0, stride_enable = 1, nchw_copy = 1, cmd_short = 0, cmd_type = 0,
  special_function = 0, fill_constant_en = 0, src_data_format = 2, mask_data_format = 0,
  eng_sync_id = 0, constant_value = 0, src_nstride = 1228800, src_cstride = 409600,
  src_hstride = 640, src_wstride = 1, dst_nstride = 21252, dst_cstride = 21252,
  dst_hstride = 322, dst_wstride = 1, src_nsize = 1, src_csize = 3, src_hsize = 66,
  src_wsize = 322, dst_nsize = 0, dst_csize = 0, dst_hsize = 0, dst_wsize = 0,
  src_start_addr_l32 = 137158656, src_start_addr_h8 = 129, dst_start_addr_l32 = 0,
  dst_start_addr_h8 = 0, mask_start_addr_l32 = 0, mask_start_addr_h8 = 0,
  localmem_mask_l32 = 4294967295, localmem_mask_h32 = 4294967295, index_cstride = 0,
  index_hstride = 0, index_csize = 0, index_hsize = 0, bias0 = 0 '\000', bias1 = 0 '\000',
  is_signed = false, zero_guard = false}

"""
test_ddr()
dma_tensor_reg = DMA_tensor_0x000__reg()
dma_tensor_reg.stride_enable = 1
dma_tensor_reg.nchw_copy = 1
dma_tensor_reg.cmd_short = 0
dma_tensor_reg.cmd_id_dep = 0
dma_tensor_reg.src_data_format = 2
dma_tensor_reg.src_nstride = 1228800
dma_tensor_reg.src_cstride = 409600
dma_tensor_reg.src_hstride = 640
dma_tensor_reg.src_wstride = 1
dma_tensor_reg.dst_nstride = 21252
dma_tensor_reg.dst_cstride = 21252
dma_tensor_reg.dst_hstride = 322
dma_tensor_reg.dst_wstride = 1
dma_tensor_reg.src_nsize = 1
dma_tensor_reg.src_csize = 3
dma_tensor_reg.src_hsize = 66
dma_tensor_reg.src_wsize = 322
dma_tensor_reg.dst_nsize = 0
dma_tensor_reg.dst_csize = 0
dma_tensor_reg.dst_hsize = 0
dma_tensor_reg.dst_wsize = 0
dma_tensor_reg.src_start_addr_l32 = 52224000
dma_tensor_reg.src_start_addr_h8 = 160
dma_tensor_reg.dst_start_addr_l32 = 0
dma_tensor_reg.dst_start_addr_h8 = 0
dma_tensor_reg.localmem_mask_l32 = 4294967295
dma_tensor_reg.localmem_mask_h32 = 4294967295

# 0x7ffece384010

print(dma_tensor_reg)
cmd = DmaCmd(dma_tensor_reg,
             buf=memoryview(bytearray(dma_tensor_reg)),
             cmd_id=0,
             param_fn=lambda x: DMA_tensor_0x000__converter(context=context, reg=x))
print(f"run {cmd}")

ori = np.zeros_like(ddr)
# runner.lib.memcpy_s2l(
#     runner.runner_p,
#     reference.ctypes.data_as(ctypes.c_void_p),
# )
# print(f"run l2s 1")
# runner.lib.memcpy_l2s(
#     runner.runner_p,
#     ori.ctypes.data_as(ctypes.c_void_p),
# )

runner.dma_compute(cmd)

# p *((float*)(continuous_mem[0][0].raw_ptr + 89128960))@12
# p *((float*)(continuous_mem[0][3].raw_ptr))@12
# p ((float*)(continuous_mem[0][3].raw_ptr))
actual = np.zeros_like(localmem)
# runner.lib.memcpy_s2l(
#     runner.runner_p,
#     reference.ctypes.data_as(ctypes.c_void_p),
# )

# print(f"run l2s 2")
runner.lib.memcpy_l2s(
    runner.runner_p,
    actual.ctypes.data_as(ctypes.c_void_p),
    0,
)
print(actual)
test_ddr2()

# 89128960
