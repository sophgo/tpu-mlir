from debugger.target_1688.device_rt import NEURON_TAG, DType, Layout, S2L_TAG
from debugger.target_1688.context import BM1688Context

import ctypes
import numpy as np

context = BM1688Context()
runner = context.get_runner(0)
memory = runner.memory

max_core_num = runner.lib.get_max_core_num(runner.runner_p)


def global_mask(addr, tag):
    if tag >= 0:
        return addr & ((1 << 32) - 1)
    else:
        return addr


def check_data(gd, address, tag=-1):
    actual = np.zeros_like(gd)

    runner.lib.memcpy_d2s(
        runner.runner_p,
        ctypes.c_uint64(global_mask(address, tag)),
        actual.size * actual.dtype.itemsize,
        actual.ctypes.data_as(ctypes.c_void_p),
        tag,
    )
    assert (np.abs(gd - actual) < 1e-9).all()


def test_ddr():
    actual = np.random.random(16 * 1024 * 1024)
    runner.lib.memcpy_s2d(
        runner.runner_p,
        ctypes.c_uint64(0),
        actual.size * actual.dtype.itemsize,
        actual.ctypes.data_as(ctypes.c_void_p),
        NEURON_TAG,
    )

    check_data(actual, 1 << 32, NEURON_TAG)

    runner.lib.memcpy_s2d(
        runner.runner_p,
        ctypes.c_uint64(global_mask(687194771456, NEURON_TAG)),
        actual.size * actual.dtype.itemsize,
        actual.ctypes.data_as(ctypes.c_void_p),
        NEURON_TAG,
    )

    check_data(actual, 687194771456, NEURON_TAG)
    print("DDR tested")


def test_localmem(core_num):
    reference = np.random.rand(1 * 32 * 512 * 64).astype(np.float32, casting='same_kind')
    reference = np.arange(reference.size, dtype=np.int32) + core_num + 5
    runner.lib.memcpy_s2l(
        runner.runner_p,
        reference.ctypes.data_as(ctypes.c_void_p),
        core_num,
    )
    check_data(reference, 0, S2L_TAG)

    actual = np.zeros_like(reference)
    runner.lib.memcpy_l2s(
        runner.runner_p,
        actual.ctypes.data_as(ctypes.c_void_p),
        core_num,
    )
    if not ((actual == reference).all()):
        breakpoint()

    print(f"localmem {core_num} tested")
    print(actual)
    print(reference)


data = np.arange(1 * 16 * 32 * 32, dtype=np.float32)

ref = context.MemRef(
    687194771456,
    shape=[1, 16, 32, 32],
    dtype=DType.f32,
    stride=[32 * 32 * 16, 32 * 32, 32, 1],
    layout=Layout.compact,
)
assert memory.set_data(ref, data)
ret = memory.get_data(ref.to_ref())

test_ddr()
for i in range(max_core_num):
    test_localmem(i)
