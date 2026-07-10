import numpy as np
import unittest
from tpulang_custom_test_base import TestTPULangCustom
import transform.TpuLang as tpul


class TestScaleAddMixed(TestTPULangCustom):
    """
    Mix the custom multi-input/multi-output op `scaleadd` with built-in ops in
    one graph:
        A, B --[scaleadd]--> sa0=A*0.5, sa1=A*0.5+B
        add(sa0, sa1)  -> t1
        mul(t1, sa1)   -> t2
        relu(t2)       -> t3
        sub(t3, sa0)   -> t4
        concat([t3, t4], axis=1) -> y   # [1,6,8,8]
    The base TestTPULangCustom.compile() runs mlir inference (plugin CPU ref for
    the custom node) vs bmodel inference (cmodel) with cmp=True, i.e. it
    auto-compares TPU output against the CPU reference.
    """

    def _build_and_compile(self, dtype):
        shape = [1, 3, 8, 8]
        a = np.random.random(shape).astype(dtype)
        b = np.random.random(shape).astype(dtype)
        A = tpul.Tensor(name="a", dtype=dtype, shape=shape, data=a)
        B = tpul.Tensor(name="b", dtype=dtype, shape=shape, data=b)

        # custom op: 2 inputs -> 2 outputs
        sa0, sa1 = tpul.custom(
            tensors_in=[A, B],
            op_name="scaleadd",
            params={"scale": 0.5},
            out_dtypes=[dtype, dtype])

        # built-in ops fed by the custom op's outputs
        t1 = tpul.add(sa0, sa1)        # A*0.5 + (A*0.5+B)
        t2 = tpul.mul(t1, sa1)         # t1 * sa1
        t3 = tpul.relu(t2)
        t4 = tpul.sub(t3, sa0)         # relu - A*0.5
        y = tpul.concat([t3, t4], axis=1)  # [1, 6, 8, 8]

        self.compile('ScaleAddMixed', [A, B], [y], dtype)

    def test_fp32(self):
        self._build_and_compile('float32')

    def test_fp16(self):
        self._build_and_compile('float16')


if __name__ == '__main__':
    unittest.main()
