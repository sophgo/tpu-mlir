import numpy as np
import unittest
from tpulang_custom_test_base import TestTPULangCustom
import transform.TpuLang as tpul
import my_tpulang_layer

class TestPreprocess(TestTPULangCustom):
    def _test(self, idtype, odtype):
        shape = [1, 3, 4, 4]
        if idtype == 'uint8':
            self.data_in = (np.random.random(shape) * 255).astype(np.uint8)
        else:
            self.data_in = np.random.random(shape).astype(idtype)
        x = tpul.Tensor(name="in", dtype=idtype, shape=shape, data=self.data_in)
        y = my_tpulang_layer.preprocess.tpulang(inputs=[x], scale=1.0, mean=1.0, dtype=odtype)[0]
        if odtype == 'int8':
            x.quantization(scale=1.0, zero_point=0)
            y.quantization(scale=1.0, zero_point=0)
        self.compile('preprocess', [x], [y], dtype=odtype)

    def test_int8(self):
        self._test('uint8','int8')

    def test_fp16(self):
        self._test('uint8','float16')
    
    def test_fp32(self):
        self._test('uint8','float32')
    
    def test_fp32_to_f16(self):
        self._test('float32','float16')
    
    # def test_fp32_to_int8(self):
    #     self._test('float32','int8') #TODO
    


if __name__ == '__main__':
    unittest.main()
