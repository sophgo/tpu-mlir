import sys
import argparse
import numpy as np
import unittest
from tpulang_custom_test_base import TestTPULangCustom
import transform.TpuLang as tpul
import my_tpulang_layer

class TestAbsAdd(TestTPULangCustom):
    def __init__(self, methodName='runTest', compile_arg=None):
        super(TestAbsAdd, self).__init__(methodName)
        self.compile_arg = compile_arg  # Store the compile argument passed from the command line

    def _test(self, dtype):
        shape = [4, 32, 36, 36]
        self.data_in = np.random.random(shape).astype(dtype)
        x = tpul.Tensor(name="in", dtype=dtype, shape=shape, data=self.data_in)
        y = my_tpulang_layer.absAdd.tpulang(inputs=[x], b=1.2, dtype=dtype)[0]
        # Use the compile argument from the command line
        self.compile('AbsAdd', [x], [y], dtype, self.compile_arg)

    def test_fp32(self):
        self._test('float32')

    def test_fp16(self):
        self._test('float16')

def main():
    parser = argparse.ArgumentParser(description="Test script for TpuLang AbsAdd functionality.")
    parser.add_argument('--is_dynamic', type=str, help="Custom argument to demonstrate CLI usage.")
    parser.add_argument('--unittest-help', action='store_true', help="Show help for unittest options")

    args, remaining_argv = parser.parse_known_args()

    if args.unittest_help:
        argv = [sys.argv[0], '-h']
    else:
        argv = [sys.argv[0]] + remaining_argv

    # Create a test suite and add tests with the custom argument
    suite = unittest.TestSuite()
    suite.addTest(TestAbsAdd('test_fp32', compile_arg=args.is_dynamic))
    suite.addTest(TestAbsAdd('test_fp16', compile_arg=args.is_dynamic))

    # Run the tests
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == '__main__':
    main()

