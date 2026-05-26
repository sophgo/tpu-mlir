//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaModOp(top::ModOp op) {
  auto inputs = op.getInputs();
  auto a = getCudaData(inputs[0]), b = getCudaData(inputs[1]);
  auto output = getCudaData(op.getOutput());
  int num = module::getNumElements(op.getOutput());
  cuda::bmMod(a, b, output, num);
}
