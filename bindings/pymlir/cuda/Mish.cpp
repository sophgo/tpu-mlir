//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaMishOp(top::MishOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  int num = module::getNumElements(op.getOutput());
  cuda::bmMish(input, output, num);
}
