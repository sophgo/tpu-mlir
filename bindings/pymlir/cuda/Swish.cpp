//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaSwishOp(top::SwishOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  float beta = op.getBeta().convertToDouble();
  int num = module::getNumElements(op.getOutput());
  cuda::bmSwish(input, output, beta, num);
}
