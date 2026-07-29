//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"
void py_cuda::cudaSignOp(top::SignOp op) {
  auto in = getCudaData(op.getInput()), out = getCudaData(op.getOutput());
  cuda::bmSign(in, out, module::getNumElements(op.getOutput()));
}
