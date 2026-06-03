//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"
void py_cuda::cudaSqrtOp(top::SqrtOp op) {
  auto in = getCudaData(op.getInput()), out = getCudaData(op.getOutput());
  cuda::bmSqrt(in, out, module::getNumElements(op.getOutput()));
}
