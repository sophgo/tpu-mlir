//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaTanOp(top::TanOp op) {
  auto in = getCudaData(op.getInput()), out = getCudaData(op.getOutput());
  cuda::bmTan(in, out, module::getNumElements(op.getOutput()));
}
