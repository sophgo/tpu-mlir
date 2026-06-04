//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"
void py_cuda::cudaSinOp(top::SinOp op) {
  auto in = getCudaData(op.getInput()), out = getCudaData(op.getOutput());
  cuda::bmSin(in, out, module::getNumElements(op.getOutput()));
}
