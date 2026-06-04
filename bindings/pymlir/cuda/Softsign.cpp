//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"
void py_cuda::cudaSoftsignOp(top::SoftsignOp op) {
  auto in = getCudaData(op.getInput()), out = getCudaData(op.getOutput());
  cuda::bmSoftsign(in, out, module::getNumElements(op.getOutput()));
}
