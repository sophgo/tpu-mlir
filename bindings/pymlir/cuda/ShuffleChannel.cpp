//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"
void py_cuda::cudaShuffleChannelOp(top::ShuffleChannelOp op) {
  int64_t n, c, h, w; module::getNCHW(op.getInput(), n, c, h, w);
  cuda::bmShuffleChannel(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                         n, c, h * w, op.getGroup());
}
