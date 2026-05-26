//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"
void py_cuda::cudaShuffleChannelOp(top::ShuffleChannelOp op) {
  int64_t n, c, h, w; module::getNCHW(op.getInput(), n, c, h, w);
  cuda::bmShuffleChannel(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                         n, c, h * w, op.getGroup());
}
void py_cuda::cudaShuffleChannelOp(tpu::ShuffleChannelOp op) {
  int64_t n, c, h, w; module::getNCHW(op.getInput(), n, c, h, w);
  auto stype = module::getStorageType(op.getOutput());
  if (stype.isF32()) {
    cuda::bmShuffleChannel(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                           n, c, h * w, op.getGroup());
  } else {
    auto num = module::getNumElements(op.getOutput());
    auto out_f32 = cuda_malloc(num * sizeof(float));
    auto in_f32 = cuda_malloc(num * sizeof(float));
    cuda::convertType(getCudaData(op.getInput()), in_f32.get(), num,
                      getCudaType(op.getInput()), cuda::DT_F32);
    cuda::bmShuffleChannel(in_f32.get(), out_f32.get(), n, c, h * w, op.getGroup());
    cuda::convertType(out_f32.get(), getCudaData(op.getOutput()), num,
                      cuda::DT_F32, getCudaType(op.getOutput()));
  }
}
