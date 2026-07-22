//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaSwapChannelOp(top::SwapChannelOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  int64_t n, c, h, w;
  module::getNCHW(op.getInput(), n, c, h, w);
  int frame_size = h * w;
  auto order_attr = module::getI64Array(op.getChannelOrder());
  std::vector<int> order(c);
  for (int i = 0; i < c; i++) order[i] = order_attr->at(i);
  auto order_d = cuda_malloc(c * sizeof(int));
  CHECK_CUDA(cudaMemcpy(order_d.get(), order.data(), c * sizeof(int), cudaMemcpyHostToDevice));
  cuda::swapChannel(input, output, order_d.get(), n, c, frame_size);
}
