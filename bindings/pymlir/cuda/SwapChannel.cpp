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
void py_cuda::cudaSwapChannelOp(tpu::SwapChannelOp op) {
  int64_t n, c, h, w;
  module::getNCHW(op.getInput(), n, c, h, w);
  int frame_size = h * w;
  auto num_elements = n * c * frame_size;
  auto order_attr = module::getI64Array(op.getChannelOrder());
  std::vector<int> order(c);
  for (int i = 0; i < c; i++) order[i] = order_attr->at(i);
  auto order_d = cuda_malloc(c * sizeof(int));
  CHECK_CUDA(cudaMemcpy(order_d.get(), order.data(), c * sizeof(int), cudaMemcpyHostToDevice));

  auto stype = module::getStorageType(op.getOutput());
  if (stype.isF32()) {
    cuda::swapChannel(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                      order_d.get(), n, c, frame_size);
  } else {
    auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
    auto output_f32 = cuda_malloc(num_elements * sizeof(float));
    cuda::swapChannel(input_f32.get(), output_f32.get(), order_d.get(), n, c, frame_size);
    cuda::convertType(output_f32.get(), getCudaData(op.getOutput()), num_elements,
                      cuda::DT_F32, getCudaType(op.getOutput()));
  }
}
