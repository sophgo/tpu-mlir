//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"
#include <vector>

void py_cuda::cudaPReluOp(top::PReluOp op) {
  int64_t n, c, h, w;
  module::getNCHW(op.getInput(), n, c, h, w);
  int num = n * c * h * w;
  int num_slope = module::getNumElements(op.getSlope());

  auto stype = module::getStorageType(op.getInput());
  std::vector<float> in_h(num);
  if (stype.isF32()) {
    CHECK_CUDA(cudaMemcpy(in_h.data(), getCudaData(op.getInput()),
                          num * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto tmp = newCudaData(op.getInput(), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(in_h.data(), tmp.get(), num * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  std::vector<float> sl_h(num_slope);
  if (getCudaType(op.getSlope()) == cuda::DT_F32) {
    CHECK_CUDA(cudaMemcpy(sl_h.data(), getCudaData(op.getSlope()),
                          num_slope * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto tmp = newCudaData(op.getSlope(), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(sl_h.data(), tmp.get(),
                          num_slope * sizeof(float), cudaMemcpyDeviceToHost));
  }

  std::vector<float> out_h(num);
  for (int i = 0; i < num; i++) {
    int ci = (num_slope > 1) ? ((i / (h * w)) % c) : 0;
    float val = in_h[i];
    out_h[i] = (val > 0) ? val : (val * sl_h[ci]);
  }

  auto out_f32 = cuda_malloc(num * sizeof(float));
  CHECK_CUDA(cudaMemcpy(out_f32.get(), out_h.data(), num * sizeof(float),
                        cudaMemcpyHostToDevice));
  if (getCudaType(op.getOutput()) != cuda::DT_F32) {
    cuda::convertType(out_f32.get(), getCudaData(op.getOutput()), num,
                      cuda::DT_F32, getCudaType(op.getOutput()));
  } else {
    CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), out_f32.get(),
                          num * sizeof(float), cudaMemcpyDeviceToDevice));
  }
}

void py_cuda::cudaPReluOp(tpu::PReluOp op) {
  // Try CV18xx path first
  if (module::isCV18xx() && module::isUniformQuantized(op.getOutput())) {
    auto shift = op.getRshift();
    auto shift_pos = op.getRshiftPos().value();
    auto multiplier_pos = op.getMultiplierPos().value();
    auto num_slope = module::getNumElements(op.getSlope());
    auto in_shape = module::getShape(op.getInput());
    int64_t num_inner = 1, num_outer = 1;
    if (in_shape.size() > 1) {
      num_outer = std::accumulate(in_shape.begin(), in_shape.begin() + 2, 1,
                                  std::multiplies<int64_t>());
      num_inner = std::accumulate(in_shape.begin() + 2, in_shape.end(), 1,
                                  std::multiplies<int64_t>());
    } else {
      num_outer = in_shape[0];
      num_inner = 1;
    }
    cuda::cvPReluInt8(getCudaData(op.getInput()), getCudaData(op.getSlope()),
                       getCudaData(op.getOutput()), num_outer, num_inner,
                       num_slope, multiplier_pos, shift_pos, shift);
    return;
  }

  // General path
  int64_t n, c, h, w;
  module::getNCHW(op.getInput(), n, c, h, w);
  int num = n * c * h * w;
  int num_slope = module::getNumElements(op.getSlope());

  auto stype = module::getStorageType(op.getInput());
  std::vector<float> in_h(num);
  if (stype.isF32()) {
    CHECK_CUDA(cudaMemcpy(in_h.data(), getCudaData(op.getInput()),
                          num * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto tmp = newCudaData(op.getInput(), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(in_h.data(), tmp.get(), num * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  std::vector<float> sl_h(num_slope);
  if (getCudaType(op.getSlope()) == cuda::DT_F32) {
    CHECK_CUDA(cudaMemcpy(sl_h.data(), getCudaData(op.getSlope()),
                          num_slope * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto tmp = newCudaData(op.getSlope(), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(sl_h.data(), tmp.get(),
                          num_slope * sizeof(float), cudaMemcpyDeviceToHost));
  }

  std::vector<float> out_h(num);
  for (int i = 0; i < num; i++) {
    int ci = (num_slope > 1) ? ((i / (h * w)) % c) : 0;
    float val = in_h[i];
    out_h[i] = (val > 0) ? val : (val * sl_h[ci]);
  }

  auto out_f32 = cuda_malloc(num * sizeof(float));
  CHECK_CUDA(cudaMemcpy(out_f32.get(), out_h.data(), num * sizeof(float),
                        cudaMemcpyHostToDevice));
  if (getCudaType(op.getOutput()) != cuda::DT_F32) {
    cuda::convertType(out_f32.get(), getCudaData(op.getOutput()), num,
                      cuda::DT_F32, getCudaType(op.getOutput()));
  } else {
    CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), out_f32.get(),
                          num * sizeof(float), cudaMemcpyDeviceToDevice));
  }
}
