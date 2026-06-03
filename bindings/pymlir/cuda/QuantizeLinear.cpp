//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"
#include <vector>

void py_cuda::cudaQuantizeLinearOp(top::QuantizeLinearOp op) {
  auto input = getCudaData(op.getInput());
  int num = module::getNumElements(op.getInput());
  auto scales_v = module::getF64Array(op.getYScale());
  auto zps_v = module::getF64Array(op.getYZeroPoint());

  // Read input as F32
  std::vector<float> in_h(num);
  if (getCudaType(op.getInput()) == cuda::DT_F32) {
    CHECK_CUDA(cudaMemcpy(in_h.data(), input, num * sizeof(float),
                          cudaMemcpyDeviceToHost));
  } else {
    auto tmp = newCudaData(op.getInput(), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(in_h.data(), tmp.get(), num * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  auto out_type = module::getStorageType(op.getOutput());
  bool is_uint8 = out_type.isUnsignedInteger(8);
  int64_t n, c, h, w;
  module::getNCHW(op.getInput(), n, c, h, w);
  int c_dim = (int)c;
  bool per_channel = (scales_v->size() > 1);

  std::vector<uint8_t> out_u8(num);
  std::vector<int8_t> out_i8(num);
  for (int i = 0; i < num; i++) {
    int ci = per_channel ? ((i / (h * w)) % c_dim) : 0;
    float scale = scales_v->at(ci);
    float zp = zps_v->at(ci % zps_v->size());
    float val = roundf(in_h[i] / scale) + zp;
    if (is_uint8) {
      val = fmaxf(0.0f, fminf(255.0f, val));
      out_u8[i] = (uint8_t)val;
    } else {
      val = fmaxf(-128.0f, fminf(127.0f, val));
      out_i8[i] = (int8_t)val;
    }
  }

  if (is_uint8) {
    CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), out_u8.data(),
                          num, cudaMemcpyHostToDevice));
  } else {
    CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), out_i8.data(),
                          num, cudaMemcpyHostToDevice));
  }
}
