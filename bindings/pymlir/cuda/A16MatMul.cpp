//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaA16MatMulOp(top::A16MatMulOp op) {
  auto p = op.parseParam();
  // Dequant weight on host (weight + scale + zp are typically small)
  int w_len = p.N * p.K;
  int q_group_size = p.q_group_size > 0 ? p.q_group_size : p.K;
  auto w_bytes = module::getNumElements(op.getWeight());
  std::vector<int8_t> h_weight(w_bytes);
  CHECK_CUDA(cudaMemcpy(h_weight.data(), getCudaData(op.getWeight()),
                        w_bytes, cudaMemcpyDeviceToHost));

  auto s_num = module::getNumElements(op.getScale());
  std::vector<float> h_scale(s_num);
  CHECK_CUDA(cudaMemcpy(h_scale.data(), getCudaData(op.getScale()),
                        s_num * sizeof(float), cudaMemcpyDeviceToHost));

  auto z_num = module::getNumElements(op.getZp());
  std::vector<int8_t> h_zp(z_num);
  CHECK_CUDA(cudaMemcpy(h_zp.data(), getCudaData(op.getZp()),
                        z_num, cudaMemcpyDeviceToHost));

  // Dequant weight to float
  std::vector<float> w_float(w_len, 0);
  if (p.weight_bits == 8) {
    for (int k = 0; k < p.K; ++k) {
      int g = k / q_group_size;
      for (int n = 0; n < p.N; ++n) {
        int idx = k * p.N + n;
        float zp_v = (float)h_zp[g * p.N + n];
        float sc_v = h_scale[g * p.N + n];
        w_float[idx] = ((float)h_weight[idx] - zp_v) * sc_v;
      }
    }
  } else {
    // W4A16: 2 elements per byte
    for (int k = 0; k < p.K; ++k) {
      int g = k / q_group_size;
      for (int n = 0; n < p.N; ++n) {
        int byte_idx = k / 2 * p.N + n;
        int lo_hi = k % 2;
        int8_t raw = h_weight[byte_idx];
        int8_t val = lo_hi ? (raw >> 4) : (raw & 0x0F);
        if (val & 0x08) val |= 0xF0; // sign extend
        float zp_v = (float)h_zp[g * p.N + n];
        float sc_v = h_scale[g * p.N + n];
        w_float[k * p.N + n] = ((float)val - zp_v) * sc_v;
      }
    }
  }

  // Upload dequantized weight to GPU
  auto d_weight = cuda_malloc(w_len * sizeof(float));
  CHECK_CUDA(cudaMemcpy(d_weight.get(), w_float.data(),
                        w_len * sizeof(float), cudaMemcpyHostToDevice));

  // MatMul: input [M, K] × weight [K, N] → output [M, N]
  int total = p.M * p.N;
  auto out_f32 = cuda_malloc(total * sizeof(float));

  cuda::mmF32(getCudaData(op.getInput()), d_weight.get(), out_f32.get(),
              p.right_transpose, p.M, p.K, p.N);

  // Add bias if present
  if (p.with_bias) {
    cuda::addAxis(out_f32.get(), getCudaData(op.getBias()), out_f32.get(),
                  p.M, p.N, 1, cuda::DT_F32);
  }

  // Copy to output
  if (module::getStorageType(op.getOutput()).isF32()) {
    CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), out_f32.get(),
                          total * sizeof(float), cudaMemcpyDeviceToDevice));
  } else {
    cuda::convertType(out_f32.get(), getCudaData(op.getOutput()), total,
                      cuda::DT_F32, getCudaType(op.getOutput()));
  }
}
