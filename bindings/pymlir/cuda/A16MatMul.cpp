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

void py_cuda::cudaA16MatMulOp(tpu::A16MatMulOp op) {
  // auto scale = getCudaData(op.getScale());
  // auto zp = getCudaData(op.getZp());
  // auto weight = getCudaData(op.getWeight());
  auto weight_shape = op.getWeight().getType().cast<RankedTensorType>().getShape();
  int K = weight_shape[0];
  int N = weight_shape[1];
  auto in_shape = op.getInput().getType().cast<RankedTensorType>().getShape();
  int64_t M = 1;
  for (int i = 0; i < in_shape.size() - 1; i++) {
    M *= in_shape[i];
  }
  int q_group_size = op.getQGroupSize() ? op.getQGroupSize() : module::getQuantGroupSize();
  auto w_transpose = op.getWTranspose();
  auto w_bits = op.getWeightBits();
  if (w_bits == 4) N *= 2;
  if (w_transpose) std::swap(K, N);
  if (q_group_size <= 0 || q_group_size > K) q_group_size = K;
  auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
  auto output_f32 = newCudaData(op.getOutput(), cuda::DT_F32);
  auto scale = newCudaData(op.getScale(), cuda::DT_F32);
  auto zp = newCudaData(op.getZp(), cuda::DT_F32);
  if (module::isDynamicQuantize()) {
    auto dynamic_quantize_type = op.getDqType();
    auto weight = getCudaData(op.getWeight());
    if (dynamic_quantize_type == "INT4") {
      cuda::mmInt4DynamicQuantize(input_f32.get(), weight, output_f32.get(),
        scale.get(), zp.get(), M, K, N, q_group_size);
    } else if (dynamic_quantize_type == "INT8") {
      cuda::mmInt8DynamicQuantize(input_f32.get(), weight, output_f32.get(),
        scale.get(), zp.get(), M, K, N, q_group_size);
    } else if (dynamic_quantize_type == "F8E4M3") {
      cuda::mmF8DynamicQuantize(input_f32.get(), weight, output_f32.get(),
        scale.get(), zp.get(), M, K, N, q_group_size);
    } else if (dynamic_quantize_type == "F4") {
      cuda::mmF4DynamicQuantize(input_f32.get(), weight, output_f32.get(),
        scale.get(), zp.get(), M, K, N, q_group_size);
    } else if (dynamic_quantize_type == "MXF4") {
      cuda::mmMXF4DynamicQuantize(input_f32.get(), weight, output_f32.get(),
        scale.get(), zp.get(), M, K, N, q_group_size);
    } else {
      UNREACHABLE_OP("unsupported dynamic quantize type", op);
    }
  } else {
    float *dequant_weight;
    cudaMalloc(&dequant_weight, K * N * sizeof(float));
    auto weight = getCudaData(op.getWeight());
    cuda::dequantA16MMWeight(
      weight, dequant_weight, scale.get(), zp.get(), K * N, q_group_size, w_bits);
    if (w_bits == 8){
      float *f16_dequant_weight;
      cudaMalloc(&f16_dequant_weight, K * N * sizeof(uint16_t));
      if (module::isF16Modes()) {
        cuda::convertType(dequant_weight, f16_dequant_weight, K * N, cuda::DT_F32, cuda::DT_F16);
        cuda::convertType(f16_dequant_weight, dequant_weight, K * N, cuda::DT_F16, cuda::DT_F32);
      } else {
        cuda::convertType(dequant_weight, f16_dequant_weight, K * N, cuda::DT_F32, cuda::DT_BF16);
        cuda::convertType(f16_dequant_weight, dequant_weight, K * N, cuda::DT_BF16, cuda::DT_F32);
      }
      cudaFree(f16_dequant_weight);
    }
    cuda::mmF32(input_f32.get(), dequant_weight, output_f32.get(), M, K, N,
      false, w_transpose, false);
    cudaFree(dequant_weight);
  }
  if (!module::isNone(op.getBias())) {
    auto bias = newCudaData(op.getBias(), cuda::DT_F32);
    cudnnTensorDescriptor_t outf32_desc, bias_desc;
    cudnnCreateTensorDescriptor(&outf32_desc);
    cudnnSetTensor4dDescriptor(outf32_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                1, 1, M, N);
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                1, 1, 1, N);
    float alpha = 1.0f, beta = 1.0f;
    CHECK_CUDNN(cudnnAddTensor(cudnn_, &alpha, bias_desc, bias.get(), &beta,
                outf32_desc, output_f32.get()));
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyTensorDescriptor(outf32_desc);
    bias.reset();
  }
  auto output = getCudaData(op.getOutput());
  if (module::isF16Modes()) {
    cuda::convertType(output_f32.get(), output, M * N, cuda::DT_F32, cuda::DT_F16);
  } else {
    cuda::convertType(output_f32.get(), output, M * N, cuda::DT_F32, cuda::DT_BF16);
  }
  input_f32.reset();
  output_f32.reset();
  scale.reset();
  zp.reset();
}

void py_cuda::cudaA16MatMulOp(top::A16MatMulOp op) {
  auto p = op.parseParam();
  auto weight_len = p.N * p.K;
  auto weight_bits = p.weight_bits;
  auto group_size = p.q_group_size;
  auto input = getCudaData(op.getInput());
  auto weight = getCudaData(op.getWeight());
  auto scale = getCudaData(op.getScale());
  auto zp = newCudaData(op.getZp(), cuda::DT_F32);
  auto output = getCudaData(op.getOutput());
  float *dequant_weight;
  cudaMalloc(&dequant_weight, weight_len * sizeof(float));
  cuda::dequantA16MMWeight(
    weight, dequant_weight, scale, zp.get(), weight_len, group_size, weight_bits);
  for (int b = 0; b < p.batch; b++) {
    auto cur_input = (float *)input + b * p.M * p.K;
    auto cur_output = (float *)output + b * p.M * p.N;
    cuda::mmF32(cur_input, dequant_weight, cur_output, p.M, p.K, p.N,
      false, p.right_transpose, false);
  }
  if (p.with_bias) {
    auto bias = getCudaData(op.getBias());
    if (p.batch != 1)
      UNREACHABLE_OP("Not support bias in batchmatmul", op);
    cudnnTensorDescriptor_t outf32_desc, bias_desc;
    cudnnCreateTensorDescriptor(&outf32_desc);
    cudnnSetTensor4dDescriptor(outf32_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                1, 1, p.M, p.N);
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                1, 1, 1, p.N);
    float alpha = 1.0f, beta = 1.0f;
    CHECK_CUDNN(cudnnAddTensor(cudnn_, &alpha, bias_desc, bias, &beta,
                outf32_desc, output));
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyTensorDescriptor(outf32_desc);
  }
  zp.reset();
}