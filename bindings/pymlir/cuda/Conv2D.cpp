//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../pycuda.h"

void py_cuda::cudaConv2D(tpu::Conv2DOp op) {
  auto p = op.parseParam();
  if (!module::isUniformQuantized(op.getOutput())) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  // --------------------------------------------------------------------------
  // 1. inference int8 => int32
  cudnnTensorDescriptor_t input_desc;
  cudnnCreateTensorDescriptor(&input_desc);
  cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT8,
                             p.n, p.ic, p.ih, p.iw);
  cudnnFilterDescriptor_t kernel_desc;
  cudnnCreateFilterDescriptor(&kernel_desc);
  cudnnSetFilter4dDescriptor(kernel_desc, CUDNN_DATA_INT8, CUDNN_TENSOR_NCHW,
                             p.oc, p.ic, p.kh, p.kw);
  cudnnTensorDescriptor_t outi32_desc;
  cudnnCreateTensorDescriptor(&outi32_desc);
  cudnnSetTensor4dDescriptor(outi32_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT32,
                             p.n, p.oc, p.oh, p.ow);
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnCreateConvolutionDescriptor(&conv_desc);
  ASSERT_OP(p.pdb == p.pdf, op); // other not supported
  ASSERT_OP(p.pwl == p.pwr, op); // other not supported
  cudnnSetConvolution2dDescriptor(conv_desc, p.phb, p.pwl, p.sh, p.sw, p.dh,
                                  p.dw, CUDNN_CONVOLUTION, CUDNN_DATA_INT32);
  // prepare input output memory
  void *input = getCudaData(op.getInput());
  void *kernel = getCudaData(op.getFilter());
  void *out_i32;
  CHECK_CUDA(cudaMalloc(&out_i32, p.n * p.oc * p.oh * p.ow * sizeof(int32_t)));
  // forward conv
  float alpha = 1.0f, beta = 0.0f;
  cudnnConvolutionForward(cudnn_, &alpha, input_desc, input, kernel_desc,
                          kernel, conv_desc,
                          CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                          nullptr, 0, &beta, outi32_desc, out_i32);
  // --------------------------------------------------------------------------
  // 2. + bias
  if (p.has_bias) {
    void *bias = getCudaData(op.getBias());
    cudnnTensorDescriptor_t bias_desc;
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT32,
                               1, p.oc, 1, 1);
    alpha = 1.0f, beta = 1.0f;
    cudnnAddTensor(cudnn_, &alpha, bias_desc, bias, &beta, outi32_desc,
                   out_i32);
  }
  // --------------------------------------------------------------------------
  // 3. multiplier + shift i32 => i8
  auto qmode = op.getQuantMode();
  if (qmode != tpu::RequantMode::QDM &&
      qmode != tpu::RequantMode::MultiplierShift) {
    UNREACHABLE_OP("Not Implemented!", op);
  }
  auto rshift_v = module::getI64Array(op.getRshift().value());
  auto multiplier_v =
      module::getI64Array(op.getMultiplier(), rshift_v->size(), 1);
  float *scale = new float[p.oc];
  // TODO(pengchao.hu): need to fix quantization mode
  if (qmode == tpu::RequantMode::QDM) {
    for (int i = 0; i < p.oc; i++) {
      scale[i] = multiplier_v->at(i) / (1 << (31 + rshift_v->at(i)));
    }
  } else {
    for (int i = 0; i < p.oc; i++) {
      scale[i] = multiplier_v->at(i) / (1 << rshift_v->at(i));
    }
  }
  auto otype = module::getStorageType(op.getOutput());
  cudnnTensorDescriptor_t trans_desc;
  cudnnCreateTensorDescriptor(&trans_desc);
  if (otype.isUnsignedInteger(8)) {
    cudnnSetTensor4dDescriptor(trans_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_UINT8,
                               p.n, p.oc, p.oh, p.ow);
  } else {
    cudnnSetTensor4dDescriptor(trans_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT8,
                               p.n, p.oc, p.oh, p.ow);
  }
  void *output = getCudaData(op.getOutput());
  cudnnTransformTensor(cudnn_, scale, outi32_desc, out_i32, scale, trans_desc,
                       output);
  delete[] scale;
  // --------------------------------------------------------------------------
  // 4. do relu
  if (otype.isUnsignedInteger(8) || op.getDoRelu() == false) {
    return;
  }
  cudnnActivationDescriptor_t relu_desc;
  cudnnCreateActivationDescriptor(&relu_desc);
  cudnnSetActivationDescriptor(relu_desc, CUDNN_ACTIVATION_RELU,
                               CUDNN_PROPAGATE_NAN, 0.0);
  alpha = 1.0, beta = 0.0;
  cudnnActivationForward(cudnn_, relu_desc, &alpha, trans_desc, output, &beta,
                         trans_desc, output);
  // --------------------------------------------------------------------------
  // 5. free mem
  CHECK_CUDA(cudaFree(out_i32));
}
