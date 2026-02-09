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

void py_cuda::cudaActiveOp(tpu::ActiveOp op) {
  int64_t n, c, h , w;
  auto in = op.getInput();
  module::getNCHW(in, n, c, h, w, false);
  auto num_out = module::getNumElements(op.getOutput());
  auto active_mode = op.getMode();
  cuda::active_mode_t mode = static_cast<cuda::active_mode_t>(active_mode);
  if (active_mode == tpu::ActiveMode::HSIGMOID) { // two coefficients
    f64_array_t coeffs = module::getF64Array(op.getCoeffs(), 2, 0);
    if (module::getStorageType(op.getInput()).isF32()) {
      cuda::bmActive(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                     num_out, mode, coeffs->at(0), coeffs->at(1));
    } else {
      auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto output_f32 = cuda_malloc(num_out * sizeof(float));
      cuda::bmActive(input_f32.get(), output_f32.get(), num_out,
                     mode, coeffs->at(0), coeffs->at(1));
      cuda::convertType(output_f32.get(), getCudaData(op.getOutput()),
                        num_out, cuda::DT_F32,
                        getCudaType(op.getOutput()));
      input_f32.reset();
      output_f32.reset();
    }
  } else if (active_mode == tpu::ActiveMode::ELU || active_mode == tpu::ActiveMode::SWISH) {
    // one coefficient
    f64_array_t coeffs = module::getF64Array(op.getCoeffs(), 1, 0);
    if (module::getStorageType(op.getInput()).isF32()) {
      cuda::bmActive(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                     num_out, mode, coeffs->at(0));
    } else {
      auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto output_f32 = cuda_malloc(num_out * sizeof(float));
      cuda::bmActive(input_f32.get(), output_f32.get(), num_out, mode, coeffs->at(0));
      cuda::convertType(output_f32.get(), getCudaData(op.getOutput()),
                        num_out, cuda::DT_F32,
                        getCudaType(op.getOutput()));
      input_f32.reset();
      output_f32.reset();
    }
  } else {
    // no coefficient
    if (module::getStorageType(op.getInput()).isF32()) {
      cuda::bmActive(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                     num_out, mode);
    } else {
      auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto output_f32 = cuda_malloc(num_out * sizeof(float));
      cuda::bmActive(input_f32.get(), output_f32.get(), num_out, mode);
      cuda::convertType(output_f32.get(), getCudaData(op.getOutput()),
                        num_out, cuda::DT_F32,
                        getCudaType(op.getOutput()));
      input_f32.reset();
      output_f32.reset();
    }
  }

}

void py_cuda::cudaSiLUOp(top::SiLUOp op) {
  int64_t n, c, h , w;
  auto in = op.getInput();
  module::getNCHW(in, n, c, h, w, false);
  auto num_out = module::getNumElements(op.getOutput());
  cudnnTensorDescriptor_t input_desc, buffer_desc, output_desc;
  cudnnActivationDescriptor_t activation_desc;
  cudnnOpTensorDescriptor_t mul_desc;

  CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&buffer_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));

  CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(buffer_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));
  CHECK_CUDNN(cudnnCreateActivationDescriptor(&activation_desc));
  CHECK_CUDNN(cudnnSetActivationDescriptor(activation_desc,
                                            CUDNN_ACTIVATION_SIGMOID,
                                            CUDNN_NOT_PROPAGATE_NAN,
                                            0.0));
  size_t bytes = num_out * sizeof(float);
  auto buffer = cuda_malloc(bytes);
  float alpha = 1.0f, beta = 0.0f;
  CHECK_CUDNN(cudnnActivationForward(cudnn_,
                                      activation_desc,
                                      &alpha,
                                      input_desc, getCudaData(op.getInput()),
                                      &beta,
                                      buffer_desc, buffer.get()));
  CHECK_CUDNN(cudnnCreateOpTensorDescriptor(&mul_desc));
  CHECK_CUDNN(cudnnSetOpTensorDescriptor(mul_desc,
                                        CUDNN_OP_TENSOR_MUL,
                                        CUDNN_DATA_FLOAT,
                                        CUDNN_NOT_PROPAGATE_NAN));
  float alpha1 = 1.0f, alpha2 = 1.0f;
  CHECK_CUDNN(cudnnOpTensor(cudnn_,
                        mul_desc,
                        &alpha1,
                        input_desc, getCudaData(op.getInput()),
                        &alpha2,
                        buffer_desc, buffer.get(),
                        &beta,
                        output_desc, getCudaData(op.getOutput())));
  buffer.reset();
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(buffer_desc);
  cudnnDestroyTensorDescriptor(output_desc);
  cudnnDestroyActivationDescriptor(activation_desc);
  cudnnDestroyOpTensorDescriptor(mul_desc);
}

void py_cuda::cudaGELUOp(top::GELUOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  auto num = module::getNumElements(op.getOutput());
  cuda::bmGELU(input, output, num);
}

void py_cuda::cudaSigmoidOp(top::SigmoidOp op) {
  int64_t n, c, h , w;
  auto in = op.getInput();
  module::getNCHW(in, n, c, h, w, false);
  cudnnTensorDescriptor_t input_desc, buffer_desc, output_desc;
  cudnnActivationDescriptor_t activation_desc;

  CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&buffer_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));

  CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(buffer_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));
  CHECK_CUDNN(cudnnCreateActivationDescriptor(&activation_desc));
  CHECK_CUDNN(cudnnSetActivationDescriptor(activation_desc,
                                            CUDNN_ACTIVATION_SIGMOID,
                                            CUDNN_NOT_PROPAGATE_NAN,
                                            0.0));
  float alpha = 1.0f, beta = 0.0f;
  CHECK_CUDNN(cudnnActivationForward(cudnn_,
                                      activation_desc,
                                      &alpha,
                                      input_desc, getCudaData(op.getInput()),
                                      &beta,
                                      output_desc, getCudaData(op.getOutput())));
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(output_desc);
  cudnnDestroyActivationDescriptor(activation_desc);
}
