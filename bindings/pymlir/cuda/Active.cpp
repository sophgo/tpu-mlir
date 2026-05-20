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
  cudnnTensorDescriptor_t input_desc, buffer_desc, output_desc;
  cudnnActivationDescriptor_t activation_desc;
  cudnnOpTensorDescriptor_t mul_desc;

  if (op.getMode() == tpu::ActiveMode::GELU) {
    if (module::getStorageType(op.getInput()).isF32()) {
      cuda::bmGELU(getCudaData(op.getInput()), getCudaData(op.getOutput()), num_out);
    } else {
      auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto output_f32 = cuda_malloc(num_out * sizeof(float));
      cuda::bmGELU(input_f32.get(), output_f32.get(), num_out);
      cuda::convertType(output_f32.get(), getCudaData(op.getOutput()),
                        num_out, cuda::DT_F32,
                        getCudaType(op.getOutput()));
      input_f32.reset();
      output_f32.reset();
    }
    return;
  }

  if (op.getMode() == tpu::ActiveMode::ABSVAL) {
    if (module::getStorageType(op.getInput()).isF32()) {
      cuda::bmAbs(getCudaData(op.getInput()), getCudaData(op.getOutput()), num_out);
    } else {
      auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto output_f32 = cuda_malloc(num_out * sizeof(float));
      cuda::bmAbs(input_f32.get(), output_f32.get(), num_out);
      cuda::convertType(output_f32.get(), getCudaData(op.getOutput()),
                        num_out, cuda::DT_F32,
                        getCudaType(op.getOutput()));
      input_f32.reset();
      output_f32.reset();
    }
    return;
  }

  if (op.getMode() == tpu::ActiveMode::ARCCOS) {
    if (module::getStorageType(op.getInput()).isF32()) {
      cuda::bmArccos(getCudaData(op.getInput()), getCudaData(op.getOutput()), num_out);
    } else {
      auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto output_f32 = cuda_malloc(num_out * sizeof(float));
      cuda::bmArccos(input_f32.get(), output_f32.get(), num_out);
      cuda::convertType(output_f32.get(), getCudaData(op.getOutput()),
                        num_out, cuda::DT_F32,
                        getCudaType(op.getOutput()));
      input_f32.reset();
      output_f32.reset();
    }
    return;
  }

  if (op.getMode() == tpu::ActiveMode::ARCTANH) {
    if (module::getStorageType(op.getInput()).isF32()) {
      cuda::bmArctanh(getCudaData(op.getInput()), getCudaData(op.getOutput()), num_out);
    } else {
      auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto output_f32 = cuda_malloc(num_out * sizeof(float));
      cuda::bmArctanh(input_f32.get(), output_f32.get(), num_out);
      cuda::convertType(output_f32.get(), getCudaData(op.getOutput()),
                        num_out, cuda::DT_F32,
                        getCudaType(op.getOutput()));
      input_f32.reset();
      output_f32.reset();
    }
    return;
  }

  if (op.getMode() == tpu::ActiveMode::COS) {
    if (module::getStorageType(op.getInput()).isF32()) {
      cuda::bmCos(getCudaData(op.getInput()), getCudaData(op.getOutput()), num_out);
    } else {
      auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto output_f32 = cuda_malloc(num_out * sizeof(float));
      cuda::bmCos(input_f32.get(), output_f32.get(), num_out);
      cuda::convertType(output_f32.get(), getCudaData(op.getOutput()),
                        num_out, cuda::DT_F32,
                        getCudaType(op.getOutput()));
      input_f32.reset();
      output_f32.reset();
    }
    return;
  }

  if (op.getMode() == tpu::ActiveMode::COSH) {
    if (module::getStorageType(op.getInput()).isF32()) {
      cuda::bmCosh(getCudaData(op.getInput()), getCudaData(op.getOutput()), num_out);
    } else {
      auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto output_f32 = cuda_malloc(num_out * sizeof(float));
      cuda::bmCosh(input_f32.get(), output_f32.get(), num_out);
      cuda::convertType(output_f32.get(), getCudaData(op.getOutput()),
                        num_out, cuda::DT_F32,
                        getCudaType(op.getOutput()));
      input_f32.reset();
      output_f32.reset();
    }
    return;
  }

  if (op.getMode() == tpu::ActiveMode::ELU) {
    float alpha = 1.0f;
    if (auto coeffs = op->getAttr("coeffs")) {
      auto arr = coeffs.cast<ArrayAttr>().getValue();
      if (!arr.empty()) {
        alpha = static_cast<float>(arr[0].cast<FloatAttr>().getValueAsDouble());
      }
    }
    if (module::getStorageType(op.getInput()).isF32()) {
      cuda::bmElu(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                  num_out, alpha);
    } else {
      auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto output_f32 = cuda_malloc(num_out * sizeof(float));
      cuda::bmElu(input_f32.get(), output_f32.get(), num_out, alpha);
      cuda::convertType(output_f32.get(), getCudaData(op.getOutput()),
                        num_out, cuda::DT_F32,
                        getCudaType(op.getOutput()));
      input_f32.reset();
      output_f32.reset();
    }
    return;
  }

  if (op.getMode() == tpu::ActiveMode::ERF) {
    if (module::getStorageType(op.getInput()).isF32()) {
      cuda::bmErf(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                  num_out);
    } else {
      auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto output_f32 = cuda_malloc(num_out * sizeof(float));
      cuda::bmErf(input_f32.get(), output_f32.get(), num_out);
      cuda::convertType(output_f32.get(), getCudaData(op.getOutput()),
                        num_out, cuda::DT_F32,
                        getCudaType(op.getOutput()));
      input_f32.reset();
      output_f32.reset();
    }
    return;
  }

  if (op.getMode() == tpu::ActiveMode::EXP) {
    if (module::getStorageType(op.getInput()).isF32()) {
      cuda::bmExpElm(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                     num_out);
    } else {
      auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto output_f32 = cuda_malloc(num_out * sizeof(float));
      cuda::bmExpElm(input_f32.get(), output_f32.get(), num_out);
      cuda::convertType(output_f32.get(), getCudaData(op.getOutput()),
                        num_out, cuda::DT_F32,
                        getCudaType(op.getOutput()));
      input_f32.reset();
      output_f32.reset();
    }
    return;
  }

  if (op.getMode() == tpu::ActiveMode::FLOOR) {
    if (module::getStorageType(op.getInput()).isF32()) {
      cuda::bmFloor(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                    num_out);
    } else {
      auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto output_f32 = cuda_malloc(num_out * sizeof(float));
      cuda::bmFloor(input_f32.get(), output_f32.get(), num_out);
      cuda::convertType(output_f32.get(), getCudaData(op.getOutput()),
                        num_out, cuda::DT_F32,
                        getCudaType(op.getOutput()));
      input_f32.reset();
      output_f32.reset();
    }
    return;
  }

  if (op.getMode() == tpu::ActiveMode::HSIGMOID) {
    float alpha = 1.0f / 6.0f, beta = 0.5f;
    if (auto coeffs = op->getAttr("coeffs")) {
      auto arr = coeffs.cast<ArrayAttr>().getValue();
      if (arr.size() >= 2) {
        beta  = static_cast<float>(arr[0].cast<FloatAttr>().getValueAsDouble());
        alpha = static_cast<float>(arr[1].cast<FloatAttr>().getValueAsDouble());
      }
    }
    if (module::getStorageType(op.getInput()).isF32()) {
      cuda::bmHardSigmoid(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                          num_out, alpha, beta);
    } else {
      auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto output_f32 = cuda_malloc(num_out * sizeof(float));
      cuda::bmHardSigmoid(input_f32.get(), output_f32.get(), num_out, alpha, beta);
      cuda::convertType(output_f32.get(), getCudaData(op.getOutput()),
                        num_out, cuda::DT_F32, getCudaType(op.getOutput()));
      input_f32.reset();
      output_f32.reset();
    }
    return;
  }

  if (op.getMode() == tpu::ActiveMode::LN) {
    if (module::getStorageType(op.getInput()).isF32()) {
      cuda::bmLog(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                  num_out);
    } else {
      auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto output_f32 = cuda_malloc(num_out * sizeof(float));
      cuda::bmLog(input_f32.get(), output_f32.get(), num_out);
      cuda::convertType(output_f32.get(), getCudaData(op.getOutput()),
                        num_out, cuda::DT_F32, getCudaType(op.getOutput()));
      input_f32.reset();
      output_f32.reset();
    }
    return;
  }

  if (op.getMode() == tpu::ActiveMode::HSWISH) {
    if (module::getStorageType(op.getInput()).isF32()) {
      cuda::bmHardSwish(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                        num_out);
    } else {
      auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto output_f32 = cuda_malloc(num_out * sizeof(float));
      cuda::bmHardSwish(input_f32.get(), output_f32.get(), num_out);
      cuda::convertType(output_f32.get(), getCudaData(op.getOutput()),
                        num_out, cuda::DT_F32, getCudaType(op.getOutput()));
      input_f32.reset();
      output_f32.reset();
    }
    return;
  }

  if (op.getMode() == tpu::ActiveMode::ROUND) {
    if (module::getStorageType(op.getInput()).isF32()) {
      cuda::bmRound(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                    num_out);
    } else {
      auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto output_f32 = cuda_malloc(num_out * sizeof(float));
      cuda::bmRound(input_f32.get(), output_f32.get(), num_out);
      cuda::convertType(output_f32.get(), getCudaData(op.getOutput()),
                        num_out, cuda::DT_F32, getCudaType(op.getOutput()));
      input_f32.reset();
      output_f32.reset();
    }
    return;
  }

  if (op.getMode() == tpu::ActiveMode::SQRT) {
    if (module::getStorageType(op.getInput()).isF32()) {
      cuda::bmSqrt(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                    num_out);
    } else {
      auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto output_f32 = cuda_malloc(num_out * sizeof(float));
      cuda::bmSqrt(input_f32.get(), output_f32.get(), num_out);
      cuda::convertType(output_f32.get(), getCudaData(op.getOutput()),
                        num_out, cuda::DT_F32, getCudaType(op.getOutput()));
      input_f32.reset();
      output_f32.reset();
    }
    return;
  }

  if (op.getMode() == tpu::ActiveMode::RSQRT) {
    if (module::getStorageType(op.getInput()).isF32()) {
      cuda::bmRsqrt(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                    num_out);
    } else {
      auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto output_f32 = cuda_malloc(num_out * sizeof(float));
      cuda::bmRsqrt(input_f32.get(), output_f32.get(), num_out);
      cuda::convertType(output_f32.get(), getCudaData(op.getOutput()),
                        num_out, cuda::DT_F32, getCudaType(op.getOutput()));
      input_f32.reset();
      output_f32.reset();
    }
    return;
  }

  CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&buffer_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));

  CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));
  CHECK_CUDNN(cudnnCreateActivationDescriptor(&activation_desc));
  switch (op.getMode()) {
    case tpu::ActiveMode::RELU:
      CHECK_CUDNN(cudnnSetActivationDescriptor(activation_desc,
                                              CUDNN_ACTIVATION_RELU,
                                              CUDNN_NOT_PROPAGATE_NAN,
                                              0.0));
      break;
    case tpu::ActiveMode::SIGMOID:
    case tpu::ActiveMode::SILU:
      CHECK_CUDNN(cudnnSetActivationDescriptor(activation_desc,
                                              CUDNN_ACTIVATION_SIGMOID,
                                              CUDNN_NOT_PROPAGATE_NAN,
                                              0.0));
      break;
    case tpu::ActiveMode::TANH:
      CHECK_CUDNN(cudnnSetActivationDescriptor(activation_desc,
                                              CUDNN_ACTIVATION_TANH,
                                              CUDNN_NOT_PROPAGATE_NAN,
                                              0.0));
      break;
    default:
      UNREACHABLE_OP("Not Implemented", op);
  }
  size_t bytes = num_out * sizeof(float);
  float alpha = 1.0f, beta = 0.0f;
  if (op.getMode() == tpu::ActiveMode::SILU) {
    auto buffer = cuda_malloc(bytes);
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(buffer_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));
    if (module::getStorageType(op.getInput()).isF32()) {
      CHECK_CUDNN(cudnnActivationForward(cudnn_,
                                        activation_desc,
                                        &alpha,
                                        input_desc, getCudaData(op.getInput()),
                                        &beta,
                                        buffer_desc, buffer.get()));
    } else  if (!module::isUniformQuantized(op.getInput())) {
      cuda_ptr input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      CHECK_CUDNN(cudnnActivationForward(cudnn_,
                                        activation_desc,
                                        &alpha,
                                        input_desc, input_f32.get(),
                                        &beta,
                                        buffer_desc, buffer.get()));
      input_f32.reset();
    } else {
      auto stype = module::getUniformQuantizedType(op.getInput());
      auto scale = stype.getScale();
      auto input_f32 = cuda_malloc(bytes);
      cuda::int8ScaleToF32(getCudaData(op.getInput()), input_f32.get(), scale,num_out, !module::getStorageType(op.getInput()).isUnsignedInteger());
      CHECK_CUDNN(cudnnActivationForward(cudnn_,
                                        activation_desc,
                                        &alpha,
                                        input_desc, input_f32.get(),
                                        &beta,
                                        buffer_desc, buffer.get()));
      input_f32.reset();
    }
    CHECK_CUDNN(cudnnCreateOpTensorDescriptor(&mul_desc));
    CHECK_CUDNN(cudnnSetOpTensorDescriptor(mul_desc,
                                          CUDNN_OP_TENSOR_MUL,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_NOT_PROPAGATE_NAN));
    float alpha1 = 1.0f, alpha2 = 1.0f;
    if (module::getStorageType(op.getOutput()).isF32()) {
      CHECK_CUDNN(cudnnOpTensor(cudnn_,
                          mul_desc,
                          &alpha1,
                          input_desc, getCudaData(op.getInput()),
                          &alpha2,
                          buffer_desc, buffer.get(),
                          &beta,
                          output_desc, getCudaData(op.getOutput())));
      cudnnDestroyOpTensorDescriptor(mul_desc);
    } else if (!module::isUniformQuantized(op.getOutput())) {
      cuda_ptr input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto buffer2 = cuda_malloc(bytes);
      CHECK_CUDNN(cudnnOpTensor(cudnn_,
                          mul_desc,
                          &alpha1,
                          input_desc, input_f32.get(),
                          &alpha2,
                          buffer_desc, buffer.get(),
                          &beta,
                          output_desc, buffer2.get()));
      input_f32.reset();
      cuda::convertType(buffer2.get(), getCudaData(op.getOutput()), num_out,
                        cuda::DT_F32, getCudaType(op.getOutput()));
      buffer2.reset();
      cudnnDestroyOpTensorDescriptor(mul_desc);
    } else {
      auto stype = module::getUniformQuantizedType(op.getOutput());
      auto scale = stype.getScale();
      cuda::f32ScaleToInt8(buffer.get(), getCudaData(op.getOutput()), scale, num_out, !module::getStorageType(op.getOutput()).isUnsignedInteger(), cuda::RD_HALF_AWAY_FROM_ZERO);
    }
    buffer.reset();
    cudnnDestroyTensorDescriptor(buffer_desc);
  } else {
    if (module::getStorageType(op.getInput()).isF32()) {
      CHECK_CUDNN(cudnnActivationForward(cudnn_,
                                        activation_desc,
                                        &alpha,
                                        input_desc, getCudaData(op.getInput()),
                                        &beta,
                                        output_desc, getCudaData(op.getOutput())));
    } else if (!module::isUniformQuantized(op.getInput())) {
      cuda_ptr input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto buffer = cuda_malloc(bytes);
      CHECK_CUDNN(cudnnSetTensor4dDescriptor(buffer_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));
      CHECK_CUDNN(cudnnActivationForward(cudnn_,
                                      activation_desc,
                                      &alpha,
                                      input_desc, input_f32.get(),
                                      &beta,
                                      buffer_desc, buffer.get()));
      cuda::convertType(buffer.get(), getCudaData(op.getOutput()), num_out,
                        cuda::DT_F32, getCudaType(op.getOutput()));
      buffer.reset();
      cudnnDestroyTensorDescriptor(buffer_desc);
    } else {
      auto stype_in = module::getUniformQuantizedType(op.getInput());
      auto scale_in = stype_in.getScale();
      auto input_f32 = cuda_malloc(bytes);
      auto buffer = cuda_malloc(bytes);
      cuda::int8ScaleToF32(getCudaData(op.getInput()), input_f32.get(), scale_in, num_out, !module::getStorageType(op.getInput()).isUnsignedInteger());
      CHECK_CUDNN(cudnnSetTensor4dDescriptor(buffer_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));

      CHECK_CUDNN(cudnnActivationForward(cudnn_,
                                      activation_desc,
                                      &alpha,
                                      input_desc, input_f32.get(),
                                      &beta,
                                      buffer_desc, buffer.get()));
      auto stype_out = module::getUniformQuantizedType(op.getOutput());
      auto scale_out = stype_out.getScale();
      cuda::f32ScaleToInt8(buffer.get(), getCudaData(op.getOutput()), scale_out, num_out, !module::getStorageType(op.getOutput()).isUnsignedInteger(), cuda::RD_HALF_AWAY_FROM_ZERO);
      buffer.reset();
      input_f32.reset();
      cudnnDestroyTensorDescriptor(buffer_desc);
    }
  }
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(output_desc);
  cudnnDestroyActivationDescriptor(activation_desc);
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
