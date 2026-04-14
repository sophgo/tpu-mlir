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

void py_cuda::cudaPool2DOp(tpu::Pool2DOp op) {
  auto p = op.parseParam();
  auto num_out = module::getNumElements(op.getOutput());
  auto out_stype = module::getStorageType(op.getOutput());
  // --------------------------------------------------------------------------
  // 1. inference int8 => float
  cudnnTensorDescriptor_t input_desc;
  cudnnCreateTensorDescriptor(&input_desc);
  cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             p.n, p.c, p.ih, p.iw);
  cudnnTensorDescriptor_t outf32_desc;
  cudnnCreateTensorDescriptor(&outf32_desc);
  cudnnSetTensor4dDescriptor(outf32_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             p.n, p.c, p.oh, p.ow);
  cudnnPoolingDescriptor_t pooling_desc;
  cudnnCreatePoolingDescriptor(&pooling_desc);
  bool is_avg = op.getPoolMode() == tpu::PoolMode::Avg;
  float alpha = 1.0f, beta = 0.0f;
  auto pool_type = CUDNN_POOLING_MAX;
  if (is_avg) {
    if (p.count_include_pad) {
      pool_type = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    } else {
      pool_type = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    }
    if (module::isUniformQuantized(op.getOutput())) {
      // for quantized average pooling, need multiply kh*kw to get sum value, cause the multiplier and shift has made the division
      alpha = p.kh * p.kw;
    }
  }
  cudnnSetPooling2dDescriptor(pooling_desc, pool_type, CUDNN_NOT_PROPAGATE_NAN,
                              p.kh, p.kw, p.pad_h, p.pad_w, p.sh, p.sw);
  auto out_f32 = cuda_malloc(num_out * sizeof(float));
  void *output = getCudaData(op.getOutput());
  if (module::isUniformQuantized(op.getInput())) {
    auto qtype_in = module::getUniformQuantizedType(op.getInput());
    auto zero_point = qtype_in.getZeroPoint();
    auto in_f32 = newCudaData(op.getInput(), cuda::DT_F32);
    if (zero_point != 0 && p.count_include_pad) {
      cuda::subConst4DF32(in_f32.get(), zero_point, in_f32.get(), false, false, p.n, p.c, p.ih, p.iw);
    }
    cudnnPoolingForward(cudnn_, pooling_desc, &alpha, input_desc, in_f32.get(),
                        &beta, outf32_desc, out_f32.get());
    if (zero_point != 0 && p.count_include_pad) {
      cuda::subConst4DF32(out_f32.get(), -zero_point*p.kh*p.kw, out_f32.get(), false, false, p.n, p.c, p.oh, p.ow);
    }
  } else {
    if (module::getStorageType(op.getInput()).isF32()) {
      auto in_f32 = getCudaData(op.getInput());
      cudnnPoolingForward(cudnn_, pooling_desc, &alpha, input_desc, in_f32,
                          &beta, outf32_desc, output);
      if (op.getDoRelu()) {
        cuda::doRelu(output, num_out, cuda::DT_F32);
      }
      out_f32.reset();
      return;
    } else {
      auto in_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      cudnnPoolingForward(cudnn_, pooling_desc, &alpha, input_desc, in_f32.get(),
                          &beta, outf32_desc, out_f32.get());
      in_f32.reset();
    }
  }
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(outf32_desc);
  cudnnDestroyPoolingDescriptor(pooling_desc);
  if (!is_avg) {
    if (op.getDoRelu()) {
      cuda::doRelu(out_f32.get(), num_out, cuda::DT_F32);
    }
    if (module::getStorageType(op.getInput()).isFloat8E4M3FN()){
      cuda::requantF8(out_f32.get(), getCudaData(op.getOutput()), 1.0,
                            1, 1, p.n, p.c, p.oh, p.ow, p.do_relu);
    } else {
      cuda::convertType(out_f32.get(), output, num_out, cuda::DT_F32,
                        getCudaType(op.getOutput()), cuda::RD_HALF_TO_EVEN);
    }
    out_f32.reset();
    return;
  }

  if (module::isUniformQuantized(op.getOutput())) {
    if (module::isAsymmetric()) {
      auto scale = op.getScale().value().convertToDouble();
      auto offset = op.getOffset().value().convertToDouble();
      cuda::rounding_mode_t rmode = static_cast<cuda::rounding_mode_t>(
        round_mode_convert(op.getRoundMode()));
      cuda::mulShiftDouble(out_f32.get(), output, scale, offset, rmode, num_out, getCudaType(op.getOutput()));

      if (op.getDoRelu()) {
        auto qtype = module::getUniformQuantizedType(op.getOutput());
        auto zero_point = qtype.getZeroPoint();
        cuda::doRelu(output, num_out, out_stype.isUnsignedInteger(8) ? cuda::DT_UINT8 : cuda::DT_INT8, zero_point);
      }
      out_f32.reset();
    } else {
      auto out_i32 = cuda_malloc(num_out * sizeof(int32_t));
      cuda::convertType(out_f32.get(), out_i32.get(), num_out, cuda::DT_F32,
                        cuda::DT_INT32, cuda::RD_HALF_UP);
      out_f32.reset();
      //-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
      // 2. multiplier + shift i32 => i8
      auto multi = op.getMultiplier().value_or(1);
      auto rs = op.getRshift().value_or(0);
      bool sign = !out_stype.isUnsignedInteger(8);
      bool relu = sign && p.do_relu;
      cuda::requantInt8(out_i32.get(), output, multi, rs, num_out, sign, false,
                        relu);
    }
  } else if (module::getStorageType(op.getInput()).isFloat8E4M3FN()){
    if (op.getDoRelu()) {
      cuda::doRelu(out_f32.get(), num_out, cuda::DT_F32);
    }
    auto scale = op.getFp8OutScale()->convertToDouble();
    cuda::requantF8(out_f32.get(), getCudaData(op.getOutput()), scale,
                          1, 1, p.n, p.c, p.oh, p.ow, p.do_relu);
    out_f32.reset();
  } else {
    if (op.getDoRelu()) {
      cuda::doRelu(out_f32.get(), num_out, cuda::DT_F32);
    }
    cuda::convertType(out_f32.get(), output, num_out, cuda::DT_F32,
                      getCudaType(op.getOutput()));
    out_f32.reset();
  }
}


void py_cuda::cudaMaxPoolOp(top::MaxPoolOp op) {
  auto p = op.parseParam();
  auto num_out = module::getNumElements(op.getOutput());
  // --------------------------------------------------------------------------
  // 1. inference int8 => float
  auto in_f32 = getCudaData(op.getInput());
  auto out_f32 = getCudaData(op.getOutput());
  cudnnTensorDescriptor_t input_desc;
  cudnnCreateTensorDescriptor(&input_desc);
  cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             p.n, p.c, p.ih, p.iw);
  cudnnTensorDescriptor_t outf32_desc;
  cudnnCreateTensorDescriptor(&outf32_desc);
  cudnnSetTensor4dDescriptor(outf32_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             p.n, p.c, p.oh, p.ow);
  cudnnPoolingDescriptor_t pooling_desc;
  cudnnCreatePoolingDescriptor(&pooling_desc);
  float alpha = 1.0f, beta = 0.0f;
  auto pool_type = CUDNN_POOLING_MAX;
  cudnnSetPooling2dDescriptor(pooling_desc, pool_type, CUDNN_NOT_PROPAGATE_NAN,
                              p.kh, p.kw, p.pad_h, p.pad_w, p.sh, p.sw);
  cudnnPoolingForward(cudnn_, pooling_desc, &alpha, input_desc, in_f32,
                      &beta, outf32_desc, out_f32);
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(outf32_desc);
  cudnnDestroyPoolingDescriptor(pooling_desc);
  //-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  if (op.getDoRelu()) {
    doRelu(out_f32, num_out, cuda::DT_F32);
  }
}

void py_cuda::cudaAvgPoolOp(top::AvgPoolOp op) {
  auto p = op.parseParam();
  auto num_out = module::getNumElements(op.getOutput());
  // --------------------------------------------------------------------------
  // 1. inference int8 => float
  auto in_f32 = getCudaData(op.getInput());
  auto out_f32 = getCudaData(op.getOutput());
  cudnnTensorDescriptor_t input_desc;
  cudnnCreateTensorDescriptor(&input_desc);
  cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             p.n, p.c, p.ih, p.iw);
  cudnnTensorDescriptor_t outf32_desc;
  cudnnCreateTensorDescriptor(&outf32_desc);
  cudnnSetTensor4dDescriptor(outf32_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             p.n, p.c, p.oh, p.ow);
  cudnnPoolingDescriptor_t pooling_desc;
  cudnnCreatePoolingDescriptor(&pooling_desc);
  float alpha = 1.0f, beta = 0.0f;
  auto pool_type = CUDNN_POOLING_MAX;
  if (op.getCountIncludePad()) {
    pool_type = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  } else {
    pool_type = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  }
  cudnnSetPooling2dDescriptor(pooling_desc, pool_type, CUDNN_NOT_PROPAGATE_NAN,
                              p.kh, p.kw, p.pad_h, p.pad_w, p.sh, p.sw);
  cudnnPoolingForward(cudnn_, pooling_desc, &alpha, input_desc, in_f32,
                      &beta, outf32_desc, out_f32);
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(outf32_desc);
  cudnnDestroyPoolingDescriptor(pooling_desc);
  //-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  if (op.getDoRelu()) {
    doRelu(out_f32, num_out, cuda::DT_F32);
  }
}
