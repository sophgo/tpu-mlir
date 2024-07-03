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
  if (!module::isUniformQuantized(op.getOutput())) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  if (p.do_relu) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  auto num_out = module::getNumElements(op.getOutput());
  auto out_stype = module::getStorageType(op.getOutput());
  // --------------------------------------------------------------------------
  // 1. inference int8 => float
  auto in_f32 = newCudaData(op.getInput(), cuda::DT_F32);
  void *output = getCudaData(op.getOutput());
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
    alpha = p.kh * p.kw;
  }
  cudnnSetPooling2dDescriptor(pooling_desc, pool_type, CUDNN_NOT_PROPAGATE_NAN,
                              p.kh, p.kw, p.pad_h, p.pad_w, p.sh, p.sw);
  auto out_f32 = cuda_malloc(num_out * sizeof(float));
  cudnnPoolingForward(cudnn_, pooling_desc, &alpha, input_desc, in_f32.get(),
                      &beta, outf32_desc, out_f32.get());
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(outf32_desc);
  cudnnDestroyPoolingDescriptor(pooling_desc);
  if (!is_avg) {
    cuda::convertType(out_f32.get(), output, num_out, cuda::DT_F32,
                      getCudaType(op.getOutput()));
    return;
  }
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
