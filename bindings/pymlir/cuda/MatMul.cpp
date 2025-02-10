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

void py_cuda::cudaMatMulOp(tpu::MatMulOp op) {
  auto p = op.parseParam();
  if (!module::isUniformQuantized(op.getOutput())) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  auto num_out = module::getNumElements(op.getOutput());
  auto out_stype = module::getStorageType(op.getOutput());
  // --------------------------------------------------------------------------
  // 1. inference int8 => float
  auto batch_elem_left = module::getNumElements(op.getInput())/p.batch;
  auto batch_elem_right = module::getNumElements(op.getInput())/p.batch;
  auto batch_elem_out = module::getNumElements(op.getOutput())/p.batch;
  auto in_f32 = newCudaData(op.getInput(), cuda::DT_F32);
  auto right_f32 = newCudaData(op.getRight(), cuda::DT_F32);
  auto out_f32 = cuda_malloc(num_out * sizeof(float));
  for (size_t b=0; b<p.batch;b++) {
    auto cur_in = (float*)in_f32.get() + b*batch_elem_left;
    auto cur_right = (float*)right_f32.get() + b*batch_elem_right;
    auto cur_out = (float*)out_f32.get() + b*batch_elem_out;
    cuda::mmF32(cur_in, cur_right, cur_out, p.M, p.K, p.N);
  }
  in_f32.reset();
  right_f32.reset();
  // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  // 2. + bias
  if (p.with_bias) {
    auto bias = newCudaData(op.getBias(), cuda::DT_F32);
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
    CHECK_CUDNN(cudnnAddTensor(cudnn_, &alpha, bias_desc, bias.get(), &beta,
                               outf32_desc, out_f32.get()));
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyTensorDescriptor(outf32_desc);
  }
  auto out_i32 =
      newCudaData(out_f32.get(), num_out, cuda::DT_F32, cuda::DT_INT32);
  out_f32.reset();
  //-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  // 3. multiplier + shift i32 => i8
  auto output = getCudaData(op.getOutput());
  auto rshift_v = module::getI64Array(op.getRshifts(), 1, 0);
  auto multiplier_v = module::getI64Array(op.getMultipliers(), 1, 1);
  int32_t multipler = multiplier_v->at(0);
  int32_t rshift = rshift_v->at(0);
  bool sign = !out_stype.isUnsignedInteger(8);
  bool qdm = op.getQuantMode() == tpu::RequantMode::QDM;
  bool relu = sign && p.do_relu;
  cuda::requantInt8(out_i32.get(), output, multipler, rshift, num_out, sign,
                    qdm, relu);
}
