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
  auto num_out = module::getNumElements(op.getOutput());
  auto out_stype = module::getStorageType(op.getOutput());
  // --------------------------------------------------------------------------
  // 1. inference int8 => float
  auto batch_elem_left = module::getNumElements(op.getInput())/p.batch;
  auto batch_elem_right = module::getNumElements(op.getRight())/p.batch;
  auto batch_elem_out = module::getNumElements(op.getOutput())/p.batch;
  auto out_f32 = cuda_malloc(num_out * sizeof(float));
  bool right_transpose = op.getRightTranspose();
  bool left_transpose = op.getLeftTranspose();
  bool out_transpose = op.getOutputTranspose();
  if (left_transpose || out_transpose)
    UNREACHABLE_OP("Not support left/out transpose in matmul", op);

  if (module::isUniformQuantized(op.getInput())) {
    for (size_t b=0; b<p.batch;b++) {
      auto cur_in = (int8_t *)getCudaData(op.getInput()) + b*batch_elem_left;
      auto cur_right = (int8_t *)getCudaData(op.getRight()) + b*batch_elem_right;
      auto cur_out = (int32_t *)out_f32.get() + b*batch_elem_out;
      bool left_signed = !module::getStorageType(op.getInput()).isUnsignedInteger(8);
      bool right_signed = !module::getStorageType(op.getRight()).isUnsignedInteger(8);
      cuda::mmInt8(cur_in, left_signed, cur_right, right_signed,
                    cur_out, right_transpose, p.M, p.K, p.N);
    }
  } else if (!module::getStorageType(op.getInput()).isF32()) {
    auto in_f32 = newCudaData(op.getInput(), cuda::DT_F32);
    auto right_f32 = newCudaData(op.getRight(), cuda::DT_F32);
    for (size_t b=0; b<p.batch;b++) {
      auto cur_in = (float*)in_f32.get() + b*batch_elem_left;
      auto cur_right = (float*)right_f32.get() + b*batch_elem_right;
      auto cur_out = (float*)out_f32.get() + b*batch_elem_out;
      cuda::mmF32(cur_in, cur_right, cur_out, right_transpose,p.M, p.K, p.N);
    }
    in_f32.reset();
    right_f32.reset();
  } else if (module::getStorageType(op.getInput()).isF32()) {
    auto in_f32 = getCudaData(op.getInput());
    auto right_f32 = getCudaData(op.getRight());
    for (size_t b=0; b<p.batch;b++) {
      auto cur_in = (float*)in_f32 + b*batch_elem_left;
      auto cur_right = (float*)right_f32 + b*batch_elem_right;
      auto cur_out = (float*)out_f32.get() + b*batch_elem_out;
      cuda::mmF32(cur_in, cur_right, cur_out, right_transpose, p.M, p.K, p.N);
    }
  } else {
    llvm_unreachable("not support matmul input type");
  }
  // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  // 2. + bias
  if (p.with_bias) {
    if (p.batch != 1)
      UNREACHABLE_OP("Not support bias in batchmatmul", op);
    if (module::isUniformQuantized(op.getInput())) {
      std::vector<int64_t> out_shape(module::getShape(op.getOutput()));
      std::vector<int64_t> bias_shape = {1,1,1,p.N};
      if (out_shape.size() > 4) {
        UNREACHABLE_OP("not support bias in matmul with out_shape>4", op);
      }
      while(out_shape.size()<4){
        out_shape.insert(out_shape.begin(),1);
      }
      cuda::add4DInt32((int32_t*)out_f32.get(), (int32_t*)getCudaData(op.getBias()), (int32_t*)out_f32.get(),
                      out_shape[0], out_shape[1], out_shape[2], out_shape[3],
                      bias_shape[0], bias_shape[1], bias_shape[2], bias_shape[3],
                      out_shape[0], out_shape[1], out_shape[2], out_shape[3]);
    } else {
      cudnnTensorDescriptor_t outf32_desc, bias_desc;
      cudnnCreateTensorDescriptor(&outf32_desc);
      cudnnSetTensor4dDescriptor(outf32_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                1, 1, p.M, p.N);
      cudnnCreateTensorDescriptor(&bias_desc);
      cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                1, 1, 1, p.N);
      float alpha = 1.0f, beta = 1.0f;
      if (!module::getStorageType(op.getInput()).isF32()) {
        // assume, only int8 bias in int32, otherwise use float bias
        auto bias = newCudaData(op.getBias(), cuda::DT_F32);
        CHECK_CUDNN(cudnnAddTensor(cudnn_, &alpha, bias_desc, bias.get(), &beta,
                                  outf32_desc, out_f32.get()));
      } else {
        auto bias = getCudaData(op.getBias());
        CHECK_CUDNN(cudnnAddTensor(cudnn_, &alpha, bias_desc, bias, &beta,
                                  outf32_desc, out_f32.get()));
      }
      cudnnDestroyTensorDescriptor(bias_desc);
      cudnnDestroyTensorDescriptor(outf32_desc);
    }
  }
  if (module::isUniformQuantized(op.getInput())) {
    // 3. multiplier + shift i32 => i8
    // should consider per-channel quant
    if(module::getStorageType(op.getOutput()).isInteger(8)) {
      auto output = getCudaData(op.getOutput());
      auto rshift_v = module::getI64Array(op.getRshifts(), 1, 0);
      auto multiplier_v = module::getI64Array(op.getMultipliers(), 1, 1);
      int32_t multipler = multiplier_v->at(0);
      int32_t rshift = rshift_v->at(0);
      bool sign = !out_stype.isUnsignedInteger(8);
      bool qdm = op.getQuantMode() == tpu::RequantMode::QDM;
      bool relu = sign && p.do_relu;
      cuda::requantInt8(out_f32.get(), output, multipler, rshift, num_out, sign,
                        qdm, relu);
    } else if (module::getStorageType(op.getOutput()).isInteger(16)) {
      auto output = getCudaData(op.getOutput());
      auto rshift_v = module::getI64Array(op.getRshifts(), 1, 0);
      auto multiplier_v = module::getI64Array(op.getMultipliers(), 1, 1);
      int32_t multipler = multiplier_v->at(0);
      int32_t rshift = rshift_v->at(0);
      bool relu = p.do_relu;
      cuda::requantInt16(out_f32.get(), output, multipler, rshift, num_out, relu);
    } else {
      llvm_unreachable("not support matmul output type other than int8/int16");
    }
  } else {
    if (!out_stype.isF32()) {
      cuda::convertType(out_f32.get(), getCudaData(op.getOutput()), num_out, cuda::DT_F32,
                        getCudaType(op.getOutput()));
      out_f32.reset();
    } else {
      cudaMemcpy(getCudaData(op.getOutput()), out_f32.get(),
                 num_out * sizeof(float), cudaMemcpyDeviceToDevice);
      out_f32.reset();
    }
  }
}

void py_cuda::cudaMatMulOp(top::MatMulOp op) {
  auto p = op.parseParam();
  auto num_out = module::getNumElements(op.getOutput());
  // --------------------------------------------------------------------------
  // 1. inference int8 => float
  auto batch_elem_left = module::getNumElements(op.getInput())/p.batch;
  auto batch_elem_right = module::getNumElements(op.getRight())/p.batch;
  auto batch_elem_out = module::getNumElements(op.getOutput())/p.batch;
  auto in_f32 = getCudaData(op.getInput());
  auto right_f32 = getCudaData(op.getRight());
  auto out_f32 = getCudaData(op.getOutput());
  bool right_transpose = op.getRightTranspose();
  bool left_transpose = op.getLeftTranspose();
  bool out_transpose = op.getOutputTranspose();
  if (left_transpose || out_transpose)
    UNREACHABLE_OP("Not support left/out transpose in matmul", op);
  for (size_t b=0; b<p.batch;b++) {
    auto cur_in = (float*)in_f32 + b*batch_elem_left;
    auto cur_right = (float*)right_f32 + b*batch_elem_right;
    auto cur_out = (float*)out_f32 + b*batch_elem_out;
    cuda::mmF32(cur_in, cur_right, cur_out, right_transpose, p.M, p.K, p.N);
  }
  // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  // 2. + bias
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
                               outf32_desc, out_f32));
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyTensorDescriptor(outf32_desc);
  }
  //-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  if (p.do_relu) {
    doRelu(out_f32, num_out, cuda::DT_F32);
  }
}
