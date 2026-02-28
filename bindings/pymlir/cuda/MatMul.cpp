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
  auto p = op.dynparseParam();
  auto num_out = module::getNumElements(op.getOutput());
  auto out_stype = module::getStorageType(op.getOutput());
  // --------------------------------------------------------------------------
  // 1. inference int8 => float
  auto batch_size = p.batch * p.batch_low;
  auto batch_elem_left = module::getNumElements(op.getInput()) / batch_size;
  auto batch_elem_right = module::getNumElements(op.getRight()) / batch_size;
  auto batch_elem_out = module::getNumElements(op.getOutput()) / batch_size;
  auto out_f32 = cuda_malloc(num_out * sizeof(float));
  auto right_zp = p.right_zp;
  auto left_zp = p.input_zp;
  bool right_transpose = op.getRightTranspose();
  bool left_transpose = op.getLeftTranspose();
  bool out_transpose = op.getOutputTranspose();
  // auto op_output = getCudaData(op.getOutput());
  cuda_ptr input_cuda_ptr, right_cuda_ptr;
  void *raw_op_input = nullptr, *raw_op_right = nullptr;
  if (module::isUniformQuantized(op.getInput())) {
    raw_op_input = getCudaData(op.getInput());
    raw_op_right = getCudaData(op.getRight());
  } else if (!module::getStorageType(op.getInput()).isF32()) {
    input_cuda_ptr = newCudaData(op.getInput(), cuda::DT_F32);
    right_cuda_ptr = newCudaData(op.getRight(), cuda::DT_F32);
    raw_op_input = input_cuda_ptr.get();
    raw_op_right = right_cuda_ptr.get();
  } else if (module::getStorageType(op.getInput()).isF32()) {
    raw_op_input = getCudaData(op.getInput());
    raw_op_right = getCudaData(op.getRight());
  } else {
    llvm_unreachable("not support matmul input type");
  }
  void *op_input = nullptr, *op_right = nullptr;
  if (p.hdim_is_batch) {
    out_transpose = false;
    int dtype_bytes;
    if (module::isUniformQuantized(op.getInput())) {
      dtype_bytes = sizeof(int8_t);
    } else if (!module::getStorageType(op.getInput()).isF32()) {
      dtype_bytes = sizeof(float);
    } else if (module::getStorageType(op.getInput()).isF32()) {
      dtype_bytes = sizeof(float);
    } else {
      llvm_unreachable("not support matmul input type");
    }
    if (left_transpose) {
      std::vector<int64_t> input_shape(module::getShape(op.getInput()));
      while (input_shape.size() < 6) {
        input_shape.push_back(1);
      }
      cudaMalloc(&op_input, batch_elem_left * batch_size * dtype_bytes);
      cuda::permute6D(raw_op_input, op_input, input_shape[0], input_shape[1],
        input_shape[2], input_shape[3], input_shape[4], input_shape[5],
        0, 2, 1, 3, 4, 5, dtype_bytes);
      left_transpose = false;
    } else {
      op_input = raw_op_input;
    }
    if (right_transpose) {
      std::vector<int64_t> right_shape(module::getShape(op.getRight()));
      while (right_shape.size() < 6) {
        right_shape.push_back(1);
      }
      cudaMalloc(&op_right, batch_elem_right * batch_size * dtype_bytes);
      cuda::permute6D(raw_op_right, op_right, right_shape[0], right_shape[1],
        right_shape[2], right_shape[3], right_shape[4], right_shape[5],
        0, 2, 1, 3, 4, 5, dtype_bytes);
      right_transpose = false;
    } else {
      op_right = raw_op_right;
    }
  } else {
    op_input = raw_op_input;
    op_right = raw_op_right;
  }
  if (module::isUniformQuantized(op.getInput())) {
    for (size_t b = 0; b < batch_size; b++) {
      auto cur_in = (int8_t *)op_input + b*batch_elem_left;
      auto cur_right = (int8_t *)op_right + b*batch_elem_right;
      auto cur_out = (int32_t *)out_f32.get() + b*batch_elem_out;
      bool left_signed = !module::getStorageType(op.getInput()).isUnsignedInteger(8);
      bool right_signed = !module::getStorageType(op.getRight()).isUnsignedInteger(8);
      cuda::mmInt8(cur_in, left_signed, cur_right, right_signed,
                    cur_out, p.M, p.K, p.N, left_transpose, right_transpose, out_transpose,
                  left_zp, right_zp);
    }
  } else {
    for (size_t b = 0; b < batch_size; b++) {
      auto cur_in = (float*)op_input + b*batch_elem_left;
      auto cur_right = (float*)op_right + b*batch_elem_right;
      auto cur_out = (float*)out_f32.get() + b*batch_elem_out;
      cuda::mmF32(cur_in, cur_right, cur_out, p.M, p.K, p.N,
        left_transpose, right_transpose, out_transpose,
        left_zp, right_zp);
    }
    if (!module::getStorageType(op.getInput()).isF32()) {
      input_cuda_ptr.reset();
      right_cuda_ptr.reset();
    }
  }
  if (p.hdim_is_batch) {
    if (op_input != raw_op_input) {
      cudaFree(op_input);
    }
    if (op_right != raw_op_right) {
      cudaFree(op_right);
    }
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
      if (module::getStorageType(op.getInput()).isF32() || module::getStorageType(op.getInput()).isFloat8E4M3FN() ||
          (module::getStorageType(op.getInput()).isBF16() && module::isCV18xx())) {
        auto bias = getCudaData(op.getBias());
        CHECK_CUDNN(cudnnAddTensor(cudnn_, &alpha, bias_desc, bias, &beta,
                                  outf32_desc, out_f32.get()));
      } else {
        // assume, only int8 bias in int32, otherwise use float bias
        auto bias = newCudaData(op.getBias(), cuda::DT_F32);
        CHECK_CUDNN(cudnnAddTensor(cudnn_, &alpha, bias_desc, bias.get(), &beta,
                                  outf32_desc, out_f32.get()));
      }
      cudnnDestroyTensorDescriptor(bias_desc);
      cudnnDestroyTensorDescriptor(outf32_desc);
    }
  }
  // void* output = getCudaData(op.getOutput());
  void *output = nullptr, *raw_output = getCudaData(op.getOutput());
  int otype_bytes;
  if (out_stype.isInteger(8) || out_stype.isFloat8E4M3FN()) {
    otype_bytes = sizeof(int8_t);
  } else if (out_stype.isInteger(16) || !out_stype.isF32()) {
    otype_bytes = sizeof(int16_t);
  } else {
    otype_bytes = sizeof(float);
  }
  if (p.hdim_is_batch  && (p.output_transpose || !module::isCV18xx())) {
    cudaMalloc(&output, num_out * otype_bytes);
  } else {
    output = raw_output;
  }
  if (module::isUniformQuantized(op.getOutput())) {
    // 3. multiplier + shift i32 => i8
    int shift_num = op.getFuseRq() ? p.N : 1;
    auto rshift_v = module::getI64Array(op.getRshifts(), shift_num, 0);
    auto multiplier_v = module::getI64Array(op.getMultipliers(), shift_num, 1);
    std::vector<int32_t> m(multiplier_v->begin(), multiplier_v->end());
    std::vector<int32_t> r(rshift_v->begin(), rshift_v->end());
    cuda_ptr cudaMults;
    cuda_ptr cudaShifts;
    int32_t multiplier = 0;
    int32_t rshift = 0;
    if (shift_num > 1) {
      cudaMults = cuda_malloc(m.size()*sizeof(int32_t));
      cudaShifts = cuda_malloc(r.size()*sizeof(int32_t));
      CHECK_CUDA(cudaMemcpy(cudaMults.get(), m.data(), m.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(cudaShifts.get(), r.data(), r.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    } else {
      multiplier = multiplier_v->at(0);
      rshift = rshift_v->at(0);
    }
    if(out_stype.isInteger(8)) {
      bool sign = !out_stype.isUnsignedInteger(8);
      bool qdm = op.getQuantMode() == tpu::RequantMode::QDM;
      bool relu = sign && p.do_relu;
      if (shift_num > 1) {
        cuda::requantInt8Perchannel(out_f32.get(), output, cudaMults.get(),
                                  cudaShifts.get(), batch_size * p.M, p.N, 1, 1, sign, qdm,
                                  relu);
      } else{
        cuda::requantInt8(out_f32.get(), output, multiplier, rshift, num_out, sign,
                          qdm, relu);
      }
    } else if (out_stype.isInteger(16)) {
      bool relu = p.do_relu;
      if (shift_num > 1) {
        cuda::requantInt16Perchannel(out_f32.get(), output, cudaMults.get(),
                                  cudaShifts.get(), batch_size * p.M, p.N, 1, 1, relu);
      } else{
        cuda::requantInt16(out_f32.get(), output, multiplier, rshift, num_out, relu);
      }
    } else {
      llvm_unreachable("not support matmul output type other than int8/int16");
    }
    if (shift_num > 1) {
      cudaMults.reset();
      cudaShifts.reset();
    }
  } else if (out_stype.isFloat8E4M3FN()) {
    f64_array_t scales = module::getF64Array(op.getOutF8Scales().value());
    if (scales->size() == 1) {
      cuda::requantF8(out_f32.get(), output, scales->at(0), 1, 1, 1, batch_size, p.M, p.N, p.do_relu);
    } else {
      std::vector<float> oscale(scales->begin(), scales->end());
      auto cudaMults = cuda_malloc(scales->size()*sizeof(float));
      CHECK_CUDA(cudaMemcpy(cudaMults.get(), oscale.data(), oscale.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
      cuda::requantF8Perchannel(out_f32.get(), output, cudaMults.get(),
                                  1, batch_size, p.M, p.N, p.do_relu, false);
      cudaMults.reset();
    }
  } else {
    if (p.do_relu) {
      cuda::doRelu(out_f32.get(), num_out, cuda::DT_F32);
    }
    if (!out_stype.isF32()) {
      cuda::convertType(out_f32.get(), output, num_out, cuda::DT_F32,
                        getCudaType(op.getOutput()));
    } else {
      cudaMemcpy(output, out_f32.get(),
                 num_out * sizeof(float), cudaMemcpyDeviceToDevice);
    }
  }
  out_f32.reset();
  if (p.hdim_is_batch && (p.output_transpose || !module::isCV18xx())) {
    std::vector<int64_t> out_shape(module::getShape(op.getOutput()));
    while (out_shape.size() < 6) {
      out_shape.push_back(1);
    }
    cuda::permute6D(output, raw_output, out_shape[0], out_shape[1], out_shape[2],
      out_shape[3], out_shape[4], out_shape[5], 0, 2, 1, 3, 4, 5, otype_bytes);
    cudaFree(output);
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
  for (size_t b=0; b<p.batch;b++) {
    auto cur_in = (float*)in_f32 + b*batch_elem_left;
    auto cur_right = (float*)right_f32 + b*batch_elem_right;
    auto cur_out = (float*)out_f32 + b*batch_elem_out;
    cuda::mmF32(cur_in, cur_right, cur_out, p.M, p.K, p.N, left_transpose, right_transpose, out_transpose);
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
