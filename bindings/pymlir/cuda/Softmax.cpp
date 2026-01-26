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

void py_cuda::cudaSoftmaxOp(tpu::SoftmaxOp op) {
  auto axis_ = op.getAxis();
  auto input_shape = module::getShape(op.getInput());
  // auto out_type = module::getStorageType(getOutput());
  // auto num_elem = module::getNumElements(getOutput());
  bool is_cv18xx = module::isCV18xx();

  int outer_dim = 1;
  for (int i = 0; i < axis_; i++) {
    outer_dim *= input_shape[i];
  }

  int inner_dim = 1;
  for (int i = axis_ + 1; i < input_shape.size(); i++) {
    inner_dim *= input_shape[i];
  }

  int axis_dim = input_shape[axis_];
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  if (is_cv18xx) { // only bf16
    auto table0 = getCudaData(op.getTable());
    auto table1 = getCudaData(op.getSlopeTable());
    auto table2 = getCudaData(op.getReciprocalTable());
    auto table3 = getCudaData(op.getReciprocalMantissaTable());
    auto buffer = cuda_malloc(outer_dim * inner_dim * sizeof(uint16_t));
    float scale = BF16(256.0 / 30.0); // EXP_BF16_LUT_RANGE
    float offset = 0.0f;
    cuda::cvSoftmax(input, buffer.get(), output, table0, table1, table2, table3,
                    outer_dim, axis_dim, inner_dim, scale, offset, op.getLog());
    buffer.reset();
  } else {
    auto buffer = cuda_malloc(outer_dim * inner_dim * sizeof(float));
    auto f32_output = cuda_malloc(outer_dim*inner_dim*axis_dim*sizeof(float));
    if (module::getStorageType(op.getInput()).isF32()) {
      cuda::bmSoftmax(input, buffer.get(), output,
                    outer_dim, axis_dim, inner_dim, op.getLog());
    } else if (!module::isUniformQuantized(op.getInput())) {
      auto f32_input = newCudaData(op.getInput(), cuda::DT_F32);
      cuda::bmSoftmax(f32_input.get(), buffer.get(), f32_output.get(),
                    outer_dim, axis_dim, inner_dim, op.getLog());
      cuda::convertType(f32_output.get(), output, outer_dim*inner_dim*axis_dim, cuda::DT_F32,
                      getCudaType(op.getOutput()));
      f32_input.reset();
    } else {
      auto exp_table = getCudaData(op.getTable());
      auto f32_input = newCudaData(op.getInput(), cuda::DT_F32);
      auto out_stype = module::getUniformQuantizedType(op.getOutput());
      auto out_scale = out_stype.getScale();
      auto out_zp = out_stype.getZeroPoint();
      cuda::bmSoftmax(f32_input.get(), buffer.get(), f32_output.get(),
                    outer_dim, axis_dim, inner_dim, exp_table, 
                    out_scale, out_zp);
      cuda::f32ScaleToInt8(f32_output.get(), output, 1.0, outer_dim * inner_dim * axis_dim, !module::getStorageType(op.getOutput()).isUnsignedInteger(), cuda::RD_HALF_AWAY_FROM_ZERO);
      f32_input.reset();
    }
    f32_output.reset();
    buffer.reset();
  }
}

void py_cuda::cudaSoftmaxOp(top::SoftmaxOp op) {
  auto axis_ = op.getAxis();
  auto input_shape = module::getShape(op.getInput());

  if (axis_ == 1 && input_shape.size() == 4) {
    cudnnSoftmaxMode_t mode;
    mode = CUDNN_SOFTMAX_MODE_CHANNEL;
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
    cudnnSoftmaxAlgorithm_t algorithm = CUDNN_SOFTMAX_ACCURATE;

    float alpha = 1.0f, beta = 0.0f;
    cudnnSoftmaxForward(cudnn_, algorithm, mode,
                        &alpha, input_desc, getCudaData(op.getInput()),
                        &beta, output_desc, getCudaData(op.getOutput()));
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
  } else {
    int outer_dim = 1;
    for (int i = 0; i < axis_; i++) {
      outer_dim *= input_shape[i];
    }

    int inner_dim = 1;
    for (int i = axis_ + 1; i < input_shape.size(); i++) {
      inner_dim *= input_shape[i];
    }

    int axis_dim = input_shape[axis_];
    auto buffer = cuda_malloc(outer_dim * inner_dim * sizeof(float));
    cuda::bmSoftmax(getCudaData(op.getInput()), buffer.get(), getCudaData(op.getOutput()),
                  outer_dim, axis_dim, inner_dim, op.getLog());
  }
}
