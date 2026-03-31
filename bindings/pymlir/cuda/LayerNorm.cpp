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

void py_cuda::cudaLayerNormOp(tpu::LayerNormOp op) {
  const int axis_ = op.getAxis();
  const float eps_ = op.getEps().convertToDouble();
  const auto input_shape = module::getShape(op.getInput());
  int outer_dim = 1;
  for (int i = 0; i < axis_; i++) {
    outer_dim *= input_shape[i];
  }

  int inner_dim = 1;
  for (int i = axis_; i < input_shape.size(); i++) {
    inner_dim *= input_shape[i];
  }

  const bool have_weight = !op.getWeight().getType().isa<mlir::NoneType>();
  const bool have_bias = !op.getBias().getType().isa<mlir::NoneType>();

  if (module::getStorageType(op.getInput()).isF32()) {
    cuda::bmLayerNorm(getCudaData(op.getInput()), getCudaData(op.getOutput()), outer_dim,
                      inner_dim, have_weight ? getCudaData(op.getWeight()) : nullptr,
                      have_bias ? getCudaData(op.getBias()) : nullptr, eps_, getCudaType(op.getInput()));
  } else {
    auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
    auto output_f32 = cuda_malloc(outer_dim * inner_dim * sizeof(float));
    auto weight_f32 = have_weight? newCudaData(op.getWeight(), cuda::DT_F32): nullptr;
    auto bias_f32 = have_bias? newCudaData(op.getBias(), cuda::DT_F32): nullptr;
    cuda::bmLayerNorm(input_f32.get(), output_f32.get(), outer_dim,
                      inner_dim, have_weight ? weight_f32.get() : nullptr,
                      have_bias ? bias_f32.get() : nullptr, eps_, getCudaType(op.getInput()));
    cuda::convertType(output_f32.get(), getCudaData(op.getOutput()),
                      outer_dim * inner_dim, cuda::DT_F32,
                      getCudaType(op.getOutput()));
    input_f32.reset();
    output_f32.reset();
    if (have_weight)
      weight_f32.reset();
    if (have_bias)
      bias_f32.reset();
  }
}

void py_cuda::cudaLayerNormOp(top::LayerNormOp op) {
  const int axis_ = op.getAxis();
  const float eps_ = op.getEps().convertToDouble();
  const auto input_shape = module::getShape(op.getInput());
  int outer_dim = 1;
  for (int i = 0; i < axis_; i++) {
    outer_dim *= input_shape[i];
  }

  int inner_dim = 1;
  for (int i = axis_; i < input_shape.size(); i++) {
    inner_dim *= input_shape[i];
  }

  const bool have_weight = !op.getWeight().getType().isa<mlir::NoneType>();
  const bool have_bias = !op.getBias().getType().isa<mlir::NoneType>();

  cuda::bmLayerNorm(getCudaData(op.getInput()), getCudaData(op.getOutput()), outer_dim,
                    inner_dim, have_weight ? getCudaData(op.getWeight()) : nullptr,
                    have_bias ? getCudaData(op.getBias()) : nullptr, eps_, getCudaType(op.getInput()));
}
