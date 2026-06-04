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

void py_cuda::cudaLayerNormTrainOp(top::LayerNormTrainOp op) {
  const auto in_shape = module::getShape(op.getInput());
  int axis = op.getAxis();
  if (axis < 0) axis += in_shape.size();
  float eps = op.getEps().convertToDouble();

  int outer_dim = 1;
  for (int i = 0; i < axis; i++) outer_dim *= in_shape[i];
  int inner_dim = 1;
  for (int i = axis; i < (int)in_shape.size(); i++) inner_dim *= in_shape[i];

  bool have_w = !op.getWeight().getType().isa<mlir::NoneType>();
  bool have_b = !op.getBias().getType().isa<mlir::NoneType>();
  void *w = have_w ? getCudaData(op.getWeight()) : nullptr;
  void *b = have_b ? getCudaData(op.getBias()) : nullptr;

  cuda::bmLayerNormTrain(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                          getCudaData(op.getMean()), getCudaData(op.getVariance()),
                          w, b, outer_dim, inner_dim, eps);
}

void py_cuda::cudaLayerNormTrainOp(tpu::LayerNormTrainOp op) {
  const auto in_shape = module::getShape(op.getInput());
  int axis = op.getAxis();
  if (axis < 0) axis += in_shape.size();
  float eps = op.getEps().convertToDouble();

  int outer_dim = 1;
  for (int i = 0; i < axis; i++) outer_dim *= in_shape[i];
  int inner_dim = 1;
  for (int i = axis; i < (int)in_shape.size(); i++) inner_dim *= in_shape[i];

  bool have_w = !op.getWeight().getType().isa<mlir::NoneType>();
  bool have_b = !op.getBias().getType().isa<mlir::NoneType>();
  void *w = have_w ? getCudaData(op.getWeight()) : nullptr;
  void *b = have_b ? getCudaData(op.getBias()) : nullptr;

  cuda::bmLayerNormTrain(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                          getCudaData(op.getMean()), getCudaData(op.getVariance()),
                          w, b, outer_dim, inner_dim, eps);
}
