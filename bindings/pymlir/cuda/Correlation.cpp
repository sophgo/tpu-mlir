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

void py_cuda::cudaCorrelationOp(top::CorrelationOp op) {
  auto inputs = op.getInputs();
  auto left = getCudaData(inputs[0]);
  auto right = getCudaData(inputs[1]);
  auto output = getCudaData(op.getOutput());

  int64_t max_disp = op.getMaxDisp();
  int64_t num_groups = op.getNumGroups();

  auto lhs_shape = module::getShape(inputs[0]);
  int64_t ic = lhs_shape[1] / num_groups;
  int64_t ih = lhs_shape[2];
  int64_t iw = lhs_shape[3];

  cuda::bmCorrelation(left, right, output, max_disp, num_groups, ic, ih, iw);
}

void py_cuda::cudaCorrelationOp(tpu::CorrelationOp op) {
  auto inputs = op.getInputs();
  auto left = getCudaData(inputs[0]);
  auto right = getCudaData(inputs[1]);
  auto output = getCudaData(op.getOutput());

  int64_t max_disp = op.getMaxDisp();
  int64_t num_groups = op.getNumGroups();

  auto lhs_shape = module::getShape(inputs[0]);
  int64_t ic = lhs_shape[1] / num_groups;
  int64_t ih = lhs_shape[2];
  int64_t iw = lhs_shape[3];

  cuda::bmCorrelation(left, right, output, max_disp, num_groups, ic, ih, iw);
}
