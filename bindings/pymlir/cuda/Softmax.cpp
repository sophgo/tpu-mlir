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
  if (is_cv18xx) {
    auto table0 = getCudaData(op.getTable());
    auto table1 = getCudaData(op.getSlopeTable());
    auto table2 = getCudaData(op.getReciprocalTable());
    auto table3 = getCudaData(op.getReciprocalMantissaTable());
    auto buffer = cuda_malloc(outer_dim * inner_dim * sizeof(uint16_t));
    float scale = BF16(256.0 / 30.0); // EXP_BF16_LUT_RANGE
    float offset = 0.0f;
    cuda::cvSoftmax(input, buffer.get(), output, table0, table1, table2, table3,
                    outer_dim, axis_dim, inner_dim, scale, offset, op.getLog());
  } else {
    UNREACHABLE_OP("Not Implemented", op);
  }
}
