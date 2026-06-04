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

void py_cuda::cudaCopyOp(top::CopyOp op) {
  auto shape = module::getI64Array(op.getShape());
  auto i_stride = module::getI64Array(op.getInputStride());
  auto o_stride = module::getI64Array(op.getOutputStride());
  int num_dims = shape->size();

  // pad strides/shape to 4D
  int64_t shape_4[4] = {1, 1, 1, 1};
  int64_t i_s[4] = {0, 0, 0, 0};
  int64_t o_s[4] = {0, 0, 0, 0};
  for (int end = num_dims - 1, idx = 3; end >= 0 && idx >= 0; end--, idx--) {
    shape_4[idx] = shape->at(end);
    i_s[idx] = i_stride->at(end);
    o_s[idx] = o_stride->at(end);
  }

  int tbytes = module::getDtypeSize(op.getOutput());
  cuda::bmCopy(getCudaData(op.getInput()), getCudaData(op.getOutput()),
               shape_4[0], shape_4[1], shape_4[2], shape_4[3],
               i_s[0], i_s[1], i_s[2], i_s[3],
               o_s[0], o_s[1], o_s[2], o_s[3], tbytes);
}

void py_cuda::cudaCopyOp(tpu::CopyOp op) {
  auto shape = module::getI64Array(op.getShape());
  auto i_stride = module::getI64Array(op.getInputStride());
  auto o_stride = module::getI64Array(op.getOutputStride());
  int num_dims = shape->size();

  int64_t shape_4[4] = {1, 1, 1, 1};
  int64_t i_s[4] = {0, 0, 0, 0};
  int64_t o_s[4] = {0, 0, 0, 0};
  for (int end = num_dims - 1, idx = 3; end >= 0 && idx >= 0; end--, idx--) {
    shape_4[idx] = shape->at(end);
    i_s[idx] = i_stride->at(end);
    o_s[idx] = o_stride->at(end);
  }

  int tbytes = module::getDtypeSize(op.getOutput());
  cuda::bmCopy(getCudaData(op.getInput()), getCudaData(op.getOutput()),
               shape_4[0], shape_4[1], shape_4[2], shape_4[3],
               i_s[0], i_s[1], i_s[2], i_s[3],
               o_s[0], o_s[1], o_s[2], o_s[3], tbytes);
}
