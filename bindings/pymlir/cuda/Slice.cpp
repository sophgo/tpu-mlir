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

void py_cuda::cudaSliceOp(tpu::SliceOp op) {
  void *input = getCudaData(op.getInput());
  void *output = getCudaData(op.getOutput());
  auto p = op.parseParam();
  auto in_shape = p.is_4;
  auto out_shape = p.os_4;
  auto offset = p.offset_4;
  auto steps = p.step_4;
  while (in_shape.size() < 6) {
    in_shape.push_back(1);
    out_shape.push_back(1);
    offset.push_back(0);
    steps.push_back(1);
  }
  for (int i=0;i<offset.size();i++) {
    if (offset[i] <0)
      offset[i] = in_shape[i] + offset[i];
  }
  cuda::slice6D(input, output, in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                in_shape[4], in_shape[5],
                offset[0], offset[1], offset[2], offset[3], offset[4], offset[5],
                steps[0], steps[1], steps[2], steps[3], steps[4], steps[5],
                out_shape[0], out_shape[1], out_shape[2], out_shape[3], out_shape[4], out_shape[5],
                module::getDtypeSize(op.getOutput()));
}

void py_cuda::cudaSliceOp(top::SliceOp op) {
  void *input = getCudaData(op.getInput());
  void *output = getCudaData(op.getOutput());
  std::vector<int64_t> in_shape = module::getShape(op.getInput());
  std::vector<int64_t> out_shape = module::getShape(op.getOutput());
  auto offset = module::getI64Array(op.getOffset());
  auto steps = module::getI64Array(op.getSteps());
  std::vector<int32_t> offset_v(offset->begin(), offset->end());
  std::vector<int32_t> steps_v(steps->begin(), steps->end());
  if (in_shape.size() != offset_v.size() || in_shape.size() != steps_v.size())
    UNREACHABLE_OP("input shape/offset/steps size not equal", op);

  while(in_shape.size() < 6) {
    in_shape.push_back(1);
    out_shape.push_back(1);
    offset_v.push_back(0);
    steps_v.push_back(1);
  }
  for (int i=0;i<offset_v.size();i++) {
    if (offset_v[i] <0)
      offset_v[i] = in_shape[i] + offset_v[i];
  }
  cuda::slice6D(input, output, in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                in_shape[4], in_shape[5],
                offset_v[0], offset_v[1], offset_v[2], offset_v[3], offset_v[4], offset_v[5],
                steps_v[0], steps_v[1], steps_v[2], steps_v[3], steps_v[4], steps_v[5],
                out_shape[0], out_shape[1], out_shape[2], out_shape[3], out_shape[4], out_shape[5],
                module::getDtypeSize(op.getOutput()));
}
