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
  cuda::slice4D(input, output, p.is_4[0], p.is_4[1], p.is_4[2], p.is_4[3],
                p.offset_4[0], p.offset_4[1], p.offset_4[2], p.offset_4[3],
                p.step_4[0], p.step_4[1], p.step_4[2], p.step_4[3], p.os_4[0],
                p.os_4[1], p.os_4[2], p.os_4[3],
                module::getDtypeSize(op.getOutput()));
}

void py_cuda::cudaSliceOp(top::SliceOp op) {
  void *input = getCudaData(op.getInput());
  void *output = getCudaData(op.getOutput());
  if (module::getShape(op.getInput()).size() > 4)
    UNREACHABLE_OP("shape size larger than 4 is not implemented", op);

  auto offset = module::getI64Array(op.getOffset());
  auto steps = module::getI64Array(op.getSteps());
  std::vector<int32_t> offset_v(offset->begin(), offset->end());
  std::vector<int32_t> steps_v(steps->begin(), steps->end());
  while(offset_v.size() < 4)
    offset_v.push_back(0);
  while(steps_v.size() < 4)
    steps_v.push_back(1);
  int64_t n_i, c_i, h_i, w_i, n_o, c_o, h_o, w_o;
  module::getNCHW(op.getInput(), n_i, c_i, h_i, w_i);
  module::getNCHW(op.getOutput(), n_o, c_o, h_o, w_o);
  cuda::slice4D(input, output, n_i, c_i, h_i, w_i,
                offset_v[0], offset_v[1], offset_v[2], offset_v[3],
                steps_v[0], steps_v[1], steps_v[2], steps_v[3],
                n_o, c_o, h_o, w_o,
                module::getDtypeSize(op.getOutput()));
}
