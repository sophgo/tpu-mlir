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

void py_cuda::cudaSwapDimInnerOp(tpu::SwapDimInnerOp op) {
  void *input = getCudaData(op.getInput());
  void *output = getCudaData(op.getOutput());
  std::vector<int64_t> in_shape = module::getShape(op.getInput());
  auto offset = module::getI64Array(op.getOffset());
  std::vector<int32_t> offset_v(offset->begin(), offset->end());
  if (in_shape.size() != offset_v.size())
    UNREACHABLE_OP("input shape/offset/steps size not equal", op);

  while(in_shape.size() < 6) {
    in_shape.push_back(1);
    offset_v.push_back(0);
  }
  for (int i=0;i<offset_v.size();i++) {
    if (offset_v[i] <0)
      offset_v[i] = in_shape[i] + offset_v[i];
  }
  cuda::swapDimInner6D(input, output, in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                in_shape[4], in_shape[5],
                offset_v[0], offset_v[1], offset_v[2], offset_v[3], offset_v[4], offset_v[5],
                module::getDtypeSize(op.getOutput()));
}

void py_cuda::cudaSwapDimInnerOp(top::SwapDimInnerOp op) {
  void *input = getCudaData(op.getInput());
  void *output = getCudaData(op.getOutput());
  std::vector<int64_t> in_shape = module::getShape(op.getInput());
  auto offset = module::getI64Array(op.getOffset());
  std::vector<int32_t> offset_v(offset->begin(), offset->end());
  if (in_shape.size() != offset_v.size())
    UNREACHABLE_OP("input shape/offset/steps size not equal", op);

  while(in_shape.size() < 6) {
    in_shape.push_back(1);
    offset_v.push_back(0);
  }
  for (int i=0;i<offset_v.size();i++) {
    if (offset_v[i] <0)
      offset_v[i] = in_shape[i] + offset_v[i];
  }
  cuda::swapDimInner6D(input, output, in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                in_shape[4], in_shape[5],
                offset_v[0], offset_v[1], offset_v[2], offset_v[3], offset_v[4], offset_v[5],
                module::getDtypeSize(op.getOutput()));
}
