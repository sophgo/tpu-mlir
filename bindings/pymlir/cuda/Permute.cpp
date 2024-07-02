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

void py_cuda::cudaPermuteOp(tpu::PermuteOp op) {
  auto p = op.parseParam();
  if (p.in_shape_fix.size() != 4) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  auto src = getCudaData(op.getInput());
  auto dst = getCudaData(op.getOutput());
  std::vector<int> shape(p.in_shape_fix.begin(), p.in_shape_fix.end());
  std::vector<int> order(p.order_fix.begin(), p.order_fix.end());
  int tbytes = module::getDtypeSize(op.getInput());
  cuda::permute4D(src, dst, shape[0], shape[1], shape[2], shape[3], order[0],
                  order[1], order[2], order[3], tbytes);
}
