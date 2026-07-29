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

void py_cuda::cudaExpandOp(top::ExpandOp op) {
  auto in_shape = module::getShape(op.getInput());
  auto out_shape = module::getShape(op.getOutput());
  int dim_in = in_shape.size();
  int dim_out = out_shape.size();
  int dim_pad = dim_out - dim_in;

  int in_n = 1, in_c = 1, in_h = 1, in_w = 1;
  int idx = dim_pad;
  for (int i = 0; i < dim_in && idx < 4; i++, idx++) {
    int64_t val = in_shape[i];
    if (idx == 0) in_n = val;
    else if (idx == 1) in_c = val;
    else if (idx == 2) in_h = val;
    else if (idx == 3) in_w = val;
  }

  int out_n = 1, out_c = 1, out_h = 1, out_w = 1;
  for (int i = 0; i < dim_out && i < 4; i++) {
    int64_t val = out_shape[i];
    if (i == 0) out_n = val;
    else if (i == 1) out_c = val;
    else if (i == 2) out_h = val;
    else if (i == 3) out_w = val;
  }

  cuda::bmExpand(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                 in_n, in_c, in_h, in_w, out_n, out_c, out_h, out_w);
}
