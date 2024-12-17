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

void py_cuda::cudaTileOp(tpu::TileOp op) {
  void *input = getCudaData(op.getInput());
  void *output = getCudaData(op.getOutput());
  std::vector<int64_t> in_shape = module::getShape(op.getInput());
  std::vector<int64_t> out_shape = module::getShape(op.getOutput());
  auto num_dims = out_shape.size();
  if (num_dims > 4) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  if (num_dims < 4) {
    for (int i = 0; i < 4 - num_dims; i++) {
      in_shape.push_back(1);
      out_shape.push_back(1);
    }
  }

  cuda::tile4D(input, output, in_shape[0], in_shape[1], in_shape[2],
               in_shape[3], out_shape[0], out_shape[1], out_shape[2],
               out_shape[3], module::getDtypeSize(op.getOutput()));
}
