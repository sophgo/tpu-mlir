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
