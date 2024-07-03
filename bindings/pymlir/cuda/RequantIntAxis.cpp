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

void py_cuda::cudaRequantIntAxisOp(tpu::RequantIntAxisOp op) {
  void *input = getCudaData(op.getInput());
  void *quant = getCudaData(op.getQuant());
  void *output = getCudaData(op.getOutput());
  auto shape = module::getShape(op.getQuant());
  if (shape.size() != 4) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  auto out_stype = module::getStorageType(op.getOutput());
  auto sign = !out_stype.isUnsignedInteger(8);
  int64_t n, c, h, w;
  module::getNCHW(op.getInput(), n, c, h, w);
  auto multipliers = cuda_malloc(shape[1] * sizeof(int32_t));
  auto shifts = cuda_malloc(shape[1] * sizeof(int32_t));
  cuda::slice4D(quant, multipliers.get(), shape[0], shape[1], shape[2],
                shape[3], 0, 0, 0, 0, 1, 1, 1, 1, shape[0], shape[1], shape[2],
                1, sizeof(int32_t));
  cuda::slice4D(quant, shifts.get(), shape[0], shape[1], shape[2], shape[3], 0,
                0, 0, 1, 1, 1, 1, 1, shape[0], shape[1], shape[2], 1,
                sizeof(int32_t));
  cuda::neg(shifts.get(), shifts.get(), shape[1], cuda::DT_INT32);
  cuda::requantInt8Perchannel(input, output, multipliers.get(), shifts.get(), n,
                              c, h, w, sign);
}
