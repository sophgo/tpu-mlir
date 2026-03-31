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
  auto shape = std::vector<int64_t>(module::getShape(op.getQuant()));
  if (shape.size() != 4) {
    // 5 for 1d conv, not support, 4 for 2d deconv, shape is 1, oc, 1, 2 or 3
    // assume zp 0
    UNREACHABLE_OP("quant shape size not equal to 4", op);
  }
  while (shape.size() < 6) {
    shape.push_back(1);
  }
  auto out_stype = module::getStorageType(op.getOutput());
  auto sign = !out_stype.isUnsignedInteger(8);
  int64_t n, c, h, w;
  module::getNCHW(op.getInput(), n, c, h, w);
  auto multipliers = cuda_malloc(shape[1] * sizeof(int32_t));
  auto shifts = cuda_malloc(shape[1] * sizeof(int32_t));
  cuda::slice6D(quant, multipliers.get(), shape[0], shape[1], shape[2],
                shape[3], shape[4], shape[5], 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, shape[0], shape[1], shape[2],
                1, shape[4], shape[5], sizeof(int32_t));
  cuda::slice6D(quant, shifts.get(), shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], 0,
                0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, shape[0], shape[1], shape[2], 1, shape[4], shape[5],
                sizeof(int32_t));
  cuda::neg(shifts.get(), shifts.get(), shape[1], cuda::DT_INT32);
  cuda::requantInt8Perchannel(input, output, multipliers.get(), shifts.get(), n,
                              c, h, w, sign);
}
