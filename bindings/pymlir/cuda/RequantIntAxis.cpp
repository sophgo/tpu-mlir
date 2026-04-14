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
  auto out_stype = module::getStorageType(op.getOutput());
  auto sign = !out_stype.isUnsignedInteger(8);
  int64_t n, c, h, w;
  module::getNCHW(op.getInput(), n, c, h, w);
  bool is_cv18xx = module::isCV18xx();
  auto multipliers = cuda_malloc(shape[1] * sizeof(int32_t));
  auto shifts = cuda_malloc(shape[1] * sizeof(int32_t));
  auto zero_points = cuda_malloc(shape[1] * sizeof(int32_t));
  if (shape.size() == 4) {
    cuda::slice6D(quant, multipliers.get(),
                  shape[0], shape[1], shape[2], shape[3], 1, 1, // in shape
                  0, 0, 0, 0, 0, 0,                             // in offset
                  1, 1, 1, 1, 1, 1,                             // in stride
                  shape[0], shape[1], shape[2], 1, 1, 1,        // out shape
                  sizeof(int32_t));
    cuda::slice6D(quant, shifts.get(),
                  shape[0], shape[1], shape[2], shape[3], 1, 1,
                  0, 0, 0, 1, 0, 0,
                  1, 1, 1, 1, 1, 1,
                  shape[0], shape[1], shape[2], 1, 1, 1,
                  sizeof(int32_t));
    if (is_cv18xx) {
      cuda::RightBitShift(shifts.get(), zero_points.get(), 16, shape[1], sizeof(int32_t));
      cuda::RightBitShift(shifts.get(), shifts.get(), -24, shape[1], sizeof(int32_t));
      cuda::RightBitShift(shifts.get(), shifts.get(), 24, shape[1], sizeof(int32_t));
    } else {
      cuda::slice6D(quant, zero_points.get(),
                    shape[0], shape[1], shape[2], shape[3], 1, 1,
                    0, 0, 0, 2, 0, 0,
                    1, 1, 1, 1, 1, 1,
                    shape[0], shape[1], shape[2], 1, 1, 1,
                    sizeof(int32_t));
    }
  } else if (shape.size() == 5) {
    cuda::slice6D(quant, multipliers.get(),
                  shape[0], shape[1], shape[2], shape[3], shape[4], 1,
                  0, 0, 0, 0, 0, 0,
                  1, 1, 1, 1, 1, 1,
                  shape[0], shape[1], shape[2], shape[3], 1, 1,
                  sizeof(int32_t));
    cuda::slice6D(quant, shifts.get(),
                  shape[0], shape[1], shape[2], shape[3], shape[4], 1,
                  0, 0, 0, 0, 1, 0,
                  1, 1, 1, 1, 1, 1,
                  shape[0], shape[1], shape[2], shape[3], 1, 1,
                  sizeof(int32_t));
    if (is_cv18xx) {
      cuda::RightBitShift(shifts.get(), zero_points.get(), 16, shape[1], sizeof(int32_t));
      cuda::RightBitShift(shifts.get(), shifts.get(), -24, shape[1], sizeof(int32_t));
      cuda::RightBitShift(shifts.get(), shifts.get(), 24, shape[1], sizeof(int32_t));
    } else {
      cuda::slice6D(quant, zero_points.get(),
                    shape[0], shape[1], shape[2], shape[3], shape[4], 1,
                    0, 0, 0, 0, 2, 0,
                    1, 1, 1, 1, 1, 1,
                    shape[0], shape[1], shape[2], shape[3], 1, 1,
                    sizeof(int32_t));
    }
  } else {
    llvm_unreachable("unsupported quant param dim");
  }
  cuda::neg(shifts.get(), shifts.get(), shape[1], cuda::DT_INT32);
  cuda::requant_mode_t rqmode = static_cast<cuda::requant_mode_t>(op.getQuantMode());
  cuda::rounding_mode_t rmode = static_cast<cuda::rounding_mode_t>(
    round_mode_convert(op.getRoundMode()));
  cuda::requantInt8Perchannel(input, output, multipliers.get(), shifts.get(), n,
                              c, h, w, sign, false, zero_points.get(), is_cv18xx,
                              rqmode, rmode);
  multipliers.reset();
  shifts.reset();
  zero_points.reset();
}
