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

void py_cuda::cudaMaxPoolWithMaskOp(top::MaxPoolWithMaskOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  auto mask = getCudaData(op.getMask());

  int64_t n, c, ih, iw, oh, ow;
  module::getNCHW(op.getInput(), n, c, ih, iw);
  module::getNCHW(op.getOutput(), n, c, oh, ow);

  auto kernel = module::getI64Array(op.getKernelShape());
  auto stride = module::getI64Array(op.getStrides());
  auto pad = module::getI64Array(op.getPads());

  int kh = kernel->at(0), kw = kernel->at(1);
  int sh = stride->at(0), sw = stride->at(1);
  int ph = pad->at(0), pw = pad->at(1);

  cuda::maxPoolWithMask(input, output, mask, n, c, ih, iw, oh, ow,
                        kh, kw, sh, sw, ph, pw);
}

void py_cuda::cudaMaxPoolWithMaskOp(tpu::MaxPoolWithMaskOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  auto mask = getCudaData(op.getMask());

  int64_t n, c, ih, iw, oh, ow;
  module::getNCHW(op.getInput(), n, c, ih, iw);
  module::getNCHW(op.getOutput(), n, c, oh, ow);

  auto kernel = module::getI64Array(op.getKernelShape());
  auto stride = module::getI64Array(op.getStrides());
  auto pad = module::getI64Array(op.getPads());

  int kh = kernel->at(0), kw = kernel->at(1);
  int sh = stride->at(0), sw = stride->at(1);
  int ph = pad->at(0), pw = pad->at(1);

  auto stype = module::getStorageType(op.getOutput());
  if (stype.isF32()) {
    cuda::maxPoolWithMask(input, output, mask, n, c, ih, iw, oh, ow,
                          kh, kw, sh, sw, ph, pw);
  } else {
    auto num_out = module::getNumElements(op.getOutput());
    auto num_mask = module::getNumElements(op.getMask());
    auto output_f32 = cuda_malloc(num_out * sizeof(float));
    auto mask_f32 = cuda_malloc(num_mask * sizeof(float));
    cuda::maxPoolWithMask(input, output_f32.get(), mask_f32.get(),
                          n, c, ih, iw, oh, ow, kh, kw, sh, sw, ph, pw);
    cuda::convertType(output_f32.get(), output, num_out, cuda::DT_F32, getCudaType(op.getOutput()));
    cuda::convertType(mask_f32.get(), mask, num_mask, cuda::DT_F32, getCudaType(op.getMask()));
  }
}
