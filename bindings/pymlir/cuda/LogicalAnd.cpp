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

void py_cuda::cudaLogicalAndOp(top::LogicalAndOp op) {
  auto lshape = module::getShape(op.getInputs()[0]);
  auto rshape = module::getShape(op.getInputs()[1]);
  auto oshape = module::getShape(op.getOutput());

  auto to4d = [](const std::vector<int64_t> &s, int &n, int &c, int &h, int &w) {
    n = c = h = w = 1;
    int sz = s.size();
    if (sz >= 4) w = s[sz-1], h = s[sz-2], c = s[sz-3], n = s[sz-4];
    else if (sz == 3) w = s[2], h = s[1], c = s[0];
    else if (sz == 2) w = s[1], h = 1, c = s[0];
    else if (sz == 1) w = s[0];
  };

  int ln, lc, lh, lw, rn, rc, rh, rw, on, oc, oh, ow;
  to4d(lshape, ln, lc, lh, lw);
  to4d(rshape, rn, rc, rh, rw);
  to4d(oshape, on, oc, oh, ow);

  int total = on * oc * oh * ow;
  auto out_f32 = cuda_malloc(total * sizeof(float));

  cuda::bmLogicalAnd(getCudaData(op.getInputs()[0]),
                      getCudaData(op.getInputs()[1]),
                      out_f32.get(),
                      ln, lc, lh, lw, rn, rc, rh, rw, on, oc, oh, ow);

  CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), out_f32.get(),
                        total * sizeof(float), cudaMemcpyDeviceToDevice));
}

void py_cuda::cudaLogicalAndOp(tpu::LogicalAndOp op) {
  auto lshape = module::getShape(op.getInputs()[0]);
  auto rshape = module::getShape(op.getInputs()[1]);
  auto oshape = module::getShape(op.getOutput());

  auto to4d = [](const std::vector<int64_t> &s, int &n, int &c, int &h, int &w) {
    n = c = h = w = 1;
    int sz = s.size();
    if (sz >= 4) w = s[sz-1], h = s[sz-2], c = s[sz-3], n = s[sz-4];
    else if (sz == 3) w = s[2], h = s[1], c = s[0];
    else if (sz == 2) w = s[1], h = 1, c = s[0];
    else if (sz == 1) w = s[0];
  };

  int ln, lc, lh, lw, rn, rc, rh, rw, on, oc, oh, ow;
  to4d(lshape, ln, lc, lh, lw);
  to4d(rshape, rn, rc, rh, rw);
  to4d(oshape, on, oc, oh, ow);

  int total = on * oc * oh * ow;
  auto out_f32 = cuda_malloc(total * sizeof(float));

  cuda::bmLogicalAnd(getCudaData(op.getInputs()[0]),
                      getCudaData(op.getInputs()[1]),
                      out_f32.get(),
                      ln, lc, lh, lw, rn, rc, rh, rw, on, oc, oh, ow);

  CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), out_f32.get(),
                        total * sizeof(float), cudaMemcpyDeviceToDevice));
}
