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

void py_cuda::cudaRangeOp(top::RangeOp op) {
  int num = module::getNumElements(op.getOutput());

  float h_start = 0.0f;
  if (!module::isNone(op.getStart()))
    cuda::copyToHost(&h_start, getCudaData(op.getStart()),
                     getCudaType(op.getStart()));

  float h_delta = 1.0f;
  if (!module::isNone(op.getDelta()))
    cuda::copyToHost(&h_delta, getCudaData(op.getDelta()),
                     getCudaType(op.getDelta()));

  cuda::bmRange(getCudaData(op.getOutput()), h_start, h_delta, num);
}

void py_cuda::cudaRangeOp(tpu::RangeOp op) {
  int num = module::getNumElements(op.getOutput());

  float h_start = 0.0f;
  if (!module::isNone(op.getStart()))
    cuda::copyToHost(&h_start, getCudaData(op.getStart()),
                     getCudaType(op.getStart()));

  float h_delta = 1.0f;
  if (!module::isNone(op.getDelta()))
    cuda::copyToHost(&h_delta, getCudaData(op.getDelta()),
                     getCudaType(op.getDelta()));

  cuda::bmRange(getCudaData(op.getOutput()), h_start, h_delta, num);
}
