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

void py_cuda::cudaClipOp(tpu::ClipOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  auto num = module::getNumElements(op.getOutput());
  float min_v = op.getMin().convertToDouble();
  float max_v = op.getMax().convertToDouble();
  cuda::bmClip(input, output, num, min_v, max_v);
}

void py_cuda::cudaClipOp(top::ClipOp op) {
  auto input = getCudaData(op.getInputs());
  auto output = getCudaData(op.getOutput());
  auto num = module::getNumElements(op.getOutput());
  float min_v = op.getMin().convertToDouble();
  float max_v = op.getMax().convertToDouble();
  cuda::bmClip(input, output, num, min_v, max_v);
}
