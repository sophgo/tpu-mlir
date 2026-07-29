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

void py_cuda::cudaHardSigmoidOp(top::HardSigmoidOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  auto num = module::getNumElements(op.getOutput());
  float alpha = static_cast<float>(op.getAlpha().convertToDouble());
  float beta = static_cast<float>(op.getBeta().convertToDouble());
  cuda::bmHardSigmoid(input, output, num, alpha, beta);
}
