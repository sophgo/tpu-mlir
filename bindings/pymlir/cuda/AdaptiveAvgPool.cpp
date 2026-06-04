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

void py_cuda::cudaAdaptiveAvgPoolOp(top::AdaptiveAvgPoolOp op) {
  auto input = op.getInput();
  auto output = op.getOutput();
  auto in_shape = module::getShape(input);
  auto out_size = module::getI64Array(op.getOutputSize());

  int n = in_shape[0], c = in_shape[1];
  int ih = in_shape[2], iw = in_shape[3];
  int oh = (int)out_size->at(0), ow = (int)out_size->at(1);

  cuda::bmAdaptiveAvgPool2D(getCudaData(input), getCudaData(output),
                             n, c, ih, iw, oh, ow);
}
