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

void py_cuda::cudaDtypeCastOp(top::DtypeCastOp op) {
  auto num = module::getNumElements(op.getOutput());
  cuda::convertType(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                    num, getCudaType(op.getInput()), getCudaType(op.getOutput()));
}

void py_cuda::cudaDtypeCastOp(tpu::DtypeCastOp op) {
  auto num = module::getNumElements(op.getOutput());
  cuda::convertType(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                    num, getCudaType(op.getInput()), getCudaType(op.getOutput()));
}
