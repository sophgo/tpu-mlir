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

void py_cuda::cudaRMSNormOp(top::RMSNormOp op) {
  auto shape = module::getShape(op.getInput());
  int dims = shape.size();
  int outer_dim = 1;
  for (int i = 0; i < dims - 1; ++i) outer_dim *= shape[i];
  int inner_dim = shape[dims - 1];
  float eps = op.getEps().convertToDouble();

  void *gamma_ptr = module::isNone(op.getGamma())
                        ? nullptr
                        : getCudaData(op.getGamma());

  cuda::bmRMSNorm(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                   outer_dim, inner_dim, gamma_ptr, eps);
}

void py_cuda::cudaRMSNormOp(tpu::RMSNormOp op) {
  auto shape = module::getShape(op.getInput());
  int dims = shape.size();
  int outer_dim = 1;
  for (int i = 0; i < dims - 1; ++i) outer_dim *= shape[i];
  int inner_dim = shape[dims - 1];
  float eps = op.getEps().convertToDouble();

  void *gamma_ptr = module::isNone(op.getGamma())
                        ? nullptr
                        : getCudaData(op.getGamma());

  cuda::bmRMSNorm(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                   outer_dim, inner_dim, gamma_ptr, eps);
}
