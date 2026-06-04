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

void py_cuda::cudaEmbDenseBwdOp(top::EmbDenseBwdOp op) {
  auto grad_shape = module::getShape(op.getGradOutput());
  auto idx_shape = module::getShape(op.getIndices());
  int num_weights = grad_shape[0];
  int embed_dim = module::getNumElements(op.getGradOutput()) / num_weights;
  int batch_size = idx_shape[0];
  cuda::bmEmbDenseBwd(getCudaData(op.getGradOutput()),
                       getCudaData(op.getIndices()),
                       getCudaData(op.getOutput()),
                       batch_size, embed_dim);
}

void py_cuda::cudaEmbDenseBwdOp(tpu::EmbDenseBwdOp op) {
  auto grad_shape = module::getShape(op.getGradOutput());
  auto idx_shape = module::getShape(op.getIndices());
  int num_weights = grad_shape[0];
  int embed_dim = module::getNumElements(op.getGradOutput()) / num_weights;
  int batch_size = idx_shape[0];
  cuda::bmEmbDenseBwd(getCudaData(op.getGradOutput()),
                       getCudaData(op.getIndices()),
                       getCudaData(op.getOutput()),
                       batch_size, embed_dim);
}
