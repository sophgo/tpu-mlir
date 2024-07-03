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

void py_cuda::cudaGatherOp(tpu::GatherOp op) {
  auto in = op.getIndices();
  auto embed = op.getInput();
  auto out = op.getOutput();
  void *in_ptr = getCudaData(in);
  void *embed_ptr = getCudaData(embed);
  void *out_ptr = getCudaData(out);
  auto in_type = getCudaType(in);
  auto out_type = getCudaType(out);
  int num_in = module::getNumElements(in);
  int num_embed = module::getNumElements(embed);
  auto embed_shape = module::getShape(embed);
  int embed_dim = embed_shape[0];
  int inner_dim = num_embed / embed_dim;
  cuda::gather(in_ptr, embed_ptr, out_ptr, num_in, embed_dim, inner_dim,
               in_type, out_type);
}
