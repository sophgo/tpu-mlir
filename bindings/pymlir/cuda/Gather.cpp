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
  // no int8 implementation, f32, f16, bf16
  auto in = op.getIndices();
  auto embed = op.getInput();
  auto out = op.getOutput();
  void *in_ptr = getCudaData(in);
  void *embed_ptr = getCudaData(embed);
  void *out_ptr = getCudaData(out);
  auto in_type = getCudaType(in);
  auto out_type = getCudaType(out);
  auto ax = op.getAxis();
  if (ax < 0) {
    ax += module::getShape(embed).size();
  }
  int num_in = module::getNumElements(in);
  auto embed_shape = module::getShape(embed);
  int outer_dims = 1;
  for (int i=0; i<ax; i++) {
    outer_dims *= embed_shape[i];
  }
  int inner_dims = 1;
  for (int i=ax+1;i<embed_shape.size();i++) {
    inner_dims *= embed_shape[i];
  }
  cuda::cudaGather(in_ptr, embed_ptr, out_ptr, num_in, outer_dims, embed_shape[ax], inner_dims,
               in_type, out_type);
}

void py_cuda::cudaGatherOp(top::GatherOp op) {
  auto in = op.getIndices();
  auto embed = op.getInput();
  auto out = op.getOutput();
  void *in_ptr = getCudaData(in);
  void *embed_ptr = getCudaData(embed);
  void *out_ptr = getCudaData(out);
  auto in_type = getCudaType(in);
  auto out_type = getCudaType(out);
  auto ax = op.getAxis();
  if (ax < 0) {
    ax += module::getShape(embed).size();
  }
  int num_in = module::getNumElements(in);
  auto embed_shape = module::getShape(embed);
  int outer_dims = 1;
  for (int i=0; i<ax; i++) {
    outer_dims *= embed_shape[i];
  }
  int inner_dims = 1;
  for (int i=ax+1;i<embed_shape.size();i++) {
    inner_dims *= embed_shape[i];
  }
  cuda::cudaGather(in_ptr, embed_ptr, out_ptr, num_in, outer_dims, embed_shape[ax], inner_dims,
               in_type, out_type);
}

