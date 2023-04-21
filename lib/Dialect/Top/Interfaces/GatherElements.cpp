//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

int64_t top::GatherElementsOp::getFLOPs() { return 0; }

LogicalResult top::GatherElementsOp::init(InferenceParameter &p) {
  return success();
}
void top::GatherElementsOp::deinit(InferenceParameter &p) {}

static inline void gather_dim3_2(float *dst, const float *src, const int *idx,
                                 const int *shape, int *org_shape) {
  int shape_1_2 = org_shape[1] * org_shape[2];
  for (int i = 0; i < shape[0]; ++i) {
    int idx_i = i * shape_1_2;
    for (int j = 0; j < shape[1]; ++j) {
      int idx_j = idx_i + j * org_shape[2];
      for (int k = 0; k < shape[2]; ++k) {
        *dst = src[idx_j + *idx];
        ++dst;
        ++idx;
      }
    }
  }
}

LogicalResult top::GatherElementsOp::inference(InferenceParameter &p) {
  const float *src = p.inputs[0];
  const float *inds = p.inputs[1];
  // convert float* inds to int*
  // int *inds_int = (int *)inds;
  int *inds_int =
      (int *)malloc(module::getNumElements(getIndices()) * sizeof(int));
  for (int i = 0; i < module::getNumElements(getIndices()); ++i) {
    inds_int[i] = (int)inds[i];
  }
  float *dst = p.outputs[0];
  auto ax = getAxis();
  auto dim = module::getShape(getInput()).size();
  int shape[dim];
  int org_shape[dim];
  module::getGlobalShape(getInput(), org_shape, dim);
  module::getGlobalShape(getIndices(), shape, dim);
  switch (dim) {
  case 3:
    if (ax == 2)
      gather_dim3_2(dst, src, inds_int, shape, org_shape);
    break;
  default:
    llvm_unreachable("Not implemented yet");
  }

  return success();
}

void top::GatherElementsOp::shape_inference() {
  auto indices_shape = module::getShape(getIndices());
  module::setShapeOrVerify(getOutput(), indices_shape);
}
