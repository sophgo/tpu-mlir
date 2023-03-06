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
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"



int64_t top::GatherOp::getFLOPs() { return 0; }

LogicalResult top::GatherOp::init(InferenceParameter &p) { return success(); }
void top::GatherOp::deinit(InferenceParameter &p) {}

LogicalResult top::GatherOp::inference(InferenceParameter &p) {
  const float *src = p.inputs[0];
  const float *inds = p.inputs[1];
  float *dst = p.outputs[0];
  auto num_indices = module::getNumElements(getIndices());
  auto ax = getAxis();
  int64_t outer_dims = 1;
  int64_t inner_dims = 1;
  auto input_shape = module::getShape(getInput());
  for (int i = 0; i < ax; ++i) {
    outer_dims *= input_shape[i];
  }
  for (int i = ax + 1; i < input_shape.size(); ++i) {
    inner_dims *= input_shape[i];
  }

  auto num_elems = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_elems))
  for (int64_t i = 0; i < outer_dims; ++i) {
    for (int64_t j = 0; j < num_indices; ++j) {
      for (int64_t k = 0; k < inner_dims; ++k) {
        int64_t src_idx = (i * input_shape[ax] + inds[j]) * inner_dims + k;
        int64_t dst_idx = (i * num_indices + j) * inner_dims + k;
        dst[dst_idx] = src[src_idx];
      }
    }
  }


  return success();
}

void top::GatherOp::shape_inference() {}
