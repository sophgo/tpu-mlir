//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"

int64_t top::GatherElementsOp::getFLOPs() { return 0; }

LogicalResult top::GatherElementsOp::init(InferenceParameter &p) {
  return success();
}
void top::GatherElementsOp::deinit(InferenceParameter &p) {}

// support dim <= 4
static inline void gather_dim4(float *dst, const float *src,
                               const float *indices, const int *indices_shape,
                               int *src_shape, int src_dim, int axis, int dst_dim) {

  int indices_shape4[4] = {1, 1, 1, 1};
  int src_shape4[4] = {1, 1, 1, 1};

  for (int i = 0; i < dst_dim; ++i) {
    indices_shape4[3 - i] = indices_shape[dst_dim - i - 1];
  }

  for (int i = 0; i < src_dim; ++i) {
    src_shape4[3 - i] = src_shape[src_dim - i - 1];
  }

  int div_src_shape = 4 - src_dim;

  int chw = src_shape4[1] * src_shape4[2] * src_shape4[3];
  int hw = src_shape4[2] * src_shape4[3];
  int w = src_shape4[3];
  int chw_arr[3] = {chw, hw ,w};
  for (int i = 0; i < div_src_shape; ++ i){
    chw_arr[i] = 0;
  }
  axis += div_src_shape;

  for (int i = 0; i < indices_shape4[0]; ++i) {
    for (int j = 0; j < indices_shape4[1]; ++j) {
      for (int k = 0; k < indices_shape4[2]; ++k) {
        for (int l = 0; l < indices_shape4[3]; ++l) {
          int idx4[4] = {i, j, k, l};
          idx4[axis] = (int)(*indices);
          int val_idx = idx4[0] * chw_arr[0] + idx4[1] * chw_arr[1] + idx4[2] * chw_arr[2] + idx4[3];
          *dst = src[val_idx];

          ++dst;
          ++indices;
        }
      }
    }
  }
}

LogicalResult top::GatherElementsOp::inference(InferenceParameter &p) {
  const float *src = p.inputs[0];
  const float *indices = p.inputs[1];
  float *dst = p.outputs[0];
  int axis = getAxis();
  auto src_dim = module::getShape(getInput()).size();
  auto dst_dim = module::getShape(getIndices()).size();
  int src_shape[src_dim];
  int indices_shape[dst_dim];
  module::getGlobalShape(getInput(), src_shape, src_dim);
  module::getGlobalShape(getIndices(), indices_shape, dst_dim);

  if (axis < 0) {
    axis += src_dim;
  }

  if (src_dim > 0 && src_dim <= 4 && axis < src_dim) {
    gather_dim4(dst, src, indices, indices_shape, src_shape, src_dim, axis, dst_dim);
  } else {
    llvm_unreachable("Not implemented yet");
  }

  return success();
}

void top::GatherElementsOp::shape_inference() {
  auto indices_shape = module::getShape(getIndices());
  module::setShapeOrVerify(getOutput(), indices_shape);
}
