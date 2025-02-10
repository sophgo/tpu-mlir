//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::ReshapeOp::init(InferenceParameter &p) { return success(); }
void tpu::ReshapeOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ReshapeOp::inference(InferenceParameter &p) {
  if (p.inputs[0] != p.outputs[0]) {
    auto num_elem = module::getNumElements(getOutput());
    memcpy(p.outputs[0], p.inputs[0], sizeof(float) * num_elem);
  }
  auto in_shape = module::getShape(getInput());
  int num = 1;
  for (int i = 0; i < in_shape.size(); i++) {
    num *= in_shape[i];
  }

  std::vector<int64_t> out_shape;
  if (!getShape().empty()) {
    auto shape = module::getI64Array(getShape());
    int shape_num = 1;
    for (int i = 0; i < shape->size(); i++) {
      shape_num *= shape->at(i);
    }
    int64_t start_dim = getFlattenStartDim();
    if (module::isPlatform(module::Platform::ONNX) && (num != shape_num) &&
        (start_dim != -1)) {
      auto outer_dims =
          std::accumulate(in_shape.begin(), in_shape.begin() + start_dim, 1,
                          std::multiplies<int64_t>());
      auto inner_dims =
          std::accumulate(in_shape.begin() + start_dim, in_shape.end(), 1,
                          std::multiplies<int64_t>());
      shape->at(0) = outer_dims;
      shape->at(1) = inner_dims;
    }
    int x = -1;
    for (int i = 0; i < shape->size(); i++) {
      auto s = shape->at(i);
      if (s > 0) {
        out_shape.push_back(s);
        num /= s;
      } else if (s == 0) {
        out_shape.push_back(in_shape[i]);
        num /= in_shape[i];
      } else if (s == -1) {
        out_shape.push_back(-1);
        x = i;
      } else {
        dump();
        llvm_unreachable("shape is illegal");
      }
    }
    if (x >= 0) {
      out_shape[x] = num;
    }
  } else {
    out_shape = module::getShape(getOutput());
  }
  module::setShape(getOutput(), out_shape);
  return success();
}

LogicalResult tpu::ReshapeOp::LocalGenSupport() {
  if (module::isCV18xx() || module::isBM1684Family()) {
    return failure();
  }
  auto ishape = module::getShape(getInput());
  auto oshape = module::getShape(getOutput());

  if (ishape.size() < 2 || oshape.size() < 2 || ishape[0] != oshape[0] ||
      ishape[1] != oshape[1]) {
    return failure();
  }
  return success();
}

LogicalResult tpu::ReshapeOp::AllowDataSplit(int64_t axis,
                                             group_type_t group_type) {
  // temp code for prevent not-inplace reshape
  // has to fix LmemAllocator in the future
  if (module::isBM1688() || module::isSG2380() || module::isMARS3() ||
      module::isSGTPUV8()) {
    return failure();
  }

  if (axis == 0 || axis == 1) {
    return success();
  }

  return failure();
}

bool tpu::ReshapeOp::support_multi_core() { return false; }
