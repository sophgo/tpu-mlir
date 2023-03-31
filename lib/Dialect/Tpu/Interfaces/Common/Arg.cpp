//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Interfaces/TypeInterface.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

LogicalResult tpu::ArgOp::init(InferenceParameter &p) { return success(); }
void tpu::ArgOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ArgOp::inference(InferenceParameter &p) {
  float *input_v = p.inputs[0];
  float *output_idx = p.outputs[0];
  const bool need_val = !getValues().getType().isa<NoneType>();
  float *output_val = need_val ? p.outputs[1] : nullptr;
  auto type_val = getMode().str();
  auto axis_val = getAxis();
  auto input_shape = module::getShape(getInput());
  // calc dims
  int start_axis = axis_val;
  int end_axis = axis_val + 1;
  int outer_dims =
      std::accumulate(input_shape.begin(), input_shape.begin() + start_axis, 1,
                      std::multiplies<int64_t>());
  int axis_dims = std::accumulate(input_shape.begin() + start_axis,
                                  input_shape.begin() + end_axis, 1,
                                  std::multiplies<int64_t>());
  int inner_dims =
      std::accumulate(input_shape.begin() + end_axis, input_shape.end(), 1,
                      std::multiplies<int64_t>());
  std::function<bool(float, float)> compare_func;
  if (type_val == "ArgMax") {
    compare_func = std::greater<float>();
  } else if (type_val == "ArgMin") {
    compare_func = std::less<float>();
  } else {
    llvm_unreachable("not support now.");
  }

  if (module::isCV18xx()) {
    int tile_size = 256;
    int tile_num = (axis_dims + tile_size - 1) / tile_size;
    for (int o = 0; o < outer_dims; o++) {
      for (int i = 0; i < inner_dims; i++) {
        for (int t = 0; t < tile_num; ++t) {
          int remain = axis_dims - tile_size * t;
          float target_val = input_v[o * axis_dims * inner_dims +
                                     t * tile_num * inner_dims + i];
          for (int k = 1; k < tile_size && k < remain; ++k) {
            auto v = input_v[o * axis_dims * inner_dims +
                             (t * tile_num + k) * inner_dims + i];
            if (compare_func(v, target_val)) {
              target_val = v;
            }
          }
          p.outputs[0][o * tile_num * inner_dims + t * inner_dims + i] =
              target_val;
        }
      }
    }
    return success();
  }
  for (int o = 0; o < outer_dims; o++) {
    for (int i = 0; i < inner_dims; i++) {
      float target_val = input_v[o * axis_dims * inner_dims + i];
      int target_idx = 0;
      for (int a = 1; a < axis_dims; a++) {
        auto v = input_v[o * axis_dims * inner_dims + a * inner_dims + i];
        if (compare_func(v, target_val)) {
          target_val = v;
          target_idx = a;
        }
      }
      output_idx[o * inner_dims + i] = target_idx;
      if (need_val) {
        output_val[o * inner_dims + i] = target_val;
      }
    }
  }
  return success();
}

mlir::Type tpu::ArgOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  auto op = getOperation();
  if (module::isCV18xx()) {
    return type_verify_case_same(op, opd_idx, mode);
  } else if (module::isNone(getValues()) == false) {
    return type_verify_case_type(op, opd_idx, getValues().getType(), mode);
  }
  return do_nothing(mode);
}
