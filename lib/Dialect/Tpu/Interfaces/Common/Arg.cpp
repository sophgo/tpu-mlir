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
  auto type_val = getMode().str();
  auto axis_val = getAxis();
  // auto out_shape = module::getShape(getOutput());
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
  for (int o = 0; o < outer_dims; o++) {
    for (int i = 0; i < inner_dims; i++) {
      if (type_val == "ArgMax" || type_val == "ArgMin") {
        float target_val = input_v[o * axis_dims * inner_dims + i];
        int target_idx = 0;
        for (int a = 1; a < axis_dims; a++) {
          auto v = input_v[o * axis_dims * inner_dims + a * inner_dims + i];
          if (type_val == "ArgMax" && v > target_val) {
            target_val = v;
            target_idx = a;
          } else if (type_val == "ArgMin" && v < target_val) {
            target_val = v;
            target_idx = a;
          }
        }
        // output_val[o * inner_dims + i] = target_val;
        output_idx[o * inner_dims + i] = target_idx;
      } else {
        llvm_unreachable("not support now.");
      }
    }
  }
  return success();
}

mlir::Type tpu::ArgOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  return do_nothing(mode);
}
