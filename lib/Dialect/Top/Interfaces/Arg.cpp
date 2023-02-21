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
#include <float.h>


int64_t top::ArgOp::getFLOPs() { return module::getNumElements(getIndices()); }

LogicalResult top::ArgOp::init(InferenceParameter &p) { return success(); }
void top::ArgOp::deinit(InferenceParameter &p) {}

LogicalResult top::ArgOp::inference(InferenceParameter &p) {
  float *input_v = p.inputs[0];
  float *output_idx = p.outputs[0];
  const bool need_val = !getValues().getType().isa<NoneType>();
  float* output_val = need_val ? p.outputs[1] : nullptr;
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
        output_idx[o * inner_dims + i] = target_idx;
        if(need_val){
          output_val[o * inner_dims + i] = target_val;
        }
      } else {
        llvm_unreachable("not support now.");
      }
    }
  }
  return success();
}
