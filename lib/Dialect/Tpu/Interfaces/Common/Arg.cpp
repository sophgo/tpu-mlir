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

typedef enum {
  ARG_MAXT,
  ARG_MINT,
} arg_mode_t;

template<typename T>
std::function<bool(T, T)> get_compare_op(arg_mode_t mode, bool select_last) {
  if (select_last) {
    if (mode == ARG_MAXT) {
      return std::greater_equal<T>();
    } else {
      return std::less_equal<T>();
    }
  } else {
    if (mode == ARG_MAXT) {
      return std::greater<T>();
    } else {
      return std::less<T>();
    }
  }
}

LogicalResult tpu::ArgOp::inference(InferenceParameter &p) {
  float *input_v = p.inputs[0];
  float *output_idx = p.outputs[0];
  const bool need_val = !module::isNone(getValues());
  float *output_val = need_val ? p.outputs[1] : nullptr;
  const auto type_val = getMode().str();
  assert(type_val == "ArgMax" || type_val == "ArgMin");
  const arg_mode_t mode = (type_val == "ArgMax") ? ARG_MAXT : ARG_MINT;
  auto axis = getAxis();
  const auto input_shape = module::getShape(getInput());
  const int input_dims = input_shape.size();
  if (axis < 0) {
    axis += input_dims;
    setAxis(axis);
  }
  assert(0 <= axis && axis < input_dims);
  int outer_dims =
      std::accumulate(input_shape.begin(), input_shape.begin() + axis, 1,
                      std::multiplies<int64_t>());
  int inner_dims =
      std::accumulate(input_shape.begin() + axis + 1, input_shape.end(), 1,
                      std::multiplies<int64_t>());
  int axis_dims = input_shape[axis];
  const auto cmp_op = get_compare_op<float>(mode, getSelectLastIndex());

  if (module::isCV18xx()) {
    assert(!need_val);
    int tile_size = 256;
    int tile_num = (axis_dims + tile_size - 1) / tile_size;
    for (int o = 0; o < outer_dims; o++) {
      for (int i = 0; i < inner_dims; i++) {
        for (int t = 0; t < tile_num; ++t) {
          int remain = axis_dims - tile_size * t;
          float target_val = input_v[o * axis_dims * inner_dims +
                                     t * tile_size * inner_dims + i];
          for (int k = 1; k < tile_size && k < remain; ++k) {
            auto v = input_v[o * axis_dims * inner_dims +
                             (t * tile_size + k) * inner_dims + i];
            if (cmp_op(v, target_val)) {
              target_val = v;
            }
          }
          p.outputs[0][o * tile_num * inner_dims + t * inner_dims + i] =
              target_val;
        }
      }
    }
  } else {
    int num_iter = outer_dims * inner_dims;
#pragma omp parallel for schedule(static, omp_schedule(num_iter))
    for (int n = 0; n < num_iter; n++) {
      const int o = n / inner_dims;
      const int i = n % inner_dims;
      const float* input_v_n = input_v + o * axis_dims * inner_dims + i;
      float target_val = input_v_n[0];
      int target_idx = 0;
      for (int a = 1; a < axis_dims; a++) {
        auto v = input_v_n[a * inner_dims];
        if (cmp_op(v, target_val)) {
          target_val = v;
          target_idx = a;
        }
      }
      output_idx[n] = target_idx;
      if (need_val) {
        output_val[n] = target_val;
      }
    }
  }
  return success();
}

mlir::Type tpu::ArgOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  auto op = getOperation();
  if (module::isCV18xx()) {
    return type_verify_case_same(op, opd_idx, mode);
  } else {
    return type_verify_case_type(op, opd_idx, Builder(op).getF32Type(), mode);
  }
  // } else if (module::isNone(getValues()) == false) {
  //   return type_verify_case_type(op, opd_idx, getValues().getType(), mode);
  // }
  return do_nothing(mode);
}
