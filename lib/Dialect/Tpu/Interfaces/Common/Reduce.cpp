//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <float.h>

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

typedef enum reduce_type {
  REDUCE_MEAN = 0,
  REDUCE_SUM = 1,
  REDUCE_MAX = 2,
  REDUCE_MIN = 3,
  REDUCE_PROD = 4,
  REDUCE_ALL = 5,
  REDUCE_ANY = 6,
  REDUCE_L2 = 7,
  REDUCE_L1 = 8,
  REDUCE_SumSquare = 9,
  REDUCE_LogSum = 10,
  REDUCE_LogSumExp
} reduce_type_t;

static void size_to_2dim(int64_t size, int64_t &h, int64_t &w) {
  h = 1;
  w = size;
  int div = std::sqrt(size);
  for (h = div; h >= 2; h--) {
    if (size % h == 0) {
      w = size / h;
      break;
    }
  }
}

tpu::ReduceOp::reduce_attr_t tpu::ReduceOp::parseParam() {
  tpu::ReduceOp::reduce_attr_t attr = {0};
  auto axes_ = Module::getI64Array(axes());
  auto input_shape = Module::getShape(input());
  int num_dims = input_shape.size();
  int num_axes = axes_->size();
  for (uint i = 1; i < num_axes; i++) {
    if (axes_->at(i) != axes_->at(i - 1) + 1) {
      llvm_unreachable("Not Implemented");
    }
    if (axes_->at(i) >= num_dims) {
      llvm_unreachable("Not Implemented");
    }
  }
  int start_axis = axes_->at(0);
  int end_axis = axes_->at(num_axes - 1) + 1;
  int64_t outer_dims =
      std::accumulate(input_shape.begin(), input_shape.begin() + start_axis, 1,
                      std::multiplies<int64_t>());
  // TODO: (outer_dims, axis_dims, inner_dims) will failure
  // so use (outer_n, outer_c, axis_dims, inner_dims)
  size_to_2dim(outer_dims, attr.outer_n, attr.outer_c);
  attr.axis_dims = std::accumulate(input_shape.begin() + start_axis,
                                   input_shape.begin() + end_axis, 1,
                                   std::multiplies<int64_t>());
  attr.inner_dims =
      std::accumulate(input_shape.begin() + end_axis, input_shape.end(), 1,
                      std::multiplies<int64_t>());
  attr.simplified = true;
  return attr;
}

LogicalResult tpu::ReduceOp::init(InferenceParameter &p) { return success(); }
void tpu::ReduceOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ReduceOp::inference(InferenceParameter &p) {
  float *input_v = p.inputs[0];
  float *output_v = p.outputs[0];
  auto type_val = mode();

  bool is_cv18xx = Module::isCV18xx();
  auto out_type = Module::getStorageType(output());
  auto attr = parseParam();
  int64_t outer_dims = attr.outer_n * attr.outer_c;
  // calc dims

  for (int o = 0; o < outer_dims; o++) {
    for (int i = 0; i < attr.inner_dims; i++) {
      if (type_val == "ReduceMean" || type_val == "ReduceSum") {
        float sum = 0.0f;
        if (attr.inner_dims == 1) {
          sum = std::accumulate(input_v + o * attr.axis_dims,
                                input_v + (o + 1) * attr.axis_dims, 0.0f);
        } else {
          for (int a = 0; a < attr.axis_dims; a++) {
            sum += input_v[o * attr.axis_dims * attr.inner_dims +
                           a * attr.inner_dims + i];
          }
        }
        if (type_val == "ReduceSum") {
          output_v[o * attr.inner_dims + i] = sum;
        } else {
          if (Quant::isUniformQuantized(output()) && is_cv18xx) {
            // divisor in multiplier
            output_v[o * attr.inner_dims + i] = sum;
          } else {
            float coeff_mean = 1.0 / attr.axis_dims;
            if (out_type.isBF16()) {
              coeff_mean = BF16(coeff_mean);
            } else if (out_type.isF16()) {
              coeff_mean = F16(coeff_mean);
            }
            output_v[o * attr.inner_dims + i] = sum * coeff_mean;
          }
        }
      } else if (type_val == "ReduceMax" || type_val == "ReduceMin") {
        float target = input_v[o * attr.axis_dims * attr.inner_dims + i];
        for (int a = 1; a < attr.axis_dims; a++) {
          auto v = input_v[o * attr.axis_dims * attr.inner_dims +
                           a * attr.inner_dims + i];
          if (type_val == "ReduceMax" && v > target) {
            target = v;
          } else if (type_val == "ReduceMin" && v < target) {
            target = v;
          }
        }
        output_v[o * attr.inner_dims + i] = target;
      } else if (type_val == "ReduceL2") {
        float sum = 0.0f;
        for (int a = 0; a < attr.axis_dims; a++) {
          sum += std::pow(input_v[o * attr.axis_dims * attr.inner_dims +
                                  a * attr.inner_dims + i],
                          2);
        }
        output_v[o * attr.inner_dims + i] = std::pow(sum, 0.5);
      } else if (type_val == "ReduceL1") {
        float sum = 0.0f;
        for (int a = 0; a < attr.axis_dims; a++) {
          sum += fabs(input_v[o * attr.axis_dims * attr.inner_dims +
                              a * attr.inner_dims + i]);
        }
        output_v[o * attr.inner_dims + i] = sum;
      } else {
        llvm_unreachable("not support now.");
      }
    }
  }
  auto num_elem = Module::getNumElements(output());
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      f32_to_bf16(p.outputs[0], p.outputs[0], num_elem, is_cv18xx);
    } else if (out_type.isF16()) {
      f32_to_f16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (Quant::isUniformQuantized(output())) {
    int64_t shift = Module::getI64Array(rshift().value())->at(0);
    int64_t multi = Module::getI64Array(multiplier().value())->at(0);
    if (shift != 0 || multi != 1) {
      for (size_t i = 0; i < num_elem; ++i) {
        int64_t v =
            applyMultiplierAndRShift(output_v[i], multi, shift, CVI_QUANT);
        output_v[i] = Quant::to_int8(v);
      }
    }
  }
  return success();
}
