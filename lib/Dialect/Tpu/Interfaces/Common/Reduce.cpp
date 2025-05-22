//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Float16.h"

#include "tpu_mlir/Support/MathUtils.h"

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

reduce_attr_t tpu::ReduceOp::parseParam() {
  reduce_attr_t attr = {0};
  auto axes_ = module::getI64Array(getAxes());
  auto input_shape = module::getShape(getInput());
  int num_dims = input_shape.size();
  int num_axes = axes_->size();
  bool neighbour = true;
  for (uint i = 1; i < num_axes; i++) {
    if (axes_->at(i) != axes_->at(i - 1) + 1) {
      neighbour = false;
    }
    if (axes_->at(i) >= num_dims) {
      UNREACHABLE_THIS("Not Implemented");
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
  if (!neighbour) {
    attr.axis_dims = 1;
    attr.inner_dims = 1;
    for (uint i = 0; i < num_axes; i++) {
      attr.axis_dims *= (*(input_shape.begin() + axes_->at(i)));
      if ((axes_->at(i) > 0) && (axes_->at(i) + 1 <= num_dims))
        attr.inner_dims *= std::accumulate(
            input_shape.begin() + (axes_->at(i) - 1),
            input_shape.begin() + axes_->at(i), 1, std::multiplies<int64_t>());
      else if ((axes_->at(i) == 0) && (axes_->at(i) + 1 <= num_dims) &&
               (i + 1 <= num_axes))
        attr.inner_dims *=
            std::accumulate(input_shape.begin() + (axes_->at(i) + 1),
                            input_shape.begin() + axes_->at(i + 1), 1,
                            std::multiplies<int64_t>());
    }
    if (axes_->size() == 3 && axes_->at(0) == 0 && axes_->at(1) == 2 &&
        axes_->at(2) == 3) {
      int length = std::accumulate(input_shape.begin(), input_shape.end(), 1,
                                   std::multiplies<int64_t>());
      attr.inner_dims = length / (attr.axis_dims * attr.outer_c * attr.outer_n);
    }
  } else {
    attr.axis_dims = std::accumulate(input_shape.begin() + start_axis,
                                     input_shape.begin() + end_axis, 1,
                                     std::multiplies<int64_t>());
    attr.inner_dims =
        std::accumulate(input_shape.begin() + end_axis, input_shape.end(), 1,
                        std::multiplies<int64_t>());
  }
  attr.simplified = true;
  return attr;
}

LogicalResult tpu::ReduceOp::init(InferenceParameter &p) { return success(); }
void tpu::ReduceOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ReduceOp::inference(InferenceParameter &p) {
  float *input_v = p.inputs[0];
  float *output_v = p.outputs[0];
  auto type_val = getMode();

  bool is_cv18xx = module::isCV18xx();
  auto out_type = module::getStorageType(getOutput());
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
          if (module::isUniformQuantized(getOutput()) && is_cv18xx) {
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
      } else if (type_val == "ReduceProd") {
        float target = input_v[o * attr.axis_dims * attr.inner_dims + i];
        for (int a = 0; a < attr.axis_dims; a++) {
          target *= input_v[o * attr.axis_dims * attr.inner_dims +
                            a * attr.inner_dims + i];
        }
        output_v[o * attr.inner_dims + i] = target;
      } else {
        llvm_unreachable("not support now.");
      }
    }
  }
  auto input_shape = module::getShape(getInput());
  auto axes_val = module::getI64Array(getAxes());
  auto num_dims = input_shape.size();
  std::vector<int64_t> out_shape;
  for (int i = 0; i < num_dims; i++) {
    if (std::find(axes_val->begin(), axes_val->end(), i) != axes_val->end()) {
      if (getKeepdims()) {
        out_shape.push_back(1);
      }
    } else {
      out_shape.push_back(input_shape[i]);
    }
  }
  /* keepdims = false, reduce at all axis,
    it need to set the shape to [1] */
  if (!out_shape.size())
    out_shape.push_back(1);
  module::setShape(getOutput(), out_shape);
  auto num_elem = module::getNumElements(getOutput());
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (module::isUniformQuantized(getOutput())) {
    int64_t shift = module::getI64Array(getRshift().value())->at(0);
    int64_t multi = module::getI64Array(getMultiplier().value())->at(0);
    if (shift != 0 || multi != 1) {
      for (size_t i = 0; i < num_elem; ++i) {
        int64_t v = applyMultiplierAndRShift(output_v[i], multi, shift);
        output_v[i] = to_int8(v);
      }
    }
  }
  return success();
}

bool tpu::ReduceOp::support_multi_core() { return false; }

LogicalResult tpu::ReduceOp::AllowDataSplit(int64_t axis,
                                            group_type_t group_type) {
  auto axes = module::getI64Array(getAxes());
  for (auto axis_val : *axes) {
    if (axis == axis_val) {
      return failure();
    }
  }
  return success();
}

LogicalResult tpu::ReduceOp::LocalGenSupport() {

  if (module::isCV18xx() || module::isBM1684Family()) {
    return failure();
  }
  auto axes = module::getI64Array(getAxes());
  if (module::getShape(getInput()).size() != 4) {
    return failure();
  }
  // Note: support other REDUCE-MODE may need to change getBufferSize().
  if (module::getStorageType(getInput()) !=
      module::getStorageType(getOutput())) {
    return failure();
  }
  if (axes->size() != 1 || axes->at(0) != 2 && axes->at(0) != 3) {
    return failure();
  }
  if (getMode() != "ReduceMean" && getMode() != "ReduceSum" &&
      getMode() != "ReduceMax") {
    return failure();
  }
  return success();
}
