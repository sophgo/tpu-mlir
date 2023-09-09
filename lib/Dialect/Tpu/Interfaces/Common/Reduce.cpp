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
      //llvm_unreachable("Not Implemented");
      neighbour = false;
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
  if (!neighbour) {
    attr.axis_dims = 1;
    attr.inner_dims = 1;
    for (uint i = 0; i < num_axes; i++) {
      attr.axis_dims *= (*(input_shape.begin() + axes_->at(i)));
      if ((axes_->at(i) > 0) && (axes_->at(i) + 1 <= num_dims))
        attr.inner_dims *= std::accumulate(input_shape.begin() + (axes_->at(i) - 1),
                                           input_shape.begin() + axes_->at(i), 1,
                                           std::multiplies<int64_t>());
      else if ((axes_->at(i) == 0) && (axes_->at(i) + 1 <= num_dims) && (i + 1 <= num_axes))
        attr.inner_dims *= std::accumulate(input_shape.begin() + (axes_->at(i) + 1),
                            input_shape.begin() + axes_->at(i + 1), 1,
                            std::multiplies<int64_t>());
    }
  }
  else {
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

/**
 * Reduce (f32) -> Cast(f32-f16) -> Cast(f16-f32) -> Active(f32)
 * Reduce (f32) -> Cast(f32-int8) -> Cast(int8-f32) -> Active(f32)
 */
LogicalResult tpu::ReduceOp::canonicalize(tpu::ReduceOp op,
                                          PatternRewriter &rewriter) {

  auto opt = op.getOutput();
  if (!opt.hasOneUse()) {
    return failure();
  }

  auto next_op_ = *opt.user_begin();
  auto castf16 = dyn_cast<tpu::CastOp>(next_op_);

  if (!castf16 ||
      (!module::getElementType(castf16.getInput()).isF32() &&
       !module::isCalibratedType(castf16.getInput()) /**for int8 cast op*/)) {
    return failure();
  }

  next_op_ = *castf16.getOutput().user_begin();
  auto castf32 = dyn_cast<tpu::CastOp>(next_op_);
  if (!castf32 || (!module::getElementType(castf32.getOutput()).isF32() &&
                   !module::isUniformQuantized(
                       castf32.getOutput()) /**for int8 cast op*/)) {
    return failure();
  }

  next_op_ = *castf32.getOutput().user_begin();
  auto active = dyn_cast<tpu::ActiveOp>(next_op_);
  if (!active) {
    return failure();
  }

  active.setOperand(opt);
  rewriter.replaceAllUsesWith(castf16.getOutput(), active.getInput());
  // erase reversed
  rewriter.eraseOp(castf32);
  rewriter.eraseOp(castf16);

  return success();
}

