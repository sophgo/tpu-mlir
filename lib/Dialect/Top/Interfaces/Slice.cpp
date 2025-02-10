//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"
#include <valarray>

int64_t top::SliceOp::getFLOPs() { return 0; }

LogicalResult top::SliceOp::init(InferenceParameter &p) { return success(); }
void top::SliceOp::deinit(InferenceParameter &p) {}

void top::SliceOp::paramConvert() {
  auto context = getContext();
  mlir::Builder builder(context);
  auto offset_ori = module::getI64Array(getOffset());
  auto steps_ori = module::getI64Array(getSteps());
  auto ends_ori = module::getI64Array(getEnds());
  auto axes_ori = module::getI64Array(getAxes());
  auto input_shapes = module::getShape(getInput());
  setHasparamConvertAxesAttr(builder.getI64ArrayAttr(*axes_ori));

  auto input_dims = input_shapes.size();
  auto slice_n = axes_ori->size();
  if (input_dims == slice_n)
    return;
  ASSERT_THIS(offset_ori->size() == slice_n && steps_ori->size() == slice_n &&
              ends_ori->size() == slice_n);
  auto offset_v = std::make_shared<std::vector<int64_t>>(input_dims, 0);
  auto steps_v = std::make_shared<std::vector<int64_t>>(input_dims, 1);
  auto ends_v = std::make_shared<std::vector<int64_t>>(input_shapes);
  for (int i = 0; i < slice_n; ++i) {
    int axis =
        axes_ori->at(i) >= 0 ? axes_ori->at(i) : axes_ori->at(i) + input_dims;
    int step = steps_ori->at(i);
    int64_t end = ends_ori->at(i);
    int64_t offset = offset_ori->at(i);
    offset_v->at(axis) = offset;
    ends_v->at(axis) = end;
    steps_v->at(axis) = step;
  }
  setOffsetAttr(builder.getI64ArrayAttr(*offset_v));
  setStepsAttr(builder.getI64ArrayAttr(*steps_v));
  setEndsAttr(builder.getI64ArrayAttr(*ends_v));
  setAxesAttr(builder.getI64ArrayAttr(std::nullopt));
}

LogicalResult top::SliceOp::inference(InferenceParameter &p) {
  auto out_num_elem = module::getNumElements(getOutput());
  auto offset_v = module::getI64Array(getOffset());
  auto steps_v = module::getI64Array(getSteps());
  std::vector<int64_t> in_shape = module::getShape(getInput());
  auto in_dims = in_shape.size();
  if (out_num_elem == 0) {
    return success();
  }
  auto ends_v_old = module::getI64Array(getEnds());
  const size_t slice_dims = offset_v->size();
  auto axes = module::getI64Array(getHasparamConvertAxesAttr());
  auto slice_n = axes->size();
  auto ends_v = ends_v_old;
  if (slice_n) {
    ends_v = std::make_shared<std::vector<int64_t>>(in_shape);
  }
  for (int i = 0; i < slice_n; ++i) {
    int axis = axes->at(i);
    if (axis < 0) {
      axis += in_dims;
    }
    int step = steps_v->at(axis);
    int64_t end = ends_v_old->at(axis);
    int64_t offset = offset_v->at(axis);
    offset_v->at(axis) = offset;
    ends_v->at(axis) = end;
    steps_v->at(axis) = step;
  }
  for (int i = 0; i < slice_dims; ++i) {
    if (offset_v->at(i) < 0) {
      offset_v->at(i) += in_shape[i];
    }
  }
  std::vector<int64_t> out_shape(in_dims);
  for (size_t i = 0; i < in_dims; ++i) {
    if (i < slice_dims) {
      auto offset = offset_v->at(i);
      auto end = ends_v->at(i);
      auto step = steps_v->at(i);
      if (end < 0) {
        end += in_shape[i];
      }
      offset = step > 0 ? std::clamp(offset, 0L, in_shape[i])
                        : std::clamp(offset, 0L, in_shape[i] - 1);
      end = step > 0 ? std::clamp(end, 0L, in_shape[i])
                     : std::clamp(end, -1L, in_shape[i] - 1);
      out_shape[i] = abs_ceiling_func(end - offset, step);
    } else {
      out_shape[i] = in_shape[i];
    }
  }
  module::setShape(getOutput(), out_shape);
  auto out_dims = out_shape.size();
  while (out_dims < in_dims) {
    out_shape.insert(out_shape.begin(), 1);
    out_dims++;
  }
  if (!(module::isNone(getOffsetT()) && module::isNone(getEndsT()) &&
        module::isNone(getStepsT()))) {
    // slice in only one aixs in such case
    for (int i = 0; i < out_dims; i++) {
      out_shape[i] = std::min(out_shape[i], in_shape[i]);
    }
    int axis = module::getI64Array(getAxes())->at(0);
    auto ends_v = module::getI64Array(getEnds());

    if (!module::isNone(getOffsetT()))
      offset_v->at(axis) = *p.inputs[1];
    if (!module::isNone(getEndsT()))
      ends_v->at(axis) = *p.inputs[2];
    if (!module::isNone(getStepsT()))
      steps_v->at(axis) = *p.inputs[3];
    if (offset_v->at(axis) < 0)
      offset_v->at(axis) += in_shape[axis];
    if (ends_v->at(axis) < 0)
      ends_v->at(axis) += in_shape[axis];
    offset_v->at(axis) =
        steps_v->at(axis) > 0
            ? std::clamp(offset_v->at(axis), 0L, in_shape[axis])
            : std::clamp(offset_v->at(axis), 0L, in_shape[axis] - 1);
    ends_v->at(axis) =
        steps_v->at(axis) > 0
            ? std::clamp(ends_v->at(axis), 0L, in_shape[axis])
            : std::clamp(ends_v->at(axis), -1L, in_shape[axis] - 1);

    out_shape[axis] =
        (ends_v->at(axis) - offset_v->at(axis)) / steps_v->at(axis);
    module::setShape(getOutput(), out_shape);
    out_num_elem = module::getNumElements(getOutput());
  }
  for (int i = 0; i < slice_dims; ++i) {
    if (offset_v->at(i) < 0) {
      offset_v->at(i) += in_shape[i];
    }
    offset_v->at(i) = steps_v->at(i) > 0
                          ? std::clamp(offset_v->at(i), 0L, in_shape[i])
                          : std::clamp(offset_v->at(i), 0L, in_shape[i] - 1);
  }
  // slice[range] -> (offset + stride)
  std::valarray<int64_t> in_stride_v(1, in_dims);
  std::valarray<int64_t> out_stride_v(1, out_dims);
  for (int i = in_stride_v.size() - 2; i >= 0; --i) {
    in_stride_v[i] *= in_stride_v[i + 1] * in_shape[i + 1];
    out_stride_v[i] *= out_stride_v[i + 1] * out_shape[i + 1];
  }
  auto in_offset_v = std::valarray<int64_t>(offset_v->data(), offset_v->size());
  auto in_offset = (in_offset_v * in_stride_v).sum();
  auto out_in_stride_v =
      std::valarray<int64_t>(steps_v->data(), steps_v->size());
  out_in_stride_v *= in_stride_v;

#pragma omp parallel for schedule(static, omp_schedule(out_num_elem))
  for (int64_t i = 0; i < out_num_elem; ++i) {
    std::valarray<int64_t> out_it(1, out_dims);
    int64_t tmp = i;
    for (int j = 0; j < out_dims; j++) {
      out_it[j] = tmp / out_stride_v[j];
      tmp = tmp % out_stride_v[j];
    }
    p.outputs[0][i] = p.inputs[0][(out_it * out_in_stride_v).sum() + in_offset];
  }

  return success();
}

void top::SliceOp::shape_inference() {
  if (!getAxes().empty())
    paramConvert();
  const auto input_shape = module::getShape(getInput());
  const size_t dims = input_shape.size();
  const auto offset_v = module::getI64Array(getOffset());
  const auto steps_v = module::getI64Array(getSteps());
  const auto ends_v = module::getI64Array(getEnds());
  const size_t slice_dims = offset_v->size();
  std::vector<int64_t> output_shape(input_shape.size());
  for (size_t i = 0; i < dims; ++i) {
    if (i < slice_dims) {
      auto offset = offset_v->at(i);
      auto end = ends_v->at(i);
      auto step = steps_v->at(i);
      if (offset < 0) {
        offset += input_shape[i];
      }
      if (end < 0) {
        end += input_shape[i];
      }
      offset = step > 0 ? std::clamp(offset, 0L, input_shape[i])
                        : std::clamp(offset, 0L, input_shape[i] - 1);
      end = step > 0 ? std::clamp(end, 0L, input_shape[i])
                     : std::clamp(end, -1L, input_shape[i] - 1);
      output_shape[i] = abs_ceiling_func(end - offset, step);
    } else {
      output_shape[i] = input_shape[i];
    }
  }
  module::setShapeOrVerify(getOutput(), output_shape);
  if (module::isShape(getInput())) {
    auto input_v = module::getShapeTensorValue(getInput());
    auto output_shape_v =
        module::commonShapeValInfer(getOperation(), {input_v}, output_shape);
    module::bindShapeTensorValue(getOutput(), output_shape_v);
  }
  if (!module::isNone(getOffsetT())) {
    // set top run mode to dynamic
    module::setTopRunMode(module::TopRunMode::DYNAMIC);
  }
}
