//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"

int64_t top::ShapeOp::getFLOPs() { return 0; }

LogicalResult top::ShapeOp::init(InferenceParameter &p) { return success(); }

void top::ShapeOp::deinit(InferenceParameter &p) {}

LogicalResult top::ShapeOp::inference(InferenceParameter &p) {
  float *output_data = p.outputs[0];
  auto input_shape = module::getShape(getInput());
  for (int i = 0; i < input_shape.size(); ++i) {
    output_data[i] = input_shape[i];
  }
  bool no_slice = true;
  int64_t input_dims = input_shape.size();
  int64_t start = getStart().has_value() ? getStart().value() : 0;
  int64_t end = getEnd().has_value() ? getEnd().value() : input_dims;
  end = std::clamp(end, 0L, input_dims);
  int64_t step = getStep().has_value() ? getStep().value() : 1;
  if (getStart().has_value()) {
    removeStartAttr();
  }
  if (getEnd().has_value()) {
    removeEndAttr();
  }
  if (start < 0 && end - start == 1) {
    start = start + input_dims;
    end = start + 1;
  }
  if (start != 0 || end != input_dims) {
    no_slice = false;
  }
  std::vector<int64_t> output_shape({(int64_t)input_shape.size()});
  if (!no_slice) {
    auto builder = OpBuilder(getContext());
    auto name = module::getName(getOutput()).str();
    auto loc = NameLoc::get(builder.getStringAttr(name + "_0"));
    auto cur_op = getOperation();
    auto none = module::getNoneOp(cur_op);
    auto cur_out = getOutput();
    cur_op->setLoc(loc);

    builder.setInsertionPointAfter(cur_op);
    std::vector<NamedAttribute> attrs;
    attrs.emplace_back(
        builder.getNamedAttr("offset", builder.getI64ArrayAttr({start})));
    attrs.emplace_back(
        builder.getNamedAttr("steps", builder.getI64ArrayAttr({step})));
    attrs.emplace_back(
        builder.getNamedAttr("ends", builder.getI64ArrayAttr({end})));
    attrs.emplace_back(
        builder.getNamedAttr("axes", builder.getI64ArrayAttr({0})));
    loc = NameLoc::get(builder.getStringAttr(name));
    auto new_op = builder.create<SliceOp>(
        loc, cur_out.getType(), ValueRange{cur_out, none, none, none}, attrs);
    cur_out.replaceAllUsesWith(new_op.getOutput());
    new_op.setOperand(0, cur_out);
    new_op.shape_inference();
    auto out_shape = module::commonShapeValInfer(
        new_op, {module::getShape(getInput()).vec()},
        module::getShape(new_op.getOutput()));
    module::setShape(new_op.getOutput(), output_shape);
  } else {
    module::setShape(getOutput(), output_shape);
  }
  return success();
}

void top::ShapeOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  bool no_slice = true;
  int64_t input_dims = input_shape.size();
  int64_t start = getStart().has_value() ? getStart().value() : 0;
  int64_t end = getEnd().has_value() ? getEnd().value() : input_dims;
  end = std::clamp(end, 0L, input_dims);
  int64_t step = getStep().has_value() ? getStep().value() : 1;
  if (getStart().has_value()) {
    removeStartAttr();
  }
  if (getEnd().has_value()) {
    removeEndAttr();
  }
  if (start < 0 && end - start == 1) {
    start = start + input_dims;
    end = start + 1;
  }
  if (start != 0 || end != input_dims) {
    no_slice = false;
  }
  std::vector<int64_t> output_shape({(int64_t)input_shape.size()});
  if (!no_slice) {
    auto builder = OpBuilder(getContext());
    auto name = module::getName(getOutput()).str();
    auto loc = NameLoc::get(builder.getStringAttr(name + "_0"));
    auto cur_op = getOperation();
    auto none = module::getNoneOp(cur_op);
    auto cur_out = getOutput();
    cur_op->setLoc(loc);

    builder.setInsertionPointAfter(cur_op);
    std::vector<NamedAttribute> attrs;
    attrs.emplace_back(
        builder.getNamedAttr("offset", builder.getI64ArrayAttr({start})));
    attrs.emplace_back(
        builder.getNamedAttr("steps", builder.getI64ArrayAttr({step})));
    attrs.emplace_back(
        builder.getNamedAttr("ends", builder.getI64ArrayAttr({end})));
    attrs.emplace_back(
        builder.getNamedAttr("axes", builder.getI64ArrayAttr({0})));
    loc = NameLoc::get(builder.getStringAttr(name));
    auto new_op = builder.create<SliceOp>(
        loc, cur_out.getType(), ValueRange{cur_out, none, none, none}, attrs);
    cur_out.replaceAllUsesWith(new_op.getOutput());
    new_op.setOperand(0, cur_out);
    module::setShapeOrVerify(getOutput(), output_shape);
    new_op.shape_inference();
    auto out_shape = module::commonShapeValInfer(
        new_op, {module::getShape(getInput()).vec()},
        module::getShape(new_op.getOutput()));
    module::bindShapeTensorValue(new_op.getOutput(), out_shape);
  } else {
    module::setShapeOrVerify(getOutput(), output_shape);
    module::bindShapeTensorValue(getOutput(), module::getShape(getInput()));
  }
  // set top run mode to dynamic
  module::setTopRunMode(module::TopRunMode::DYNAMIC);
}
