//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::SplitOp::getFLOPs() { return 0; }

LogicalResult top::SplitOp::init(InferenceParameter &p) { return success(); }
void top::SplitOp::deinit(InferenceParameter &p) {}

LogicalResult top::SplitOp::inference(InferenceParameter &p) {
  // auto out_num_elem = module::getNumElements(getOutput());
  int out_num = getNum();
  int split_axis = getAxis();
  auto in_shape = module::getShape(getInput());
  auto split_size = module::getI64Array(getSplitSizeAttr());
  int64_t outer_num_elem = 1, inner_num_elem = 1;
  for (int i = 0; i < split_axis; ++i) {
    outer_num_elem *= in_shape[i];
  }
  for (int i = split_axis + 1; i < in_shape.size(); ++i) {
    inner_num_elem *= in_shape[i];
  }

  for (int i = 0; i < outer_num_elem; ++i) {
    int64_t index = i * in_shape[split_axis] * inner_num_elem;
    int split_num = 0;
    for (int j = 0; j < out_num; ++j) {
      memcpy(p.outputs[j] + i * split_size->at(j) * inner_num_elem,
             p.inputs[0] + index + split_num * inner_num_elem,
             split_size->at(j) * inner_num_elem * sizeof(float));
      split_num += split_size->at(j);
    }
  }

  return success();
}

void top::SplitOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  auto num = getNum();
  auto axis = getAxis();
  if (axis < 0) {
    axis += in_shape.size();
    setAxis(axis);
  }
  int64_t out_max_size = (in_shape[axis] + num - 1) / num;
  auto split_size = module::getI64Array(getSplitSize(), num, out_max_size);
  std::vector<int64_t> out_size(*split_size);
  auto length = std::accumulate(out_size.begin(), out_size.end(), 0);
  out_size[num - 1] = out_size[num - 1] + in_shape[axis] - length;
  OpBuilder builder(module::getCtx());
  setSplitSizeAttr(builder.getI64ArrayAttr(out_size));
  ASSERT_THIS(num == getOutputs().size());
  std::vector<int64_t> out_shape = in_shape;

  for (int i = 0; i < num; ++i) {
    auto out = getResult(i);
    out_shape[axis] = out_size[i];
    module::setShapeOrVerify(out, out_shape);
  }
}
