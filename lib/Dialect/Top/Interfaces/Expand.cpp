//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::ExpandOp::getFLOPs() { return 0; }

LogicalResult top::ExpandOp::init(InferenceParameter &p) { return success(); }
void top::ExpandOp::deinit(InferenceParameter &p) {}

LogicalResult top::ExpandOp::inference(InferenceParameter &p) {
  llvm_unreachable("Should be convert to other ops in it's canonicalize pass.");
  return success();
}

void top::ExpandOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  std::vector<int64_t> expand_shape;
  if (!getShapeT()) {
    auto shape_v = module::getI64Array(getShape());
    expand_shape = *shape_v;
  } else if (auto shape_w =
                 dyn_cast<top::WeightOp>(getShapeT().getDefiningOp())) {
    auto shape_v = shape_w.read_as_float();
    std::transform(shape_v->begin(), shape_v->end(),
                   std::back_inserter(expand_shape),
                   [](auto &v) { return static_cast<int64_t>(v); });
  } else if (module::isShape(getShapeT())) {
    expand_shape = module::getShapeTensorValue(getShapeT());
  } else {
    llvm_unreachable("out_shape is illegal");
  }

  int dim_in = in_shape.size();
  int dim_out = expand_shape.size();
  int dim_pad = dim_out - dim_in;
  ASSERT_THIS(dim_pad >= 0);

  std::vector<int64_t> out_shape(expand_shape.begin(), expand_shape.end());
  for (int i = dim_pad; i < dim_out; i++) {
    out_shape[i] =
        in_shape[i - dim_pad] == 1 ? expand_shape[i] : in_shape[i - dim_pad];
  }
  module::setShapeOrVerify(getOutput(), out_shape);

  // if (module::isShape(getInput())) {
  //   std::vector<std::vector<int64_t>> input_shapes_v;
  //   auto input_shape_v = module::getShapeTensorValue(getInput());
  //   input_shapes_v.push_back(input_shape_v);
  //   input_shapes_v.push_back(expand_shape);
  //   auto output_shape_v =
  //       module::commonShapeValInfer(getOperation(), input_shapes_v,
  //       out_shape);
  //   module::bindShapeTensorValue(getOutput(), output_shape_v);
  // }
}
