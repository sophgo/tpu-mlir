//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"

int64_t top::CompareOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::CompareOp::init(InferenceParameter &p) {

  std::map<std::string, algorithm> map_mode = {
      {"Equal", algorithm::binary_eq},
      {"Greater", algorithm::binary_gt},
      {"GreaterOrEqual", algorithm::binary_ge},
      {"Less", algorithm::binary_lt},
      {"LessOrEqual", algorithm::binary_le},
      {"NotEqual", algorithm::binary_ne},
      {"Xor", algorithm::binary_ne},
      {"And", algorithm::binary_mul}};

  auto binary = new Binary();
  auto lhs_shape = module::getShape(getOperand(0));
  auto rhs_shape = module::getShape(getOperand(1));
  auto max_ndim = std::max(lhs_shape.size(), rhs_shape.size());
  auto input0_shape = shape_expand_dim(lhs_shape, max_ndim);
  auto input1_shape = shape_expand_dim(rhs_shape, max_ndim);
  auto iter = map_mode.find(getModeAttr().str());
  algorithm compare_mode;
  if (iter != map_mode.end()) {
    compare_mode = iter->second;
  }

  (*binary)
      .lhs(p.inputs[0], input0_shape)
      .rhs(p.inputs[1], input1_shape)
      .dst(p.outputs[0], module::getShape(getOutput()))
      .algorithem(compare_mode)
      .setup();

  p.handle = (void *)binary;

  return success();
}

void top::CompareOp::deinit(InferenceParameter &p) {}

LogicalResult top::CompareOp::inference(InferenceParameter &p) {

  if (p.handle == nullptr) {
    return failure();
  }
  auto binary = (Binary *)p.handle;
  binary->run();

  return success();
}

void top::CompareOp::shape_inference() {
  broadcast_shape_inference(getOperation());
  for (int i = 0; i < getNumOperands(); i++) {
    auto value = getOperation()->getOperand(i);
    broadcast_tensor_reshape(getOutput(), value);
  }
  auto inputs = {getLhs(), getRhs()};
  // shape value inference can only support shape and weight
  bool need_shape_val_infer =
      std::all_of(inputs.begin(), inputs.end(), [](auto in_op) {
        return module::isShape(in_op) || module::isWeight(in_op);
      });
  if (need_shape_val_infer) {
    std::vector<std::vector<int64_t>> input_shapes_v;
    for (auto in_op : inputs) {
      if (module::isShape(in_op)) {
        auto input_shape_v = module::getShapeTensorValue(in_op);
        input_shapes_v.push_back(input_shape_v);
      } else if (module::isWeight(in_op)) {
        auto data = in_op.getDefiningOp<top::WeightOp>().read_as_float();
        std::vector<int64_t> data_v(data->begin(), data->end());
        input_shapes_v.push_back(data_v);
      }
    }
    auto out_shape = module::getShape(getOutput());
    if (out_shape.size() > 1 && module::isTrain()) {
      return;
    }
    auto output_shape_v =
        module::commonShapeValInfer(getOperation(), input_shapes_v, out_shape);
    module::bindShapeTensorValue(getOutput(), output_shape_v);
  }
}
