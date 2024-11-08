//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"

int64_t top::SubOp::getFLOPs() {
  return module::getNumElements(getOutput()) *
         (getInputs().size() - 1 + (getDoRelu() ? 1 : 0));
}

LogicalResult top::SubOp::init(InferenceParameter &p) {
  auto binary = new Binary();
  int index0 = 0, index1 = 1;
  if (getIsReverse()) {
    index0 = 1, index1 = 0;
  }
  auto lhs_shape = module::getShape(getInputs()[index0]);
  auto rhs_shape = module::getShape(getInputs()[index1]);

  (*binary)
      .hs(p.inputs[index0], p.inputs[index1], lhs_shape, rhs_shape)
      .dst(p.outputs[0], module::getShape(getOutput()))
      .do_relu(getDoRelu())
      .relu_limit(getReluLimit().convertToDouble())
      .algorithem(algorithm::binary_sub)
      .setup();

  p.handle = (void *)binary;

  return success();
}
void top::SubOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult top::SubOp::inference(InferenceParameter &p) {
  auto output_shape = computer_broadcast_shape(getOperation());
  module::setShape(getOutput(), output_shape);
  if (p.handle == nullptr) {
    return failure();
  }
  auto binary = (Binary *)p.handle;
  int index0 = 0, index1 = 1;
  if (getIsReverse()) {
    index0 = 1, index1 = 0;
  }
  auto lhs_shape = module::getShape(getInputs()[index0]);
  auto rhs_shape = module::getShape(getInputs()[index1]);

  (*binary)
      .hs(p.inputs[index0], p.inputs[index1], lhs_shape, rhs_shape)
      .dst(p.outputs[0], module::getShape(getOutput()))
      .do_relu(getDoRelu())
      .relu_limit(getReluLimit().convertToDouble())
      .algorithem(algorithm::binary_sub)
      .setup();
  binary->run();
  return success();
}

void top::SubOp::shape_inference() {
  broadcast_shape_inference(getOperation());
  bool has_scalar = false;
  for (int i = 0; i < getNumOperands(); i++) {
    auto value = getInputs()[i];
    broadcast_tensor_reshape(getOutput(), value);
    has_scalar = has_scalar || module::isScalar(value.getDefiningOp());
  }
  auto out_shape = module::getShape(getOutput());
  if (out_shape.size() == 1 && out_shape[0] == 1 && has_scalar) {
    auto context = getContext();
    mlir::Builder builder(context);
    setIsScalarAttr(builder.getBoolAttr(true));
  }
  if (llvm::find_if(getOperands(), module::isShape) != getOperands().end()) {
    auto inputs = getInputs();
    std::vector<std::vector<int64_t>> input_shapes_v;
    for (auto in_op : inputs) {
      if (module::isShape(in_op)) {
        auto input_shape_v = module::getShapeTensorValue(in_op);
        input_shapes_v.push_back(input_shape_v);
      } else if (module::isWeight(in_op)) {
        auto data = in_op.getDefiningOp<top::WeightOp>().read_as_float();
        std::vector<int64_t> data_v(data->begin(), data->end());
        input_shapes_v.push_back(data_v);
      } else {
        llvm_unreachable("Dynamic type is illegal");
      }
    }
    if (out_shape.size() == 1 || out_shape.size() == 0) {
      auto output_shape_v = module::commonShapeValInfer(
          getOperation(), input_shapes_v, out_shape);
      module::bindShapeTensorValue(getOutput(), output_shape_v);
    } else {
      dump();
      llvm::errs() << "WARNING: Shape Type Tensor is calculating with a Tensor "
                      "dimension > 1\n";
    }
  }
}
