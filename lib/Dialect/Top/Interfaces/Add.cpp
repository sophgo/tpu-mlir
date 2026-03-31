//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"

int64_t top::AddOp::getFLOPs() {
  return module::getNumElements(getOutput()) *
         (getInputs().size() - 1 + (getDoRelu() ? 1 : 0));
}

LogicalResult top::AddOp::init(InferenceParameter &p) {
  if (getInputs().size() == 2) {
    auto binary = new Binary();
    // auto lhs_shape = module::getShape(getInputs()[0]);
    // auto rhs_shape = module::getShape(getInputs()[1]);

    // (*binary)
    //     .hs(p.inputs[0], p.inputs[1], lhs_shape, rhs_shape)
    //     .dst(p.outputs[0], module::getShape(getOutput()))
    //     .do_relu(getDoRelu())
    //     .relu_limit(getReluLimit().convertToDouble())
    //     .algorithem(algorithm::binary_add)
    //     .setup();

    p.handle = (void *)binary;
  } else {
    p.handle = nullptr;
  }
  return success();
}
void top::AddOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult top::AddOp::inference(InferenceParameter &p) {
  auto output_shape = computer_broadcast_shape(getOperation());
  module::setShape(getOutput(), output_shape);
  if (getInputs().size() == 2) {
    if (p.handle == nullptr) {
      return failure();
    }

    auto binary = (Binary *)p.handle;

    auto lhs_shape = module::getShape(getInputs()[0]);
    auto rhs_shape = module::getShape(getInputs()[1]);

    (*binary)
        .hs(p.inputs[0], p.inputs[1], lhs_shape, rhs_shape)
        .dst(p.outputs[0], module::getShape(getOutput()))
        .do_relu(getDoRelu())
        .relu_limit(getReluLimit().convertToDouble())
        .algorithem(algorithm::binary_add)
        .setup();

    binary->run();
  } else {
    auto num_elem = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      p.outputs[0][i] = 0;
      for (auto in : p.inputs) {
        if (in != nullptr) {
          p.outputs[0][i] += in[i];
        }
      }
    }
    if (getDoRelu()) {
      auto limit = getReluLimit().convertToDouble();
      function_relu(p.outputs[0], p.outputs[0], num_elem, limit);
    }
  }

  return success();
}

void top::AddOp::shape_inference() {
  broadcast_shape_inference(getOperation());
  bool has_scalar = false;
  for (int i = 0; i < getNumOperands(); i++) {
    auto value = getInputs()[i];
    broadcast_tensor_reshape(getOutput(), value);
    has_scalar = has_scalar && module::isScalar(value.getDefiningOp());
  }
  auto inputs = getInputs();
  // shape value inference can only support shape and weight
  bool need_shape_val_infer =
      std::all_of(inputs.begin(), inputs.end(),
                  [](auto in_op) {
                    return module::isWeight(in_op) || module::isShape(in_op);
                  }) &&
      std::any_of(inputs.begin(), inputs.end(),
                  [](auto in_op) { return module::isShape(in_op); });
  need_shape_val_infer &=
      std::any_of(inputs.begin(), inputs.end(),
                  [](auto in_op) { return module::isShape(in_op); });
  auto out_shape = module::getShape(getOutput());
  if (out_shape.size() == 1 && out_shape[0] == 1 && has_scalar) {
    auto context = getContext();
    mlir::Builder builder(context);
    setIsScalarAttr(builder.getBoolAttr(true));
  }
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

    auto output_shape_v =
        module::commonShapeValInfer(getOperation(), input_shapes_v, out_shape);
    module::bindShapeTensorValue(getOutput(), output_shape_v);
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
