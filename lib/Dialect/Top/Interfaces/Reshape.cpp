//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

int64_t top::ReshapeOp::getFLOPs() { return 0; }

LogicalResult top::ReshapeOp::init(InferenceParameter &p) { return success(); }
void top::ReshapeOp::deinit(InferenceParameter &p) {}

LogicalResult top::ReshapeOp::inference(InferenceParameter &p) {
  if (p.inputs[0] != p.outputs[0]) {
    auto num_elem = module::getNumElements(getOutput());
    memcpy(p.outputs[0], p.inputs[0], num_elem * sizeof(float));
  }
  return success();
}

void top::ReshapeOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  auto num = module::getNumElements(getInput());
  std::vector<int64_t> out_shape;
  if (getShape().has_value()) {
    auto shape = module::getI64Array(getShape().value());
    int x = -1;
    for (int i = 0; i < shape->size(); i++) {
      auto s = shape->at(i);
      if (s > 0) {
        out_shape.push_back(s);
        num /= s;
      } else if (s == 0) {
        out_shape.push_back(in_shape[i]);
        num /= in_shape[i];
      } else if (s == -1) {
        out_shape.push_back(-1);
        x = i;
      } else {
        dump();
        llvm_unreachable("shape is illegal");
      }
    }
    if (x >= 0) {
      out_shape[x] = num;
    }
    module::setShapeOrVerify(getOutput(), out_shape);
  } else {
    /* for unranked tensor as below, sema is ok, don;t check it
      %294 = "top.Reshape"(%293) : (tensor<1xf32>) -> tensor<*xf32> loc(#loc294) */
    //assert(module::isUnranked(getOutput()) == false);
  }

  if (!module::isUnranked(getOutput())) {
    auto num_input = module::getNumElements(getInput());
    auto num_output = module::getNumElements(getOutput());
    assert(num_input == num_output);
  }

  removeShapeAttr();
}
