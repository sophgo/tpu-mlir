//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/CastUtils.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Float8.h"

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Interfaces/IndexingMapsInterface.h"

LogicalResult tpu::CastAddOp::init(InferenceParameter &p) {
  p.handle = nullptr;
  return success();
}

void tpu::CastAddOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult tpu::CastAddOp::inference(InferenceParameter &p) {
  auto output_shape = computer_broadcast_shape(getOperation());
  module::setShape(getOutput(), output_shape);

  auto in0_shape = module::getShape(getInputs()[0]);
  auto in1_shape = module::getShape(getInputs()[1]);
  auto binary = new Binary();
  // fix me. naive impletment.
  // It should be o = alpha * i0 + beta * i1
  bool is_add = true;

  auto num_elem_in0 = module::getNumElements(getInputs()[0]);
  auto num_elem_in1 = module::getNumElements(getInputs()[1]);
  float temp_input_0[num_elem_in0];
  float temp_input_1[num_elem_in1];
  for (int64_t i = 0; i < num_elem_in0; i++) {
    temp_input_0[i] = p.inputs[0][i];
  }

  for (int64_t i = 0; i < num_elem_in1; i++) {
    temp_input_1[i] = p.inputs[1][i];
  }
  if (module::isMARS3()) {
    auto in0_type = module::getStorageType(getInputs()[0]);
    auto in1_type = module::getStorageType(getInputs()[1]);
    bool flagPatternA =
        module::isUniformQuantized(getInputs()[0]) && in1_type.isBF16();
    bool flagPatternB =
        in0_type.isBF16() && module::isUniformQuantized(getInputs()[1]);
    if (flagPatternA || flagPatternB) {
      if (module::isUniformQuantized(getInputs()[0])) {
        auto qtype_in0 = module::getUniformQuantizedType(getInputs()[0]);
        for (int64_t i = 0; i < num_elem_in0; i++) {
          p.inputs[0][i] = dequant(p.inputs[0][i], qtype_in0);
        }
      }
      if (module::isUniformQuantized(getInputs()[1])) {
        auto qtype_in1 = module::getUniformQuantizedType(getInputs()[1]);
        for (int64_t i = 0; i < num_elem_in1; i++) {
          p.inputs[1][i] = dequant(p.inputs[1][i], qtype_in1);
        }
      }
    }
  }

  (*binary)
      .hs(p.inputs[0], p.inputs[1], in0_shape, in1_shape)
      .dst(p.outputs[0], module::getShape(getOutput()))
      .do_relu(getDoRelu())
      .relu_limit(getReluLimit().convertToDouble())
      .algorithem(is_add ? algorithm::binary_add : algorithm::binary_sub)
      .setup();
  p.handle = (void *)binary;

  auto num_elem = module::getNumElements(getOutput());
  auto out_type = module::getStorageType(getOutput());
  memset(p.outputs[0], 0, num_elem * sizeof(float));
  if (out_type.isa<FloatType>()) {
    auto binary = (Binary *)p.handle;
    binary->run();
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isFloat8E5M2()) {
      F8E5M2(p.outputs[0], p.outputs[0], num_elem, 1.0, true);
    }
  } else if (out_type.isInteger(32)) {
    auto binary = (Binary *)p.handle;
    binary->run();
  } else {
    UNREACHABLE_THIS(0);
  }

  return success();
}

bool tpu::CastAddOp::support_multi_core() { return false; }
