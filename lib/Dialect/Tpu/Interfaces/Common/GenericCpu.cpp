//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::GenericCpuOp::init(InferenceParameter &p) {
  return success();
}
void tpu::GenericCpuOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::GenericCpuOp::inference(InferenceParameter &p) {
  std::string func_name = operation_name().str();
  if (func_name == "quant") {
    assert(inputs().size() == 1);
    auto num_elem = Module::getNumElements(output());
    auto in_type = Module::getStorageType(inputs()[0]);
    auto out_type = Module::getStorageType(output());
    bool isInQuant = Quant::isUniformQuantized(inputs()[0]);
    bool isOutQuant = Quant::isUniformQuantized(output());
    auto op = getOperation();
    if (in_type.isF32() && out_type.isSignedInteger()) {
      auto qtype = Quant::getUniformQuantizedType(output());
      quantizeToInt8(p.inputs[0], p.outputs[0], num_elem, 1. / qtype.getScale(),
                     ROUNDING_HALF_DOWN);
    } else {
      llvm_unreachable("not supported!\n");
    }

  } else {
    llvm_unreachable("generic cpu func not supported!\n");
  }
  return success();
}
