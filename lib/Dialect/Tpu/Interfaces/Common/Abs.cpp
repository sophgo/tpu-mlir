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
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Float16.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::AbsOp::init(InferenceParameter &p) { return success(); }
void tpu::AbsOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::AbsOp::inference(InferenceParameter &p) {
  auto module = Module::getModuleOp(getOperation());
  auto num_elem = Module::getNumElements(output());
  auto out_type = Module::getStorageType(output());
  memset(p.outputs[0], 0, num_elem * sizeof(float));
  auto num_element = Module::getNumElements(input());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::abs(val);
  }

  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      f32_to_bf16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      f32_to_f16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (out_type.isSignedInteger(8)){
     std::cout << "================Abs Op can't support int8=================" << std::endl;
  }
  return success();
}

LogicalResult tpu::AbsOp::LocalGenSupport() {
  // BackwardH and BackwardN can not handle more than one input right now.
  // The same n_slice and h_slice value will propagate to each inputs.
  // Thus, the local layer is only safe when we do not need to slice n and h
  // dimensions.
  return success();
}
