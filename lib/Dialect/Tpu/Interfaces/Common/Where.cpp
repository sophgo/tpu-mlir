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

LogicalResult tpu::WhereOp::init(InferenceParameter &p) { return success(); }
void tpu::WhereOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::WhereOp::inference(InferenceParameter &p) {
  const auto num_element = Module::getNumElements(output());
  const float tbrn_const = tbrn_const_val().convertToDouble();
  const float fbrn_const = fbrn_const_val().convertToDouble();
  int idx = 1;
  const int tbrn_idx = tbrn_is_const() ? -1 : (idx++);
  const int fbrn_idx = fbrn_is_const() ? -1 : (idx++);
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    const float tbrn = tbrn_is_const() ? tbrn_const : p.inputs[tbrn_idx][i];
    const float fbrn = fbrn_is_const() ? fbrn_const : p.inputs[fbrn_idx][i];
    p.outputs[0][i] = p.inputs[0][i] ? tbrn : fbrn;
  }
  return success();
}
