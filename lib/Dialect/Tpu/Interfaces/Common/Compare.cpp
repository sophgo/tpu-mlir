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

LogicalResult tpu::CompareOp::init(InferenceParameter &p) { return success(); }
void tpu::CompareOp::deinit(InferenceParameter &p) {}

static bool eq(float a, float b) { return a == b; }
static bool gt(float a, float b) { return a > b; }
static bool ge(float a, float b) { return a >= b; }
static bool lt(float a, float b) { return a < b; }
static bool le(float a, float b) { return a <= b; }

LogicalResult tpu::CompareOp::inference(InferenceParameter &p) {
  auto num_element = Module::getNumElements(output());
  typedef bool(*cmp_func_t)(float, float);
  const cmp_func_t cmp_funcs[] = {eq, gt, ge, lt, le};
  const cmp_func_t compare = cmp_funcs[type()];
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    p.outputs[0][i] = compare(p.inputs[0][i], p.inputs[1][i]);
  }
  return success();
}
