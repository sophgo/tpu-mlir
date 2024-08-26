//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/LutFunc.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

LogicalResult tpu::LutOp::init(InferenceParameter &p) { return success(); }
void tpu::LutOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::LutOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getInput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    int offset = p.inputs[0][i];
    if (offset < 0) {
      offset += 256;
    }
    assert(offset >= 0 && offset <= 255);
    p.outputs[0][i] = p.inputs[1][offset];
  }
  return success();
}

LogicalResult tpu::LutOp::LocalGenSupport() { return success(); }

void tpu::LutOp::assign_fw_param(void *param) {
  fw_lut_layer_param_t layer_param = {0};
  layer_param.lut_is_coeff = module::isWeight(getTable());
  layer_param.index_is_coeff = module::isWeight(getInput());
  layer_param.index_dim = module::getShape(getInput()).size();
  uint8_t stmode_dtype =
      BM1684::getStoreMode(getInput()) == STORE_MODE_1N ? 1 : 0;
  auto out_dtype = BM1684::getDataType(getOutput());
  stmode_dtype +=
      out_dtype == DTYPE_INT8 || out_dtype == DTYPE_UINT8 ? 1 << 2 : 0;
  stmode_dtype +=
      BM1684::getStoreMode(getOutput()) == STORE_MODE_4N ? 1 << 4 : 0;
  layer_param.stmode_dtype = stmode_dtype;
  layer_param.dtype = get_dynamic_compiler_tensor_datasize(getOutput());
  int input_shape[MAX_SHAPE_DIMS];
  module::getGlobalShape(getInput(), input_shape);
  memcpy(layer_param.index_shape, input_shape,
         sizeof(int) * layer_param.index_dim);
  memcpy(param, &layer_param, sizeof(fw_lut_layer_param_t));
}

ArrayAttr tpu::LutOp::getIndexingMaps() {
  return getBinaryIndexingMaps(getOperation());
};

bool tpu::LutOp::support_multi_core() { return false; }
