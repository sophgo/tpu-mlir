//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"

using namespace tpu_mlir::backend;

void tpu::Weight2ActivationOp::codegen_global_bm1684() {
  auto input_addr = module::getAddress(getInput());
  auto output_addr = module::getAddress(getOutput());
  assert(input_addr == output_addr);
}

uint32_t
tpu::Weight2ActivationOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  ir_layer_info_t *layer_info = (ir_layer_info_t *)ir_layer_info;
  dynamic_common_ir_layer_info(layer_info, getInput(), getOutput());
  layer_info->fw_layer_param_u.fw_coeff2neuron_layer_param.dims =
      module::getShape(getInput()).size();
  layer_info->fw_layer_param_u.fw_coeff2neuron_layer_param.dtype =
      BM1684::getDataType(getInput());
  for (int i = 0; i < module::getShape(getInput()).size(); ++i) {
    layer_info->fw_layer_param_u.fw_coeff2neuron_layer_param.shape[i] =
        module::getShape(getInput())[i];
  }
  return sizeof(fw_coeff2neuron_layer_param_t);
}

int64_t tpu::Weight2ActivationOp::get_fw_type_bm1684() {
  return FW_BMNET_COEFF2NEURON;
}
