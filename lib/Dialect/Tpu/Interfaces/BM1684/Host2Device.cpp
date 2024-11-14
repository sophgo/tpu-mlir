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

void tpu::Host2DeviceOp::codegen_global_bm1684() {
  llvm_unreachable("Not supported now");
}

uint32_t tpu::Host2DeviceOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  int fw_ir_length = 0;
  ir_layer_info_t *add_layer_info = (ir_layer_info_t *)ir_layer_info;
  dynamic_common_ir_layer_info(add_layer_info, getInput(), getOutput());
  auto extra_len = sizeof(int);
  u8 extra_version = 0; // for upgrade
  auto extra_buffer =
      (int *)add_layer_info->set_extra_buffer(extra_len, extra_version);
  extra_buffer[0] = 0;
  if (add_layer_info->extra_len > 0) {
    fw_ir_length += sizeof(u32);
    fw_ir_length += add_layer_info->extra_len;
  }
  fw_ir_length += sizeof(int);
  return fw_ir_length;
}

int64_t tpu::Host2DeviceOp::get_fw_type_bm1684() {
  return FW_BMNET_HOST2DEVICE;
}
