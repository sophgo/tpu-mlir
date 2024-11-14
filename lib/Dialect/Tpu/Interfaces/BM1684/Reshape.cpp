//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

void tpu::ReshapeOp::codegen_global_bm1684() {
  auto in_addr = module::getAddress(getInput());
  auto out_addr = module::getAddress(getOutput());
  if (in_addr == out_addr) {
    return;
  }
  int64_t in, ic, ih, iw, on, oc, oh, ow;
  module::getNCHW(getInput(), in, ic, ih, iw);
  module::getNCHW(getOutput(), on, oc, oh, ow);
  if (on != in) {
    // 4N->1N
    auto buffer_addr = module::getAddress(getBuffer());
    BM1684::instance().dl_nodechip_reshape_fix8b(
        in_addr, buffer_addr, in, ic, ih, iw, in, ic, ih, iw, 2, 0,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
    // 1N->4N
    BM1684::instance().dl_nodechip_reshape_fix8b(
        buffer_addr, out_addr, on, oc, oh, ow, on, oc, oh, ow, 0, 2,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    int total_num = align_up(on, 4l) * oc * oh * ow;
    BM1684::instance().dl_nodechip_global_memcpy_ex(
        in_addr, out_addr, 1, total_num, total_num, DTYPE_FP32, DTYPE_FP32,
        total_num, (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
}

// ======================================
// LocalGenInterface
// ======================================
int64_t tpu::ReshapeOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::ReshapeOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                          local_sec_info_t &sec_info) {
  // do nothing
  llvm_unreachable("Not supported now");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================

uint32_t tpu::ReshapeOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  uint32_t fw_ir_length = 0;
  fw_reshape_layer_param_t fw_reshape_layer_param = {0};
  fw_reshape_layer_param.bottom_tensor_id = get_tensor_id(getInput());
  fw_reshape_layer_param.new_dims = module::getShape(getOutput()).size();
  module::getGlobalShape(getOutput(), fw_reshape_layer_param.new_shape);
  fw_reshape_layer_param.global_buffer_addr = 0; // not support buffer
  int input_shape[MAX_SHAPE_DIMS];
  module::getGlobalShape(getInput(), input_shape);
  fw_reshape_layer_param.bottom_n = input_shape[0];
  fw_reshape_layer_param.bottom_c = input_shape[1];

  ir_layer_info_t *reshape_layer_info = (ir_layer_info_t *)ir_layer_info;
  reshape_layer_info->data_size =
      get_dynamic_compiler_tensor_datasize(getInput());
  reshape_layer_info->intensor_store_mode = BM168x::getStoreMode(getInput());
  reshape_layer_info->outtensor_store_mode = BM168x::getStoreMode(getOutput());
  reshape_layer_info->fw_layer_param_u.fw_reshape_layer_param =
      fw_reshape_layer_param;
  fw_ir_length += sizeof(fw_reshape_layer_param_t);
  return fw_ir_length;
}

int64_t tpu::ReshapeOp::get_fw_type_bm1684() { return FW_BMNET_RESHAPE; }

// ======================================
// Dynamic LocalGenInterface
// ======================================

int32_t tpu::ReshapeOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  // do nothing
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
