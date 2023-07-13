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

void tpu::SoftmaxOp::codegen_global_bm1684() {
  auto input = getInput();
  auto output = getOutput();
  auto axis = getAxis();
  auto log = getLog();
  auto in_addr = module::getAddress(input);
  auto out_addr = module::getAddress(output);
  uint64_t buffer_addr = 0;
  auto in_dtype = module::getStorageType(input);
  double in_scale = 0.f;
  int in_tensor_global_store_mode = 0;
  int bottom_prec = 0;
  if (module::isUniformQuantized(input)) {
    buffer_addr = module::getAddress(getBuffer());
    auto qtype = module::getUniformQuantizedType(input);
    in_scale = qtype.getScale();
    in_tensor_global_store_mode = 2;
    bottom_prec = in_dtype.isSignedInteger() ? 1 : 2;
  }
  if (module::isUniformQuantized(output)) {
    llvm_unreachable("Not supported now");
    return;
  }
  int *input_shape = new int[MAX_SHAPE_DIMS];
  module::getGlobalShape(input, input_shape);
  int outer_num = 1;
  int inner_num = 1;
  int softmax_num = input_shape[axis];
  int input_dim = module::getShape(input).size();
  for (int i = 0; i < axis; i++) {
    outer_num *= input_shape[i];
  }
  for (int i = axis + 1; i < input_dim; ++i) {
    inner_num *= input_shape[i];
  }
  BM1684::instance().dl_nodechip_softmax_forward_parallel(
      in_addr, out_addr, outer_num, softmax_num, inner_num, 1, input_shape[0],
      input_shape[1], input_shape[2], input_shape[3],
      in_tensor_global_store_mode, buffer_addr, bottom_prec, (float)in_scale,
      log, (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  delete[] input_shape;
}

// =========================================
// LocalGenInterface
// =========================================
int64_t tpu::SoftmaxOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::SoftmaxOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                          local_sec_info_t &sec_info) {
  llvm_unreachable("Not supported now");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================

uint32_t tpu::SoftmaxOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  uint32_t fw_ir_length = 0;
  ir_layer_info_t *softmax_layer_info = (ir_layer_info_t *)ir_layer_info;
  dynamic_common_ir_layer_info(softmax_layer_info, getInput(), getOutput());
  fw_softmax_layer_param_t fw_softmax_layer_param = {0};
  fw_softmax_layer_param.softmax_dim = (uint32_t)getAxis() << 28;
  fw_softmax_layer_param.scale_val = 0.f;
  if (module::isUniformQuantized(getInput())) {
    auto qtype = module::getUniformQuantizedType(getInput());
    fw_softmax_layer_param.scale_val = qtype.getScale();
  }
  fw_softmax_layer_param.log = getLog();

  if (softmax_layer_info->intensor_store_mode == 2) {
    fw_softmax_layer_param.global_offset_1N_buf =
        module::getAddress(getBuffer());
  }
  softmax_layer_info->fw_layer_param_u.fw_softmax_layer_param =
      fw_softmax_layer_param;
  fw_ir_length += sizeof(fw_softmax_layer_param_t);
  return fw_ir_length;
}

int64_t tpu::SoftmaxOp::get_fw_type_bm1684() { return FW_BMNET_SOFTMAX; }

int32_t tpu::SoftmaxOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  llvm_unreachable("not implement");
}