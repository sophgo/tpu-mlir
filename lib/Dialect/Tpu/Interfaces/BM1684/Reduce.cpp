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

int64_t tpu::ReduceOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  llvm_unreachable("Not supported now");
  return 0;
}

void tpu::ReduceOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                         local_sec_info_t &sec_info) {
  llvm_unreachable("unimplemented local reduceOp.");
}

void tpu::ReduceOp::codegen_global_bm1684() {
  int i_dims = module::getShape(getInput()).size();
  auto in_addr = module::getAddress(getInput());
  auto out_addr = module::getAddress(getOutput());
  // int input_shapes[i_dims];
  // module::getGlobalShape(getInput(), input_shapes);
  uint32_t *input_shape = new uint32_t[MAX_SHAPE_DIMS];
  for (auto v : llvm::enumerate(module::getShape(getInput())))
    input_shape[v.index()] = (uint32_t)v.value();
  int method = BM1684::get_reduce_type(getMode());
  auto &&axes = getAxes();
  int axis_num = axes.size();
  int axis_list[axis_num];
  for (int i = 0; i < axes.size(); i++)
    axis_list[i] = (axes[i].cast<IntegerAttr>().getInt());
  auto buffer_addr = module::getAddress(getBuffer());
  if (false == module::isUniformQuantized(getInput())) {
    BM1684::instance().dl_nodechip_reduce_full_v3(
        in_addr, out_addr, input_shape, i_dims, axis_list, axis_num, method,
        buffer_addr, 0, (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    int keep_dims = getKeepdims() ? 1 : 0;
    int bottom_sign = module::isSign(getInput()) ? 1 : 0;
    int store_mode = STORE_MODE_4N;
    float bottom_scale = 1.0f;
    float top_scale = 1.0f;
    BM1684::instance().dl_nodechip_reduce_full_fix8b(
        in_addr, out_addr, buffer_addr, input_shape, i_dims, axis_list,
        axis_num, method, keep_dims, bottom_sign, store_mode, bottom_scale,
        top_scale, (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
}

uint32_t tpu::ReduceOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  ir_layer_info_t *layer_info = (ir_layer_info_t *)ir_layer_info;
  dynamic_common_ir_layer_info(layer_info, getInput(), getOutput());
  fw_reduce_full_layer_param_t layer_param = {0};
  layer_param.reduce_method = BM1684::get_reduce_type(getMode());
  layer_param.axis_num = getAxes().size();
  short version = 1;
  layer_param.keep_dims =
      ((uint8_t)version << 8) | ((uint8_t)getKeepdims()); // reduce_full_v3
  auto axes = module::getI64Array(getAxes());
  for (int i = 0; i < getAxes().size(); i++) {
    layer_param.axis_list[i] = axes->at(i);
  }
  layer_param.input_sign = module::isSign(getInput());
  layer_param.input_scale = 1.f; // not implement
  layer_param.output_scale = 1.f;
  layer_info->fw_layer_param_u.fw_reduce_full_layer_param = layer_param;
  return sizeof(fw_reduce_full_layer_param_t);
}

int64_t tpu::ReduceOp::get_fw_type_bm1684() { return FW_BMNET_REDUCE_FULL; }
