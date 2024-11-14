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

void tpu::ConcatOp::codegen_global_bm1684() {
  if (getOnlyMerge() &&
      module::getAddress(getInputs()[0]) == module::getAddress(getOutput())) {
    return;
  }
  int num_input = getInputs().size();
  int(*bottomtensor_shape)[MAX_SHAPE_DIMS] = new int[num_input][MAX_SHAPE_DIMS];
  SmallVector<int> is_st_concat_way(num_input, 0);
  SmallVector<uint64_t> in_addr(num_input, 0);
  auto out_addr = module::getAddress(getOutput());
  for (int i = 0; i < num_input; ++i) {
    in_addr[i] = module::getAddress(getInputs()[i]);
    module::getGlobalShape(getInputs()[i], bottomtensor_shape[i]);
  }
  int out_shape[MAX_SHAPE_DIMS] = {0};
  module::getGlobalShape(getOutput(), out_shape);
  if (false == module::isUniformQuantized(getOutput())) {
    BM1684::instance().dl_nodechip_concat_md(
        getAxis(), module::getShape(getInputs()[0]).size(), getInputs().size(),
        in_addr.data(), out_addr, bottomtensor_shape, out_shape,
        is_st_concat_way.data(), (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    BM1684::instance().dl_nodechip_concat_md_fix8b(
        getAxis(), module::getShape(getInputs()[0]).size(), getInputs().size(),
        in_addr.data(), out_addr, bottomtensor_shape, out_shape,
        is_st_concat_way.data(), 2 /* in_stmode == out_stmode */, 2,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
  delete[] bottomtensor_shape;
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::ConcatOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::ConcatOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                         local_sec_info_t &sec_info) {
  auto gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  int num_inputs = getInputs().size();
  SmallVector<int> is_st_concat_way(num_inputs, 0);
  SmallVector<uint32_t> in_addr(num_inputs, 0);
  auto bottomtensor_shape = new int *[num_inputs];
  for (int i = 0; i < num_inputs; i++) {
    auto ingi = LocalGenInterface::getGroupInfo(getInputs()[i], n_step, h_step);
    in_addr[i] = ingi.out_addr;
    bottomtensor_shape[i] = new int[4];
    module::getLocalShape(getInputs()[i], n_step, h_step,
                          bottomtensor_shape[i]);
  }
  int out_shape[MAX_SHAPE_DIMS] = {0};
  module::getLocalShape(getOutput(), n_step, h_step, out_shape);
  if (false == module::isUniformQuantized(getOutput())) {
    BM1684::instance().dl_nodechip_concat_local_v2(
        in_addr.data(), gi.out_addr, bottomtensor_shape, num_inputs,
        is_st_concat_way.data(), out_shape, getAxis(),
        (CMD_ID_NODE *)BM1684::instance()->bdc_node,
        (CMD_ID_NODE *)BM1684::instance()->gdma_node);
  } else {
    BM1684::instance().dl_nodechip_concat_fix8b_local_v2(
        in_addr.data(), gi.out_addr, bottomtensor_shape, num_inputs,
        is_st_concat_way.data(), out_shape, getAxis(),
        (CMD_ID_NODE *)BM1684::instance()->bdc_node,
        (CMD_ID_NODE *)BM1684::instance()->gdma_node);
  }
  for (int i = 0; i < num_inputs; ++i) {
    delete[] bottomtensor_shape[i];
  }
  delete[] bottomtensor_shape;
}

uint32_t tpu::ConcatOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  uint32_t fw_ir_length = 0;
  ir_layer_info_t *concat_layer_info = (ir_layer_info_t *)ir_layer_info;
  dynamic_common_ir_layer_info(concat_layer_info, getInputs()[0], getOutput());
  assign_fw_param(
      (void *)&concat_layer_info->fw_layer_param_u.fw_concat_layer_param);
  auto extra_len = sizeof(fw_concat_input_info_t) * getInputs().size();
  u8 extra_version = 0;
  auto extra_info =
      (fw_concat_input_info_t *)concat_layer_info->set_extra_buffer(
          extra_len, extra_version);
  for (int i = 0; i < getInputs().size(); ++i) {
    extra_info[i].is_coeff = module::isWeight(getInputs()[i]);
    extra_info[i].concat_size = module::getShape(getInputs()[i])[getAxis()];
    extra_info[i].st_way = 0; // no use
  }
  fw_ir_length += sizeof(fw_concat_layer_param_t);
  assert(concat_layer_info->extra_len > 0);
  fw_ir_length += sizeof(uint32_t);
  fw_ir_length += concat_layer_info->extra_len;
  return fw_ir_length;
}

int64_t tpu::ConcatOp::get_fw_type_bm1684() { return FW_BMNET_CONCAT; }

int32_t tpu::ConcatOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  int fw_ir_length = 0;
  ir_layer_info_t *concat_layer_info = (ir_layer_info_t *)ir_layer_info;
  dynamic_common_ir_layer_info(concat_layer_info, getInputs()[0], getOutput());

  fw_concat_loc_layer_param_t fw_concat_layer_param = {0};
  int out_shape[MAX_SHAPE_DIMS];
  module::getLocalShape(getOutput(), 0, 0, out_shape);
  fw_concat_layer_param.c = out_shape[1];
  fw_concat_layer_param.h = out_shape[2];
  fw_concat_layer_param.w = out_shape[3];
  fw_concat_layer_param.version = 2;
  fw_concat_layer_param.concat_axis = getAxis();
  concat_layer_info->fw_layer_param_u.fw_concat_loc_layer_param =
      fw_concat_layer_param;
  fw_ir_length += sizeof(fw_concat_loc_layer_param_t);

  // output
  dynamic_push_back_local_tensor(concat_layer_info->ir_tensor_info_v,
                                 getOutput());
  // input
  int in_tensors_num = getInputs().size();
  auto extra_len = sizeof(fw_concat_input_info_t) * in_tensors_num;
  auto extra_info =
      (fw_concat_input_info_t *)concat_layer_info->set_extra_buffer(extra_len);
  memset(extra_info, 0, extra_len);
  for (int i = 0; i < in_tensors_num; ++i) {
    dynamic_push_back_local_tensor(concat_layer_info->ir_tensor_info_v,
                                   getInputs()[i]);
    extra_info[i].is_coeff = module::isWeight(getInputs()[i]);
    extra_info[i].concat_size = module::getShape(getInputs()[i])[getAxis()];
    extra_info[i].st_way = 0; // no use
  }

  // compute fw ir info length for loc concat input and output
  fw_ir_length +=
      (sizeof(uint32_t) + align_up(in_tensors_num, 2) / 2 * sizeof(uint32_t) +
       (1 + in_tensors_num) * sizeof(uint32_t)) +
      in_tensors_num * sizeof(fw_concat_input_info_t);
  fw_ir_length += sizeof(uint32_t);
  return fw_ir_length;
}
