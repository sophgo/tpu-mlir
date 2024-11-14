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

void tpu::Pool2DOp::codegen_global_bm1684() {
  auto attr = parseParam();
  bool is_avg_pooling = getPoolMode() == tpu::PoolMode::Avg;
  auto in_addr = module::getAddress(getInput());
  auto out_addr = module::getAddress(getOutput());
  if (module::isUniformQuantized(getInput())) {
    auto in_sign = module::isSign(getInput());
    auto out_sign = module::isSign(getOutput());
    BM1684::instance()
        .dl_nodechip_pooling_fix8b_forward_parallel_with_data_split(
            in_addr, out_addr, attr.n, attr.c, attr.ih, attr.iw, attr.kh,
            attr.kw, attr.pad_h, attr.pad_h_after, attr.pad_w, attr.pad_w_after,
            attr.sh, attr.sw, 0, 0, is_avg_pooling ? 1 : 0, 0, 0, 0, 0, in_sign,
            0, 0, out_sign, attr.do_relu ? 1 : 0,
            (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    // F32
    BM1684::instance().dl_nodechip_pooling_forward_parallel_with_data_split(
        in_addr, out_addr, attr.n, attr.c, attr.ih, attr.iw, attr.oh, attr.ow,
        attr.kh, attr.kw, attr.pad_h, attr.pad_w, attr.sh, attr.sw,
        is_avg_pooling ? 1 : 0, attr.count_include_pad ? 0 : 1,
        attr.do_relu ? 1 : 0, (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
}

int64_t tpu::Pool2DOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::Pool2DOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                         local_sec_info_t &sec_info) {
  auto out_gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto p = parseParam();
  int bottom_dim[4] = {(int)in_gi.n_slice, (int)p.c, (int)in_gi.h_slice,
                       (int)p.iw};
  int top_dim[4] = {(int)out_gi.n_slice, (int)p.c, (int)out_gi.h_slice,
                    (int)p.ow};
  bool is_avg_pooling = getPoolMode() == tpu::PoolMode::Avg;
  int pad_h_t = (in_gi.h_idx == 0 ? p.pad_h : 0);
  int pad_h_b = (in_gi.h_idx + in_gi.h_slice == p.ih ? p.pad_h_after : 0);
  int pad_w_l = (in_gi.w_idx == 0 ? p.pad_w : 0);
  int pad_w_r = (in_gi.w_idx + in_gi.w_slice == p.iw ? p.pad_w_after : 0);
  if (module::isUniformQuantized(getInput())) {
    int bottom_sign = module::isSign(getInput()) ? 1 : 0;
    int top_sign = module::isSign(getOutput()) ? 1 : 0;
    BM1684::instance().dl_nodechip_pooling_fix8b_forward_local(
        in_gi.out_addr, /*weight*/ 0, /*bias*/ 0, out_gi.out_addr, bottom_dim,
        top_dim, p.kh, p.kw, pad_h_t, pad_h_b, pad_w_l, pad_w_r, p.sh, p.sw,
        /*ins_h*/ 0, /*ins_w*/ 0, is_avg_pooling ? 1 : 0,
        p.count_include_pad ? 0 : 1, // avg_pooling_mode should be always 0,
                                     // BM1684 fixpoint not support
        /*shift*/ 0, /*bias*/ 0, /*shift_type*/ 0, bottom_sign, 0, 0, top_sign,
        p.do_relu, (CMD_ID_NODE *)BM1684::instance()->bdc_node);
  } else {
    BM1684::instance().dl_nodechip_pooling_forward_local(
        in_gi.out_addr, out_gi.out_addr, bottom_dim, top_dim, p.kh, p.kw,
        pad_h_t, pad_h_b, pad_w_l, pad_w_r, p.sh, p.sw, is_avg_pooling ? 1 : 0,
        p.count_include_pad ? 0 : 1,
        (CMD_ID_NODE *)BM1684::instance()->bdc_node, p.do_relu);
  }
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================

uint32_t tpu::Pool2DOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  uint32_t fw_ir_length = 0;
  ir_layer_info_t *pool_layer_info = (ir_layer_info_t *)ir_layer_info;
  dynamic_common_ir_layer_info(pool_layer_info, getInput(), getOutput());

  fw_pool_layer_param_t fw_pool_layer_param = {0};
  assign_fw_param((void *)&fw_pool_layer_param);
  pool_layer_info->fw_layer_param_u.fw_pool_layer_param = fw_pool_layer_param;
  fw_ir_length += sizeof(fw_pool_layer_param_t);
  return fw_ir_length;
}

int64_t tpu::Pool2DOp::get_fw_type_bm1684() { return FW_BMNET_POOL; }

// ======================================
// Dynamic LocalGenInterface
// ======================================

int32_t tpu::Pool2DOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  int fw_ir_length = 0;
  ir_layer_info_t *pool_layer_info = (ir_layer_info_t *)ir_layer_info;
  dynamic_common_ir_layer_info(pool_layer_info, getInput(), getOutput());

  // get fw layer param
  fw_pool_layer_param_t fw_pool_layer_param = {0};
  assign_fw_param((void *)&fw_pool_layer_param);
  pool_layer_info->fw_layer_param_u.fw_pool_layer_param = fw_pool_layer_param;
  fw_ir_length += sizeof(fw_pool_layer_param_t);

  // input tensor
  dynamic_push_back_local_tensor(pool_layer_info->ir_tensor_info_v, getInput());
  // output
  dynamic_push_back_local_tensor(pool_layer_info->ir_tensor_info_v,
                                 getOutput());

  // compute fw ir info length for pooling input and output
  fw_ir_length += (sizeof(uint32_t) + 2 * sizeof(uint32_t));

  // add fw ir length for output consumer number
  fw_ir_length += sizeof(u32);

  return fw_ir_length;
}
