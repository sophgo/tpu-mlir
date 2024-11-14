//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Deconv.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"

using namespace tpu_mlir::backend;

void tpu::DeconvOp::codegen_global_bm1684() {
  auto attr = parseParam();
  auto in_addr = module::getAddress(getInput());
  auto out_addr = module::getAddress(getOutput());
  auto filter_addr = module::getAddress(getFilter());
  auto bias_addr = module::getAddress(getBias());
  if (module::isUniformQuantized(getInput())) {
    auto shift_v = module::getI64Array(getRshift(), 1, 0);
    auto shift = shift_v->at(0);
    auto in_sign = module::isSign(getInput());
    auto filter_sign = module::isSign(getFilter());
    auto bias_sign = attr.with_bias ? module::isSign(getBias()) : 0;
    if (attr.is_dw) {
      int kh_ext = (attr.kh - 1) * attr.dh + 1;
      int kw_ext = (attr.kw - 1) * attr.dw + 1;
      BM1684::instance().dl_nodechip_depthwise_fix8b_forward_parallel(
          in_addr, out_addr, filter_addr, bias_addr, attr.n, attr.ic, attr.ih,
          attr.iw, attr.kh, attr.kw, kh_ext - 1 - attr.pad_h,
          kh_ext - 1 - attr.pad_h_after, kw_ext - 1 - attr.pad_w,
          kw_ext - 1 - attr.pad_w_after, 1, 1, attr.sw - 1, attr.sh - 1, shift,
          attr.with_bias, 0 /*rshft_type*/, in_sign, filter_sign, bias_sign,
          module::isSign(getOutput()), attr.do_relu, attr.relu_limit,
          (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
    } else {
      BM1684::instance().dl_nodechip_deconv_fix8b_forward_parallel(
          in_addr, out_addr, filter_addr, bias_addr, attr.n, attr.ic, attr.ih,
          attr.iw, attr.oc, attr.g, attr.kh, attr.kw, attr.dh, attr.dw,
          attr.pad_h, attr.pad_h_after, attr.pad_w, attr.pad_w_after, attr.sh,
          attr.sw, attr.output_pad_h, attr.output_pad_w, attr.with_bias ? 1 : 0,
          attr.do_relu ? 1 : 0, shift, 1, in_sign, filter_sign, bias_sign,
          (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
    }
  } else {
    BM1684::instance().dl_nodechip_deconv_forward_parallel_with_data_split_v2(
        in_addr, out_addr, filter_addr, bias_addr, attr.n, attr.ic, attr.ih,
        attr.iw, attr.g, attr.oc, attr.kh, attr.kw, attr.dh, attr.dw,
        attr.pad_h, attr.pad_h_after, attr.pad_w, attr.pad_w_after, attr.sh,
        attr.sw, attr.output_pad_h, attr.output_pad_w, attr.with_bias ? 1 : 0,
        0, attr.do_relu ? 1 : 0, 1, 1,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
}

int64_t tpu::DeconvOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  // TODO for spicial situation
  return 0;
}

void tpu::DeconvOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                         local_sec_info_t &sec_info) {
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto f_gi = LocalGenInterface::getGroupInfo(getFilter());
  auto b_gi = LocalGenInterface::getGroupInfo(getBias());
  auto gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto p = parseParam();
  int bottom_dim[4] = {(int)in_gi.n_slice, (int)p.ic, (int)in_gi.h_slice,
                       (int)p.iw};
  int top_dim[4] = {(int)gi.n_slice, (int)p.oc, (int)gi.h_slice, (int)p.ow};
  int kh_ext = (p.kh - 1) * p.dh + 1;
  if (auto deconv_in_slice =
          DeconvSlice(gi.h_idx, gi.h_slice, p.sh, kh_ext, p.ih, p.pad_h)) {
    p.pad_h = deconv_in_slice.value()[0];
    p.pad_h_after = deconv_in_slice.value()[1];
  } else {
    p.pad_h = p.kh - p.pad_h - 1;
    p.pad_h_after = p.kh - p.pad_h_after - 1 + p.output_pad_h;
  }
  p.pad_w = p.kw - p.pad_w - 1;
  p.pad_w_after = p.kw - p.pad_w_after - 1 + p.output_pad_w;
  if (module::isUniformQuantized(getInput())) {
    auto shift_v = module::getI64Array(getRshift(), 1, 0);
    auto shift = shift_v->at(0);
    auto in_sign = module::isSign(getInput());
    auto filter_sign = module::isSign(getFilter());
    auto bias_sign = p.with_bias ? module::isSign(getBias()) : 0;
    if (p.is_dw) {
      BM1684::instance().dl_nodechip_pooling_fix8b_forward_local(
          in_gi.out_addr, f_gi.out_addr, b_gi.out_addr, gi.out_addr, bottom_dim,
          top_dim, p.kh, p.kw, p.pad_h, p.pad_h_after, p.pad_w, p.pad_w_after,
          1, 1, p.sh - 1, p.sw - 1, 2 /*is depthwise*/, 0, shift, p.with_bias,
          1 /*shift type*/, in_sign, filter_sign, bias_sign, 0 /*res sign*/,
          p.do_relu, (CMD_ID_NODE *)BM1684::instance()->bdc_node);
    } else {
      BM1684::instance().dl_nodechip_deconv_fix8b_forward_local(
          in_gi.out_addr, f_gi.out_addr, b_gi.out_addr, gi.out_addr, bottom_dim,
          top_dim, p.g, p.kh, p.kw, p.dh, p.dw, p.pad_h, p.pad_h_after, p.pad_w,
          p.pad_w_after, p.sh - 1, p.sw - 1, p.with_bias ? 1 : 0,
          p.do_relu ? 1 : 0, shift, in_sign, filter_sign, bias_sign,
          (CMD_ID_NODE *)BM1684::instance()->bdc_node);
    }
  } else {
    BM1684::instance().dl_nodechip_deconv_forward_local_v2(
        in_gi.out_addr, f_gi.out_addr, b_gi.out_addr, gi.out_addr, bottom_dim,
        top_dim, p.g, p.kh, p.kw, p.dh, p.dw, p.pad_h, p.pad_h_after, p.pad_w,
        p.pad_w_after, p.sh - 1, p.sw - 1, p.with_bias, 0, p.do_relu ? 1 : 0, 1,
        BM1684::instance()->bdc_node);
  }
}

uint32_t tpu::DeconvOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  GLOBAL_IR_COMMON(deconv);
}

int64_t tpu::DeconvOp::get_fw_type_bm1684() { return FW_BMNET_DECONV; }

int32_t tpu::DeconvOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  int fw_ir_length = 0;
  IR_PARAM_COMMON(deconv);
  // input tensor
  dynamic_push_back_local_tensor(layer_info->ir_tensor_info_v, getInput());
  // weight
  dynamic_push_back_local_tensor(layer_info->ir_tensor_info_v, getFilter());
  // bias
  if (getWithBias()) {
    dynamic_push_back_local_tensor(layer_info->ir_tensor_info_v, getBias());
  }
  // output
  dynamic_push_back_local_tensor(layer_info->ir_tensor_info_v, getOutput());
  // compute fw ir info length for deconv input and output
  fw_ir_length += (sizeof(uint32_t) + 3 * sizeof(uint32_t) +
                   getWithBias() * sizeof(uint32_t));
  fw_ir_length += sizeof(uint32_t);
  return fw_ir_length;
}
