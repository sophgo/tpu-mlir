//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

void tpu::Conv3DOp::codegen_global_bm1684() {
  auto attr = parseParam();
  auto in_addr = module::getAddress(getInput());
  auto out_addr = module::getAddress(getOutput());
  auto filter_addr = module::getAddress(getFilter());
  auto bias_addr = module::getAddress(getBias());

  if (module::isUniformQuantized(getInput())) {
    // Int8
    auto shift_v = module::getI64Array(getRshift(), 1, 0);
    auto shift = shift_v->at(0);
    auto in_sign = module::isSign(getInput());
    auto filter_sign = module::isSign(getFilter());
    auto bias_sign = attr.has_bias ? module::isSign(getBias()) : 0;
    auto out_sign = module::isSign(getOutput());
    BM1684::instance().dl_nodechip_conv3d_fix8b_parallel(
        in_addr, out_addr, filter_addr, bias_addr, attr.n, attr.ic, attr.id,
        attr.ih, attr.iw, attr.groups, attr.oc, attr.kd, attr.kh, attr.kw,
        attr.dd, attr.dh, attr.dw, attr.pdf, attr.pdb, attr.pht, attr.phb,
        attr.pwl, attr.pwr, attr.sd, attr.sh, attr.sw, attr.has_bias ? 1 : 0,
        attr.do_relu ? 1 : 0, attr.relu_limit, in_sign, out_sign, filter_sign,
        bias_sign, shift, (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    // F32
    // refer to
    // bmcompiler/src/interface/bmcompiler_net_interface.cpp:5284
    int ic_threshold = 10;
    int method = 0;
    if (attr.dd > 1)
      method = 2; // nodechip not implemented
    else if (attr.ic / attr.groups > ic_threshold || attr.dh > 1)
      method = 1;

    BM1684::instance().dl_nodechip_conv3d_parallel(
        in_addr, out_addr, filter_addr, bias_addr, attr.n, attr.ic, attr.id,
        attr.ih, attr.iw, attr.groups, attr.oc, attr.kd, attr.kh, attr.kw,
        attr.dd, attr.dh, attr.dw, attr.pdf, attr.pdb, attr.pht, attr.phb,
        attr.pwl, attr.pwr, attr.sd, attr.sh, attr.sw, attr.has_bias ? 1 : 0,
        attr.do_relu ? 1 : 0, attr.relu_limit, method,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
}

int64_t tpu::Conv3DOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  // TODO for spicial situation
  return 0;
}

void tpu::Conv3DOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                         local_sec_info_t &sec_info) {
  auto out_gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto f_gi = LocalGenInterface::getGroupInfo(getFilter());
  auto b_gi = LocalGenInterface::getGroupInfo(getBias());
  auto attr = parseParam();
  auto in_addr = in_gi.out_addr;
  auto out_addr = out_gi.out_addr;
  auto filter_addr = f_gi.out_addr;
  auto bias_addr = b_gi.out_addr;
  int input_shape[5] = {(int)in_gi.n_slice, (int)attr.ic, (int)in_gi.d_slice,
                        (int)in_gi.h_slice, (int)attr.iw};
  int output_shape[5] = {(int)in_gi.n_slice, (int)attr.oc, (int)out_gi.d_slice,
                         (int)out_gi.h_slice, (int)attr.ow};
  int pad_d = (in_gi.d_idx == 0 ? attr.pdf : 0);
  int pad_d_after = (in_gi.d_idx + in_gi.d_slice == attr.id ? attr.pdb : 0);
  int pad_h = (in_gi.h_idx == 0 ? attr.pht : 0);
  int pad_h_after = (in_gi.h_idx + in_gi.h_slice == attr.ih ? attr.phb : 0);
  int pad_w = (in_gi.w_idx == 0 ? attr.pwl : 0);
  int pad_w_after = (in_gi.w_idx + in_gi.w_slice == attr.iw ? attr.pwr : 0);
  if (module::isUniformQuantized(getInput())) {
    auto in_sign = module::isSign(getInput());
    auto filter_sign = module::isSign(getFilter());
    auto bias_sign = attr.has_bias ? module::isSign(getBias()) : 0;
    auto out_sign = module::isSign(getOutput());
    auto shift_v = module::getI64Array(getRshift(), 1, 0);
    auto shift = shift_v->at(0);
    BM1684::instance().dl_nodechip_conv3d_fix8b_local(
        in_addr, filter_addr, bias_addr, out_addr, input_shape, output_shape,
        attr.kd, attr.kh, attr.kw, attr.dd, attr.dh, attr.dw, pad_d,
        pad_d_after, pad_h, pad_h_after, pad_w, pad_w_after, attr.sd, attr.sh,
        attr.sw, attr.has_bias ? 1 : 0, attr.do_relu ? 1 : 0, in_sign,
        filter_sign, bias_sign, out_sign, shift,
        (CMD_ID_NODE *)BM1684::instance()->bdc_node);
  } else {
    BM1684::instance().dl_nodechip_conv3d_local(
        in_addr, filter_addr, bias_addr, out_addr, input_shape, output_shape,
        attr.groups, attr.kd, attr.kh, attr.kw, attr.dd, attr.dh, attr.dw,
        pad_d, pad_d_after, pad_h, pad_h_after, pad_w, pad_w_after, attr.sd,
        attr.sh, attr.sw, attr.has_bias ? 1 : 0, attr.do_relu ? 1 : 0,
        attr.relu_limit, (CMD_ID_NODE *)BM1684::instance()->bdc_node);
  }
}

uint32_t tpu::Conv3DOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

int64_t tpu::Conv3DOp::get_fw_type_bm1684() { return -1; }

int32_t tpu::Conv3DOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
