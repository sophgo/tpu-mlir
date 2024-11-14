//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::Pool3DOp::codegen_global_bm1684() {
  auto attr = parseParam();
  auto in_addr = module::getAddress(getInput());
  auto out_addr = module::getAddress(getOutput());
  auto buffer_addr = module::getAddress(getBuffer());

  int is_avg_pooling = getPoolMode() == tpu::PoolMode::Avg ? 1 : 0;
  int avg_pooling_mode = attr.count_include_pad ? 0 : 1;
  if (module::isUniformQuantized(getInput())) {
    auto in_sign = module::isSign(getInput());
    auto out_sign = module::isSign(getOutput());
    if (is_avg_pooling == 0) {
      BM1684::instance().dl_nodechip_pooling3d_fix8b_forward_parallel(
          in_addr, buffer_addr, out_addr, attr.n, attr.c, attr.id, attr.ih,
          attr.iw, attr.kd, attr.kh, attr.kw, attr.pad_d, attr.pad_d_after,
          attr.pad_h, attr.pad_h_after, attr.pad_w, attr.pad_w_after, attr.sd,
          attr.sh, attr.sw, is_avg_pooling, avg_pooling_mode, 0, 0, 0, in_sign,
          0, 0, out_sign, attr.do_relu ? 1 : 0, attr.relu_limit,
          (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
    } else {
      llvm_unreachable("nodechip currently not supported.");
    }
  } else {
    BM1684::instance().dl_nodechip_pooling3d_forward_parallel(
        in_addr, buffer_addr, out_addr, attr.n, attr.c, attr.id, attr.ih,
        attr.iw, attr.od, attr.oh, attr.ow, attr.kd, attr.kh, attr.kw,
        attr.pad_d, attr.pad_d_after, attr.pad_h, attr.pad_h_after, attr.pad_w,
        attr.pad_w_after, attr.sd, attr.sh, attr.sw, is_avg_pooling,
        avg_pooling_mode, attr.do_relu ? 1 : 0, attr.relu_limit,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
}

int64_t tpu::Pool3DOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  auto attr = parseParam();
  int64_t channel = attr.c;
  int64_t ih = attr.ih;
  int64_t iw = attr.iw;
  int64_t it = attr.id;
  int64_t oh = (ih - attr.kh + attr.pad_h + attr.pad_h_after) / attr.sh + 1;
  int64_t ow = (iw - attr.kw + attr.pad_w + attr.pad_w_after) / attr.sw + 1;
  int64_t ot = (it - attr.kd + attr.pad_d + attr.pad_d_after) / attr.sd + 1;
  int64_t buffer_size = it * attr.n * ceiling_func(channel, BM168x::NPU_NUM) *
                        align_up(oh * ow, BM168x::EU_BYTES) * sizeof(float);
  buffer_size += attr.n * ceiling_func(channel, BM168x::NPU_NUM) *
                 align_up(oh * ow * it, BM168x::EU_BYTES) * sizeof(float);

  if (oh * ow >= (1 << 12)) {
    buffer_size += attr.n * ceiling_func(channel, BM168x::NPU_NUM) *
                   align_up(4095 * ot, BM168x::EU_BYTES) * sizeof(float);
  }

  if (attr.pad_d == 0 && attr.pad_d_after == 0 &&
      (it == 1 || (attr.kd == 1 && attr.sd == 1))) {
    buffer_size = 0;
  }
  return buffer_size;
}

void tpu::Pool3DOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                         local_sec_info_t &sec_info) {
  auto out_gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto p = parseParam();
  bool is_avg_pooling = getPoolMode() == tpu::PoolMode::Avg;
  int avg_pooling_mode = p.count_include_pad ? 0 : 1;
  int pad_d = (in_gi.d_idx == 0 ? p.pad_d : 0);
  int pad_d_after = (in_gi.d_idx + in_gi.d_slice == p.ih ? p.pad_d_after : 0);
  int pad_h = (in_gi.h_idx == 0 ? p.pad_h : 0);
  int pad_h_after = (in_gi.h_idx + in_gi.h_slice == p.ih ? p.pad_h_after : 0);
  int pad_w = (in_gi.w_idx == 0 ? p.pad_w : 0);
  int pad_w_after = (in_gi.w_idx + in_gi.w_slice == p.iw ? p.pad_w_after : 0);
  if (module::isUniformQuantized(getInput())) {
    UNREACHABLE_THIS("Not Implemented");
  } else {
    BM1684::instance().dl_nodechip_pooling3d_local(
        in_gi.out_addr, out_gi.out_addr, out_gi.buffer_addr, (int)in_gi.n_slice,
        (int)p.c, (int)in_gi.d_slice, (int)in_gi.h_slice, (int)in_gi.w_slice,
        (int)out_gi.d_slice, (int)out_gi.h_slice, (int)out_gi.w_slice, p.kd,
        p.kh, p.kw, p.sd, p.sh, p.sw, pad_d, pad_d_after, pad_h, pad_h_after,
        pad_w, pad_w_after, is_avg_pooling, avg_pooling_mode, p.do_relu,
        p.relu_limit, (CMD_ID_NODE *)BM1684::instance()->bdc_node);
  }
}

uint32_t tpu::Pool3DOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
int64_t tpu::Pool3DOp::get_fw_type_bm1684() { return -1; }

int32_t tpu::Pool3DOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
