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

using namespace tpu_mlir::backend;

void tpu::Pool1DOp::codegen_global_bm1684() {
  auto attr = parseParam();
  bool is_avg_pooling = getPoolMode() == tpu::PoolMode::Avg;
  if (module::isUniformQuantized(getInput())) {
    BM1684::instance()
        .dl_nodechip_pooling_fix8b_forward_parallel_with_data_split(
            module::getAddress(getInput()), module::getAddress(getOutput()),
            attr.n, attr.c, attr.ih, attr.iw, attr.kh, attr.kw, attr.pad_h,
            attr.pad_h_after, attr.pad_w, attr.pad_w_after, attr.sh, attr.sw, 0,
            0, is_avg_pooling, 0, 0, 0, 0, 1, 0, 0, 1, attr.do_relu ? 1 : 0,
            (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    BM1684::instance().dl_nodechip_pooling_forward_parallel_with_data_split(
        module::getAddress(getInput()), module::getAddress(getOutput()), attr.n,
        attr.c, attr.ih, attr.iw, attr.oh, attr.ow, attr.kh, attr.kw,
        attr.pad_h, attr.pad_w, attr.sh, attr.sw, is_avg_pooling ? 1 : 0,
        attr.count_include_pad ? 0 : 1, attr.do_relu ? 1 : 0,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
}

int64_t tpu::Pool1DOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::Pool1DOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                         local_sec_info_t &sec_info) {
  auto out_gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto p = parseParam();
  int bottom_dim[4] = {(int)in_gi.n_slice, (int)p.c, (int)in_gi.h_slice,
                       (int)p.iw};
  int top_dim[4] = {(int)out_gi.n_slice, (int)p.c, (int)out_gi.h_slice,
                    (int)p.ow};
  bool is_avg_pooling = getPoolMode() == tpu::PoolMode::Avg;
  if (module::isUniformQuantized(getInput())) {
    BM1684::instance().dl_nodechip_pooling_fix8b_forward_local(
        in_gi.out_addr, /*weight*/ 0, /*bias*/ 0, out_gi.out_addr, bottom_dim,
        top_dim, p.kh, p.kw, p.pad_h, p.pad_h_after, p.pad_w, p.pad_w_after,
        p.sh, p.sw,
        /*ins_h*/ 0, /*ins_w*/ 0, is_avg_pooling ? 1 : 0,
        p.count_include_pad ? 0 : 1, // avg_pooling_mode should be always 0,
                                     // BM1684 fixpoint not support
        /*shift*/ 0, /*bias*/ 0, /*shift_type*/ 0,
        module::isSign(getInput()) ? 1 : 0, 0, 0,
        module::isSign(getOutput()) ? 1 : 0, p.do_relu,
        (CMD_ID_NODE *)BM1684::instance()->bdc_node);
  } else {
    BM1684::instance().dl_nodechip_pooling_forward_local(
        in_gi.out_addr, out_gi.out_addr, bottom_dim, top_dim, p.kh, p.kw,
        p.pad_h, p.pad_h_after, p.pad_w, p.pad_w_after, p.sh, p.sw,
        is_avg_pooling ? 1 : 0, p.count_include_pad ? 0 : 1,
        (CMD_ID_NODE *)BM1684::instance()->bdc_node, p.do_relu);
  }
}

uint32_t tpu::Pool1DOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
int64_t tpu::Pool1DOp::get_fw_type_bm1684() { return -1; }

int32_t tpu::Pool1DOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
