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

void tpu::Deconv3DOp::codegen_global_bm1684() {
  auto attr = parseParam();
  BM1684::instance().dl_nodechip_deconv3d(
      module::getAddress(getInput()), module::getAddress(getOutput()),
      module::getAddress(getFilter()),
      attr.with_bias ? module::getAddress(getBias()) : 0, attr.n, attr.ic,
      attr.id, attr.ih, attr.iw, attr.g, attr.oc, attr.kd, attr.kh, attr.kw,
      attr.dd, attr.dh, attr.dw, attr.pad_d, attr.pad_d_after, attr.pad_h,
      attr.pad_h_after, attr.pad_w, attr.pad_w_after, attr.sd, attr.sh, attr.sw,
      attr.output_pad_d, attr.output_pad_h, attr.output_pad_w, attr.with_bias,
      attr.do_relu, attr.relu_limit,
      (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
}

int64_t tpu::Deconv3DOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  // TODO for spicial situation
  return 0;
}

void tpu::Deconv3DOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                           local_sec_info_t &sec_info) {
  UNREACHABLE_THIS("Not Implemented");
}

uint32_t tpu::Deconv3DOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  GLOBAL_IR_COMMON(deconv3d);
}

int64_t tpu::Deconv3DOp::get_fw_type_bm1684() { return FW_BMNET_DECONV3D; }
