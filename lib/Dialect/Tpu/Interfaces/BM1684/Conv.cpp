//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

void tpu::ConvOp::weight_reorder_int8_bm1684() {
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, relu;
  float relu_upper_limit;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, relu, relu_upper_limit);
  auto filterOp = cast<top::WeightOp>(filter().getDefiningOp());
  auto filter_int8 = filterOp.read<int8_t>();
  int new_size = oc * (align_up(ic, 4l)) * kh * kw;
  auto filter_new = std::make_shared<std::vector<int8_t>>(new_size, 0);
  for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
    for (int ic_idx = 0; ic_idx < ic; ic_idx++) {
      for (int k_idx = 0; k_idx < kh * kw; k_idx++) {
        int orig_offset = ic_idx * kh * kw + k_idx + oc_idx * kh * kw * ic;
        int trans_offset = ic_idx + k_idx * align_up(ic, 4l) +
                           oc_idx * (kh * kw * align_up(ic, 4l));
        filter_new->at(trans_offset) = filter_int8->at(orig_offset);
      }
    }
  }
  auto filter_type = filterOp.getType().cast<RankedTensorType>();
  std::vector<int64_t> new_shape = {1, oc, kh * kw * align_up(ic, 4l), 1};
  auto new_type =
      RankedTensorType::get(new_shape, filter_type.getElementType());
  auto new_filter = top::WeightOp::create(filter().getDefiningOp(), "reorderd",
                                          *filter_new, new_type);
  setOperand(1, new_filter);
}

void tpu::ConvOp::codegen_global_int8_bm1684() {
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, relu;
  float relu_upper_limit;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, relu, relu_upper_limit);
  if (is_dw) {
    BM1684::instance().dl_nodechip_depthwise_fix8b_forward_parallel(
        Module::getAddress(input()), Module::getAddress(output()),
        Module::getAddress(filter()),
        with_bias ? Module::getAddress(bias()) : 0, n, ic, ih, iw, kh, kw, pt,
        pb, pl, pr, sh, sw, ins_h, ins_w,
        rshift().getValue()[0].cast<IntegerAttr>().getInt(), with_bias ? 1 : 0,
        0, 1, 1, 1, 1, relu ? 1 : 0,
        (CMD_ID_NODE *)BM1684::instance().cmdid_node);
  } else {
    auto weight_addr = Module::getAddress(filter());
    auto bias_offset = align_up(ic / g, 4l) * kh * kw;
    BM1684::instance().dl_nodechip_conv_forward_parallel_fix8b_with_data_split(
        Module::getAddress(input()), Module::getAddress(output()),
        Module::getAddress(filter()),
        with_bias ? Module::getAddress(bias()) : 0, n, ic, ih, iw, g, oc, kh,
        kw, dh, dw, pt, pb, pl, pr, sh, sw, with_bias ? 1 : 0, 0, relu ? 1 : 0,
        0, 1, 0, 0, rshift().getValue()[0].cast<IntegerAttr>().getInt(), 1, 1,
        1, 3, 0, 0, 0, 0, 0, (CMD_ID_NODE *)BM1684::instance().cmdid_node);
  }
}

int64_t tpu::ConvOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {
  // TODO for spicial situation
  return 0;
}

void tpu::ConvOp::codegen_local_int8_bm1684(int64_t n_step, int64_t h_step) {
  llvm_unreachable("support later");
}
