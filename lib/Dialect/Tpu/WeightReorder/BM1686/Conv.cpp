//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Backend/BM168x/BM1686.h"
#include "sophgo/Interfaces/WeightReorderInterface.h"
#include "sophgo/Interfaces/CodegenInterface.h"
#include "sophgo/Support/MathUtils.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;
using namespace sophgo::backend;

// convert (1, oc, 1, w) to (1, NPU_NUM, 1, DIV_UP(oc, NPU_NUM) * w)
static void reshape_coeff_for_broadcast_channel(
    std::shared_ptr<std::vector<uint8_t>> &filter_data,
    std::vector<int64_t> &shape, bool align = false) {
  int64_t n, c, h, w;
  Module::getNCHW(shape, n, c, h, w);
  if (n != 1 || h != 1 || c <= BM1686::NPU_NUM) {
    return;
  }
  // convert (1, oc, 1, w) to (1, NPU_NUM, 1, DIV_UP(oc, NPU_NUM) * w)
  int64_t new_c = BM1686::NPU_NUM;
  int type_len = 1; // int8
  auto c2w = ceiling_func(c, new_c);
  auto old_w_align = ALIGN(w, BM1686::get_eu_num(type_len));
  int new_w = (align ? old_w_align : w) * (c2w - 1) + w;
  int64_t new_size = new_w * new_c * type_len;
  auto filter_new = std::make_shared<std::vector<uint8_t>>(new_size, 0);
  for (int i = 0; i < c2w; i++) {
    for (int j = 0; j < new_c; j++) {
      for (int k = 0; k < w; k++) {
        int src_idx = i * new_c * w + j * w + k;
        int dst_idx = j * new_w + i * (align ? old_w_align : w) + k;
        filter_new->at(dst_idx) = filter_data->at(src_idx);
      }
    }
  }
  shape = {1, new_c, 1, new_w};
  filter_data = filter_new;
}

// refer to net_compiler: bool BM1686CoeffArranger::ConvWeightArr(GraphEdge*
// edge)
void tpu::ConvOp::weight_reorder_int8_bm1686() {
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, relu);
  auto filterOp = cast<top::WeightOp>(filter().getDefiningOp());
  if (is_dw) {
    llvm_unreachable("depthwise should support !!");
  }
  auto type = Module::getStorageType(filter());
  int64_t IC_PARALLEL = 64;
  auto type_bytes = type.getIntOrFloatBitWidth() / 8;
  size_t new_bytes = ALIGN(ic, IC_PARALLEL) * oc * kh * kw * type_bytes;
  auto filter_i8 = filterOp.read_as_byte();
  auto filter_new = std::make_shared<std::vector<uint8_t>>(new_bytes, 0);
  auto kernel_hw = kh * kw;
  int64_t new_ic = ceiling_func(ic, IC_PARALLEL);
  int64_t new_hw = kernel_hw * IC_PARALLEL;
  for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
    for (int ic_idx = 0; ic_idx < new_ic; ic_idx++) {
      for (int k_idx = 0; k_idx < kernel_hw; k_idx++) {
        for (int inner = 0; inner < IC_PARALLEL; inner++) {
          if (ic_idx * IC_PARALLEL + inner >= ic)
            break;
          int orig_offset = oc_idx * ic * kh * kw +
                            (ic_idx * IC_PARALLEL + inner) * kernel_hw + k_idx;
          int trans_offset = oc_idx * new_ic * new_hw + ic_idx * new_hw +
                             k_idx * IC_PARALLEL + inner;
          filter_new->at(trans_offset) = filter_i8->at(orig_offset);
        }
      }
    }
  }
  auto filter_type = filterOp.getType().cast<RankedTensorType>();
  std::vector<int64_t> filter_shape = {1, oc, 1, new_ic * new_hw};
  // refer to net_compier: reshape_coeff_for_broadcast_channel(weight, false);
  reshape_coeff_for_broadcast_channel(filter_new, filter_shape);
  auto new_type =
      RankedTensorType::get(filter_shape, filter_type.getElementType());
  auto new_filter = top::WeightOp::create(filter().getDefiningOp(), "reorderd",
                                          *filter_new, new_type);
  setOperand(1, new_filter);

  if (with_bias) {
    // do nothing for int32
  }
}
