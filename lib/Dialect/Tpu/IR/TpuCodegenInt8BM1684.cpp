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
#include "sophgo/Backend/BM1684.h"
#include "sophgo/Interfaces/CodegenInterface.h"
#include "sophgo/Support/MathUtils.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;
using namespace sophgo::backend;

#define ALIGN(x, a) ((((x) + (a)-1) / (a)) * (a))

typedef enum {
  STORE_MODE_1N = 0,
  STORE_MODE_2N = 1,
  STORE_MODE_4N = 2,
} STORE_MODE_T;

void tpu::ConvOp::codegen_int8_bm1684() {
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw, idt, wdt, bdt, odt, rshift;
  bool is_dw, with_bias, relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, relu, idt, wdt, bdt, odt, rshift);
  if (is_dw) {
    BM1684::instance().dl_nodechip_depthwise_fix8b_forward_parallel(
        Module::getAddress(input()), Module::getAddress(output()),
        Module::getAddress(filter()),
        with_bias ? Module::getAddress(bias()) : 0, n, ic, ih, iw, kh, kw, pt,
        pb, pl, pr, sh, sw, ins_h, ins_w, rshift, with_bias ? 1 : 0, 0, 1, 1,
        1, 1, relu ? 1 : 0, BM1684::instance().get_cmd_id_node());
  } else {
    auto weight_addr = Module::getAddress(filter());
    auto bias_offset = ALIGN(ic / g, 4) * kh * kw;
    BM1684::instance().dl_nodechip_conv_forward_parallel_fix8b_with_data_split(
        Module::getAddress(input()), Module::getAddress(output()),
        Module::getAddress(filter()),
        with_bias ? Module::getAddress(bias()) : 0, n, ic, ih, iw, g, oc, kh,
        kw, dh, dw, pt, pb, pl, pr, sh, sw, with_bias ? 1 : 0, 0, relu ? 1 : 0,
        0, 1, 0, 0, rshift, 1, 1, 1, 3, 0, 0, 0, 0, 0,
        BM1684::instance().get_cmd_id_node());
  }
}

typedef enum {
  FcPerLayerShift = 0,
  FcPerLayerScale = 1,
  FcPerChannelScale = 2,
} FcQScale;

typedef struct {
  float perlayer_scale;
  int if_asymmetic;
  int weight_offset;
  int output_offset;
  int if_bias_float;
} FcQParams;

void tpu::MatMulOp::codegen_int8_bm1684() {
  int64_t batch, M, K, N, ldt, rdt, bdt, odt, rshift;
  bool with_bias;
  parseParam(batch, M, K, N, with_bias, ldt, rdt, bdt, odt, rshift);
  int res_16b = 0;
  int out_sign = 1;
  int opd0_sign = 1;
  int opd1_sign = 1;
  int using_bias = with_bias ? 1 : 0;
  int bias_sign = 1;
  int if_relu = do_relu();
  FcQParams quant_param{0, 0, 0, 0, 0};
  BM1684::instance().dl_nodechip_fc_fix8b_forward_parallel(
      Module::getAddress(input()), Module::getAddress(right()),
      with_bias ? Module::getAddress(bias()) : 0, Module::getAddress(output()),
      0, M, K, N, 0, using_bias, 1, 1, 1, rshift, 0, do_relu() ? 1 : 0, 1,
      isa<top::WeightOp>(right().getDefiningOp()) ? 0 : 1, 1, 0, FcPerLayerShift, &quant_param,
      BM1684::instance().get_cmd_id_node());
}

void tpu::AvgPoolOp::codegen_int8_bm1684() {
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value, dt;
  bool is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             is_global, count_include_pad, dt);
  BM1684::instance().dl_nodechip_pooling_fix8b_forward_parallel_with_data_split(
      Module::getAddress(input()), Module::getAddress(output()), n, c, ih, iw,
      kh, kw, pt, pb, pl, pr, sh, sw, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1,
      do_relu() ? 1 : 0, BM1684::instance().get_cmd_id_node());
}

void tpu::MaxPoolOp::codegen_int8_bm1684() {
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value, dt;
  bool is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             is_global, count_include_pad, dt);
  BM1684::instance().dl_nodechip_pooling_fix8b_forward_parallel_with_data_split(
      Module::getAddress(input()), Module::getAddress(output()), n, c, ih, iw,
      kh, kw, pt, pb, pl, pr, sh, sw, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
      do_relu() ? 1 : 0, BM1684::instance().get_cmd_id_node());
}

void tpu::ReshapeOp::codegen_int8_bm1684() {
  auto in_addr = Module::getAddress(input());
  auto out_addr = Module::getAddress(output());
  if (in_addr == out_addr) {
    return;
  }
  int64_t in, ic, ih, iw, on, oc, oh, ow;
  Module::getNCHW(input(), in, ic, ih, iw);
  Module::getNCHW(output(), on, oc, oh, ow);
  BM1684::instance().dl_nodechip_reshape_fix8b(
      in_addr, out_addr, in, ic, ih, iw, on, oc, oh, ow, STORE_MODE_4N,
      STORE_MODE_4N, BM1684::instance().get_cmd_id_node());
}

void tpu::AddOp::codegen_int8_bm1684() {
  int input_num = inputs().size();
  assert(input_num == 2);
  int64_t n, c, h, w;
  Module::getNCHW(output(), n, c, h, w);
  int op_code = 1; // (0: Product; 1: Sum; 2: Max)

  std::vector<float> coeff_v;
  if (coeff().hasValue()) {
    for (auto data : coeff().getValue()) {
      coeff_v.push_back(data.cast<FloatAttr>().getValueAsDouble());
    }
  } else {
    coeff_v = {0, 0};
  }
  std::vector<int> rshift_v;
  for (auto data : rshifts()) {
    rshift_v.push_back(data.cast<IntegerAttr>().getInt());
  }

  BM1684::instance().dl_nodechip_eltwise_fix8b_forward_parallel(
      Module::getAddress(inputs()[0]),     // u64    bottom_A_global_addr,
      Module::getAddress(inputs()[1]),     // u64    bottom_B_global_addr,
      Module::getAddress(output()),        // u64    top_global_addr,
      n,                                   // int    tensor_n,
      c,                                   // int    tensor_c,
      h,                                   // int    tensor_h,
      w,                                   // int    tensor_w,
      op_code,                             // int    op_code,
      coeff_v[0],                          // int    scale_A,
      coeff_v[1],                          // int    scale_B,
      1,                                   // int    sign_A,
      1,                                   // int    sign_B,
      rshift_v[0],                         // int    rshift_A,
      rshift_v[1],                         // int    rshift_B,
      do_relu() ? 1 : 0,                   // int    do_relu(),
      BM1684::instance().get_cmd_id_node() // CMD_ID_NODE *id_node
  );
}
