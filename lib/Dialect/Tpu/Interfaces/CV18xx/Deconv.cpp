//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Deconv.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/CV18xx/WeightReorder.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/TPUCompressUtil.h"

using namespace tpu_mlir::backend;
using namespace tpu_mlir::cv18xx;

// common for weight
template <typename T>
static void transposeConvolutionFilter(std::shared_ptr<std::vector<T>> &w,
                                       const std::vector<int64_t> &s) {
  // shape[oc, ic, kh, kw] => [oc, kh, kw, ic]
  int64_t oc, ic, ks;
  if (s.size() == 4) {
    oc = s[0];
    ic = s[1];
    ks = s[2] * s[3];
  } else {
    llvm_unreachable("unsupported shape size");
  }
  auto w_t = std::make_shared<std::vector<T>>(w->size());
  if (ks == 1 || ic == 1) {
    return;
  } else {
    // for other conv, transpose ic <-> kh*kw
    for (int64_t i = 0; i < oc; i++) {
      for (int64_t j = 0; j < ic; j++) {
        for (int64_t k = 0; k < ks; k++) {
          w_t->at(i * ic * ks + k * ic + j) = w->at(i * ic * ks + j * ks + k);
        }
      }
    }
    w->assign(w_t->begin(), w_t->end());
  }
}

template <typename T>
static void rotateConvolutionFilter(std::shared_ptr<std::vector<T>> &w,
                                    const std::vector<int64_t> &s) {
  int64_t oc, ic, kh, kw;
  if (s.size() == 4) {
    oc = s[0];
    ic = s[1];
    kh = s[2];
    kw = s[3];
    // } else if (s.size() == 5) {
    //   // g, oc/g, ic/g, kh, kw
    //   oc = s[0] * s[1];
    //   ic = s[2];
    //   kh = s[3];
    //   kw = s[4];
  } else {
    llvm_unreachable("unsupported shape size");
  }
  auto w_t = std::make_shared<std::vector<T>>(w->size());
  if (kh == 1 && kw == 1) {
    return;
  } else {
    // for other conv, rotate 180
    for (int64_t i = 0; i < oc * ic; i++) {
      for (int64_t j = 0; j < kh; j++) {
        for (int64_t k = 0; k < kw; k++) {
          w_t->at(i * kh * kw + (kh - 1 - j) * kw + (kw - 1) - k) =
              w->at(i * kh * kw + j * kw + k);
        }
      }
    }
    w->assign(w_t->begin(), w_t->end());
  }
}

// for int8 bias
static std::unique_ptr<std::vector<uint8_t>>
packWeight(i32_array_t &bias, i64_array_t &rshift, i64_array_t &multiplier,
           int64_t oc, std::vector<int64_t> &shape) {
  if (bias) {
    assert(bias->size() == (size_t)oc);
  }
  assert(rshift->size() == (size_t)oc);
  assert(multiplier->size() == (size_t)oc);

  int64_t isz = bias ? 9 : 5;
  shape = std::vector<int64_t>{1, oc, 1, isz};

  auto packed = std::make_unique<std::vector<uint8_t>>(oc * isz);

  uint8_t *ptr = packed->data();
  for (int i = 0; i < oc; i++) {
    if (bias) {
      uint32_t val = (uint32_t)bias->at(i);
      *ptr = (uint8_t)(val & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 8) & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 16) & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 24) & 0xff);
      ptr++;
    }

    {
      uint32_t val = (uint32_t)multiplier->at(i);
      *ptr = (uint8_t)(val & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 8) & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 16) & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 24) & 0xff);
      ptr++;
    }

    {
      uint8_t val = (uint8_t)rshift->at(i);
      *ptr = (uint8_t)val;
      ptr++;
    }
  }
  return packed;
}

// for bf16 bias
static void
transposeBiasFp32(const std::shared_ptr<std::vector<float>> &bias_f32,
                  std::vector<uint16_t> &bias_u16) {
  // Split into high/low part
  std::vector<uint16_t> bias_fp32_high;
  std::vector<uint16_t> bias_fp32_low;
  float *biasFloatPtr = bias_f32->data();
  int size = bias_f32->size();
  for (int i = 0; i < size; ++i) {
    unsigned short *temp_short_ptr =
        reinterpret_cast<unsigned short *>(biasFloatPtr + i);
    bias_fp32_high.push_back(temp_short_ptr[1]);
    bias_fp32_low.push_back(temp_short_ptr[0]);
  }
  std::vector<uint16_t> bias_reshape_fp32;
  bias_reshape_fp32.insert(bias_reshape_fp32.end(), bias_fp32_high.begin(),
                           bias_fp32_high.end());
  bias_reshape_fp32.insert(bias_reshape_fp32.end(), bias_fp32_low.begin(),
                           bias_fp32_low.end());
  // then copy into uint32_t
  assert(bias_u16.size() == 2 * bias_f32->size());
  memcpy(bias_u16.data(), bias_reshape_fp32.data(), size * sizeof(uint32_t));
}

// ======================================
// Weight reorder
// ======================================

template <>
LogicalResult WeightReorder<tpu::DeconvOp, int8_t>::matchAndRewrite(
    tpu::DeconvOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isInteger(8))
    return failure();

  auto attr = op.parseParam();
  // lower weight  for groups weight's shape is (oc, ic/g, kh, kw)
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto filter_i8 = filterOp.read<int8_t>();
  std::vector<int64_t> filter_shape = {attr.oc, attr.ic / attr.g, attr.kh,
                                       attr.kw};
  if (attr.dh > 1 || attr.dw > 1) {
    // TODO do dilation here, ins in top/tpu_common interpreter
    llvm_unreachable("Not supported now");
  }
  rotateConvolutionFilter(filter_i8, filter_shape);
  transposeConvolutionFilter(filter_i8, filter_shape);
  // rewrite weightOp shape (oc, ic/g, kh, kw) -> (1, oc, kh*kw, ic/g)
  std::vector<int64_t> new_filter_shape = {1, attr.oc, attr.kh * attr.kw,
                                           attr.ic / attr.g};
  if (attr.is_dw) {
    new_filter_shape = {1, attr.oc, attr.kh, attr.kw};
  }
  auto elem_type = module::getStorageType(op.getFilter());
  auto filter_type = RankedTensorType::get(new_filter_shape, elem_type);
  auto weight_op =
      top::WeightOp::create(op, "filter_reordered", *filter_i8, filter_type);
  op->setOperand(1, weight_op);

  // merge conv rshift/multiplier/bias into one packed tensor
  i32_array_t bias_new;
  std::vector<int64_t> bias_shape = {1, attr.oc, 1, 1};
  if (attr.with_bias) {
    auto biasOp = op.getBias().getDefiningOp<top::WeightOp>();
    bias_new = biasOp.read<int32_t>();
  }
  auto m_data = module::getI64Array(op.getMultiplier(), attr.oc, 1);
  auto r_data = module::getI64Array(op.getRshift(), attr.oc, 0);
  std::vector<int64_t> packedShape;
  auto packed = packWeight(bias_new, r_data, m_data, attr.oc, packedShape);
  auto packed_type =
      RankedTensorType::get(packedShape, rewriter.getIntegerType(8));
  auto pack_op = top::WeightOp::create(op, "bias_packed", *packed, packed_type);
  op->removeAttr("rshift");
  op->removeAttr("multiplier");
  op->setOperand(2, pack_op);
  return success();
}

template <>
LogicalResult WeightReorder<tpu::DeconvOp, BFloat16Type>::matchAndRewrite(
    tpu::DeconvOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isBF16())
    return failure();

  auto attr = op.parseParam();
  // first lower weight
  auto shape = module::getShape(op.getFilter());
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  std::vector<int64_t> filter_shape = {attr.oc, attr.ic / attr.g, attr.kh,
                                       attr.kw};
  auto filter_u16 = filterOp.read<uint16_t>();
  if (attr.dh > 1 || attr.dw > 1) {
    // TODO do dilation here, ins in top/tpu_common interpreter
    llvm_unreachable("dilation is not supported now");
  }
  rotateConvolutionFilter(filter_u16, filter_shape);
  transposeConvolutionFilter(filter_u16, filter_shape);
  // rewrite weightOp shape (oc, ic/g, kh, kw) -> (1, oc, kh*kw, ic/g)
  std::vector<int64_t> new_filter_shape = {1, attr.oc, attr.kh * attr.kw,
                                           attr.ic / attr.g};
  if (attr.is_dw) {
    new_filter_shape = {1, attr.oc, attr.kh, attr.kw};
  }
  auto filter_type =
      RankedTensorType::get(new_filter_shape, rewriter.getBF16Type());
  auto weight_op =
      top::WeightOp::create(op, "filter_reordered", *filter_u16, filter_type);
  op->setOperand(1, weight_op);
  // second lower bias if exist
  if (attr.with_bias) {
    auto biasOp = op.getBias().getDefiningOp<top::WeightOp>();
    auto bias_f32 = biasOp.read<float>();
    std::vector<uint16_t> bias_new(bias_f32->size() * 2);
    transposeBiasFp32(bias_f32, bias_new);
    // rewrite biasOp
    // rewrite weightOp shape (oc) f32 -> (2, oc, 1, 1) uint16
    std::vector<int64_t> new_bias_shape = {2, attr.oc, 1, 1};
    auto new_bias_type = RankedTensorType::get(
        new_bias_shape, rewriter.getIntegerType(16, false));
    auto lbias_op =
        top::WeightOp::create(op, "bias_reordered", bias_new, new_bias_type);
    op->setOperand(2, lbias_op);
  }
  return success();
}

// ======================================
// GlobalGenInterface
// ======================================

void tpu::DeconvOp::codegen_global_cv18xx(int64_t layer_id) {

  auto attr = parseParam();
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  gaddr_t ga_filter = module::getAddress(getFilter());
  gaddr_t ga_pc_info = GA_INVALID;
  if (module::isUniformQuantized(getOutput()) || attr.with_bias) {
    ga_pc_info = module::getAddress(getBias());
  }

  int kh_ext = (attr.kh - 1) * attr.dh + 1;
  int kw_ext = (attr.kw - 1) * attr.dw + 1;
  int ins_h = attr.sh - 1;
  int ins_w = attr.sw - 1;
  int pad_t = kh_ext - attr.pad_h - 1;
  int pad_l = kw_ext - attr.pad_w - 1;
  int pad_b = attr.oh + attr.pad_h - (attr.ih - 1) * attr.sh - 1;
  int pad_r = attr.ow + attr.pad_w - (attr.iw - 1) * attr.sw - 1;
  int sh = 1;
  int sw = 1;

  if (module::isUniformQuantized(getOutput())) {
    cvi_backend_tg_fixed_conv_kernel(layer_id,   // layer_id,
                                     ga_input,   // input_data_gaddr,
                                     ga_output,  // output_data_gaddr,
                                     ga_filter,  // weight_data_gaddr,
                                     ga_pc_info, // bias_data_gaddr,
                                     attr.n, attr.ic, attr.ih, attr.iw,
                                     attr.g, // group,
                                     attr.oc, attr.kh, attr.kw, attr.dh,
                                     attr.dw, pad_t, pad_b, pad_l, pad_r, ins_h,
                                     ins_w, sh, sw,
                                     attr.with_bias,       // bias_term,
                                     attr.do_relu ? 1 : 0, // do_activation,
                                     nullptr,              // activation_arg,
                                     0, // activation_gt_scale,
                                     0, // activation_gt_rshift,
                                     0, // activation_le_scale,
                                     0, // activation_le_rshift,
                                     0, // (int)rshift[0], //right_shift_width,
                                     true, // do_chl_quan
                                     false // do_ic_alignment,
    );
  } else {
    cvi_backend_tg_bf16_conv_kernel(layer_id,   // layer_id
                                    ga_input,   // input_data_gaddr,
                                    ga_output,  // output_data_gaddr,
                                    ga_filter,  // weight_data_gaddr,
                                    ga_pc_info, // bias_data_gaddr,
                                    attr.n, attr.ic, attr.ih, attr.iw,
                                    attr.g, // group
                                    attr.oc, attr.kh, attr.kw, attr.dh, attr.dw,
                                    pad_t, pad_b, pad_l, pad_r, ins_h, ins_w,
                                    sh, sw,
                                    attr.with_bias,       // bias_term,
                                    attr.do_relu ? 1 : 0, // do_activation,
                                    false                 // fp32_output
    );
  }
}

// ======================================
// LocalGenInterface
// ======================================

int64_t tpu::DeconvOp::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::DeconvOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                         int64_t layer_id) {
  auto attr = parseParam();
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto out_gi = LocalGenInterface::getGroupInfo(getOutput(), n_step, h_step);
  auto w_gi = LocalGenInterface::getGroupInfo(getFilter());
  auto b_gi = LocalGenInterface::getGroupInfo(getBias());

  laddr_t la_input = in_gi.out_addr;
  laddr_t la_output = out_gi.out_addr;
  laddr_t la_weight = w_gi.out_addr;
  laddr_t la_bias = b_gi.out_addr;

  bool do_ic_alignment = false;

  int n = in_gi.n_slice;
  int ih = in_gi.h_slice;
  int oh = out_gi.h_slice;

  int pad_h_top, pad_h_bottom;
  int pad_w_left, pad_w_right;
  int kh_ext = (attr.kh - 1) * attr.dh + 1;
  int kw_ext = (attr.kw - 1) * attr.dw + 1;
  int ins_last_h = 0;
  int ins_last_w = (attr.ow + attr.pad_w + attr.pad_w_after - kw_ext) % attr.sw;
  int ins_h = attr.sh - 1;
  int ins_w = attr.sw - 1;
  if (auto deconv_in_slice =
          DeconvSlice(gi.h_idx, gi.h_slice, attr.sh, kh_ext, attr.pad_h)) {
    pad_h_top = deconv_in_slice.value()[0];
    pad_h_bottom = deconv_in_slice.value()[1];

  } else {
    pad_h_top = attr.kh - attr.pad_h - 1;
    pad_h_bottom = attr.kh - attr.pad_h_after - 1;
  }
  pad_w_left = attr.kw - attr.pad_w - 1;
  pad_w_right = attr.kw - attr.pad_w_after - 1;
  // hw limitation once set ins_w / ins_h that input w/h should > 1
  if (ins_h && ih == 1) {
    ins_last_h += ins_h;
    ins_h = 0;
    if (pad_h_top) {
      ins_last_h = 0; // included in pad_h_top
    }
  }
  if (ins_w && attr.iw == 1) {
    ins_last_w += ins_w;
    ins_w = 0;
    if (pad_w_left) {
      // TODO: need to verify
      ins_last_w = 0; // included in pad_w_left
    }
  }
  assert(ins_last_h < 16 && ins_last_w < 16);
  assert(pad_h_top < 16 && pad_h_bottom < 16 && pad_w_left < 16 && pad_w_right < 16);
  if (module::isUniformQuantized(getOutput())) {
    cvi_backend_tl_deconv(
        layer_id, la_input, la_output, la_weight, la_bias, n, attr.ic, ih,
        attr.iw, attr.g, attr.oc, oh, attr.ow, attr.kh, attr.kw, attr.dh,
        attr.dw, ins_h, ins_last_h, ins_w, ins_last_w, pad_h_top, pad_h_bottom,
        pad_w_left, pad_w_right, attr.sh, attr.sw, attr.with_bias,
        getDoRelu(), // do_activation,
        0,           // right_shift_width,
        attr.oc,     // right_shift_array_len
        do_ic_alignment);
  } else {
    cvi_backend_tl_bf16_deconv(
        layer_id, la_input, la_output, la_weight, la_bias, n, attr.ic, ih,
        attr.iw, attr.g, attr.oc, oh, attr.ow, attr.kh, attr.kw, attr.dh,
        attr.dw, ins_h, ins_last_h, ins_w, ins_last_w, pad_h_top, pad_h_bottom,
        pad_w_left, pad_w_right, attr.sh, attr.sw, attr.with_bias, getDoRelu());
  }
  return;
}
