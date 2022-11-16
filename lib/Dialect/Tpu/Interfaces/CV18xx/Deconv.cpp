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
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/TPUCompressUtil.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

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
packWeight(std::shared_ptr<std::vector<int32_t>> &bias,
           std::shared_ptr<std::vector<int64_t>> &rshift,
           std::shared_ptr<std::vector<int64_t>> &multiplier, int64_t oc,
           std::vector<int64_t> &shape) {
  if (bias) {
    assert(bias->size() == (size_t)oc);
  }
  assert(rshift->size() == (size_t)oc);
  assert(multiplier->size() == (size_t)oc);

  int64_t isz = bias ? 9 : 5;
  shape = std::vector<int64_t>{oc, 1, isz};

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
                  std::vector<uint32_t> &bias_u32) {
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
  assert(bias_u32.size() == bias_f32->size());
  memcpy(bias_u32.data(), bias_reshape_fp32.data(), size * sizeof(uint32_t));
}

// ======================================
// Weight reorder
// ======================================

void tpu::DeconvOp::weight_reorder_int8_cv18xx() {
  deconv_attr_t attr = {0};
  parseParam(&attr);
  OpBuilder builder(getContext());
  auto op = getOperation();
  // lower weight  for groups weight's shape is (oc, ic/g, kh, kw)
  auto filterOp = filter().getDefiningOp<top::WeightOp>();
  auto filter_i8 = filterOp.read<int8_t>();
  std::vector<int64_t> filter_shape = {attr.oc, attr.ic / attr.g, attr.kh,
                                       attr.kw};
  if (attr.dh > 1 || attr.dw > 1) {
    // TODO do dilation here, ins in top/tpu_common interpreter
    llvm_unreachable("Not supported now");
  }
  rotateConvolutionFilter(filter_i8, filter_shape);
  transposeConvolutionFilter(filter_i8, filter_shape);
  auto elem_type = Module::getStorageType(filter());
  auto filter_type = RankedTensorType::get(filter_shape, elem_type);
  auto weight_op =
      top::WeightOp::create(op, "filter_reordered", *filter_i8, filter_type);
  op->setOperand(1, weight_op);

  // merge conv rshift/multiplier/bias into one packed tensor
  std::shared_ptr<std::vector<int32_t>> bias_new;
  std::vector<int64_t> bias_shape = {1, attr.oc, 1, 1};
  if (attr.with_bias) {
    auto biasOp = bias().getDefiningOp<top::WeightOp>();
    bias_new = biasOp.read<int32_t>();
  }
  auto m_data = Module::getI64Array(multiplier(), attr.oc, 1);
  auto r_data = Module::getI64Array(rshift(), attr.oc, 0);
  std::vector<int64_t> packedShape;
  auto packed = packWeight(bias_new, r_data, m_data, attr.oc, packedShape);
  auto packed_type =
      RankedTensorType::get(packedShape, builder.getIntegerType(8));
  auto pack_op = top::WeightOp::create(op, "bias_packed", *packed, packed_type);
  op->removeAttr("rshift");
  op->removeAttr("multiplier");
  op->setOperand(2, pack_op);
}

void tpu::DeconvOp::weight_reorder_bf16_cv18xx() {
  deconv_attr_t attr = {0};
  parseParam(&attr);
  OpBuilder builder(getContext());
  auto op = getOperation();
  // first lower weight
  auto shape = Module::getShape( filter());
  auto filterOp = filter().getDefiningOp<top::WeightOp>();
  std::vector<int64_t> filter_shape = {attr.oc, attr.ic / attr.g, attr.kh,
                                       attr.kw};
  auto filter_u16 = filterOp.read<uint16_t>();
  if (attr.dh > 1 || attr.dw > 1) {
    // TODO do dilation here, ins in top/tpu_common interpreter
    llvm_unreachable("dilation is not supported now");
  }
  rotateConvolutionFilter(filter_u16, filter_shape);
  transposeConvolutionFilter(filter_u16, filter_shape);
  // rewrite weightOp
  auto filter_type = RankedTensorType::get(filter_shape, builder.getBF16Type());
  auto weight_op =
      top::WeightOp::create(op, "filter_reordered", *filter_u16, filter_type);
  op->setOperand(1, weight_op);
  // second lower bias if exist
  if (attr.with_bias) {
    auto biasOp = bias().getDefiningOp<top::WeightOp>();
    auto bias_f32 = biasOp.read<float>();
    std::vector<uint32_t> bias_new(bias_f32->size());
    transposeBiasFp32(bias_f32, bias_new);
    // rewrite biasOp
    auto new_bias_type = RankedTensorType::get(Module::getShape(bias()),
                                               builder.getIntegerType(32));
    auto lbias_op =
        top::WeightOp::create(op, "bias_reordered", bias_new, new_bias_type);
    op->setOperand(2, lbias_op);
  }
}

// ======================================
// GlobalGenInterface
// ======================================

void tpu::DeconvOp::codegen_global_cv18xx(void *ctx, int64_t layer_id) {
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  deconv_attr_t attr = {0};
  parseParam(&attr);
  gaddr_t ga_input = Module::getAddress(input());
  gaddr_t ga_output = Module::getAddress(output());
  gaddr_t ga_filter = Module::getAddress(filter());
  gaddr_t ga_pc_info = GA_INVALID;
  if (Quant::isUniformQuantized(output()) || attr.with_bias) {
    ga_pc_info = Module::getAddress(bias());
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

  if (Quant::isUniformQuantized(output())) {
    cvi_backend_tg_fixed_conv_kernel(*backend_ctx,
                                     layer_id,   // layer_id,
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
    cvi_backend_tg_bf16_conv_kernel(*backend_ctx,
                                    layer_id,   // layer_id
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
  llvm_unreachable("Not supported now");
  return 0;
}

void tpu::DeconvOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step) {
  llvm_unreachable("Not supported now");
}
