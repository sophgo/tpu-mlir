//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Conv.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/TPUCompressUtil.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::backend;
using namespace tpu_mlir::helper;

// ======================================
// WeightReorderInterface
// ======================================

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
    w = w_t;
  }
}

// for int8 double conv
static void refactorOddIcConv(std::shared_ptr<std::vector<int8_t>> &w,
                              std::vector<int64_t> &s, const int32_t g,
                              bool &do_ic_alignment) {
  int kn = s[0];
  int kc = s[1];
  int kh = s[2];
  int kw = s[3];
  if ((kc % 2 != 0) && (g == 1) && (kc > 1)) {
    // Support only kc is odd && isConvolutionOp && kc>=3
    int new_ic = kc + 1;
    int64_t newFilterSize = kn * new_ic * kh * kw;
    auto n_w = std::make_shared<std::vector<int8_t>>(newFilterSize);
    for (int n_counter = 0; n_counter < kn; n_counter++) {
      for (int h_counter = 0; h_counter < kh; h_counter++) {
        for (int w_counter = 0; w_counter < kw; w_counter++) {
          for (int c_counter = 0; c_counter < new_ic; c_counter++) {
            uint32_t index_old = c_counter + w_counter * kc +
                                 h_counter * kc * kw + n_counter * kh * kc * kw;
            uint32_t index_new = c_counter + w_counter * new_ic +
                                 h_counter * new_ic * kw +
                                 n_counter * kh * new_ic * kw;
            if (c_counter == kc) {
              n_w->at(index_new) = 0;
            } else {
              n_w->at(index_new) = w->at(index_old);
            }
          }
        }
      }
    }
    w = n_w;
    s[1] += 1;
    do_ic_alignment = true;
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

void tpu::Conv2DOp::weight_reorder_int8_cv18xx() {
  conv_attr_t attr = {0};
  parseParam(&attr);
  OpBuilder builder(getContext());
  auto op = getOperation();
  // first, merge conv rshift/multiplier/bias into one packed tensor
  std::shared_ptr<std::vector<int32_t>> bias_new;
  std::vector<int64_t> bias_shape = {1, attr.oc, 1, 1};
  if (attr.has_bias) {
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
  // second lower weight  for groups onnx weight's shape (oc, ic/g, kh, kw)
  auto filterOp = filter().getDefiningOp<top::WeightOp>();
  auto filter_i8 = filterOp.read<int8_t>();
  std::vector<int64_t> filter_shape = {attr.oc, attr.ic / attr.groups, attr.kh,
                                       attr.kw};
  if (attr.dh > 1 || attr.dw > 1) {
    // TODO do dilation here, ins in top/tpu_common interpreter
    llvm_unreachable("Not supported now");
  }
  transposeConvolutionFilter(filter_i8, filter_shape);
  // third padding odd ic to even to enable double conv
  bool do_ic_alignment = false;
  refactorOddIcConv(filter_i8, filter_shape, attr.groups, do_ic_alignment);
  if (do_ic_alignment) {
    op->setAttr("use_3ic_optimize", builder.getI64IntegerAttr(4));
  }
  // rewrite weightOp
  auto elem_type = Module::getStorageType(filter());
  auto filter_type = RankedTensorType::get(filter_shape, elem_type);
  auto weight_op =
      top::WeightOp::create(op, "filter_reordered", *filter_i8, filter_type);
  op->setOperand(1, weight_op);
}

void tpu::Conv2DOp::weight_reorder_bf16_cv18xx() {
  conv_attr_t attr = {0};
  parseParam(&attr);
  OpBuilder builder(getContext());
  auto op = getOperation();
  // first lower weight
  auto filterOp = filter().getDefiningOp<top::WeightOp>();
  std::vector<int64_t> filter_shape = {attr.oc, attr.ic / attr.groups, attr.kh,
                                       attr.kw};
  auto filter_u16 = filterOp.read<uint16_t>();
  if (attr.dh > 1 || attr.dw > 1) {
    // TODO do dilation here, ins in top/tpu_common interpreter
    llvm_unreachable("dilation is not supported now");
  }
  transposeConvolutionFilter(filter_u16, filter_shape);
  // rewrite weightOp
  auto filter_type = RankedTensorType::get(filter_shape, builder.getBF16Type());
  auto weight_op =
      top::WeightOp::create(op, "filter_reordered", *filter_u16, filter_type);
  op->setOperand(1, weight_op);
  // second lower bias if exist
  if (attr.has_bias) {
    auto biasOp = bias().getDefiningOp<top::WeightOp>();
    auto bias_f32 = biasOp.read<float>();
    std::vector<uint32_t> bias_new(bias_f32->size());
    transposeBiasFp32(bias_f32, bias_new);
    // rewrite biasOp
    auto new_bias_type = RankedTensorType::get(Module::getShape(bias()),
                                               builder.getIntegerType(32));
    // bias().setType(new_bias_type);
    auto lbias_op =
        top::WeightOp::create(op, "bias_reordered", bias_new, new_bias_type);
    op->setOperand(2, lbias_op);
  }
}

// ======================================
// GlobalGenInterface
// ======================================

void tpu::Conv2DOp::codegen_global_cv18xx(void *ctx, int64_t layer_id) {
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  conv_attr_t attr = {0};
  parseParam(&attr);
  gaddr_t ga_input = Module::getAddress(input());
  gaddr_t ga_output = Module::getAddress(output());
  gaddr_t ga_filter = Module::getAddress(filter());
  gaddr_t ga_pc_info = GA_INVALID;
  if (Quant::isUniformQuantized(output()) || attr.has_bias) {
    ga_pc_info = Module::getAddress(bias());
  }

  auto fliter_shape = Module::getShape(filter());
  bool do_compress = attr.groups > 1 ? false : true;
  WeightCompresser weight_opt(this->getOperation(), do_compress); // fix me
  if (Quant::isUniformQuantized(output())) {
    bool do_ic_alignment = use_3ic_optimize() ? true : false;
    gaddr_t ga_scale_lut = GA_INVALID;
    // todo leakyrelu
    int fused_leakyrelu_pos_rshift = 0;
    int fused_leakyrelu_pos_m_i8 = 0;
    int fused_leakyrelu_neg_rshift = 0;
    int fused_leakyrelu_neg_m_i8 = 0;
    float fused_negative_slope = 0.0f; // Todo this->do_leaky_relu()

    cvi_backend_tg_fixed_conv_kernel(
        *backend_ctx,
        layer_id,   // layer_id,
        ga_input,   // input_data_gaddr,
        ga_output,  // output_data_gaddr,
        ga_filter,  // weight_data_gaddr,
        ga_pc_info, // bias_data_gaddr,
        attr.n, attr.ic, attr.ih, attr.iw,
        attr.groups, // group,
        attr.oc, attr.kh, attr.kw, attr.dh, attr.dw, attr.pht, attr.phb,
        attr.pwl,
        attr.pwr,               // pad (t, b, l, r)
        attr.ins_h, attr.ins_w, // ins_h, ins_w
        attr.sh, attr.sw,
        attr.has_bias,                                  // bias_term,
        attr.do_relu ? 1 : 0,                           // do_activation,
        attr.do_relu ? &fused_negative_slope : nullptr, // activation_arg,
        fused_leakyrelu_pos_m_i8,                       // activation_gt_scale,
        fused_leakyrelu_pos_rshift,                     // activation_gt_rshift,
        fused_leakyrelu_neg_m_i8,                       // activation_le_scale,
        fused_leakyrelu_neg_rshift,                     // activation_le_rshift,
        0,               // (int)rshift[0], //right_shift_width,
        true,            // do_chl_quan
        do_ic_alignment, // do_ic_alignment,
        &weight_opt.old_data, &weight_opt.new_data,
        attr.pad_value, // pad_value
        ga_scale_lut);
  } else {
    bool do_quant = false;
    gaddr_t ga_scale = GA_INVALID;
    gaddr_t ga_zeropoint = GA_INVALID;
    cvi_backend_tg_bf16_conv_kernel(*backend_ctx,
                                    layer_id,   // layer_id
                                    ga_input,   // input_data_gaddr,
                                    ga_output,  // output_data_gaddr,
                                    ga_filter,  // weight_data_gaddr,
                                    ga_pc_info, // bias_data_gaddr,
                                    attr.n, attr.ic, attr.ih, attr.iw,
                                    attr.groups, // group
                                    attr.oc, attr.kh, attr.kw, attr.dh, attr.dw,
                                    attr.pht, attr.phb, attr.pwl,
                                    attr.pwr,               // pad (t, b, l, r)
                                    attr.ins_h, attr.ins_w, // ins_h, ins_w
                                    attr.sh, attr.sw,
                                    attr.has_bias,        // bias_term,
                                    attr.do_relu ? 1 : 0, // do_activation,
                                    false,                // fp32_output
                                    &weight_opt.old_data, &weight_opt.new_data,
                                    do_quant, ga_scale, ga_zeropoint); // TODO
  }
}

// ======================================
// LocalGenInterface
// ======================================

int64_t tpu::Conv2DOp::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  int64_t sz = out_lmem_bytes * sizeof(int32_t);
  llvm_unreachable("Not supported now");
  return 0;
}

void tpu::Conv2DOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step) {
  llvm_unreachable("Not supported now");
}
