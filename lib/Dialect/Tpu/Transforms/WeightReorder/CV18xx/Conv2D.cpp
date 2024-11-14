//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../WeightReorder.h"

using namespace cv18xx;

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
  assert(bias_u16.size() == 2 * size);
  memcpy(bias_u16.data(), bias_reshape_fp32.data(), size * sizeof(uint32_t));
}

template <>
LogicalResult WeightReorder<tpu::Conv2DOp, int8_t>::matchAndRewriteImpl(
    tpu::Conv2DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isInteger(8))
    return failure();

  auto attr = op.parseParam();
  // first, merge conv rshift/multiplier/bias into one packed tensor
  i32_array_t bias_new;
  std::vector<int64_t> bias_shape = {1, attr.oc, 1, 1};
  if (attr.has_bias) {
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
  // second lower weight  for groups onnx weight's shape (oc, ic/g, kh, kw)
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto filter_i8 = filterOp.read<int8_t>();
  std::vector<int64_t> filter_shape = {attr.oc, attr.ic / attr.groups, attr.kh,
                                       attr.kw};
  if (attr.dh > 15 || attr.dw > 15) {
    // TODO ins in top/tpu_common interpreter
    llvm_unreachable("Not supported now");
  }
  transposeConvolutionFilter(filter_i8, filter_shape);
  // third padding odd ic to even to enable double conv
  bool do_ic_alignment = false;
  refactorOddIcConv(filter_i8, filter_shape, attr.groups, do_ic_alignment);
  if (do_ic_alignment) {
    op->setAttr("use_3ic_optimize", rewriter.getI64IntegerAttr(4));
  }
  // rewrite weightOp shape (oc, ic/g, kh, kw) -> (1, oc, kh*kw, ic/g)
  std::vector<int64_t> new_filter_shape = {
      1, attr.oc, attr.kh * attr.kw,
      (attr.ic + (do_ic_alignment ? 1 : 0)) / attr.groups};

  if (attr.is_dw) {
    new_filter_shape = {1, attr.oc, attr.kh, attr.kw};
  }
  // rewrite weightOp
  auto elem_type = module::getStorageType(op.getFilter());
  auto filter_type = RankedTensorType::get(new_filter_shape, elem_type);
  auto weight_op =
      top::WeightOp::create(op, "filter_reordered", *filter_i8, filter_type);
  op->setOperand(1, weight_op);
  return success();
}

template <>
LogicalResult WeightReorder<tpu::Conv2DOp, BFloat16Type>::matchAndRewriteImpl(
    tpu::Conv2DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isBF16() ||
      !module::isWeight(op.getFilter()))
    return failure();

  auto attr = op.parseParam();
  // first lower weight
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  std::vector<int64_t> filter_shape = {attr.oc, attr.ic / attr.groups, attr.kh,
                                       attr.kw};
  auto filter_u16 = filterOp.read<uint16_t>();
  if (attr.dh > 15 || attr.dw > 15) {
    // TODO do dilation here, ins in top/tpu_common interpreter
    llvm_unreachable("dilation is not supported now");
  }
  transposeConvolutionFilter(filter_u16, filter_shape);
  // rewrite weightOp shape (oc, ic/g, kh, kw) -> (1, oc, kh*kw, ic/g)
  std::vector<int64_t> new_filter_shape = {1, attr.oc, attr.kh * attr.kw,
                                           attr.ic / attr.groups};
  if (attr.is_dw) {
    new_filter_shape = {1, attr.oc, attr.kh, attr.kw};
  }

  auto filter_type =
      RankedTensorType::get(new_filter_shape, rewriter.getBF16Type());
  auto weight_op =
      top::WeightOp::create(op, "filter_reordered", *filter_u16, filter_type);
  op->setOperand(1, weight_op);
  // second lower bias if exist
  if (attr.has_bias) {
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
