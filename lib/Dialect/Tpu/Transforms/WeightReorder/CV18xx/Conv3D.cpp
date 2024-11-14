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

static void get_strides_from_shapes5d(int strides[5], const int shapes[5],
                                      int ws) {
  strides[5 - 1] = ws;
  for (int i = 5 - 2; i >= 0; i--)
    strides[i] = shapes[i + 1] * strides[i + 1];
}

static int get_tensor5d_offset(int poss[5], const int strides[5]) {
  int offset = 0;
  for (int i = 0; i < 5; i++)
    offset += poss[i] * strides[i];

  return offset;
}
// ======================================
// WeightReorderInterface
// ======================================

// (oc, ic, kd, kh, kw) -> (kd, oc, kh, kw, ic)
template <typename T>
static void transposeConvolution3dFilter(std::shared_ptr<std::vector<T>> &w,
                                         const std::vector<int64_t> &s) {
  int oc, ic, kd, kh, kw;
  if (s.size() == 5) {
    // oc, ic, kd, kh, kw
    oc = (int)s[0];
    ic = (int)s[1];
    kd = (int)s[2];
    kh = (int)s[3];
    kw = (int)s[4];
  } else {
    llvm_unreachable("unsupported shape size");
  }

  std::vector<T> w_t(w->size());
  int cpu_shapes[5] = {oc, ic, kd, kh, kw};
  int tpu_shapes[5] = {kd, oc, kh, kw, ic};

  // logical stride, in unit of float
  int cpu_strides[5], tpu_strides[5];
  get_strides_from_shapes5d(cpu_strides, cpu_shapes, 1);
  get_strides_from_shapes5d(tpu_strides, tpu_shapes, 1);

  // (oc, ic, id, kh, kw) -> (id, oc, khxkw, ic)
  for (int i = 0; i < cpu_shapes[0]; i++) {
    for (int j = 0; j < cpu_shapes[1]; j++) {
      for (int z = 0; z < cpu_shapes[2]; z++) {
        for (int y = 0; y < cpu_shapes[3]; y++) {
          for (int x = 0; x < cpu_shapes[4]; x++) {
            int cpu_poss[5] = {i, j, z, y, x};
            int tpu_poss[5] = {z, i, y, x, j};
            int cpu_offset = get_tensor5d_offset(cpu_poss, cpu_strides);
            int tpu_offset = get_tensor5d_offset(tpu_poss, tpu_strides);
            w_t[tpu_offset] = w->at(cpu_offset);
          }
        }
      }
    }
  }

  w->assign(w_t.begin(), w_t.end());
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
LogicalResult WeightReorder<tpu::Conv3DOp, BFloat16Type>::matchAndRewriteImpl(
    tpu::Conv3DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isBF16())
    return failure();

  auto attr = op.parseParam();
  // first lower weight
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto filter_shape = module::getShape(filterOp.getOutput());
  auto filter_u16 = filterOp.read<uint16_t>();
  if (attr.dh > 15 || attr.dw > 15) {
    // TODO do dilation here, ins in top/tpu_common interpreter
    llvm_unreachable("dilation is not supported now");
  }
  transposeConvolution3dFilter(filter_u16, filter_shape.vec());
  // rewrite weightOp shape (oc, ic/g, kh, kw) -> (1, oc, kh*kw, ic/g)
  auto filter_type =
      RankedTensorType::get(filter_shape, rewriter.getBF16Type());
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
