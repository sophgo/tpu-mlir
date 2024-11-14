//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

static const unsigned char dev_table_i[256] = {
    1,   1,   1,   171, 1,   205, 171, 73,  1,   57,  205, 93,  171, 79,  73,
    137, 1,   241, 57,  27,  205, 195, 93,  89,  171, 41,  79,  19,  73,  141,
    137, 33,  1,   31,  241, 117, 57,  221, 27,  105, 205, 25,  195, 191, 93,
    91,  89,  87,  171, 167, 41,  161, 79,  155, 19,  149, 73,  9,   141, 139,
    137, 67,  33,  65,  1,   63,  31,  245, 241, 237, 117, 231, 57,  7,   221,
    109, 27,  213, 105, 207, 205, 101, 25,  197, 195, 193, 191, 47,  93,  23,
    91,  45,  89,  11,  87,  43,  171, 169, 167, 165, 41,  81,  161, 159, 79,
    39,  155, 153, 19,  75,  149, 37,  73,  145, 9,   71,  141, 35,  139, 69,
    137, 135, 67,  133, 33,  131, 65,  129, 1,   127, 63,  125, 31,  123, 245,
    243, 241, 239, 237, 59,  117, 29,  231, 229, 57,  113, 7,   223, 221, 55,
    109, 217, 27,  107, 213, 211, 105, 209, 207, 103, 205, 51,  101, 201, 25,
    199, 197, 49,  195, 97,  193, 3,   191, 189, 47,  187, 93,  185, 23,  183,
    91,  181, 45,  179, 89,  177, 11,  175, 87,  173, 43,  43,  171, 85,  169,
    21,  167, 83,  165, 165, 41,  163, 81,  161, 161, 5,   159, 79,  79,  157,
    39,  155, 155, 77,  153, 19,  19,  151, 75,  75,  149, 37,  37,  147, 73,
    73};

static const unsigned char dev_table_e[256] = {
    0,  0,  1,  9,  2,  10, 10, 9,  3,  9,  11, 10, 11, 10, 10, 11, 4,  12, 10,
    9,  12, 12, 11, 11, 12, 10, 11, 9,  11, 12, 12, 10, 5,  10, 13, 12, 11, 13,
    10, 12, 13, 10, 13, 13, 12, 12, 12, 12, 13, 13, 11, 13, 12, 13, 10, 13, 12,
    9,  13, 13, 13, 12, 11, 12, 6,  12, 11, 14, 14, 14, 13, 14, 12, 9,  14, 13,
    11, 14, 13, 14, 14, 13, 11, 14, 14, 14, 14, 12, 13, 11, 13, 12, 13, 10, 13,
    12, 14, 14, 14, 14, 12, 13, 14, 14, 13, 12, 14, 14, 11, 13, 14, 12, 13, 14,
    10, 13, 14, 12, 14, 13, 14, 14, 13, 14, 12, 14, 13, 14, 7,  14, 13, 14, 12,
    14, 15, 15, 15, 15, 15, 13, 14, 12, 15, 15, 13, 14, 10, 15, 15, 13, 14, 15,
    12, 14, 15, 15, 14, 15, 15, 14, 15, 13, 14, 15, 12, 15, 15, 13, 15, 14, 15,
    9,  15, 15, 13, 15, 14, 15, 12, 15, 14, 15, 13, 15, 14, 15, 11, 15, 14, 15,
    13, 13, 15, 14, 15, 12, 15, 14, 15, 15, 13, 15, 14, 15, 15, 10, 15, 14, 14,
    15, 13, 15, 15, 14, 15, 12, 12, 15, 14, 14, 15, 13, 13, 15, 14, 14};

void AvgPoolLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::AvgPoolOp op) const {
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Avg));
  if (op.getKernelShape().size() == 3) {
    lowering_common_f32<tpu::Pool3DOp>(rewriter, op, 2);
  } else if (op.getKernelShape().size() == 2) {
    lowering_common_f32<tpu::Pool2DOp>(rewriter, op);
  } else {
    lowering_common_f32<tpu::Pool1DOp>(rewriter, op);
  }
}

void AvgPoolLowering::LoweringINT8(PatternRewriter &rewriter, top::AvgPoolOp op,
                                   bool asymmetric) const {
  auto p = op.parseParam();
  auto k = p.kd * p.kh * p.kw;
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Avg));
  if (k <= 225) {
    op->setAttr("multiplier", rewriter.getSI32IntegerAttr(dev_table_i[k]));
    op->setAttr("rshift", rewriter.getSI32IntegerAttr(dev_table_e[k]));
    if (op.getKernelShape().size() == 3) {
      lowering_common_int8<tpu::Pool3DOp>(rewriter, op, asymmetric, 2);
    } else if (op.getKernelShape().size() == 2) {
      lowering_common_int8<tpu::Pool2DOp>(rewriter, op);
    } else {
      lowering_common_int8<tpu::Pool1DOp>(rewriter, op);
    }
  } else {
    if (op.getKernelShape().size() == 3) {
      lowering_common_f32<tpu::Pool3DOp>(rewriter, op, 2);
    } else if (op.getKernelShape().size() == 2) {
      lowering_common_f32<tpu::Pool2DOp>(rewriter, op);
    } else {
      lowering_common_f32<tpu::Pool1DOp>(rewriter, op);
    }
  }
}

} // namespace bm1684
} // namespace tpu_mlir
