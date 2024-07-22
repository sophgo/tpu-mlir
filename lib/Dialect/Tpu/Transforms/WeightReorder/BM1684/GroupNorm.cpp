//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../WeightReorder.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"

using namespace bm1684;
using namespace backend;

// ======================================
// WeightReorderInterface
// ======================================
// W =  [1, c, 1, 1]
// R =  [1, c, 1, 1]
// => W0 = [n, c, 1, 1]
//    R0 = [n, c, 1, 1]
static void groupnorm_filter_broadcast(const std::vector<int64_t> &filter_shape,
                                       const void *filter_orig,
                                       void *filter_trans) {
  int n = filter_shape[0];
  int c = filter_shape[1];
  for (int kn = 0; kn < n; kn++) {
    for (int kc = 0; kc < c; kc++) {
      uint64_t src = kc;
      uint64_t dst = kn * c + kc;
      *((float *)filter_trans + dst) = *((float *)filter_orig + src);
    }
  }
}

template <>
LogicalResult WeightReorder<tpu::GroupNormOp, Float32Type>::matchAndRewriteImpl(
    tpu::GroupNormOp op, PatternRewriter &rewriter) const {
  /// do broadcast for weight and bias
  int64_t N, C, H, W;
  module::getNCHW(op.getOutput(), N, C, H, W);
  int new_filter_count = N * C;
  auto out_type = module::getStorageType(op.getOutput());
  std::vector<int64_t> new_filter_shape = {N, C, 1, 1};
  if (module::getStorageType(op.getWeight()).isF32()) {
    auto filterOp = op.getWeight().getDefiningOp<top::WeightOp>();
    auto weight_data = filterOp.read_as_byte();
    auto new_weight = std::make_shared<std::vector<float>>(new_filter_count, 0);
    groupnorm_filter_broadcast(new_filter_shape, weight_data->data(),
                               new_weight->data());
    auto new_type = RankedTensorType::get(new_filter_shape, out_type);
    auto new_weightOp = top::WeightOp::create(
        op.getWeight().getDefiningOp(), "reorderd", *new_weight, new_type);
    op.setOperand(1, new_weightOp);
  }
  if (module::getStorageType(op.getBias()).isF32()) {
    auto biasOp = op.getBias().getDefiningOp<top::WeightOp>();
    auto bias_data = biasOp.read_as_byte();
    auto new_bias = std::make_shared<std::vector<float>>(new_filter_count, 0);
    groupnorm_filter_broadcast(new_filter_shape, bias_data->data(),
                               new_bias->data());
    auto new_type = RankedTensorType::get(new_filter_shape, out_type);
    auto new_biasOp = top::WeightOp::create(op.getBias().getDefiningOp(),
                                            "reorderd", *new_bias, new_type);
    op.setOperand(2, new_biasOp);
    return success();
  }
  return failure();
}
