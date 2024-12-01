//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace tpu_mlir::top;

struct TopFusePad : public OpRewriterPatternEx<PadOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  TopFusePad(mlir::MLIRContext *context)
      : OpRewriterPatternEx<PadOp>(context, "TopFusePad") {}

  LogicalResult matchAndRewriteImpl(PadOp op,
                                    PatternRewriter &rewriter) const override {

    if (op.getInput().getType().dyn_cast<TensorType>().getShape().size() < 3)
      return failure();
    if (op.getPaddingsT())
      return failure();
    auto paddings = module::getI64Array(op.getPaddings());
    // without batch or channel padding
    int tensor_dim = paddings.get()->size() / 2;
    for (int i = 0; i < 2; ++i) {
      if (paddings.get()->at(i) != 0 || paddings.get()->at(i + tensor_dim) != 0)
        return failure();
    }
    int pad_dim = tensor_dim - 2;

    // only const pad
    auto pad_mode = op.getMode();
    if (pad_mode != "constant")
      return failure();

    // check next op, pad_value and pad algo
    double pad_value = op->getAttr("val").cast<FloatAttr>().getValueAsDouble();
    for (auto nextOp_iter = op->user_begin(); nextOp_iter != op->user_end();
         nextOp_iter++) {
      auto nextOp = *nextOp_iter;
      if (isa<ConvOp>(nextOp)) {
        if (pad_value != 0)
          return failure();
      } else if (isa<MaxPoolOp, AvgPoolOp>(nextOp)) {
        auto nextOp_pad_value =
            nextOp->getAttr("pad_value").cast<IntegerAttr>().getInt();
        if (pad_value != double(nextOp_pad_value))
          return failure();
        auto nextOp_count_include_pad =
            nextOp->getAttr("count_include_pad").cast<BoolAttr>().getValue();
        if (!nextOp_count_include_pad)
          return failure();
      } else
        return failure();
    }

    // remove batch padding and channel padding
    std::vector<int64_t> paddings_(pad_dim * 2, 0);
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < pad_dim; ++j) {
        paddings_[i * pad_dim + j] = paddings.get()->at(i * tensor_dim + j + 2);
      }
    }

    // check tensor dims and paddings after merged
    for (auto nextOp_iter = op->user_begin(); nextOp_iter != op->user_end();
         nextOp_iter++) {
      auto nextOp = *nextOp_iter;
      auto kernel_shape = nextOp->getAttr("kernel_shape").dyn_cast<ArrayAttr>();
      if (kernel_shape.size() != pad_dim)
        return failure();
      auto next_paddings =
          module::getI64Array(nextOp->getAttr("pads").dyn_cast<ArrayAttr>());
      for (int i = 0; i < pad_dim * 2; ++i) {
        // chip limit
        if (next_paddings.get()->at(i) + paddings_[i] > 15)
          return failure();
      }
    }

    // merge paddings
    for (auto nextOp_iter = op->user_begin(); nextOp_iter != op->user_end();
         nextOp_iter++) {
      std::vector<int64_t> new_paddings(pad_dim * 2, 0);
      auto nextOp = *nextOp_iter;
      auto next_paddings =
          module::getI64Array(nextOp->getAttr("pads").dyn_cast<ArrayAttr>());
      for (int i = 0; i < pad_dim * 2; ++i) {
        new_paddings[i] = next_paddings.get()->at(i) + paddings_[i];
      }
      nextOp->setAttr("pads", rewriter.getI64ArrayAttr(new_paddings));
    }

    // remove the pad Op
    rewriter.replaceOp(op, {op.getInput()});
    return success();
  }
};

void PadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<TopFusePad>(context);
}
