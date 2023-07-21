//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

namespace tpu_mlir {
namespace tpu {

// only >= 4MB distribute to multi devices
static const int64_t WEIGHT_LIMIT = 0x400000;

class MatMulDistributePattern : public OpRewritePattern<tpu::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto num_devices = module::getDeviceNum();
    if (num_devices <= 1) {
      return failure();
    }
    if (op->hasOneUse()) {
      auto user = *op->getUsers().begin();
      if (isa<tpu::ConnectOp>(user)) {
        return failure();
      }
    }
    if (module::isWeight(op.getRight()) == false) {
      return failure();
    }
    auto out_stype = module::getStorageType(op.getOutput());
    if (!out_stype.isa<FloatType>()) {
      return failure();
    }
    auto num_right = module::getNumElements(op.getRight());
    if (num_right <= WEIGHT_LIMIT) {
      return failure();
    }
    auto attrs = op->getAttrs();
    auto type = op.getOutput().getType();
    auto outputShape = module::getShape(op.getOutput());
    auto inputShape = module::getShape(op.getInput());
    auto filterOp = op.getRight().getDefiningOp<top::WeightOp>();
    auto filterShape = module::getShape(filterOp.getOutput());
    auto has_bias = !module::isNone(op.getBias());
    auto num_dims = filterShape.size();
    auto K = filterShape[num_dims - 2];
    auto N = filterShape[num_dims - 1];
    auto name = module::getName(op.getOperation()).str();
    std::vector<Value> connect_operands;
    auto ctx = rewriter.getContext();
    if (K <= N || op.getDoRelu()) {
      auto slice_n = ceiling_func(N, num_devices);
      for (int i = 0; i < num_devices; i++) {
        auto offset = i * slice_n;
        auto length = std::min(slice_n, N - offset);
        auto newFilter =
            module::opSliceAxis(op.getRight(), num_dims - 1, offset, length);
        std::vector<Value> operands;
        operands.push_back(op.getInput());
        operands.push_back(newFilter);
        if (has_bias) {
          auto new_bias =
              module::opSliceAxis(op.getBias(), num_dims - 1, offset, length);
          operands.push_back(new_bias);
        } else {
          operands.push_back(op.getBias());
        }
        auto new_name = name + "_" + std::to_string(i);
        auto new_loc = NameLoc::get(rewriter.getStringAttr(new_name));
        std::vector<int64_t> new_shape = outputShape;
        new_shape[new_shape.size() - 1] = length;
        auto new_type = RankedTensorType::get(
            new_shape, module::getElementType(op.getOutput()));
        auto new_mm =
            rewriter.create<tpu::MatMulOp>(new_loc, new_type, operands, attrs);
        connect_operands.push_back(new_mm.getOutput());
      }
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr(
          "mode", tpu::ConnectModeAttr::get(ctx, tpu::ConnectMode::Concat)));
      rewriter.replaceOpWithNewOp<tpu::ConnectOp>(op, type, connect_operands,
                                                  attrs);
    } else {
      auto slice_k = ceiling_func(K, num_devices);
      for (int i = 0; i < num_devices; i++) {
        auto offset = i * slice_k;
        auto length = std::min(slice_k, K - offset);
        auto inputSlice = module::opSliceAxis(
            op.getInput(), inputShape.size() - 1, offset, length);
        auto newFilter =
            module::opSliceAxis(op.getRight(), num_dims - 2, offset, length);
        std::vector<Value> operands;
        operands.push_back(inputSlice);
        operands.push_back(newFilter);
        if (has_bias) {
          auto bias = op.getBias().getDefiningOp<top::WeightOp>();
          auto new_bias = bias.clone(std::to_string(i));
          operands.push_back(new_bias);
        } else {
          operands.push_back(op.getBias());
        }
        auto new_name = name + "_" + std::to_string(i);
        auto new_loc = NameLoc::get(rewriter.getStringAttr(new_name));
        auto new_mm =
            rewriter.create<tpu::MatMulOp>(new_loc, type, operands, attrs);
        connect_operands.push_back(new_mm.getOutput());
      }
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr(
          "mode", tpu::ConnectModeAttr::get(ctx, tpu::ConnectMode::Add)));
      rewriter.replaceOpWithNewOp<tpu::ConnectOp>(op, type, connect_operands,
                                                  attrs);
    }
    return success();
  }
};

class MatMulDistributePattern : public OpRewritePattern<tpu::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto num_devices = module::getDeviceNum();
    if (num_devices <= 1) {
      return failure();
    }
  }
};

void populateDistributeBM1684XPatterns(RewritePatternSet *patterns) {
  patterns->add<MatMulDistributePattern>(patterns->getContext());
};

} // namespace tpu
} // namespace tpu_mlir
