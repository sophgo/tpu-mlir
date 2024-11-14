//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace llvm;

namespace tpu_mlir {
namespace tpu {

// op -> reshape + op + reshape
// (1, 1, H, W) -> (1, H, W)
struct InsertReshapePattern : public OpRewriterPatternEx3 {
  InsertReshapePattern(MLIRContext *context, int level)
      : OpRewriterPatternEx3(context, "InsertReshapePattern", level) {}
  LogicalResult matchAndRewriteImpl(Operation *op,
                                    PatternRewriter &rewriter) const override {

    if (false == isa<tpu::MatMulOp, tpu::CastOp, tpu::RequantIntAxisOp,
                     tpu::RequantIntOp, tpu::BinaryConstShiftOp, tpu::LutOp,
                     tpu::LayerNormOp, tpu::ActiveOp>(op)) {
      return failure();
    }
    if (op->getNumResults() != 1) {
      return failure();
    }
    auto shape = module::getShape(op->getOperand(0));
    auto out_shape = module::getShape(op->getResult(0));
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 1 || shape[2] == 1 ||
        shape[3] == 1) {
      return failure();
    }
    if (out_shape.size() != 4 || out_shape[0] != 1 || out_shape[1] != 1 ||
        out_shape[2] == 1 || out_shape[3] == 1) {
      return failure();
    }
    // if (auto bianry_op = dyn_cast<tpu::BinaryShiftOp>(op)) {
    //   auto shape1 = module::getShape(op->getOperand(1));
    //   if (shape.size() != shape1.size()) {
    //     return failure();
    //   }
    //   for (int i = 0; i < shape.size(); ++i) {
    //     if (shape[i] != shape1[i]) {
    //       return failure();
    //     }
    //   }
    // }
    if (auto matmul_op = dyn_cast<tpu::MatMulOp>(op)) {
      if (module::isWeight(matmul_op.getRight()) == false) {
        return failure();
      }
      if (module::isNone(matmul_op.getBias()) == false) {
        auto b_shape = module::getShape(matmul_op.getBias());
        if (b_shape.size() == 4) {
          auto b_type = RankedTensorType::get(
              b_shape.drop_front(),
              module::getElementType(matmul_op.getBias()));
          matmul_op.getBias().setType(b_type);
        }
      }
    }
    if (auto rq_op = dyn_cast<tpu::RequantIntAxisOp>(op)) {
      auto axis = rq_op.getRqAxis();
      axis = axis < 0 ? axis + shape.size() : axis;
      auto new_axis = axis > 0 ? axis - 1 : axis;
      rq_op.setRqAxis(new_axis);
      auto q_shape = module::getShape(rq_op.getQuant());
      auto q_new_shape = axis > 0 ? q_shape.drop_front() : q_shape.drop_back();
      auto q_type = RankedTensorType::get(
          q_new_shape, module::getElementType(rq_op.getQuant()));
      rq_op.getQuant().setType(q_type);
    }
    if (auto norm_op = dyn_cast<tpu::LayerNormOp>(op)) {
      auto axis = norm_op.getAxis();
      axis = axis < 0 ? axis + shape.size() : axis;
      auto new_axis = axis > 0 ? axis - 1 : axis;
      norm_op.setAxis(new_axis);
    }

    // squeeze dim
    rewriter.setInsertionPointAfterValue(op->getOperand(0));
    auto sq_loc = NameLoc::get(rewriter.getStringAttr(
        module::getName(op->getResult(0)).str() + "_squeeze_dim"));
    std::vector<int64_t> new_shape(shape.size() - 1, 1);
    new_shape[0] = shape[0];
    for (int i = 1; i < shape.size() - 1; ++i) {
      new_shape[i] = shape[i + 1];
    }
    auto sq_type = RankedTensorType::get(
        new_shape, module::getElementType(op->getOperand(0)));
    std::vector<NamedAttribute> sq_attrs;
    sq_attrs.emplace_back(
        rewriter.getNamedAttr("shape", rewriter.getI64ArrayAttr(new_shape)));
    auto squeeze_op = rewriter.create<tpu::ReshapeOp>(
        sq_loc, sq_type, ValueRange{op->getOperand(0)}, sq_attrs);

    // expand dim
    rewriter.setInsertionPointAfterValue(op->getResult(0));
    std::vector<NamedAttribute> ed_attrs;
    ed_attrs.emplace_back(
        rewriter.getNamedAttr("shape", rewriter.getI64ArrayAttr(shape)));
    auto expand_op = rewriter.create<tpu::ReshapeOp>(
        op->getLoc(), op->getResult(0).getType(), ValueRange{op->getResult(0)},
        ed_attrs);

    // op
    std::vector<int64_t> out_new_shape(out_shape.size() - 1, 1);
    out_new_shape[0] = out_shape[0];
    for (int i = 1; i < out_shape.size() - 1; ++i) {
      out_new_shape[i] = out_shape[i + 1];
    }
    auto op_loc = NameLoc::get(rewriter.getStringAttr(
        module::getName(op->getResult(0)).str() + "_op_squeeze_dim"));
    op->setLoc(op_loc);
    op->setOperands(0, 1, {squeeze_op.getOutput()});
    auto new_type = RankedTensorType::get(
        out_new_shape, module::getElementType(op->getResult(0)));
    op->getResult(0).setType(new_type);
    op->getResult(0).replaceAllUsesWith(expand_op.getOutput());
    expand_op->setOperands(0, 1, {op->getResult(0)});

    return success();
  }
  bool shouldPrint(Operation *op) const override { return false; }
};

class ShapeOptimizePass : public ShapeOptimizeBase<ShapeOptimizePass> {
public:
  ShapeOptimizePass() {}
  void runOnOperation() override {
    auto ctx = &getContext();
    auto modules = module::getAllModules();
    for (auto s : *modules) {
      for (auto func : s.getOps<FuncOp>()) {
        RewritePatternSet patterns(ctx);
        patterns.add<InsertReshapePattern>(ctx, 1);
        applyPatternsAndFoldGreedily(func, std::move(patterns));
        // special for attention
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createShapeOptimizePass() {
  return std::make_unique<ShapeOptimizePass>();
}
} // namespace tpu
} // namespace tpu_mlir
