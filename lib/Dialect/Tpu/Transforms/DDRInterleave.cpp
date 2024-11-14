//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineExpr.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace llvm;

namespace tpu_mlir {
namespace tpu {

class MatMulDDRInterleave : public OpRewriterPatternEx<MatMulOp> {
public:
  MatMulDDRInterleave(mlir::MLIRContext *context)
      : OpRewriterPatternEx<MatMulOp>(context, "MatMulDDRInterleave") {}

  LogicalResult matchAndRewriteImpl(MatMulOp Op,
                                    PatternRewriter &rewriter) const override {

    for (auto user : Op->getUsers())
      if (user->hasTrait<OpTrait::IsTerminator>())
        return failure();

    auto tensorType = cast<mlir::RankedTensorType>(Op.getOutput().getType());
    if (isa_and_present<tpu::DDRInterleaveAttr>(tensorType.getEncoding()))
      return failure();

    auto context = getContext();
    auto d0 = mlir::getAffineDimExpr(0, context);
    auto d1 = mlir::getAffineDimExpr(1, context);
    auto s8 = mlir::getAffineConstantExpr(8, context);

    auto stride = mlir::AffineMap::get(2, 0, {s8, d0, d1.ceilDiv(8)}, context);

    auto ddrAttr = tpu::DDRInterleaveAttr::get(context, stride, {0}, 0);

    auto ddrType = mlir::RankedTensorType::get(
        tensorType.getShape(), tensorType.getElementType(), ddrAttr);
    Op.getOutput().setType(ddrType);

    if (isa<top::WeightOp>(Op.getRight().getDefiningOp())) {
      auto tensorType = cast<mlir::RankedTensorType>(Op.getRight().getType());
      auto ddrType = mlir::RankedTensorType::get(
          tensorType.getShape(), tensorType.getElementType(), ddrAttr);
      Op.getRight().setType(ddrType);
    }
    return success();
  };
  bool shouldPrint(MatMulOp Op) const override { return false; }
};

class DDRInterleavePass : public DDRInterleaveBase<DDRInterleavePass> {
public:
  DDRInterleavePass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    module::applyPatternOnce<MatMulDDRInterleave>(mOp);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createDDRInterleavePass() {
  return std::make_unique<DDRInterleavePass>();
};

} // namespace tpu
} // namespace tpu_mlir
