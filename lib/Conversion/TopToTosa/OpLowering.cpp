//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTosa/OpLowering.h"

namespace tpu_mlir {

void populateTopToTosaConversionPatterns(RewritePatternSet *patterns) {
  patterns->add<
      // clang-format off
        WeightLowering,
        AddLowering,
        ConvLowering,
        AvgPoolLowering,
        MaxPoolLowering
      // clang-format on
      >(patterns->getContext());
}

//===------------------------------------------------------------===//
// WeightLowering
//===------------------------------------------------------------===//
void WeightLowering::Lowering(PatternRewriter &rewriter,
                              top::WeightOp op) const {
  assert(op->getNumResults() == 1);
  auto out_Type = op->getResult(0).getType();
  auto val = op.read_as_float();
  DenseElementsAttr attr = DenseElementsAttr::get(
      out_Type, llvm::makeArrayRef(val->data(), val->size()));
  rewriter.replaceOpWithNewOp<mlir::tosa::ConstOp>(op, attr.getType(), attr);
}

//===------------------------------------------------------------===//
// AddLowering
//===------------------------------------------------------------===//
void AddLowering::Lowering(PatternRewriter &rewriter, top::AddOp op) const {
  assert(op->getNumResults() == 1);
  auto newType = op->getResult(0).getType();
  std::vector<Value> operands;
  for (auto in : op->getOperands()) {
    // if(auto InOp =
    //          dyn_cast<top::InputOp>(in.getDefiningOp())){
    if (isa<top::InputOp>(in.getDefiningOp())) {
      auto InOp = in.getDefiningOp<top::InputOp>();
      operands.push_back(InOp->getOperand(0));
    } else
      operands.push_back(in);
  }
  rewriter.replaceOpWithNewOp<mlir::tosa::AddOp>(op, newType, operands);
}

//===------------------------------------------------------------===//
// ConvLowering
//===------------------------------------------------------------===//
void ConvLowering::Lowering(PatternRewriter &rewriter, top::ConvOp op) const {
  assert(op->getNumResults() == 1);
  auto newType = op->getResult(0).getType();
  std::vector<Value> operands;
  for (auto in : op->getOperands()) {
    if (isa<top::InputOp>(in.getDefiningOp())) {
      auto InOp = in.getDefiningOp<top::InputOp>();
      operands.push_back(InOp->getOperand(0));
    } else
      operands.push_back(in);
  }
  std::vector<NamedAttribute> attrs;
  auto x1 =
      op.getPadsAttr().getValue()[0].cast<mlir::IntegerAttr>().getInt(); // top
  auto x2 = op.getPadsAttr()
                .getValue()[2]
                .cast<mlir::IntegerAttr>()
                .getInt(); // bottom
  auto x3 =
      op.getPadsAttr().getValue()[1].cast<mlir::IntegerAttr>().getInt(); // left
  auto x4 = op.getPadsAttr()
                .getValue()[3]
                .cast<mlir::IntegerAttr>()
                .getInt(); // right
  std::vector<int64_t> newValues{x1, x2, x3, x4};
  attrs.push_back(
      rewriter.getNamedAttr("pad", rewriter.getI64ArrayAttr(newValues)));
  attrs.push_back(rewriter.getNamedAttr("stride", op.getStridesAttr()));
  attrs.push_back(rewriter.getNamedAttr("dilation", op.getDilationsAttr()));
  rewriter.replaceOpWithNewOp<mlir::tosa::Conv2DOp>(op, newType, operands,
                                                    attrs);
}

//===------------------------------------------------------------===//
// AvgPoolLowering
//===------------------------------------------------------------===//
void AvgPoolLowering::Lowering(PatternRewriter &rewriter,
                               top::AvgPoolOp op) const {
  assert(op->getNumResults() == 1);
  auto newType = op->getResult(0).getType();
  std::vector<Value> operands;
  for (auto in : op->getOperands()) {
    if (isa<top::InputOp>(in.getDefiningOp())) {
      auto InOp = in.getDefiningOp<top::InputOp>();
      operands.push_back(InOp->getOperand(0));
    } else
      operands.push_back(in);
  }
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("kernel", op.getKernelShapeAttr()));
  attrs.push_back(rewriter.getNamedAttr("stride", op.getStridesAttr()));
  auto x1 =
      op.getPadsAttr().getValue()[0].cast<mlir::IntegerAttr>().getInt(); // top
  auto x2 = op.getPadsAttr()
                .getValue()[2]
                .cast<mlir::IntegerAttr>()
                .getInt(); // bottom
  auto x3 =
      op.getPadsAttr().getValue()[1].cast<mlir::IntegerAttr>().getInt(); // left
  auto x4 = op.getPadsAttr()
                .getValue()[3]
                .cast<mlir::IntegerAttr>()
                .getInt(); // right
  std::vector<int64_t> newValues{x1, x2, x3, x4};
  attrs.push_back(
      rewriter.getNamedAttr("pad", rewriter.getI64ArrayAttr(newValues)));
  rewriter.replaceOpWithNewOp<mlir::tosa::AvgPool2dOp>(op, newType, operands,
                                                       attrs);
}

//===------------------------------------------------------------===//
// MaxPoolLowering
//===------------------------------------------------------------===//
void MaxPoolLowering::Lowering(PatternRewriter &rewriter,
                               top::MaxPoolOp op) const {
  assert(op->getNumResults() == 1);
  auto newType = op->getResult(0).getType();
  std::vector<Value> operands;
  for (auto in : op->getOperands()) {
    if (isa<top::InputOp>(in.getDefiningOp())) {
      auto InOp = in.getDefiningOp<top::InputOp>();
      operands.push_back(InOp->getOperand(0));
    } else
      operands.push_back(in);
  }
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("kernel", op.getKernelShapeAttr()));
  attrs.push_back(rewriter.getNamedAttr("stride", op.getStridesAttr()));
  auto x1 =
      op.getPadsAttr().getValue()[0].cast<mlir::IntegerAttr>().getInt(); // top
  auto x2 = op.getPadsAttr()
                .getValue()[2]
                .cast<mlir::IntegerAttr>()
                .getInt(); // bottom
  auto x3 =
      op.getPadsAttr().getValue()[1].cast<mlir::IntegerAttr>().getInt(); // left
  auto x4 = op.getPadsAttr()
                .getValue()[3]
                .cast<mlir::IntegerAttr>()
                .getInt(); // right
  std::vector<int64_t> newValues{x1, x2, x3, x4};
  attrs.push_back(
      rewriter.getNamedAttr("pad", rewriter.getI64ArrayAttr(newValues)));
  rewriter.replaceOpWithNewOp<mlir::tosa::MaxPool2dOp>(op, newType, operands,
                                                       attrs);
}

} // namespace tpu_mlir
