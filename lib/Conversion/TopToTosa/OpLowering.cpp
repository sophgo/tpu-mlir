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
        InputLowering,
        AddLowering,
        ConvLowering,
        AvgPoolLowering,
        MaxPoolLowering,
        SoftmaxLowering
      // clang-format on
      >(patterns->getContext());
}

//===------------------------------------------------------------===//
// InputLowering
//===------------------------------------------------------------===//
void InputLowering::Lowering(PatternRewriter &rewriter, top::InputOp op) const {
  assert(op->getNumResults() == 1);
  auto outType = change_dataformat(op->getResult(0).getType());
  std::vector<Value> operands;
  operands.push_back(op->getOperand(0));
  std::vector<int32_t> perms = {0, 2, 3, 1};
  auto const_ty = RankedTensorType::get({4}, rewriter.getI32Type());
  DenseElementsAttr attr = DenseElementsAttr::get(
      const_ty, llvm::makeArrayRef(perms.data(), perms.size()));
  auto constop =
      rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr);
  operands.push_back(constop->getResult(0));
  rewriter.replaceOpWithNewOp<mlir::tosa::TransposeOp>(op, outType, operands);
}

//===------------------------------------------------------------===//
// AddLowering
//===------------------------------------------------------------===//
void AddLowering::Lowering(PatternRewriter &rewriter, top::AddOp op) const {
  assert(op->getNumResults() == 1);
  auto newType = change_dataformat(op->getResult(0).getType());
  std::vector<Value> operands;
  for (auto in : op->getOperands()) {
    operands.push_back(in);
    /*
    // if(auto InOp =
    //          dyn_cast<top::InputOp>(in.getDefiningOp())){
    if (isa<top::InputOp>(in.getDefiningOp())) {
      auto InOp = in.getDefiningOp<top::InputOp>();
      operands.push_back(InOp->getOperand(0));
    } else
      operands.push_back(in);
    */
  }
  rewriter.replaceOpWithNewOp<mlir::tosa::AddOp>(op, newType, operands);
}

//===------------------------------------------------------------===//
// ConvLowering
//===------------------------------------------------------------===//
void ConvLowering::Lowering(PatternRewriter &rewriter, top::ConvOp op) const {
  assert(op->getNumResults() == 1);
  auto newType = change_dataformat(op->getResult(0).getType());
  std::vector<Value> operands;
  for (auto in : op->getOperands()) {
    operands.push_back(in);
    /*
    if (isa<top::InputOp>(in.getDefiningOp())) {
      auto InOp = in.getDefiningOp<top::InputOp>();
      operands.push_back(InOp->getOperand(0));
    } else
      operands.push_back(in);
    */
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
  auto newType = change_dataformat(op->getResult(0).getType());
  std::vector<Value> operands;
  for (auto in : op->getOperands()) {
    operands.push_back(in);
    /*
    if (isa<top::InputOp>(in.getDefiningOp())) {
      auto InOp = in.getDefiningOp<top::InputOp>();
      operands.push_back(InOp->getOperand(0));
    } else
      operands.push_back(in);
    */
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
  auto newType = change_dataformat(op->getResult(0).getType());
  std::vector<Value> operands;
  for (auto in : op->getOperands()) {
    operands.push_back(in);
    /*
    if (isa<top::InputOp>(in.getDefiningOp())) {
      auto InOp = in.getDefiningOp<top::InputOp>();
      operands.push_back(InOp->getOperand(0));
    } else
      operands.push_back(in);
    */
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

//===------------------------------------------------------------===//
// SoftmaxLowering
//===------------------------------------------------------------===//
void SoftmaxLowering::Lowering(PatternRewriter &rewriter,
                               top::SoftmaxOp op) const {
  assert(op->getNumResults() == 1);
  auto preType = op->getResult(0).getType();
  auto newType = change_dataformat(preType);
  auto size = preType.cast<RankedTensorType>().getShape().size();
  uint64_t new_axis, axis = op.getAxis();
  if (size == 4) {
    if (axis == 1 || axis == -3)
      new_axis = 3; // C
    else if (axis == 2 || axis == -2)
      new_axis = 1; // H
    else if (axis == 3 || axis == -1)
      new_axis = 2; // W
    else
      new_axis = axis; // N
  }
  // ExpOp (beta = 1 by default)
  auto exp = rewriter.create<mlir::tosa::ExpOp>(
      op->getLoc(),
      //                     op->getOperand(0).getType(), op->getOperands());
      newType, op->getOperand(0).getDefiningOp()->getResults());
  // ResuceSumOp
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("axis", rewriter.getI64IntegerAttr(new_axis)));
  auto exp_ty = exp->getResult(0).getType().cast<RankedTensorType>();
  std::vector<int64_t> out_shape(exp_ty.getShape());
  out_shape[new_axis] = 1;
  auto out_type = RankedTensorType::get(out_shape, exp_ty.getElementType());
  auto reducesum = rewriter.create<mlir::tosa::ReduceSumOp>(
      exp->getLoc(), out_type, exp->getResults(), attrs);
  // ReciprocalOp
  auto reducesum_ty = reducesum->getResult(0).getType();
  auto reciprocal = rewriter.create<mlir::tosa::ReciprocalOp>(
      reducesum->getLoc(), reducesum_ty, reducesum->getResults());
  // MulOp
  auto mul = rewriter.create<mlir::tosa::MulOp>(
      reciprocal->getLoc(), newType, exp->getResult(0),
      reciprocal->getResult(0), rewriter.getI32IntegerAttr(0));
  rewriter.replaceOp(op, mul->getResults());
}



} // namespace tpu_mlir
