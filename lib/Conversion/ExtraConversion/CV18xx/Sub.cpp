//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/ExtraConversion/ExtraConvertCV18XX.h"
namespace tpu_mlir {
namespace cv18xx {
static int is_bcast(top::SubOp op) {
  int bcast = 0;
  auto shape0 = module::getShape(op.getInputs()[0]);
  auto shape1 = module::getShape(op.getInputs()[1]);
  auto prod0 = std::accumulate(shape0.begin(), shape0.end(), 1,
                               std::multiplies<int64_t>());
  auto prod1 = std::accumulate(shape1.begin(), shape1.end(), 1,
                               std::multiplies<int64_t>());
  auto sub = prod0 - prod1;
  if (sub < 0) {
    bcast = 1; //left bcast
  } else if (sub > 0) {
    bcast = 2; //right bcast
  }
  if (bcast) {
    auto len = std::min(shape0.size(), shape1.size());
    for (int i = 0; i < len; i++) {
      int dim_a = shape0[shape0.size() - 1 - i];
      int dim_b = shape1[shape1.size() - 1 - i];
      if (dim_a != dim_b &&
          ((sub > 0 && dim_b != 1) || (sub < 0 && dim_a != 1))) {
        llvm_unreachable("Broadcast dim should be 1");
      }
    }
  }
  return bcast;
}

LogicalResult ConvertSubOp::matchAndRewrite(top::SubOp op,
                              PatternRewriter &rewriter) const{
  //if not bcast,convert it to AddOp
  //if left_bcast, convert it to MulConstOp + AddOp
  //if right_bcast, not convert
  auto fn = module::getMainFuncOp();
  assert(op.getNumOperands() == 2);
  int bcast = is_bcast(op);
  auto coeff_v = module::getF64Array(op.getCoeff(), 2, 1.0);
  assert(coeff_v->at(0) == 1 && coeff_v->at(1) == 1);
  std::vector<NamedAttribute> attrs;
  if (bcast == 0) {
    attrs.push_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
    attrs.push_back(rewriter.getNamedAttr("relu_limit", op.getReluLimitAttr()));
    attrs.push_back(
        rewriter.getNamedAttr("coeff", rewriter.getF64ArrayAttr({1., -1.})));
    rewriter.replaceOpWithNewOp<top::AddOp>(op, op.getOutput().getType().cast<RankedTensorType>(),
                                            op.getOperands(), attrs);
    fn.dump();
    return success();
  } else if (bcast == 1) {
    auto left_operand = op.getOperands()[0];
    auto right_operand = op.getOperands()[1];
    assert(!isa<top::WeightOp>(right_operand.getDefiningOp()));
    rewriter.setInsertionPointAfterValue(right_operand);
    std::vector<Value> operands;
    attrs.emplace_back(rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(-1.0)));
    operands.emplace_back(right_operand);
    std::string name = module::getName(op.getOutput()).str();
    auto loc1 = NameLoc::get(rewriter.getStringAttr(name + "_mulconst"));
    auto type1 = right_operand.getType().cast<RankedTensorType>();
    auto mulconstOp = rewriter.create<top::MulConstOp>(loc1, type1, operands, attrs);
    auto out1 = mulconstOp.getOutput();
    attrs.clear();
    operands.clear();
    attrs.push_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
    attrs.push_back(rewriter.getNamedAttr("relu_limit", op.getReluLimitAttr()));
    operands.emplace_back(left_operand);
    operands.emplace_back(out1);
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<top::AddOp>(op, op.getOutput().getType().cast<RankedTensorType>(),
                                            operands, attrs);
    return success();

  } else if (bcast == 2) {
    return failure();
  }
  return success();
}
}
}
