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

LogicalResult
ConvertMaskedFillOp::matchAndRewrite(top::MaskedFillOp op,
                                     PatternRewriter &rewriter) const {
  bool inverse = op.getInversed();
  double const_val = op.getConstVal().convertToDouble();
  Value input0 = op.getOperand(0);
  Value input1 = op.getOperand(1);
  Value ori_out = op.getOutput();
  std::string name = module::getName(ori_out).str();
  std::vector<int64_t> output_shape = module::getShape(ori_out);
  std::vector<int64_t> input0_shape = module::getShape(input0);
  std::vector<int64_t> input1_shape = module::getShape(input1);
  // cv18xx only support one operand broadcast now.
  assert((input0_shape == output_shape || input1_shape == output_shape));
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  rewriter.setInsertionPointAfterValue(ori_out);
  if (inverse) {
    // out = input[0] * const_val + (1 - input[0]) * input[1]

    // create input[0] * const_val
    operands.emplace_back(input0);
    attrs.emplace_back(rewriter.getNamedAttr(
        "const_val", rewriter.getF64FloatAttr(const_val)));
    auto loc1 = NameLoc::get(rewriter.getStringAttr(name + "_mulconst1"));
    auto type1 = RankedTensorType::get(input0_shape, rewriter.getF32Type());
    auto mulconstOp1 =
        rewriter.create<top::MulConstOp>(loc1, type1, operands, attrs);
    auto out1 = mulconstOp1.getOutput();

    // create (input[0]) * input[1]
    operands.clear();
    attrs.clear();
    rewriter.setInsertionPointAfterValue(out1);
    operands.emplace_back(input0);
    operands.emplace_back(input1);
    auto loc2 = NameLoc::get(rewriter.getStringAttr(name + "_mul1"));
    auto type2 = RankedTensorType::get(output_shape, rewriter.getF32Type());
    auto mulOp1 = rewriter.create<top::MulOp>(loc2, type2, operands, attrs);
    auto out2 = mulOp1.getOutput();

    // create input[1] - input[0] * input[1]
    attrs.clear();
    operands.clear();
    rewriter.setInsertionPointAfterValue(out2);
    operands.emplace_back(input1);
    operands.emplace_back(out2);
    auto loc3 = NameLoc::get(rewriter.getStringAttr(name + "_sub1"));
    // auto type4 = RankedTensorType::get(output_shape, rewriter.getF32Type());
    auto addOp1 = rewriter.create<top::SubOp>(loc3, type2, operands, attrs);
    auto out3 = addOp1.getOutput();

    // create (input[0] * const_val)+ (input[1] - input[0] * input[1])
    attrs.clear();
    operands.clear();
    rewriter.setInsertionPointAfterValue(out3);
    operands.emplace_back(out3);
    operands.emplace_back(out1);
    auto loc4 = NameLoc::get(rewriter.getStringAttr(name));
    auto addOp2 = rewriter.create<top::AddOp>(loc4, type2, operands, attrs);
    rewriter.replaceAllUsesWith(ori_out, addOp2.getOutput());
    rewriter.eraseOp(op);

  } else {
    // out = input[0] * input[1] + (1 - input[0]) * const_val

    // create input[0] * input[1]
    operands.emplace_back(input0);
    operands.emplace_back(input1);
    auto loc1 = NameLoc::get(rewriter.getStringAttr(name + "_mul1"));
    auto type1 = RankedTensorType::get(output_shape, rewriter.getF32Type());
    auto mulOp1 = rewriter.create<top::MulOp>(loc1, type1, operands, attrs);
    auto out1 = mulOp1.getOutput();
    // out1.setLoc(op.getLoc());

    // create -const_val * input[0]
    operands.clear();
    attrs.emplace_back(rewriter.getNamedAttr(
        "const_val", rewriter.getF64FloatAttr(-const_val)));
    operands.emplace_back(input0);
    auto loc2 = NameLoc::get(rewriter.getStringAttr(name + "mulconst1"));
    auto type2 = RankedTensorType::get(input0_shape, rewriter.getF32Type());
    rewriter.setInsertionPointAfterValue(out1);
    auto mulconstOp1 =
        rewriter.create<top::MulConstOp>(loc2, type2, operands, attrs);
    auto out2 = mulconstOp1.getOutput();

    // create (-const_val * input[0]) + const_val
    operands.clear();
    attrs.clear();
    operands.emplace_back(out2);
    attrs.emplace_back(rewriter.getNamedAttr(
        "const_val", rewriter.getF64FloatAttr(const_val)));
    auto loc3 = NameLoc::get(rewriter.getStringAttr(name + "addconst1"));
    rewriter.setInsertionPointAfterValue(out2);
    auto addconstOp1 =
        rewriter.create<top::AddConstOp>(loc3, type2, operands, attrs);
    auto out3 = addconstOp1.getOutput();

    // create (input[0] * input[1]) + ((-const_val * input[0]) + const_val)
    operands.clear();
    attrs.clear();
    operands.emplace_back(out1);
    operands.emplace_back(out3);
    auto loc4 = NameLoc::get(rewriter.getStringAttr(name));
    rewriter.setInsertionPointAfterValue(out3);
    auto addOp1 = rewriter.create<top::AddOp>(loc4, type1, operands, attrs);
    rewriter.replaceAllUsesWith(ori_out, addOp1.getOutput());
    rewriter.eraseOp(op);
  }
  return success();
}

} // namespace cv18xx
} // namespace tpu_mlir
