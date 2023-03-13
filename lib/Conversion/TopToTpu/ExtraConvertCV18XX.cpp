//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/ExtraConvertCV18XX.h"

namespace tpu_mlir {

namespace cv18xx {

LogicalResult ConvertMaskedFillOp::matchAndRewrite(top::MaskedFillOp op,
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
  //cv18xx only support one operand broadcast now.
  assert((input0_shape == output_shape || input1_shape == output_shape));
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  rewriter.setInsertionPointAfterValue(ori_out);
  if (inverse) {
    //out = input[0] * const_val + (1 - input[0]) * input[1]

    //create input[0] * const_val
    operands.emplace_back(input0);
    attrs.emplace_back(rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(const_val)));
    auto loc1 = NameLoc::get(rewriter.getStringAttr(name + "_mulconst1"));
    auto type1 = RankedTensorType::get(input0_shape, rewriter.getF32Type());
    auto mulconstOp1 = rewriter.create<top::MulConstOp>(loc1, type1, operands, attrs);
    auto out1 = mulconstOp1.getOutput();

    //create (input[0]) * input[1]
    operands.clear();
    attrs.clear();
    rewriter.setInsertionPointAfterValue(out1);
    operands.emplace_back(input0);
    operands.emplace_back(input1);
    auto loc2 = NameLoc::get(rewriter.getStringAttr(name + "_mul1"));
    auto type2 = RankedTensorType::get(output_shape, rewriter.getF32Type());
    auto mulOp1 = rewriter.create<top::MulOp>(loc2, type2, operands, attrs);
    auto out2 = mulOp1.getOutput();

    //create input[1] - input[0] * input[1]
    attrs.clear();
    operands.clear();
    rewriter.setInsertionPointAfterValue(out2);
    operands.emplace_back(input1);
    operands.emplace_back(out2);
    attrs.emplace_back(rewriter.getNamedAttr("coeff", rewriter.getF64ArrayAttr({1.0, -1.0})));
    auto loc3 = NameLoc::get(rewriter.getStringAttr(name + "_add1"));
    //auto type4 = RankedTensorType::get(output_shape, rewriter.getF32Type());
    auto addOp1 = rewriter.create<top::AddOp>(loc3, type2, operands, attrs);
    auto out3 = addOp1.getOutput();

    //create (input[0] * const_val)+ (input[1] - input[0] * input[1])
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
    //out = input[0] * input[1] + (1 - input[0]) * const_val

    //create input[0] * input[1]
    operands.emplace_back(input0);
    operands.emplace_back(input1);
    auto loc1 = NameLoc::get(rewriter.getStringAttr(name + "_mul1"));
    auto type1 = RankedTensorType::get(output_shape, rewriter.getF32Type());
    auto mulOp1 = rewriter.create<top::MulOp>(loc1, type1, operands, attrs);
    auto out1 = mulOp1.getOutput();
    //out1.setLoc(op.getLoc());

    //create -const_val * input[0]
    operands.clear();
    attrs.emplace_back(rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(-const_val)));
    operands.emplace_back(input0);
    auto loc2 = NameLoc::get(rewriter.getStringAttr(name + "mulconst1"));
    auto type2 = RankedTensorType::get(input0_shape, rewriter.getF32Type());
    rewriter.setInsertionPointAfterValue(out1);
    auto mulconstOp1 = rewriter.create<top::MulConstOp>(loc2, type2, operands, attrs);
    auto out2 = mulconstOp1.getOutput();

    //create (-const_val * input[0]) + const_val
    operands.clear();
    attrs.clear();
    operands.emplace_back(out2);
    attrs.emplace_back(rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(const_val)));
    auto loc3 = NameLoc::get(rewriter.getStringAttr(name + "addconst1"));
    rewriter.setInsertionPointAfterValue(out2);
    auto addconstOp1 = rewriter.create<top::AddConstOp>(loc3, type2, operands, attrs);
    auto out3 = addconstOp1.getOutput();

    //create (input[0] * input[1]) + ((-const_val * input[0]) + const_val)
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

LogicalResult ConvertWhereOp::matchAndRewrite(top::WhereOp op,
                                PatternRewriter &rewriter) const {
  //out = input[0] * input[1] + (1 - input[0]) * input[2]
  Value input0 = op.getOperand(0);
  Value input1 = op.getOperand(1);
  Value input2 = op.getOperand(2);
  Value ori_out = op.getOutput();
  std::string name = module::getName(ori_out).str();
  std::vector<int64_t> output_shape = module::getShape(ori_out);
  std::vector<int64_t> input0_shape = module::getShape(input0);
  std::vector<int64_t> input1_shape = module::getShape(input1);
  std::vector<int64_t> input2_shape = module::getShape(input2);
  int64_t num_input0 = module::getNumElements(input0);
  int64_t num_input1 = module::getNumElements(input1);
  int64_t num_input2 = module::getNumElements(input2);
  //cv18xx only support one operand broadcast now.
  assert((input0_shape == output_shape || input1_shape == output_shape || input2_shape == output_shape));
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  rewriter.setInsertionPointAfterValue(ori_out);

  //create input[0] * input[1]
  operands.emplace_back(input0);
  operands.emplace_back(input1);
  std::vector<int64_t> out1_shape = (num_input0 > num_input1) ? input0_shape : input1_shape;
  auto loc1 = NameLoc::get(rewriter.getStringAttr(name + "_mul1"));
  auto type1 = RankedTensorType::get(out1_shape, rewriter.getF32Type());
  auto mulOp1 = rewriter.create<top::MulOp>(loc1, type1, operands, attrs);
  auto out1 = mulOp1.getOutput();


  //create input[0] * input[2]
  operands.clear();
  attrs.clear();
  operands.emplace_back(input0);
  operands.emplace_back(input2);
  rewriter.setInsertionPointAfterValue(out1);
  std::vector<int64_t> out2_shape = (num_input0 > num_input2) ? input0_shape : input2_shape;
  auto loc2 = NameLoc::get(rewriter.getStringAttr(name + "mul2"));
  auto type2 = RankedTensorType::get(out2_shape, rewriter.getF32Type());
  auto mulOp2 = rewriter.create<top::MulOp>(loc2, type2, operands, attrs);
  auto out2 = mulOp2.getOutput();

  //create input[2] - input[0] * input[2]
  attrs.clear();
  operands.clear();
  operands.emplace_back(input2);
  operands.emplace_back(out2);
  attrs.emplace_back(rewriter.getNamedAttr("coeff", rewriter.getF64ArrayAttr({1.0, -1.0})));
  rewriter.setInsertionPointAfterValue(out2);
  auto loc3 = NameLoc::get(rewriter.getStringAttr(name + "_add1"));
  auto addOp1 = rewriter.create<top::AddOp>(loc3, type2, operands, attrs);
  auto out3 = addOp1.getOutput();

  //create (input[0] * input[1]) + (input[2] - input[0] * input[2])
  attrs.clear();
  operands.clear();
  operands.emplace_back(out1);
  operands.emplace_back(out3);
  rewriter.setInsertionPointAfterValue(out3);
  auto loc4 = NameLoc::get(rewriter.getStringAttr(name));
  auto type4 = RankedTensorType::get(output_shape, rewriter.getF32Type());
  auto add2Op = rewriter.create<top::AddOp>(loc4, type4, operands, attrs);
  auto out4 = add2Op.getOutput();
  rewriter.replaceAllUsesWith(ori_out, out4);
  rewriter.eraseOp(op);

  return success();
}

LogicalResult ConvertGatherOp::matchAndRewrite(top::GatherOp op,
                                PatternRewriter &rewriter) const {
  //return success();
  //for transform decode's index op
  Value input = op.getInput();
  Value indices = op.getIndices();
  Value ori_out = op.getOutput();
  std::string name = module::getName(ori_out).str();
  uint64_t axis = op.getAxis();
  std::vector<int64_t> input_shape = module::getShape(input);
  std::vector<int64_t> indices_shape = module::getShape(indices);
  bool need_convert = (axis == 1 && indices_shape.size() == 0 && input_shape.size() == 3 && input_shape[0] == 1
                          && !(isa<top::WeightOp>(input.getDefiningOp())));
  if (need_convert) {
    //conver to reshapeOp + new GatherOp
    rewriter.setInsertionPointAfterValue(ori_out);
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    operands.emplace_back(input);
    auto loc1 = NameLoc::get(rewriter.getStringAttr(name + "_reshape"));
    auto type1 = RankedTensorType::get({input_shape[1], input_shape[2]}, rewriter.getF32Type());
    auto reshapeOp = rewriter.create<top::ReshapeOp>(loc1, type1, operands, attrs);
    auto out1 = reshapeOp.getOutput();
    operands.clear();
    operands.emplace_back(out1);
    operands.emplace_back(indices);
    attrs.emplace_back(rewriter.getNamedAttr("axis", rewriter.getI64IntegerAttr(0)));
    auto loc2 = NameLoc::get(rewriter.getStringAttr(name));
    auto type2 = ori_out.getType().cast<RankedTensorType>();
    auto newOp = rewriter.create<top::GatherOp>(loc2, type2, operands, attrs);
    auto newOut = newOp.getOutput();
    rewriter.replaceAllUsesWith(ori_out, newOut);
    rewriter.eraseOp(op);
  } else {
    return failure();
  }
  return success();
}

void populateDoExtraConversionPatterns(RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
      ConvertMaskedFillOp,
      ConvertWhereOp,
      ConvertGatherOp
  >(patterns->getContext());
  // clang-format on
}
}
}
