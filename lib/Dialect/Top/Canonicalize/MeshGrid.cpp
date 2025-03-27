
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace tpu_mlir::top;

struct MeshGrid2Mul : public OpRewriterPatternEx<MeshGridOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  MeshGrid2Mul(mlir::MLIRContext *context)
      : OpRewriterPatternEx<MeshGridOp>(context, "MeshGrid2Mul") {}

  LogicalResult matchAndRewriteImpl(MeshGridOp op,
                                    PatternRewriter &rewriter) const override {

    rewriter.setInsertionPointAfter(op);

    auto out_shape = module::getShape(op.getOutputs()[0]);
    auto stype = module::getStorageType(op.getInputs()[0]);
    int64_t length = 1;
    for (int i = 0; i < out_shape.size(); ++i) {
      length *= out_shape[i];
    }
    std::vector<float> data(length, 1);
    int64_t input_num = op.getInputs().size();
    std::vector<NamedAttribute> attrs;
    for (int64_t i = 0; i < input_num; ++i) {
      int64_t idx = op.getIsReverse() ? (input_num - 1 - i) : i;
      auto input = op.getInputs()[idx];
      auto output = op.getOutputs()[idx];
      auto shape = module::getShape(input);
      std::vector<int64_t> r_shape(input_num, 1);
      r_shape[idx] = shape.size();
      auto type_r = RankedTensorType::get(r_shape, stype);
      std::vector<NamedAttribute> attrs_reshape;
      attrs_reshape.push_back(
          rewriter.getNamedAttr("shape", rewriter.getI64ArrayAttr(r_shape)));
      std::string out_name = module::getName(op.getOutputs()[idx]).data();
      std::string name_r = out_name + "_r_reshape_" + std::to_string(i);
      auto name_loc_r = NameLoc::get(rewriter.getStringAttr(name_r));
      auto reshape = rewriter.create<ReshapeOp>(
          name_loc_r, type_r, ValueRange{input}, attrs_reshape);
      auto name_loc = NameLoc::get(rewriter.getStringAttr(out_name));
      auto mul = rewriter.create<MulOp>(name_loc, op.getOutputs().getType(),
                                        ValueRange{reshape}, attrs);
      output.replaceAllUsesWith(mul);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

void MeshGridOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<MeshGrid2Mul>(context);
}
