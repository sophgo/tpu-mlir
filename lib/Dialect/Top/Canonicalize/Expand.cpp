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
#include "tpu_mlir/Support/Patterns.h"

using namespace tpu_mlir::top;

struct ConvertExpand : public OpRewriterPatternEx<ExpandOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  ConvertExpand(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ExpandOp>(context, "ConvertExpand") {}

  LogicalResult matchAndRewriteImpl(ExpandOp op,
                                    PatternRewriter &rewriter) const override {

    if (!op.getShapeT() || !module::isActive(op.getShapeT())) {
      auto output_shape = module::getShape(op.getOutput());
      auto input_shape = module::getShape(op.getInput());
      auto output_dims = output_shape.size();
      assert(output_dims >= input_shape.size());
      Value new_op = op.getInput();
      std::string name = module::getName(op.getOutput()).str();
      auto elt_type = module::getElementType(op.getOutput());

      rewriter.setInsertionPoint(op);
      // remove leading 1
      auto len_diff = output_dims - input_shape.size();
      std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
      if (len_diff != 0) {
        std::vector<int64_t> input_shape_ex(len_diff, 1);
        for (auto s : input_shape) {
          input_shape_ex.push_back(s);
        }
        auto newType = RankedTensorType::get(input_shape_ex, elt_type);
        auto loc = NameLoc::get(rewriter.getStringAttr(name + "_r_reshape"));
        new_op = rewriter.create<ReshapeOp>(loc, newType, ValueRange{new_op});
        out_shape = input_shape_ex;
      }
      int32_t count = 0;
      for (uint32_t i = 0; i < output_dims; i++) {
        if (out_shape[i] != output_shape[i]) {
          ++count;
        }
      }
      if (count == 0) {
        op.replaceAllUsesWith(new_op.getDefiningOp());
        rewriter.eraseOp(op);
        return success();
      }
      std::vector<int64_t> weight_tile(output_dims, 1);
      for (uint32_t i = 0; i < output_dims; i++) {
        if (out_shape[i] == output_shape[i])
          continue;
        int64_t tile = output_shape[i] / out_shape[i];
        weight_tile[i] = tile;
      }
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("tile", rewriter.getI64ArrayAttr(weight_tile)));
      auto newType = RankedTensorType::get(output_shape, elt_type);
      auto loc = NameLoc::get(rewriter.getStringAttr(name));
      new_op =
          rewriter.create<top::TileOp>(loc, newType, ValueRange{new_op}, attrs);
      op.replaceAllUsesWith(new_op.getDefiningOp());
      rewriter.eraseOp(op);
      return success();
    } else {
      std::string name = module::getName(op.getOutput()).str();
      auto consF_loc =
          NameLoc::get(rewriter.getStringAttr(name + "_constFill"));
      auto storage_type = module::getStorageType(op.getOutput());
      if (storage_type.isIntOrIndex()) {
        std::vector<NamedAttribute> attrs_consF;
        attrs_consF.push_back(
            rewriter.getNamedAttr("value", rewriter.getF64FloatAttr(1.0)));
        auto p_type =
            UnrankedTensorType::get(module::getElementType(op.getOutput()));
        auto consF_op = rewriter.create<ConstantFillOp>(
            consF_loc, p_type, ValueRange(op.getShapeT()), attrs_consF);
        consF_op.shape_inference();
        auto right = consF_op.getResult();

        auto mul_loc = NameLoc::get(rewriter.getStringAttr(name));
        std::vector<Value> operands;
        operands.push_back(op.getInput());
        operands.push_back(right);
        std::vector<NamedAttribute> attrs;
        attrs.push_back(
            rewriter.getNamedAttr("mode", rewriter.getStringAttr("Mul")));
        attrs.push_back(
            rewriter.getNamedAttr("shift", rewriter.getSI32IntegerAttr(0)));
        auto mul = rewriter.create<BinaryShiftOp>(
            mul_loc, op.getOutput().getType(), operands, attrs);
        op.getOutput().replaceAllUsesWith(mul);
        rewriter.eraseOp(op);
        return success();
      }
      std::vector<NamedAttribute> attrs_consF;
      attrs_consF.push_back(
          rewriter.getNamedAttr("value", rewriter.getF64FloatAttr(1.0)));
      auto p_type =
          UnrankedTensorType::get(module::getElementType(op.getShapeT()));
      auto consF_op = rewriter.create<ConstantFillOp>(
          consF_loc, p_type, ValueRange(op.getShapeT()), attrs_consF);
      consF_op.shape_inference();
      auto right = consF_op.getResult();

      auto mul_loc = NameLoc::get(rewriter.getStringAttr(name));
      std::vector<Value> operands;
      operands.push_back(op.getInput());
      operands.push_back(right);
      std::vector<NamedAttribute> attrs;
      auto mul = rewriter.create<MulOp>(mul_loc, op.getOutput().getType(),
                                        operands, attrs);
      op.getOutput().replaceAllUsesWith(mul);
      rewriter.eraseOp(op);
      return success();
    }
  }
};

void ExpandOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<ConvertExpand>(context);
}
