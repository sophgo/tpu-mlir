//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::top;

struct ConvertEinsum : public OpRewritePattern<EinsumOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(EinsumOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2 || module::isWeight(op.getInputs()[0])) {
      llvm_unreachable("Not support now.");
      // return failure();
    }
    auto none = module::getNoneOp(op);
    auto mode = op.getMode().str();
    auto lhs = op.getInputs()[0];
    auto rhs = op.getInputs()[1];
    auto lshape = module::getShape(lhs);
    auto rshape = module::getShape(rhs);
    std::string lname = module::getName(lhs).str();
    std::string rname = module::getName(rhs).str();
    std::string name = module::getName(op.getOutput()).str();

    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    if (mode == "a,b->ab") {
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto newType = RankedTensorType::get({lshape[0], 1}, module::getElementType(lhs));
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_to2dim"));
      auto lrsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{lhs});
      operands.push_back(lrsOp);
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      newType = RankedTensorType::get({1, rshape[0]}, module::getElementType(rhs));
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_to2dim"));
      auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
      operands.push_back(rrsop);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      auto matmulOp = rewriter.create<MatMulOp>(op.getLoc(), op.getType(), operands, attrs);
      op.replaceAllUsesWith(matmulOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,cde->abe") {
      // lhs_reshape_rst = [lhs_shape[0] * lhs_shape[1], lhs_shape[2] * lhs_shape[3]]
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto newType = RankedTensorType::get({lshape[0] * lshape[1], lshape[2] * lshape[3]}, module::getElementType(lhs));
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_to2dim"));
      auto lrsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{lhs});
      operands.push_back(lrsOp);
      newType = RankedTensorType::get({rshape[0] * rshape[1], rshape[2]}, module::getElementType(rhs));
      if (module::isWeight(rhs)) {
        rhs.setType(newType);
        operands.push_back(rhs);
      } else {
        rewriter.setInsertionPointAfter(rhs.getDefiningOp());
        loc = NameLoc::get(rewriter.getStringAttr(rname + "_to2dim"));
        auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
        operands.push_back(rrsop);
      }
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      newType = RankedTensorType::get({lshape[0] * lshape[1], rshape[2]}, module::getElementType(op));
      loc = NameLoc::get(rewriter.getStringAttr(name + "_matmul"));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      auto orsOp = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(), ValueRange{matmulOp});
      op.replaceAllUsesWith(orsOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,bed->abce") {
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      // batch matmul does not support broadcast
      // temporary solution
      // [h, k, c] -> [1, h, k, c] -> [b, h, k, c]
      operands.push_back(lhs);
      RankedTensorType newType;
      if (auto wOp = dyn_cast<top::WeightOp>(rhs.getDefiningOp())) {
        auto storage_type = module::getStorageType(rhs);
        assert(storage_type.isF32() && "Todo, supoort more weight type");
        auto data = wOp.read_as_byte();
        uint8_t *dptr;
        newType = RankedTensorType::get({lshape[0], rshape[0], rshape[1], rshape[2]}, module::getElementType(rhs));
        std::vector<float_t> new_filter(newType.getNumElements(), 0);
        dptr = (uint8_t *)new_filter.data();
        for (int32_t i = 0; i < lshape[0]; i++) {
          auto offset = i * data->size();
          memcpy(dptr + offset, data->data(), data->size());
        }
        auto new_op = top::WeightOp::create(op, "folder", new_filter, newType);
        wOp.replaceAllUsesWith(new_op.getDefiningOp());
        operands.push_back(new_op);
        rewriter.eraseOp(wOp);
      } else {
        auto loc = NameLoc::get(rewriter.getStringAttr(rname + "_reshape"));
        newType = RankedTensorType::get({1, rshape[0], rshape[1], rshape[2]}, module::getElementType(rhs));
        auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
        newType = RankedTensorType::get({lshape[0], rshape[0], rshape[1], rshape[2]}, module::getElementType(rhs));
        loc = NameLoc::get(rewriter.getStringAttr(rname + "_tile"));
        attrs.push_back(rewriter.getNamedAttr("tile", rewriter.getI64ArrayAttr({lshape[0], 1, 1, 1})));
        auto tileOp = rewriter.create<TileOp>(loc, newType, ValueRange{rrsop}, attrs);
        attrs.clear();
        operands.push_back(tileOp);
      }
      operands.push_back(none);
      // [b, h, w, c] * [b, h, k, c]^T -> [b, h, w, k]
      attrs.push_back(rewriter.getNamedAttr("right_transpose", rewriter.getBoolAttr(true)));
      rewriter.setInsertionPoint(op);
      auto matmulOp = rewriter.create<MatMulOp>(op.getLoc(), op.getType(), operands, attrs);
      op.replaceAllUsesWith(matmulOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,ced->abce") {
      // dumb implementation
      // [b, h, w, c] -> [b, w, h, c]
      // trans_shape = [lhs_shape[0], lhs_shape[2], lhs_shape[1], lhs_shape[3]]
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_trans"));
      auto newType = RankedTensorType::get({lshape[0], lshape[2], lshape[1], lshape[3]}, module::getElementType(lhs));
      attrs.push_back(rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1, 3})));
      auto tranOp = rewriter.create<PermuteOp>(loc, newType, ValueRange{lhs}, attrs);
      attrs.clear();
      operands.push_back(tranOp);
      // [w, k, c] -> [1, w, k, c] -> [b, w, k, c]
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      if (auto wOp = dyn_cast<top::WeightOp>(rhs.getDefiningOp())) {
        auto storage_type = module::getStorageType(rhs);
        assert(storage_type.isF32() && "Todo, supoort more weight type");
        auto data = wOp.read_as_byte();
        uint8_t *dptr;
        newType = RankedTensorType::get({lshape[0], rshape[0], rshape[1], rshape[2]}, module::getElementType(rhs));
        std::vector<float_t> new_filter(newType.getNumElements(), 0);
        dptr = (uint8_t *)new_filter.data();
        for (int32_t i = 0; i < lshape[0]; i++) {
          auto offset = i * data->size();
          memcpy(dptr + offset, data->data(), data->size());
        }
        auto new_op = top::WeightOp::create(op, "folder", new_filter, newType);
        wOp.replaceAllUsesWith(new_op.getDefiningOp());
        operands.push_back(new_op);
        rewriter.eraseOp(wOp);
      } else {
        loc = NameLoc::get(rewriter.getStringAttr(rname + "_reshape"));
        newType = RankedTensorType::get({1, rshape[0], rshape[1], rshape[2]}, module::getElementType(rhs));
        auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
        loc = NameLoc::get(rewriter.getStringAttr(rname + "_tile"));
        attrs.push_back(rewriter.getNamedAttr("tile", rewriter.getI64ArrayAttr({lshape[0], 1, 1, 1})));
        newType = RankedTensorType::get({lshape[0], rshape[0], rshape[1], rshape[2]}, module::getElementType(rhs));
        auto tileOp = rewriter.create<TileOp>(loc, newType, ValueRange{rrsop}, attrs);
        attrs.clear();
        operands.push_back(tileOp);
      }
      operands.push_back(none);
      // [b, w, h, c] * [b, w, k, c]^T -> [b, w, h, k]
      newType = RankedTensorType::get({lshape[0], lshape[2], lshape[1], rshape[1]}, module::getElementType(op));
      attrs.push_back(rewriter.getNamedAttr("right_transpose", rewriter.getBoolAttr(true)));
      rewriter.setInsertionPoint(op);
      loc = NameLoc::get(rewriter.getStringAttr(name + "_matmul"));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      attrs.clear();
      // [b, w, h, k] -> [b, h, w, k]
      attrs.push_back(rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1, 3})));
      auto tranBackOp = rewriter.create<PermuteOp>(op.getLoc(), op.getType(), ValueRange{matmulOp}, attrs);
      op.replaceAllUsesWith(tranBackOp.getOperation());
      rewriter.eraseOp(op);
    } else {
      llvm_unreachable("Einsum not support this mode now");
    }
    return success();
  }
};


void EinsumOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  // llvm_unreachable("getCanonicalizationPatterns not Implemented");
  results.insert<ConvertEinsum>(context);
}
