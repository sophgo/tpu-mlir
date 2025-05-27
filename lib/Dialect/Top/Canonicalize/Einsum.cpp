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

struct ConvertEinsum : public OpRewriterPatternEx<EinsumOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  ConvertEinsum(mlir::MLIRContext *context)
      : OpRewriterPatternEx<EinsumOp>(context, "ConvertEinsum") {}

  LogicalResult matchAndRewriteImpl(EinsumOp op,
                                    PatternRewriter &rewriter) const override {
    // if (op.getInputs().size() != 2 || module::isWeight(op.getInputs()[0])) {
    //   llvm_unreachable("Not support now.");
    //   // return failure();
    // }
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
      auto newType =
          RankedTensorType::get({lshape[0], 1}, module::getElementType(lhs));
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_r_to2dim"));
      auto lrsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{lhs});
      operands.push_back(lrsOp);
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      newType =
          RankedTensorType::get({1, rshape[0]}, module::getElementType(rhs));
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_to2dim"));
      auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
      operands.push_back(rrsop);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      auto matmulOp =
          rewriter.create<MatMulOp>(op.getLoc(), op.getType(), operands, attrs);
      op.replaceAllUsesWith(matmulOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "ab,ab->a") {
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto newType = RankedTensorType::get({lshape[0], 1, lshape[1]},
                                           module::getElementType(lhs));
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_r_to3dim"));
      auto lrsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{lhs});
      operands.push_back(lrsOp);
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      newType = RankedTensorType::get({rshape[0], rshape[1], 1},
                                      module::getElementType(rhs));
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_to3dim"));
      auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
      operands.push_back(rrsop);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      newType =
          RankedTensorType::get({lshape[0], 1, 1}, module::getElementType(op));
      loc = NameLoc::get(rewriter.getStringAttr(name + "_r_matmul"));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      auto reshapeOp = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(),
                                                  ValueRange{matmulOp});
      op.replaceAllUsesWith(reshapeOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "ab,acb->ac") {
      // matmul([a,1,1,b],[a,c,b,1])->[a,c,1,1]
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto newType = RankedTensorType::get({lshape[0], 1, 1, lshape[1]},
                                           module::getElementType(lhs));
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_r_to4dim"));
      auto lrsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{lhs});
      operands.push_back(lrsOp);
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      newType = RankedTensorType::get({rshape[0], rshape[1], rshape[2], 1},
                                      module::getElementType(rhs));
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_to4dim"));
      auto rrsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
      operands.push_back(rrsOp);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);

      loc = NameLoc::get(rewriter.getStringAttr(name + "_r_matmul"));
      newType = RankedTensorType::get({lshape[0], rshape[1], 1, 1},
                                      module::getElementType(op));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      auto reshapeOp = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(),
                                                  ValueRange{matmulOp});
      op.replaceAllUsesWith(reshapeOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "ab,abc->ac") {
      // matmul([a,1,b],[a,b,c])->[a,1,c]
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto newType = RankedTensorType::get({lshape[0], 1, lshape[1]},
                                           module::getElementType(lhs));
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_r_to3dim"));
      auto lrsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{lhs});
      operands.push_back(lrsOp);
      operands.push_back(rhs);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);

      loc = NameLoc::get(rewriter.getStringAttr(name + "_r_matmul"));
      newType = RankedTensorType::get({lshape[0], 1, rshape[2]},
                                      module::getElementType(op));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      auto reshapeOp = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(),
                                                  ValueRange{matmulOp});
      op.replaceAllUsesWith(reshapeOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "ab,cdb->acd") {
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto newType = RankedTensorType::get({1, lshape[0], lshape[1]},
                                           module::getElementType(lhs));
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_r_to3dim"));
      auto lrsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{lhs});
      operands.push_back(lrsOp);
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      newType = RankedTensorType::get({1, rshape[0] * rshape[1], rshape[2]},
                                      module::getElementType(rhs));
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_to3dim"));
      auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_trans"));
      newType = RankedTensorType::get({1, rshape[2], rshape[0] * rshape[1]},
                                      module::getElementType(rrsop));
      attrs.push_back(
          rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1})));
      auto tranOp =
          rewriter.create<PermuteOp>(loc, newType, ValueRange{rrsop}, attrs);
      operands.push_back(tranOp);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);

      loc = NameLoc::get(rewriter.getStringAttr(name + "_r_matmul"));
      newType = RankedTensorType::get({1, lshape[0], rshape[0] * rshape[1]},
                                      module::getElementType(op));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      auto reshapeOp = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(),
                                                  ValueRange{matmulOp});
      op.replaceAllUsesWith(reshapeOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abc,db->adc") {
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      auto newType = RankedTensorType::get({1, rshape[0], rshape[1]},
                                           module::getElementType(rhs));
      auto loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_to3dim"));
      auto rrsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
      operands.push_back(rrsOp);
      operands.push_back(lhs);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      auto matmulOp =
          rewriter.create<MatMulOp>(op.getLoc(), op.getType(), operands, attrs);
      op.replaceAllUsesWith(matmulOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,abce->acde") {
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_r_permute"));
      auto newType =
          RankedTensorType::get({lshape[0], lshape[2], lshape[3], lshape[1]},
                                module::getElementType(lhs));
      attrs.push_back(rewriter.getNamedAttr(
          "order", rewriter.getI64ArrayAttr({0, 2, 3, 1})));
      auto ltranOp =
          rewriter.create<PermuteOp>(loc, newType, ValueRange{lhs}, attrs);
      operands.push_back(ltranOp);
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      attrs.clear();
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_permute"));
      newType =
          RankedTensorType::get({rshape[0], rshape[2], lshape[1], rshape[3]},
                                module::getElementType(rhs));
      attrs.push_back(rewriter.getNamedAttr(
          "order", rewriter.getI64ArrayAttr({0, 2, 1, 3})));
      auto rtranOp =
          rewriter.create<PermuteOp>(loc, newType, ValueRange{rhs}, attrs);
      operands.push_back(rtranOp);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      attrs.clear();
      auto matmulOp =
          rewriter.create<MatMulOp>(op.getLoc(), op.getType(), operands, attrs);
      op.replaceAllUsesWith(matmulOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,acd->abc") {
      // TODO : check whether tile can be optimized
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto newType = RankedTensorType::get(
          {lshape[0] * lshape[1] * lshape[2], 1, lshape[3]},
          module::getElementType(lhs));
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_r_to3dim"));
      auto lrsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{lhs});
      operands.push_back(lrsOp);
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      newType = RankedTensorType::get({rshape[0], 1, rshape[1], rshape[2], 1},
                                      module::getElementType(rhs));
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_to5dim"));
      auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
      newType =
          RankedTensorType::get({rshape[0], lshape[1], rshape[1], rshape[2], 1},
                                module::getElementType(rhs));
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_tile"));
      attrs.push_back(rewriter.getNamedAttr(
          "tile", rewriter.getI64ArrayAttr({1, lshape[1], 1, 1, 1})));
      auto tileOp =
          rewriter.create<TileOp>(loc, newType, ValueRange{rrsop}, attrs);
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_to3dim"));
      newType = RankedTensorType::get(
          {rshape[0] * lshape[1] * rshape[1], rshape[2], 1},
          module::getElementType(rhs));
      auto reshapeOp =
          rewriter.create<ReshapeOp>(loc, newType, ValueRange{tileOp});
      operands.push_back(reshapeOp);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      attrs.clear();
      loc = NameLoc::get(rewriter.getStringAttr(name + "_r_matmul"));
      newType = RankedTensorType::get({lshape[0] * lshape[1] * lshape[2], 1, 1},
                                      module::getElementType(op));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      reshapeOp = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(),
                                             ValueRange{matmulOp});
      op.replaceAllUsesWith(reshapeOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,cde->abe") {
      // lhs_reshape_rst = [lhs_shape[0] * lhs_shape[1], lhs_shape[2] *
      // lhs_shape[3]]
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto newType =
          RankedTensorType::get({lshape[0] * lshape[1], lshape[2] * lshape[3]},
                                module::getElementType(lhs));
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_r_to2dim"));
      auto lrsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{lhs});
      operands.push_back(lrsOp);
      newType = RankedTensorType::get({rshape[0] * rshape[1], rshape[2]},
                                      module::getElementType(rhs));
      if (module::isWeight(rhs)) {
        rhs.setType(newType);
        operands.push_back(rhs);
      } else {
        rewriter.setInsertionPointAfter(rhs.getDefiningOp());
        loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_to2dim"));
        auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
        operands.push_back(rrsop);
      }
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      newType = RankedTensorType::get({lshape[0] * lshape[1], rshape[2]},
                                      module::getElementType(op));
      loc = NameLoc::get(rewriter.getStringAttr(name + "_r_matmul"));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      auto orsOp = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(),
                                              ValueRange{matmulOp});
      op.replaceAllUsesWith(orsOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,bed->abce") {
      // TODO ： remove right_transpose to meet sg2380 gdma update
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
        newType =
            RankedTensorType::get({lshape[0], rshape[0], rshape[1], rshape[2]},
                                  module::getElementType(rhs));
        std::vector<float_t> new_filter(newType.getNumElements(), 0);
        dptr = (uint8_t *)new_filter.data();
        for (int32_t i = 0; i < lshape[0]; i++) {
          auto offset = i * data->size();
          memcpy(dptr + offset, data->data(), data->size());
        }
        auto new_op = top::WeightOp::create(op, "folder", new_filter, newType);
        wOp.replaceAllUsesWith(new_op.getDefiningOp());
        auto loc =
            NameLoc::get(rewriter.getStringAttr(rname + "weight_reorder"));
        attrs.clear();
        attrs.push_back(rewriter.getNamedAttr(
            "order", rewriter.getI64ArrayAttr({0, 1, 3, 2})));

        newType =
            RankedTensorType::get({lshape[0], rshape[0], rshape[2], rshape[1]},
                                  module::getElementType(rhs));
        auto tranOp =
            rewriter.create<PermuteOp>(loc, newType, ValueRange{new_op}, attrs);

        operands.push_back(tranOp);
        rewriter.eraseOp(wOp);
      } else {
        auto loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_reshape"));
        newType = RankedTensorType::get({1, rshape[0], rshape[1], rshape[2]},
                                        module::getElementType(rhs));
        auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
        newType =
            RankedTensorType::get({lshape[0], rshape[0], rshape[1], rshape[2]},
                                  module::getElementType(rhs));
        loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_tile"));
        attrs.push_back(rewriter.getNamedAttr(
            "tile", rewriter.getI64ArrayAttr({lshape[0], 1, 1, 1})));
        auto tileOp =
            rewriter.create<TileOp>(loc, newType, ValueRange{rrsop}, attrs);
        loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_permute"));
        attrs.clear();
        attrs.push_back(rewriter.getNamedAttr(
            "order", rewriter.getI64ArrayAttr({0, 1, 3, 2})));

        newType =
            RankedTensorType::get({lshape[0], rshape[0], rshape[2], rshape[1]},
                                  module::getElementType(rhs));
        auto tranOp =
            rewriter.create<PermuteOp>(loc, newType, ValueRange{tileOp}, attrs);
        attrs.clear();
        operands.push_back(tranOp);
      }
      operands.push_back(none);
      attrs.clear();
      rewriter.setInsertionPoint(op);
      auto matmulOp =
          rewriter.create<MatMulOp>(op.getLoc(), op.getType(), operands, attrs);
      op.replaceAllUsesWith(matmulOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,ced->abce") {
      // dumb implementation
      // [b, h, w, c] -> [b, w, h, c]
      // trans_shape = [lhs_shape[0], lhs_shape[2], lhs_shape[1], lhs_shape[3]]
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_r_trans"));
      auto newType =
          RankedTensorType::get({lshape[0], lshape[2], lshape[1], lshape[3]},
                                module::getElementType(lhs));
      attrs.push_back(rewriter.getNamedAttr(
          "order", rewriter.getI64ArrayAttr({0, 2, 1, 3})));
      auto tranOp =
          rewriter.create<PermuteOp>(loc, newType, ValueRange{lhs}, attrs);
      attrs.clear();
      operands.push_back(tranOp);
      // [w, k, c] -> [1, w, k, c] -> [b, w, k, c]
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      if (auto wOp = dyn_cast<top::WeightOp>(rhs.getDefiningOp())) {
        auto storage_type = module::getStorageType(rhs);
        assert(storage_type.isF32() && "Todo, supoort more weight type");
        auto data = wOp.read_as_byte();
        uint8_t *dptr;
        newType =
            RankedTensorType::get({lshape[0], rshape[0], rshape[1], rshape[2]},
                                  module::getElementType(rhs));
        std::vector<float_t> new_filter(newType.getNumElements(), 0);
        dptr = (uint8_t *)new_filter.data();
        for (int32_t i = 0; i < lshape[0]; i++) {
          auto offset = i * data->size();
          memcpy(dptr + offset, data->data(), data->size());
        }
        auto new_op = top::WeightOp::create(op, "folder", new_filter, newType);
        wOp.replaceAllUsesWith(new_op.getDefiningOp());
        auto loc =
            NameLoc::get(rewriter.getStringAttr(rname + "weight_reorder"));
        attrs.clear();
        attrs.push_back(rewriter.getNamedAttr(
            "order", rewriter.getI64ArrayAttr({0, 1, 3, 2})));

        newType =
            RankedTensorType::get({lshape[0], rshape[0], rshape[2], rshape[1]},
                                  module::getElementType(rhs));
        auto tranOp =
            rewriter.create<PermuteOp>(loc, newType, ValueRange{new_op}, attrs);

        operands.push_back(tranOp);
        rewriter.eraseOp(wOp);
      } else {
        loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_reshape"));
        newType = RankedTensorType::get({1, rshape[0], rshape[1], rshape[2]},
                                        module::getElementType(rhs));
        auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
        loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_tile"));
        attrs.push_back(rewriter.getNamedAttr(
            "tile", rewriter.getI64ArrayAttr({lshape[0], 1, 1, 1})));
        newType =
            RankedTensorType::get({lshape[0], rshape[0], rshape[1], rshape[2]},
                                  module::getElementType(rhs));
        auto tileOp =
            rewriter.create<TileOp>(loc, newType, ValueRange{rrsop}, attrs);
        attrs.clear();
        loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_permute"));
        attrs.push_back(rewriter.getNamedAttr(
            "order", rewriter.getI64ArrayAttr({0, 1, 3, 2})));

        newType =
            RankedTensorType::get({lshape[0], rshape[0], rshape[2], rshape[1]},
                                  module::getElementType(rhs));
        auto tranOp =
            rewriter.create<PermuteOp>(loc, newType, ValueRange{tileOp}, attrs);
        attrs.clear();
        operands.push_back(tranOp);
      }
      operands.push_back(none);
      // [b, w, h, c] * [b, w, k, c]^T -> [b, w, h, k]
      newType =
          RankedTensorType::get({lshape[0], lshape[2], lshape[1], rshape[1]},
                                module::getElementType(op));
      rewriter.setInsertionPoint(op);
      attrs.clear();
      loc = NameLoc::get(rewriter.getStringAttr(name + "_r_matmul"));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      attrs.clear();
      // [b, w, h, k] -> [b, h, w, k]
      attrs.push_back(rewriter.getNamedAttr(
          "order", rewriter.getI64ArrayAttr({0, 2, 1, 3})));
      auto tranBackOp = rewriter.create<PermuteOp>(op.getLoc(), op.getType(),
                                                   ValueRange{matmulOp}, attrs);
      op.replaceAllUsesWith(tranBackOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,abed->abce" || mode == "abcd,abde->abce") {
      // lhs(abcd) * rhs(abed)^T -> abce
      // lhs(abcd) * rhs(abde) -> abce

      auto newType =
          RankedTensorType::get({lshape[0], lshape[1], lshape[2], rshape[2]},
                                module::getElementType(op));
      rewriter.setInsertionPointAfter(op);
      operands.push_back(lhs);
      if (mode == "abcd,abde->abce") {
        newType =
            RankedTensorType::get({lshape[0], lshape[1], lshape[2], rshape[3]},
                                  module::getElementType(op));
        operands.push_back(rhs);
      }

      if (mode == "abcd,abed->abce") {
        // rhs(abed)^T
        attrs.clear();
        auto loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_permute"));
        attrs.push_back(rewriter.getNamedAttr(
            "order", rewriter.getI64ArrayAttr({0, 1, 3, 2})));

        newType =
            RankedTensorType::get({rshape[0], rshape[1], rshape[3], rshape[2]},
                                  module::getElementType(rhs));
        auto transOp =
            rewriter.create<PermuteOp>(loc, newType, ValueRange{rhs}, attrs);
        operands.push_back(transOp);
        newType =
            RankedTensorType::get({lshape[0], lshape[1], lshape[2], rshape[2]},
                                  module::getElementType(rhs));
      }
      operands.push_back(none);
      auto loc = NameLoc::get(rewriter.getStringAttr(name));

      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      op.replaceAllUsesWith(matmulOp.getOperation());
      attrs.clear();
      rewriter.eraseOp(op);

    } else if (mode == "abc,adc->abd") {

      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      auto type_permute = RankedTensorType::get(
          {rshape[0], rshape[2], rshape[1]}, module::getElementType(rhs));
      auto loc_permute =
          NameLoc::get(rewriter.getStringAttr(rname + "_r_permute"));
      attrs.push_back(
          rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1})));
      auto permuteOp = rewriter.create<PermuteOp>(loc_permute, type_permute,
                                                  ValueRange{rhs}, attrs);
      attrs.clear();

      rewriter.setInsertionPointAfter(op);
      auto loc = NameLoc::get(rewriter.getStringAttr(name));

      auto newType = RankedTensorType::get({lshape[0], lshape[1], rshape[1]},
                                           module::getElementType(op));
      operands.push_back(lhs);
      operands.push_back(permuteOp);
      operands.push_back(none);
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands);
      op.replaceAllUsesWith(matmulOp.getOperation());
      rewriter.eraseOp(op);

    } else if (mode == "abc,adc->adb") {
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_r_trans"));
      auto newType = RankedTensorType::get({lshape[0], lshape[2], lshape[1]},
                                           module::getElementType(lhs));
      attrs.push_back(
          rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1})));
      auto tranOp =
          rewriter.create<PermuteOp>(loc, newType, ValueRange{lhs}, attrs);
      operands.push_back(rhs);
      operands.push_back(tranOp);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      auto matmulOp =
          rewriter.create<MatMulOp>(op.getLoc(), op.getType(), operands, attrs);
      op.replaceAllUsesWith(matmulOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abc,abd->acd") {
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_r_trans"));
      auto newType = RankedTensorType::get({lshape[0], lshape[2], lshape[1]},
                                           module::getElementType(lhs));
      attrs.push_back(
          rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1})));
      auto tranOp =
          rewriter.create<PermuteOp>(loc, newType, ValueRange{lhs}, attrs);
      operands.push_back(tranOp);
      operands.push_back(rhs);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      auto matmulOp =
          rewriter.create<MatMulOp>(op.getLoc(), op.getType(), operands, attrs);
      op.replaceAllUsesWith(matmulOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,aecd->aeb") {
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto lreshape_loc =
          NameLoc::get(rewriter.getStringAttr(lname + "_r_reshape"));
      auto lreshape_type =
          RankedTensorType::get({lshape[0], lshape[1], lshape[2] * lshape[3]},
                                module::getElementType(lhs));
      auto lreshape_op = rewriter.create<top::ReshapeOp>(
          lreshape_loc, lreshape_type, ValueRange{lhs});
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      auto rreshape_loc =
          NameLoc::get(rewriter.getStringAttr(rname + "_r_reshape"));
      auto rreshape_type =
          RankedTensorType::get({rshape[0], rshape[1], rshape[2] * rshape[3]},
                                module::getElementType(rhs));
      auto rreshape_op = rewriter.create<top::ReshapeOp>(
          rreshape_loc, rreshape_type, ValueRange{rhs});

      rewriter.setInsertionPointAfter(lreshape_op);
      auto type_permute =
          RankedTensorType::get({lshape[0], lshape[2] * lshape[3], lshape[1]},
                                module::getElementType(lhs));
      auto loc_permute =
          NameLoc::get(rewriter.getStringAttr(lname + "_r_permute"));
      attrs.push_back(
          rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1})));
      auto permuteOp = rewriter.create<PermuteOp>(
          loc_permute, type_permute, ValueRange{lreshape_op}, attrs);
      attrs.clear();

      rewriter.setInsertionPointAfter(op);
      auto loc = NameLoc::get(rewriter.getStringAttr(name));
      auto newType = RankedTensorType::get({lshape[0], rshape[1], lshape[1]},
                                           module::getElementType(op));
      operands.push_back(rreshape_op);
      operands.push_back(permuteOp);
      operands.push_back(none);
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands);
      op.replaceAllUsesWith(matmulOp.getOperation());
      rewriter.eraseOp(op);

    } else if (mode == "abcd,aecd->abec") {
      // permute(a,b,c,d) -> (a,c,b,d)
      // permute(a,e,c,d) -> (a,c,d,e)
      // matmul((a,c,b,d),(a,c,d,e)) -> (a,c,b,e)
      // permute(a,c,b,e) -> (a,b,e,c)
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto l_loc = NameLoc::get(rewriter.getStringAttr(lname + "_l_trans"));
      auto l_type =
          RankedTensorType::get({lshape[0], lshape[1], lshape[2], lshape[3]},
                                module::getElementType(lhs));
      attrs.push_back(rewriter.getNamedAttr(
          "order", rewriter.getI64ArrayAttr({0, 2, 1, 3})));
      auto l_tranOp =
          rewriter.create<PermuteOp>(l_loc, l_type, ValueRange{lhs}, attrs);
      rewriter.setInsertionPointAfter(l_tranOp);
      attrs.clear();

      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      auto r_loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_trans"));
      auto r_type =
          RankedTensorType::get({rshape[0], rshape[1], rshape[2], rshape[3]},
                                module::getElementType(rhs));
      attrs.push_back(rewriter.getNamedAttr(
          "order", rewriter.getI64ArrayAttr({0, 2, 3, 1})));
      auto r_tranOp =
          rewriter.create<PermuteOp>(r_loc, r_type, ValueRange{rhs}, attrs);
      rewriter.setInsertionPointAfter(r_tranOp);
      attrs.clear();

      rewriter.setInsertionPointAfter(op);
      auto matmul_loc = NameLoc::get(rewriter.getStringAttr(name + "_matmul"));
      auto matmul_type =
          RankedTensorType::get({lshape[0], rshape[2], lshape[1], rshape[1]},
                                module::getElementType(op));
      operands.push_back(l_tranOp);
      operands.push_back(r_tranOp);
      operands.push_back(none);
      auto matmulOp =
          rewriter.create<MatMulOp>(matmul_loc, matmul_type, operands);
      auto newType =
          RankedTensorType::get({lshape[0], lshape[1], rshape[1], rshape[2]},
                                module::getElementType(matmulOp));
      attrs.push_back(rewriter.getNamedAttr(
          "order", rewriter.getI64ArrayAttr({0, 2, 3, 1})));
      auto transOp = rewriter.create<PermuteOp>(op.getLoc(), newType,
                                                ValueRange{matmulOp}, attrs);
      op.replaceAllUsesWith(transOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,aecd->acbe") {
      // permute(a,b,c,d) -> (a,c,b,d)
      // reshape(a,c,b,d) -> (ac,b,d)
      // permute(a,e,c,d) -> (a,c,d,e)
      // reshape(a,c,d,e) -> (ac,d,e)
      // matmul((ac,b,d),(ac,d,e)) -> (ac,b,e)
      // reshape(ac,b,e)->(a,c,b,e)
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto l_loc = NameLoc::get(rewriter.getStringAttr(lname + "_l_trans"));
      auto l_type =
          RankedTensorType::get({lshape[0], lshape[1], lshape[2], lshape[3]},
                                module::getElementType(lhs));
      attrs.push_back(rewriter.getNamedAttr(
          "order", rewriter.getI64ArrayAttr({0, 2, 1, 3})));
      auto l_tranOp =
          rewriter.create<PermuteOp>(l_loc, l_type, ValueRange{lhs}, attrs);
      rewriter.setInsertionPointAfter(l_tranOp);
      auto l_reshape_loc =
          NameLoc::get(rewriter.getStringAttr(lname + "_l_reshape"));
      auto l_reshape_type =
          RankedTensorType::get({lshape[0] * lshape[2], lshape[1], lshape[3]},
                                module::getElementType(lhs));
      auto l_reshape_Op = rewriter.create<ReshapeOp>(
          l_reshape_loc, l_reshape_type, ValueRange{lhs});
      attrs.clear();

      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      auto r_loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_trans"));
      auto r_type =
          RankedTensorType::get({rshape[0], rshape[1], rshape[2], rshape[3]},
                                module::getElementType(rhs));
      attrs.push_back(rewriter.getNamedAttr(
          "order", rewriter.getI64ArrayAttr({0, 2, 3, 1})));
      auto r_tranOp =
          rewriter.create<PermuteOp>(r_loc, r_type, ValueRange{rhs}, attrs);
      rewriter.setInsertionPointAfter(r_tranOp);
      auto r_reshape_loc =
          NameLoc::get(rewriter.getStringAttr(rname + "_r_reshape"));
      auto r_reshape_type =
          RankedTensorType::get({rshape[0] * rshape[2], rshape[3], rshape[1]},
                                module::getElementType(rhs));
      auto r_reshape_Op = rewriter.create<ReshapeOp>(
          r_reshape_loc, r_reshape_type, ValueRange{rhs});

      attrs.clear();

      rewriter.setInsertionPointAfter(op);
      auto matmul_loc = NameLoc::get(rewriter.getStringAttr(name + "_matmul"));
      auto matmul_type =
          RankedTensorType::get({lshape[0] * rshape[2], lshape[1], rshape[1]},
                                module::getElementType(op));
      operands.push_back(l_reshape_Op);
      operands.push_back(r_reshape_Op);
      operands.push_back(none);
      auto matmulOp =
          rewriter.create<MatMulOp>(matmul_loc, matmul_type, operands);
      auto reshape_loc =
          NameLoc::get(rewriter.getStringAttr(name + "_matmul_reshape"));
      auto reshape_type =
          RankedTensorType::get({lshape[0], rshape[2], lshape[1], rshape[1]},
                                module::getElementType(op.getOutput()));
      auto reshape_Op = rewriter.create<top::ReshapeOp>(
          reshape_loc, reshape_type, ValueRange{matmulOp});
      op.replaceAllUsesWith(reshape_Op.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,acde->abde") {
      // permute(a,b,c,d) -> (a,d,b,c)
      // reshape(a,d,b,c) -> (ad,b,c)
      // permute(a,c,d,e) -> (a,d,c,e)
      // reshape(a,d,c,e) -> (ad,c,e)
      // matmul((ad,b,c),(ad,c,e)) -> (ad,b,e)
      // reshape(ad,b,e)->(a,d,b,e)
      // permute(a,d,b,e) -> (a,b,d,e)
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto l_loc = NameLoc::get(rewriter.getStringAttr(lname + "_l_trans"));
      auto l_type =
          RankedTensorType::get({lshape[0], lshape[1], lshape[2], lshape[3]},
                                module::getElementType(lhs));
      attrs.push_back(rewriter.getNamedAttr(
          "order", rewriter.getI64ArrayAttr({0, 3, 1, 2})));
      auto l_tranOp =
          rewriter.create<PermuteOp>(l_loc, l_type, ValueRange{lhs}, attrs);
      rewriter.setInsertionPointAfter(l_tranOp);
      auto l_reshape_loc =
          NameLoc::get(rewriter.getStringAttr(lname + "_l_reshape"));
      auto l_reshape_type =
          RankedTensorType::get({lshape[0] * lshape[3], lshape[1], lshape[2]},
                                module::getElementType(lhs));
      auto l_reshape_Op = rewriter.create<ReshapeOp>(
          l_reshape_loc, l_reshape_type, ValueRange{lhs});
      attrs.clear();
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      auto r_loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_trans"));
      auto r_type =
          RankedTensorType::get({rshape[0], rshape[1], rshape[2], rshape[3]},
                                module::getElementType(lhs));
      attrs.push_back(rewriter.getNamedAttr(
          "order", rewriter.getI64ArrayAttr({0, 2, 1, 3})));
      auto r_tranOp =
          rewriter.create<PermuteOp>(r_loc, r_type, ValueRange{rhs}, attrs);
      rewriter.setInsertionPointAfter(r_tranOp);
      auto r_reshape_loc =
          NameLoc::get(rewriter.getStringAttr(rname + "_r_reshape"));
      auto r_reshape_type =
          RankedTensorType::get({rshape[0] * rshape[2], rshape[1], rshape[3]},
                                module::getElementType(lhs));
      auto r_reshape_Op = rewriter.create<ReshapeOp>(
          r_reshape_loc, r_reshape_type, ValueRange{rhs});

      attrs.clear();
      rewriter.setInsertionPointAfter(op);
      auto matmul_loc = NameLoc::get(rewriter.getStringAttr(name + "_matmul"));
      auto matmul_type =
          RankedTensorType::get({lshape[0] * rshape[2], lshape[1], rshape[3]},
                                module::getElementType(op));
      operands.push_back(l_reshape_Op);
      operands.push_back(r_reshape_Op);
      operands.push_back(none);
      auto matmulOp =
          rewriter.create<MatMulOp>(matmul_loc, matmul_type, operands);
      auto reshape_loc =
          NameLoc::get(rewriter.getStringAttr(name + "_matmul_reshape"));
      auto reshape_type =
          RankedTensorType::get({lshape[0], rshape[2], lshape[1], rshape[3]},
                                module::getElementType(op.getOutput()));
      auto reshape_Op = rewriter.create<top::ReshapeOp>(
          reshape_loc, reshape_type, ValueRange{matmulOp});
      auto newType =
          RankedTensorType::get({lshape[0], lshape[1], rshape[2], rshape[3]},
                                module::getElementType(reshape_Op));
      attrs.push_back(rewriter.getNamedAttr(
          "order", rewriter.getI64ArrayAttr({0, 2, 1, 3})));
      auto transOp = rewriter.create<PermuteOp>(op.getLoc(), newType,
                                                ValueRange{reshape_Op}, attrs);
      op.replaceAllUsesWith(transOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abc,cde->abde") {
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto lreshape_loc =
          NameLoc::get(rewriter.getStringAttr(lname + "_r_reshape"));
      auto lreshape_type = RankedTensorType::get(
          {lshape[0] * lshape[1], lshape[2]}, module::getElementType(lhs));
      auto lreshape_op = rewriter.create<top::ReshapeOp>(
          lreshape_loc, lreshape_type, ValueRange{lhs});
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      auto rreshape_loc =
          NameLoc::get(rewriter.getStringAttr(rname + "_r_reshape"));
      auto rreshape_type = RankedTensorType::get(
          {rshape[0], rshape[1] * rshape[2]}, module::getElementType(rhs));
      auto rreshape_op = rewriter.create<top::ReshapeOp>(
          rreshape_loc, rreshape_type, ValueRange{rhs});

      rewriter.setInsertionPointAfter(op);
      auto matmul_loc =
          NameLoc::get(rewriter.getStringAttr(name + "_r_matmul"));
      auto matmul_type =
          RankedTensorType::get({lshape[0] * lshape[1], rshape[1] * rshape[2]},
                                module::getElementType(op));
      operands.push_back(lreshape_op);
      operands.push_back(rreshape_op);
      operands.push_back(none);
      auto matmulOp =
          rewriter.create<MatMulOp>(matmul_loc, matmul_type, operands);
      auto loc = NameLoc::get(rewriter.getStringAttr(name));
      auto newType =
          RankedTensorType::get({lshape[0], lshape[1], rshape[1], rshape[2]},
                                module::getElementType(matmulOp));
      auto reshape_op =
          rewriter.create<top::ReshapeOp>(loc, newType, ValueRange{matmulOp});
      op.replaceAllUsesWith(reshape_op.getOperation());
      rewriter.eraseOp(op);

    } else if (mode == "abcd,cde->abce") {

      // lhs :
      //     abcd -> acbd(pemute)
      // rhs :
      //     cde  -> 1cde(reshape)
      //     acde -> acde(tile)
      // matmul:
      //   lhs(acbd) * rhs(acde) = result(acbe)
      // result:
      //     acbe -> abce(pemute)
      // success!

      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_r_trans"));
      auto newType =
          RankedTensorType::get({lshape[0], lshape[2], lshape[1], lshape[3]},
                                module::getElementType(lhs));
      attrs.push_back(rewriter.getNamedAttr(
          "order", rewriter.getI64ArrayAttr({0, 2, 1, 3})));
      auto tranOp =
          rewriter.create<PermuteOp>(loc, newType, ValueRange{lhs}, attrs);
      attrs.clear();
      operands.push_back(tranOp);
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      if (auto wOp = dyn_cast<top::WeightOp>(rhs.getDefiningOp())) {

        auto data = wOp.read_as_byte();
        uint8_t *dptr;
        newType =
            RankedTensorType::get({lshape[0], rshape[0], rshape[1], rshape[2]},
                                  module::getElementType(rhs));
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
        loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_reshape"));
        newType = RankedTensorType::get({1, rshape[0], rshape[1], rshape[2]},
                                        module::getElementType(rhs));
        auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
        loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_tile"));
        attrs.push_back(rewriter.getNamedAttr(
            "tile", rewriter.getI64ArrayAttr({lshape[0], 1, 1, 1})));
        newType =
            RankedTensorType::get({lshape[0], rshape[0], rshape[1], rshape[2]},
                                  module::getElementType(rhs));
        auto tileOp =
            rewriter.create<TileOp>(loc, newType, ValueRange{rrsop}, attrs);
        attrs.clear();
        operands.push_back(tileOp);
      }
      operands.push_back(none);
      newType =
          RankedTensorType::get({lshape[0], lshape[2], lshape[1], rshape[2]},
                                module::getElementType(op));
      rewriter.setInsertionPoint(op);
      loc = NameLoc::get(rewriter.getStringAttr(name + "_r_matmul"));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      attrs.clear();
      attrs.push_back(rewriter.getNamedAttr(
          "order", rewriter.getI64ArrayAttr({0, 2, 1, 3})));
      auto tranBackOp = rewriter.create<PermuteOp>(op.getLoc(), op.getType(),
                                                   ValueRange{matmulOp}, attrs);
      op.replaceAllUsesWith(tranBackOp.getOperation());
      rewriter.eraseOp(op);

    } else if (mode == "abc,acde->abde") {
      // lhs :
      //     abc
      // rhs :
      //     acde -> ac(de)(reshape)
      // matmul:
      //   lhs(abc) * rhs(ac(de)) = result(ab(de))
      // result:
      //     ab(de) -> abde(reshape)
      // success!
      operands.push_back(lhs);
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      auto newType =
          RankedTensorType::get({rshape[0], rshape[1], rshape[2] * rshape[3]},
                                module::getElementType(op));
      if (module::isWeight(rhs)) {
        rhs.setType(newType);
        operands.push_back(rhs);
      } else {
        auto loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_to3dim"));
        auto rhsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
        operands.push_back(rhsOp);
      }
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      newType =
          RankedTensorType::get({lshape[0], lshape[1], rshape[2] * rshape[3]},
                                module::getElementType(op));
      auto loc = NameLoc::get(rewriter.getStringAttr(name + "_r_matmul"));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      auto reshapeOp = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(),
                                                  ValueRange{matmulOp});
      op.replaceAllUsesWith(reshapeOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abc,abde->acde") {
      // lhs :
      //     abc -> acb(permute)
      // rhs :
      //     abde -> ab(de)(reshape)
      // matmul:
      //   lhs(acb) * rhs(ab(de)) = result(ac(de))
      // result:
      //     ab(de) -> abde(reshape)
      // success!
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_r_trans"));
      auto newType = RankedTensorType::get({lshape[0], lshape[2], lshape[1]},
                                           module::getElementType(lhs));
      attrs.push_back(
          rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1})));
      auto tranOp =
          rewriter.create<PermuteOp>(loc, newType, ValueRange{lhs}, attrs);
      operands.push_back(tranOp.getOutput());
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      newType =
          RankedTensorType::get({rshape[0], rshape[1], rshape[2] * rshape[3]},
                                module::getElementType(op));
      if (module::isWeight(rhs)) {
        rhs.setType(newType);
        operands.push_back(rhs);
      } else {
        loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_to3dim"));
        auto rhsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
        operands.push_back(rhsOp);
      }
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      newType =
          RankedTensorType::get({lshape[0], lshape[2], rshape[2] * rshape[3]},
                                module::getElementType(op));
      loc = NameLoc::get(rewriter.getStringAttr(name + "_r_matmul"));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      auto reshapeOp = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(),
                                                  ValueRange{matmulOp});
      op.replaceAllUsesWith(reshapeOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abc,bd->abcd") {

      // lhs :
      //     abc -> abc1(reshape)
      // rhs :
      //     bd -> 1b1d(reshape)
      //     1b1d -> ab1d(tile)
      // matmul:
      //   lhs(abc1) * rhs(ab1d) = result(abcd)
      // success!

      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto loc =
          NameLoc::get(rewriter.getStringAttr(lname + name + "_r_to4dim"));
      auto newType = RankedTensorType::get({lshape[0], lshape[1], lshape[2], 1},
                                           module::getElementType(op));
      auto lhsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{lhs});
      operands.push_back(lhsOp.getOutput());

      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      loc = NameLoc::get(rewriter.getStringAttr(rname + name + "_r_to4dim"));
      newType = RankedTensorType::get({1, rshape[0], 1, rshape[1]},
                                      module::getElementType(op));
      auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});

      loc = NameLoc::get(rewriter.getStringAttr(rname + name + "_r_tile"));
      attrs.push_back(rewriter.getNamedAttr(
          "tile", rewriter.getI64ArrayAttr({lshape[0], 1, 1, 1})));
      newType = RankedTensorType::get({lshape[0], rshape[0], 1, rshape[1]},
                                      module::getElementType(rhs));
      auto rhs_tileOp =
          rewriter.create<TileOp>(loc, newType, ValueRange{rrsop}, attrs);
      attrs.clear();
      operands.push_back(rhs_tileOp);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      newType =
          RankedTensorType::get({lshape[0], lshape[1], lshape[2], rshape[1]},
                                module::getElementType(op));
      loc = NameLoc::get(rewriter.getStringAttr(name));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      auto reshapeOp = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(),
                                                  ValueRange{matmulOp});
      op.replaceAllUsesWith(reshapeOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abc,abc->ab") {

      // einsum('abc, abc -> ab', L, H) => sum(L * H, dim = -1)
      rewriter.setInsertionPointAfter(op);
      auto loc = NameLoc::get(rewriter.getStringAttr(name + "_r_Mul"));
      auto newType = RankedTensorType::get({lshape[0], lshape[1], lshape[2]},
                                           module::getElementType(op));
      std::vector<Value> mul_operands;
      mul_operands.clear();
      mul_operands.push_back(lhs);
      mul_operands.push_back(rhs);
      auto mul_op = rewriter.create<MulOp>(loc, newType, mul_operands, attrs);

      rewriter.setInsertionPointAfter(mul_op);
      loc = NameLoc::get(rewriter.getStringAttr(name));

      newType = RankedTensorType::get({lshape[0], lshape[1]},
                                      module::getElementType(op));
      attrs.clear();
      attrs.push_back(
          rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr({2})));
      attrs.push_back(
          rewriter.getNamedAttr("keepdims", rewriter.getBoolAttr(false)));
      attrs.push_back(
          rewriter.getNamedAttr("mode", rewriter.getStringAttr("ReduceSum")));
      auto sumOp =
          rewriter.create<ReduceOp>(loc, newType, mul_op.getOutput(), attrs);
      op.getOutput().replaceAllUsesWith(sumOp);
      rewriter.eraseOp(op);
    } else if (mode == "abc,abdc,abc->abcd") {

      // einsum('abc, abdc, abc -> abcd', L, N, D) => (L.unsqueeze(2) * N *
      // D.unsqueeze(2)).permute(0,1,3,2)
      //
      // lhs :
      //     abc -> ab1c(reshape)
      // rhs :
      //     do nothing
      // dhs:
      //     abc -> ab1c(reshape)
      //
      // lhs(ab1c) * rhs(abdc) * dhs(ab1c) => result0(abdc).permute(0,1,3,2) ==>
      // result1(abcd) success!

      auto dhs = op.getInputs()[2];
      auto dshape = module::getShape(dhs);
      std::string dname = module::getName(dhs).str();

      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_r_to4dim"));
      auto newType = RankedTensorType::get({lshape[0], lshape[1], 1, lshape[2]},
                                           module::getElementType(op));
      auto lhsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{lhs});

      rewriter.setInsertionPointAfter(dhs.getDefiningOp());
      loc = NameLoc::get(rewriter.getStringAttr(dname + "_r_to4dim"));
      newType = RankedTensorType::get({dshape[0], dshape[1], 1, dshape[2]},
                                      module::getElementType(op));
      auto dhsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{dhs});

      rewriter.setInsertionPointAfter(op);
      loc = NameLoc::get(rewriter.getStringAttr(name + "_LN_Mul"));
      std::vector<Value> mul_operands;
      mul_operands.push_back(lhsOp.getOutput());
      mul_operands.push_back(rhs);
      std::vector<NamedAttribute> attrs;
      newType =
          RankedTensorType::get({rshape[0], rshape[1], rshape[2], rshape[3]},
                                module::getElementType(op));
      auto mul_op = rewriter.create<MulOp>(loc, newType, mul_operands, attrs);

      rewriter.setInsertionPointAfter(mul_op);
      loc = NameLoc::get(rewriter.getStringAttr(name + "_r_LRD_Mul"));
      mul_operands.clear();
      mul_operands.push_back(mul_op.getOutput());
      mul_operands.push_back(dhsOp.getOutput());
      auto mul_op2 = rewriter.create<MulOp>(loc, newType, mul_operands, attrs);

      rewriter.setInsertionPointAfter(mul_op2);
      loc = NameLoc::get(rewriter.getStringAttr(name));
      newType =
          RankedTensorType::get({rshape[0], rshape[1], rshape[3], rshape[2]},
                                module::getElementType(op));
      attrs.clear();
      attrs.push_back(rewriter.getNamedAttr(
          "order", rewriter.getI64ArrayAttr({0, 1, 3, 2})));
      auto tranOp =
          rewriter.create<PermuteOp>(loc, newType, mul_op2.getOutput(), attrs);
      op.getOutput().replaceAllUsesWith(tranOp);
      rewriter.eraseOp(op);
    } else if (mode == "abcd,acde,abc->abce") {
      // TODO : check whether tile can be optimized
      auto dhs = op.getInputs()[2];
      auto dshape = module::getShape(dhs);
      std::string dname = module::getName(dhs).str();

      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto newType = RankedTensorType::get(
          {lshape[0] * lshape[1] * lshape[2], 1, lshape[3]},
          module::getElementType(lhs));
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_r_to3dim"));
      auto lrsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{lhs});
      operands.push_back(lrsOp);
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      newType =
          RankedTensorType::get({rshape[0], 1, rshape[1], rshape[2], rshape[3]},
                                module::getElementType(rhs));
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_to5dim"));
      auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
      newType = RankedTensorType::get(
          {rshape[0], lshape[1], rshape[1], rshape[2], rshape[3]},
          module::getElementType(rhs));
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_tile"));
      attrs.push_back(rewriter.getNamedAttr(
          "tile", rewriter.getI64ArrayAttr({1, lshape[1], 1, 1, 1})));
      auto tileOp =
          rewriter.create<TileOp>(loc, newType, ValueRange{rrsop}, attrs);
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_to3dim"));
      newType = RankedTensorType::get(
          {rshape[0] * lshape[1] * rshape[1], rshape[2], rshape[3]},
          module::getElementType(rhs));
      auto reshapeOp =
          rewriter.create<ReshapeOp>(loc, newType, ValueRange{tileOp});
      operands.push_back(reshapeOp);
      operands.push_back(none);

      rewriter.setInsertionPoint(op);
      attrs.clear();
      loc = NameLoc::get(rewriter.getStringAttr(name + "_r_matmul"));
      newType = RankedTensorType::get(
          {lshape[0] * lshape[1] * lshape[2], 1, rshape[3]},
          module::getElementType(op));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);

      rewriter.setInsertionPointAfter(matmulOp);
      loc = NameLoc::get(rewriter.getStringAttr(name + "_r_reshape"));
      newType =
          RankedTensorType::get({lshape[0], lshape[1], lshape[2], rshape[3]},
                                module::getElementType(op));
      auto matmul_reshape_op =
          rewriter.create<ReshapeOp>(loc, newType, ValueRange{matmulOp});

      // rewriter.setInsertionPointAfter(dhs);
      newType = RankedTensorType::get({dshape[0], dshape[1], dshape[2], 1},
                                      module::getElementType(dhs));
      loc = NameLoc::get(rewriter.getStringAttr(dname + "_r_to4dim"));
      auto dhs_reshape_op =
          rewriter.create<ReshapeOp>(loc, newType, ValueRange{dhs});

      // rewriter.setInsertionPointAfter(matmul_reshape_op);
      operands.clear();
      operands.push_back(matmul_reshape_op.getOutput());
      operands.push_back(dhs_reshape_op.getOutput());
      auto mul_op =
          rewriter.create<MulOp>(op.getLoc(), op.getType(), operands, attrs);
      op.replaceAllUsesWith(mul_op.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,aeb->aecd") {
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto lreshape_loc =
          NameLoc::get(rewriter.getStringAttr(lname + "_r_reshape"));
      auto lreshape_type =
          RankedTensorType::get({lshape[0], lshape[1], lshape[2] * lshape[3]},
                                module::getElementType(lhs));
      auto lreshape_op = rewriter.create<top::ReshapeOp>(
          lreshape_loc, lreshape_type, ValueRange{lhs});

      rewriter.setInsertionPointAfter(op);
      auto newType =
          RankedTensorType::get({rshape[0], rshape[1], lshape[2] * lshape[3]},
                                module::getElementType(op));
      operands.push_back(rhs);
      operands.push_back(lreshape_op);
      operands.push_back(none);
      auto loc = NameLoc::get(rewriter.getStringAttr(name + "_r_matmul"));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands);
      auto reshapeOp = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(),
                                                  ValueRange{matmulOp});
      op.replaceAllUsesWith(reshapeOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcde,afbc->abdef") {
      // TODO : top/tpu matmul inference set left_transpose false by defalut,
      // maybe can support true
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto lreshape_loc =
          NameLoc::get(rewriter.getStringAttr(lname + "_r_reshape"));
      auto lreshape_type = RankedTensorType::get(
          {lshape[0], lshape[1], lshape[2], lshape[3] * lshape[4]},
          module::getElementType(lhs));
      auto lreshape_op = rewriter.create<top::ReshapeOp>(
          lreshape_loc, lreshape_type, ValueRange{lhs});
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_r_trans"));
      auto newType = RankedTensorType::get(
          {lshape[0], lshape[1], lshape[3] * lshape[4], lshape[2]},
          module::getElementType(lreshape_op));
      attrs.push_back(rewriter.getNamedAttr(
          "order", rewriter.getI64ArrayAttr({0, 1, 3, 2})));
      auto ltranOp = rewriter.create<PermuteOp>(loc, newType,
                                                ValueRange{lreshape_op}, attrs);
      attrs.clear();
      operands.push_back(ltranOp);
      // operands.push_back(lreshape_op);
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      // if (auto wOp = dyn_cast<top::WeightOp>(rhs.getDefiningOp())) {
      //   // TODO: check weight type other F32 can satisfy
      //   auto storage_type = module::getStorageType(rhs);
      //   assert(storage_type.isF32() && "Todo, supoort more weight type");
      //   auto data = wOp.read_as_byte();
      //   uint8_t *dptr;
      //   newType = RankedTensorType::get({rshape[0], rshape[2], rshape[1],
      //   rshape[3]}, module::getElementType(rhs)); std::vector<float_t>
      //   new_filter(newType.getNumElements(), 0); dptr = (uint8_t
      //   *)new_filter.data();
      //   // TODO : use continious data copy
      //   for (int32_t i = 0; i < rshape[0]; i++) {
      //       for (int32_t j = 0; j < rshape[2]; j++) {
      //           for (int32_t k = 0; k < rshape[1]; k++) {
      //               for (int32_t l = 0; l < rshape[3]; l++) {
      //                   auto offset = (i * rshape[2] * rshape[1] * rshape[3]
      //                   + j * rshape[1] * rshape[3] + k * rshape[3] + l) *
      //                   data->size(); memcpy(dptr + offset, data->data(),
      //                   data->size());
      //               }
      //           }
      //       }
      //   }

      //   auto new_op = top::WeightOp::create(op, "folder", new_filter,
      //   newType); wOp.replaceAllUsesWith(new_op.getDefiningOp());
      //   operands.push_back(new_op);
      //   rewriter.eraseOp(wOp);
      // } else {
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_trans"));
      newType =
          RankedTensorType::get({rshape[0], rshape[2], rshape[3], rshape[1]},
                                module::getElementType(rhs));
      attrs.push_back(rewriter.getNamedAttr(
          "order", rewriter.getI64ArrayAttr({0, 2, 3, 1})));
      auto rtranOp =
          rewriter.create<PermuteOp>(loc, newType, ValueRange{rhs}, attrs);
      attrs.clear();
      operands.push_back(rtranOp);
      // }
      operands.push_back(none);
      newType = RankedTensorType::get(
          {lshape[0], lshape[1], lshape[3] * lshape[4], rshape[1]},
          module::getElementType(op));
      rewriter.setInsertionPoint(op);
      loc = NameLoc::get(rewriter.getStringAttr(name + "_r_matmul"));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      auto reshapeOp = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(),
                                                  ValueRange{matmulOp});
      op.replaceAllUsesWith(reshapeOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "ab,cbdef->cadef") {
      rewriter.setInsertionPointAfter(op);
      auto rreshape_loc =
          NameLoc::get(rewriter.getStringAttr(rname + "_r_to3dim"));
      auto rreshape_type = RankedTensorType::get(
          {rshape[0], rshape[1], rshape[2] * rshape[3] * rshape[4]},
          module::getElementType(rhs));
      auto rreshape_op = rewriter.create<top::ReshapeOp>(
          rreshape_loc, rreshape_type, ValueRange{rhs});
      auto loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_permute"));
      auto newType = RankedTensorType::get(
          {rshape[1], rshape[0], rshape[2] * rshape[3] * rshape[4]},
          module::getElementType(rhs));
      attrs.push_back(
          rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({1, 0, 2})));
      auto rtranOp = rewriter.create<PermuteOp>(loc, newType,
                                                ValueRange{rreshape_op}, attrs);
      attrs.clear();

      rreshape_loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_to2dim"));
      rreshape_type = RankedTensorType::get(
          {rshape[1], rshape[0] * rshape[2] * rshape[3] * rshape[4]},
          module::getElementType(rhs));
      rreshape_op = rewriter.create<top::ReshapeOp>(rreshape_loc, rreshape_type,
                                                    ValueRange{rtranOp});
      operands.push_back(lhs);
      operands.push_back(rreshape_op);
      operands.push_back(none);
      //   rewriter.setInsertionPoint(op);
      //   attrs.clear();
      loc = NameLoc::get(rewriter.getStringAttr(name + "_r_matmul"));
      auto matmul_type = RankedTensorType::get(
          {lshape[0], rshape[0] * rshape[2] * rshape[3] * rshape[4]},
          module::getElementType(op.getOutput()));
      auto matmulOp =
          rewriter.create<MatMulOp>(loc, matmul_type, operands, attrs);
      rreshape_loc = NameLoc::get(rewriter.getStringAttr(name + "_r_to5dim"));
      rreshape_type = RankedTensorType::get(
          {lshape[0], rshape[0], rshape[2], rshape[3], rshape[4]},
          module::getElementType(op.getOutput()));
      rreshape_op = rewriter.create<top::ReshapeOp>(rreshape_loc, rreshape_type,
                                                    ValueRange{matmulOp});

      newType = RankedTensorType::get(
          {rshape[1], rshape[0], rshape[2] * rshape[3] * rshape[4]},
          module::getElementType(rhs));
      attrs.push_back(rewriter.getNamedAttr(
          "order", rewriter.getI64ArrayAttr({1, 0, 2, 3, 4})));
      rtranOp = rewriter.create<PermuteOp>(op.getLoc(), op.getType(),
                                           ValueRange{rreshape_op}, attrs);
      op.replaceAllUsesWith(rtranOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,cebd->abce" || mode == "abcd,ecbd->abec") {
      rewriter.setInsertionPointAfter(op);
      auto lreshape_loc =
          NameLoc::get(rewriter.getStringAttr(lname + "_r_to5dim"));
      auto lreshape_type = RankedTensorType::get(
          {lshape[0], lshape[1], mode == "abcd,cebd->abce" ? lshape[2] : 1,
           mode == "abcd,cebd->abce" ? 1 : lshape[2], lshape[3]},
          module::getElementType(lhs));
      auto lreshape_op = rewriter.create<top::ReshapeOp>(
          lreshape_loc, lreshape_type, ValueRange{lhs});
      operands.push_back(lreshape_op);

      attrs.clear();
      auto loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_permute"));
      auto newType =
          RankedTensorType::get({rshape[2], rshape[0], rshape[1], rshape[3]},
                                module::getElementType(rhs));
      attrs.push_back(rewriter.getNamedAttr(
          "order", rewriter.getI64ArrayAttr({2, 0, 1, 3})));
      auto rtranOp =
          rewriter.create<PermuteOp>(loc, newType, ValueRange{rhs}, attrs);

      lreshape_loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_to5dim"));
      lreshape_type =
          RankedTensorType::get({1, rshape[2], rshape[0], rshape[1], rshape[3]},
                                module::getElementType(rtranOp));
      lreshape_op = rewriter.create<top::ReshapeOp>(lreshape_loc, lreshape_type,
                                                    ValueRange{rtranOp});
      operands.push_back(lreshape_op);

      attrs.clear();
      loc = NameLoc::get(rewriter.getStringAttr(name + "_r_Mul"));
      newType = RankedTensorType::get(
          {lshape[0], lshape[1], rshape[0], rshape[1], lshape[3]},
          module::getElementType(op));
      auto mul_op = rewriter.create<MulOp>(loc, newType, operands, attrs);

      //   rewriter.setInsertionPointAfter(mul_op);
      loc = NameLoc::get(rewriter.getStringAttr(name));
      newType =
          RankedTensorType::get({lshape[0], lshape[1], rshape[0], rshape[1]},
                                module::getElementType(op));
      attrs.clear();
      attrs.push_back(
          rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr({4})));
      attrs.push_back(
          rewriter.getNamedAttr("keepdims", rewriter.getBoolAttr(false)));
      attrs.push_back(
          rewriter.getNamedAttr("mode", rewriter.getStringAttr("ReduceSum")));
      auto sumOp =
          rewriter.create<ReduceOp>(loc, newType, mul_op.getOutput(), attrs);
      op.getOutput().replaceAllUsesWith(sumOp);
      rewriter.eraseOp(op);
    } else if (mode == "abcd,cdbe->abce") {
      rewriter.setInsertionPointAfter(op);
      auto lreshape_loc =
          NameLoc::get(rewriter.getStringAttr(lname + "_r_to5dim"));
      auto lreshape_type =
          RankedTensorType::get({lshape[0], lshape[1], lshape[2], 1, lshape[3]},
                                module::getElementType(lhs));
      auto lreshape_op = rewriter.create<top::ReshapeOp>(
          lreshape_loc, lreshape_type, ValueRange{lhs});
      operands.push_back(lreshape_op);

      attrs.clear();
      auto loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_permute"));
      auto newType =
          RankedTensorType::get({rshape[2], rshape[0], rshape[3], rshape[1]},
                                module::getElementType(rhs));
      attrs.push_back(rewriter.getNamedAttr(
          "order", rewriter.getI64ArrayAttr({2, 0, 3, 1})));
      auto rtranOp =
          rewriter.create<PermuteOp>(loc, newType, ValueRange{rhs}, attrs);

      lreshape_loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_to5dim"));
      lreshape_type =
          RankedTensorType::get({1, rshape[2], rshape[0], rshape[3], rshape[1]},
                                module::getElementType(rtranOp));
      lreshape_op = rewriter.create<top::ReshapeOp>(lreshape_loc, lreshape_type,
                                                    ValueRange{rtranOp});
      operands.push_back(lreshape_op);

      attrs.clear();
      loc = NameLoc::get(rewriter.getStringAttr(name + "_r_Mul"));
      newType = RankedTensorType::get(
          {lshape[0], lshape[1], rshape[0], rshape[3], lshape[3]},
          module::getElementType(op));
      auto mul_op = rewriter.create<MulOp>(loc, newType, operands, attrs);

      //   rewriter.setInsertionPointAfter(mul_op);
      loc = NameLoc::get(rewriter.getStringAttr(name));
      newType =
          RankedTensorType::get({lshape[0], lshape[1], rshape[0], rshape[3]},
                                module::getElementType(op));
      attrs.clear();
      attrs.push_back(
          rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr({4})));
      attrs.push_back(
          rewriter.getNamedAttr("keepdims", rewriter.getBoolAttr(false)));
      attrs.push_back(
          rewriter.getNamedAttr("mode", rewriter.getStringAttr("ReduceSum")));
      auto sumOp =
          rewriter.create<ReduceOp>(loc, newType, mul_op.getOutput(), attrs);
      op.getOutput().replaceAllUsesWith(sumOp);
      rewriter.eraseOp(op);
    } else if (mode == "abcde,abfge->abcdfg") {
      // reshape(abcde) -> (ab,cd,e)
      // reshape(abfge) -> (ab,fg,e)
      // permute(ab,fg,e) -> (ab,e,fg)
      // batch_matmul((ab,cd,e),(ab,e,fg)) -> (ab,cd,fg)
      // reshape(ab,cd,fg) = (a,b,c,d,f,g)
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto lhs_loc = NameLoc::get(rewriter.getStringAttr(lname + "_r_reshape"));
      auto lhs_new_type = RankedTensorType::get(
          {lshape[0] * lshape[1], lshape[2] * lshape[3], lshape[4]},
          module::getElementType(lhs));
      auto lhsOp =
          rewriter.create<ReshapeOp>(lhs_loc, lhs_new_type, ValueRange{lhs});

      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      auto rhs_loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_reshape"));
      auto rhs_new_type = RankedTensorType::get(
          {rshape[0] * rshape[1], rshape[2] * rshape[3], rshape[4]},
          module::getElementType(rhs));
      auto rhsOp =
          rewriter.create<ReshapeOp>(rhs_loc, rhs_new_type, ValueRange{rhs});

      rewriter.setInsertionPointAfter(rhsOp);
      auto rhsOpPermute = RankedTensorType::get(
          {rshape[0] * rshape[1], rshape[4], rshape[2] * rshape[3]},
          module::getElementType(rhs));
      auto rhsPermuteLoc =
          NameLoc::get(rewriter.getStringAttr(rname + "_r_permute"));
      attrs.push_back(
          rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1})));
      auto RPermuteOp = rewriter.create<PermuteOp>(rhsPermuteLoc, rhsOpPermute,
                                                   ValueRange{rhsOp}, attrs);
      attrs.clear();

      rewriter.setInsertionPointAfter(op);
      auto mm_loc = NameLoc::get(rewriter.getStringAttr(name + "_r_matmul"));
      auto mm_type = RankedTensorType::get(
          {lshape[0] * lshape[1], lshape[2] * lshape[3], rshape[2] * rshape[3]},
          module::getElementType(op));
      operands.push_back(lhsOp);
      operands.push_back(RPermuteOp);
      operands.push_back(none);
      auto matmulOp = rewriter.create<MatMulOp>(mm_loc, mm_type, operands);
      auto reshapeOp = rewriter.create<ReshapeOp>(
          op.getLoc(), op.getType(), ValueRange{matmulOp.getOutput()});
      op.replaceAllUsesWith(reshapeOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,abef->acdef") {
      // reshape(abcd) -> (a,b,cd)
      // reshape(abef) -> (a,b,ef)
      // permute(a,b,cd) -> (a,cd,b)
      // batch_matmul((a,cd,b),(a,b,ef)) -> (a,cd,ef)
      // reshape(a,cd,ef) = (a,c,d,e,f)
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto lhs_loc = NameLoc::get(rewriter.getStringAttr(lname + "_l_reshape"));
      auto lhs_new_type =
          RankedTensorType::get({lshape[0], lshape[1], lshape[2] * lshape[3]},
                                module::getElementType(lhs));
      auto lhsOp =
          rewriter.create<ReshapeOp>(lhs_loc, lhs_new_type, ValueRange{lhs});

      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      auto rhs_loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_reshape"));
      auto rhs_new_type =
          RankedTensorType::get({rshape[0], rshape[1], rshape[2] * rshape[3]},
                                module::getElementType(rhs));
      auto rhsOp =
          rewriter.create<ReshapeOp>(rhs_loc, rhs_new_type, ValueRange{rhs});

      rewriter.setInsertionPointAfter(lhsOp);
      auto lhsOpPermute =
          RankedTensorType::get({lshape[0], lshape[2] * lshape[3], lshape[1]},
                                module::getElementType(lhs));
      auto lhsPermuteLoc =
          NameLoc::get(rewriter.getStringAttr(rname + "_l_permute"));
      attrs.push_back(
          rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1})));
      auto LPermuteOp = rewriter.create<PermuteOp>(lhsPermuteLoc, lhsOpPermute,
                                                   ValueRange{lhsOp}, attrs);
      attrs.clear();

      rewriter.setInsertionPointAfter(op);
      auto mm_loc = NameLoc::get(rewriter.getStringAttr(name + "_l_matmul"));
      auto mm_type = RankedTensorType::get(
          {lshape[0], lshape[2] * lshape[3], rshape[2] * rshape[3]},
          module::getElementType(op));
      operands.push_back(LPermuteOp);
      operands.push_back(rhsOp);
      operands.push_back(none);
      auto matmulOp = rewriter.create<MatMulOp>(mm_loc, mm_type, operands);
      auto reshapeOp = rewriter.create<ReshapeOp>(
          op.getLoc(), op.getType(), ValueRange{matmulOp.getOutput()});
      op.replaceAllUsesWith(reshapeOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,abce->abde") {
      // reshape(abcd) -> (ab,c,d)
      // reshape(abce) -> (ab,c,e)
      // permute(ab,c,d) -> (ab,d,c)
      // batch_matmul((ab,d,c),(ab,c,e)) -> (ab,d,e)
      // reshape(ab,d,e) = (a,b,d,e)
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto lhs_loc = NameLoc::get(rewriter.getStringAttr(lname + "_l_reshape"));
      auto lhs_new_type =
          RankedTensorType::get({lshape[0] * lshape[1], lshape[2], lshape[3]},
                                module::getElementType(lhs));
      auto lhsOp =
          rewriter.create<ReshapeOp>(lhs_loc, lhs_new_type, ValueRange{lhs});

      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      auto rhs_loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_reshape"));
      auto rhs_new_type =
          RankedTensorType::get({rshape[0] * rshape[1], rshape[2], rshape[3]},
                                module::getElementType(rhs));
      auto rhsOp =
          rewriter.create<ReshapeOp>(rhs_loc, rhs_new_type, ValueRange{rhs});

      rewriter.setInsertionPointAfter(lhsOp);
      auto lhsOpPermute =
          RankedTensorType::get({lshape[0] * lshape[1], lshape[3], lshape[2]},
                                module::getElementType(lhs));
      auto lhsPermuteLoc =
          NameLoc::get(rewriter.getStringAttr(rname + "_l_permute"));
      attrs.push_back(
          rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1})));
      auto LPermuteOp = rewriter.create<PermuteOp>(lhsPermuteLoc, lhsOpPermute,
                                                   ValueRange{lhsOp}, attrs);
      attrs.clear();

      rewriter.setInsertionPointAfter(op);
      auto mm_loc = NameLoc::get(rewriter.getStringAttr(name + "_l_matmul"));
      auto mm_type =
          RankedTensorType::get({lshape[0] * lshape[1], lshape[3], rshape[3]},
                                module::getElementType(op));
      operands.push_back(LPermuteOp);
      operands.push_back(rhsOp);
      operands.push_back(none);
      auto matmulOp = rewriter.create<MatMulOp>(mm_loc, mm_type, operands);
      auto reshapeOp = rewriter.create<ReshapeOp>(
          op.getLoc(), op.getType(), ValueRange{matmulOp.getOutput()});
      op.replaceAllUsesWith(reshapeOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,adbe->acbe") {
      // reshape(a,b,c,d) -> (ab,c,d)
      // permute(a,d,b,e) -> (a,b,d,e)
      // reshape(a,b,d,e) -> (ab,d,e)
      // matmul((ab,c,d),(ab,d,e)) -> (ab,c,e)
      // reshape(ab,c,e)->(a,b,c,e)
      // permute(a,b,c,e) -> (a,c,b,e)
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto l_reshape_loc =
          NameLoc::get(rewriter.getStringAttr(lname + "_l_reshape"));
      auto l_reshape_type =
          RankedTensorType::get({lshape[0] * lshape[1], lshape[2], lshape[3]},
                                module::getElementType(lhs));
      auto l_reshape_Op = rewriter.create<ReshapeOp>(
          l_reshape_loc, l_reshape_type, ValueRange{lhs});
      attrs.clear();

      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      auto r_loc = NameLoc::get(rewriter.getStringAttr(rname + "_r_trans"));
      auto r_type =
          RankedTensorType::get({rshape[0], rshape[1], rshape[2], rshape[3]},
                                module::getElementType(rhs));
      attrs.push_back(rewriter.getNamedAttr(
          "order", rewriter.getI64ArrayAttr({0, 2, 1, 3})));
      auto r_tranOp =
          rewriter.create<PermuteOp>(r_loc, r_type, ValueRange{rhs}, attrs);
      rewriter.setInsertionPointAfter(r_tranOp);
      auto r_reshape_loc =
          NameLoc::get(rewriter.getStringAttr(rname + "_r_reshape"));
      auto r_reshape_type =
          RankedTensorType::get({rshape[0] * rshape[2], rshape[1], rshape[3]},
                                module::getElementType(rhs));
      auto r_reshape_Op = rewriter.create<ReshapeOp>(
          r_reshape_loc, r_reshape_type, ValueRange{rhs});

      attrs.clear();

      rewriter.setInsertionPointAfter(op);
      auto matmul_loc = NameLoc::get(rewriter.getStringAttr(name + "_matmul"));
      auto matmul_type =
          RankedTensorType::get({lshape[0] * lshape[1], lshape[2], rshape[3]},
                                module::getElementType(op));
      operands.push_back(l_reshape_Op);
      operands.push_back(r_reshape_Op);
      operands.push_back(none);
      auto matmulOp =
          rewriter.create<MatMulOp>(matmul_loc, matmul_type, operands);
      auto reshape_loc =
          NameLoc::get(rewriter.getStringAttr(name + "_matmul_reshape"));
      auto reshape_type =
          RankedTensorType::get({lshape[0], lshape[1], lshape[2], rshape[3]},
                                module::getElementType(op.getOutput()));
      auto reshape_Op = rewriter.create<top::ReshapeOp>(
          reshape_loc, reshape_type, ValueRange{matmulOp});
      auto newType =
          RankedTensorType::get({lshape[0], lshape[2], lshape[1], rshape[3]},
                                module::getElementType(reshape_Op));
      attrs.push_back(rewriter.getNamedAttr(
          "order", rewriter.getI64ArrayAttr({0, 2, 1, 3})));
      auto transOp = rewriter.create<PermuteOp>(op.getLoc(), newType,
                                                ValueRange{reshape_Op}, attrs);
      op.replaceAllUsesWith(transOp.getOperation());
      rewriter.eraseOp(op);
    } else {
      llvm_unreachable("Einsum not support this mode now");
    }
    return success();
  }
};

void EinsumOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<ConvertEinsum>(context);
}
