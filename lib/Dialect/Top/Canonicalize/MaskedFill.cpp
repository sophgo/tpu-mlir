//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#define TILE_COND 1
#define TILE_BRN 2
using namespace tpu_mlir::top;

struct MaskedFillBroadcast : public OpRewritePattern<MaskedFillOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MaskedFillOp op,
                                PatternRewriter &rewriter) const override {
    auto input_shape = module::getShape(op.getBrn());
    auto condition_shape = module::getShape(op.getCond());
    auto dims = input_shape.size();
    int *broadcast = new int[dims];
    for (int i = 0; i < dims; ++i) {
      if (input_shape[i] != condition_shape[i]) {
        if (input_shape[i] == 1) {
          broadcast[i] = TILE_BRN;
        } else if (condition_shape[i] == 1) {
          broadcast[i] = TILE_COND;
        } else {
          llvm_unreachable("input shape and condition shape mismatch");
        }
      } else {
        broadcast[i] = 0;
      }
    }
    if (std::all_of(broadcast, broadcast + dims,
                    [](int i) { return i == 0; })) {
      return failure();
    }

    rewriter.setInsertionPointAfter(op);
    auto cond = op.getCond();
    auto input = op.getBrn();
    // insert tile to broadcast
    std::vector<NamedAttribute> attrs;
    for (int i = 0; i < dims; ++i) {
      if (broadcast[i] != 0) {
        auto tile_input = broadcast[i] == TILE_BRN ? input : cond;
        auto tile_input_shape = module::getShape(tile_input);
        attrs.push_back(
            rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(i)));
        auto input_op_name = module::getName(tile_input);
        auto tile_loc =
            NameLoc::get(rewriter.getStringAttr(input_op_name.str() + "_tile"));
        auto stype = module::getStorageType(tile_input);
        auto tile_output_shape = std::vector<int64_t>(tile_input_shape);
        tile_output_shape[i] =
            broadcast[i] == TILE_BRN ? condition_shape[i] : input_shape[i];
        attrs.push_back(rewriter.getNamedAttr(
            "tile", rewriter.getI64IntegerAttr(tile_output_shape[i])));
        auto tile_output_type = RankedTensorType::get(tile_output_shape, stype);
        bool reuse_tile = false;
        for (auto j : tile_input.getUsers()) {
          if (isa<top::TileOp>(j)) {
            auto tile_op = dyn_cast<top::TileOp>(j);
            auto pre_tile_attrs = tile_op->getAttrs();
            if (pre_tile_attrs.equals(attrs)) {
              reuse_tile = true;
              if (broadcast[i] == TILE_BRN) {
                input = tile_op.getOutput();
              } else {
                cond = tile_op.getOutput();
              }
              attrs.clear();
            }
          }
        }
        if (!reuse_tile) {
          auto tile_op = rewriter.create<top::TileOp>(
              tile_loc, tile_output_type, ValueRange{tile_input}, attrs);
          attrs.clear();
          if (broadcast[i] == TILE_BRN) {
            input = tile_op.getOutput();
          } else {
            cond = tile_op.getOutput();
          }
        }
      }
    }
    for (auto &attr : op->getAttrs()) {
      attrs.push_back(attr);
    }
    auto maskedfill_op = rewriter.create<top::MaskedFillOp>(
        op.getLoc(), op.getType(), ValueRange{cond, input}, attrs);
    op.replaceAllUsesWith(maskedfill_op.getOperation());
    op.erase();
    return success();
  }
};

void MaskedFillOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<MaskedFillBroadcast>(context);
}
