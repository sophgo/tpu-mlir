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

using namespace tpu_mlir::top;

struct TopFuseTile : public OpRewriterPatternEx<TileOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  TopFuseTile(mlir::MLIRContext *context)
      : OpRewriterPatternEx<TileOp>(context, "TopFuseTile") {}

  LogicalResult matchAndRewriteImpl(TileOp op,
                                    PatternRewriter &rewriter) const override {
    Value output = op.getOutput();
    Value input = op.getInput();

    // Return failure if output has no users
    if (output.use_empty()) {
      return failure();
    }

    auto inputShape = module::getShape(input);
    auto outputShape = module::getShape(output);

    // Check broadcast compatibility: input can be broadcast to output shape
    bool isBcastCompatible = true;
    if (inputShape.size() != outputShape.size()) {
      isBcastCompatible = false;
    } else {
      for (int i = 0; i < inputShape.size(); ++i) {
        // Dimensions must be either equal or have at least one 1 for
        // broadcasting
        if (inputShape[i] != outputShape[i] &&
            std::min(inputShape[i], outputShape[i]) != 1) {
          isBcastCompatible = false;
          break;
        }
      }
    }

    // Categorize users: broadcast-compatible vs incompatible
    SmallVector<Operation *> bcastUsers;
    SmallVector<Operation *> nonBcastUsers;
    bool hasValidBcastUser = false;

    for (Operation *user : output.getUsers()) {
      // Check for supported broadcast operations
      if (isa<AddOp, SubOp, MulOp, MinOp, MaxOp>(user)) {
        // Special handling for AddOp: exclude cases with weight operands
        if (isa<AddOp>(user)) {
          bool hasWeight = false;
          for (Value operand : user->getOperands()) {
            if (auto defOp = operand.getDefiningOp()) {
              if (isa<WeightOp>(defOp)) {
                hasWeight = true;
                break;
              }
            }
          }
          // Only qualify as broadcast user if no weight and
          // broadcast-compatible
          if (!hasWeight && isBcastCompatible) {
            bcastUsers.push_back(user);
            hasValidBcastUser = true;
            continue;
          }
        }
        // Handle other broadcast-compatible operations
        else if (isBcastCompatible) {
          bcastUsers.push_back(user);
          hasValidBcastUser = true;
          continue;
        }
      }

      // All other users are considered non-broadcast users
      nonBcastUsers.push_back(user);
    }

    // Fail optimization if no valid broadcast users found
    if (!hasValidBcastUser) {
      return failure();
    }

    // Case 1: All users are broadcast-compatible - remove TileOp entirely
    if (nonBcastUsers.empty()) {
      rewriter.replaceOp(op, {input});
      return success();
    }

    // Case 2: Mixed users - clone TileOp for non-broadcast users
    auto loc = op.getLoc();
    auto newTileOp = rewriter.create<TileOp>(loc, output.getType(), input);
    Value newOutput = newTileOp.getOutput();

    // Replace operands for non-broadcast users with new TileOp output
    for (Operation *user : nonBcastUsers) {
      user->replaceUsesOfWith(output, newOutput);
    }

    // Replace operands for broadcast users with original input
    for (Operation *user : bcastUsers) {
      user->replaceUsesOfWith(output, input);
    }

    // Clean up original TileOp if it has no remaining users
    if (output.use_empty()) {
      rewriter.eraseOp(op);
    }

    return success();
  }
};

struct ReplaceWithWeightInput : public OpRewriterPatternEx<TileOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  ReplaceWithWeightInput(mlir::MLIRContext *context)
      : OpRewriterPatternEx<TileOp>(context, "ReplaceWithWeightInput") {}

  LogicalResult matchAndRewriteImpl(TileOp op,
                                    PatternRewriter &rewriter) const override {
    if (op.getTileT()) {
      return failure();
    }
    if (isa<WeightOp>(op.getInput().getDefiningOp())) {
      auto storage_type = module::getStorageType(op.getOutput());
      auto weight = dyn_cast<WeightOp>(op.getInput().getDefiningOp());
      auto w = weight.read_as_float();
      auto shape0 = module::getShape(op.getInput());
      auto shape1 = module::getShape(op.getOutput());
      bool updated = false;
      for (int i = shape0.size() - 1; i >= 0; i--) {
        if (shape0[i] == shape1[i])
          continue;
        int tile = shape1[i] / shape0[i];
        size_t inner = shape0[i];
        for (int j = i + 1; j < shape1.size(); j++)
          inner *= shape1[j];
        size_t outer = 1;
        for (int j = i - 1; j >= 0; j--)
          outer *= shape0[j];

        std::shared_ptr<std::vector<float>> new_w =
            std::make_shared<std::vector<float>>();
        for (int j = 0; j < outer; j++) {
          for (int k = 0; k < tile; k++) {
            new_w.get()->insert(new_w.get()->end(),
                                w.get()->begin() + j * inner,
                                w.get()->begin() + (j + 1) * inner);
          }
        }
        w = new_w;
        updated = true;
      }
      if (updated) {
        auto w_op =
            WeightOp::create_float(op, module::getName(op.getOutput()).str(),
                                   *w, shape1, storage_type);
        rewriter.replaceOp(op, w_op);
        return success();
      }
    }
    return failure();
  }
};

void TileOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<TopFuseTile, ReplaceWithWeightInput>(context);
}
