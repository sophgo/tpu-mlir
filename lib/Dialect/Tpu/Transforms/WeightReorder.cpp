//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Backend/Arch.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace llvm;

using namespace tpu_mlir::backend;
namespace tpu_mlir {
namespace tpu {
extern void populateWeightReorderCV18xxPatterns(RewritePatternSet *patterns);
extern void populateWeightReorderBM1684Patterns(RewritePatternSet *patterns);
extern void populateWeightReorderBM1684XPatterns(RewritePatternSet *patterns);

class WeightTypePattern : public OpRewriterPatternEx<top::WeightOp> {
public:
  WeightTypePattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<top::WeightOp>(context, "WeightTypePattern") {}

  LogicalResult matchAndRewriteImpl(top::WeightOp op,
                                    PatternRewriter &rewriter) const override {
    if (op.getPath().has_value()) {
      return failure();
    }
    if (op.use_empty()) {
      return failure();
    }
    auto user_op = *op->user_begin();
    std::string path = user_op->getName().getStringRef().str();
    auto result = op.getResult();
    if (auto user = dyn_cast<tpu::A16MatMulOp>(user_op)) {
      if (result == user.getInput()) {
        path += ".input";
      } else if (result == user.getWeight()) {
        path += ".weight";
      } else if (result == user.getScale()) {
        path += ".scale";
      } else if (result == user.getZp()) {
        path += ".zp";
      } else if (result == user.getBias()) {
        path += ".bias";
      }
    } else if (auto user = dyn_cast<tpu::MatMulOp>(user_op)) {
      if (result == user.getInput()) {
        path += ".input";
      } else if (result == user.getRight()) {
        path += ".weight";
      } else if (result == user.getBias()) {
        path += ".bias";
      }
    } else if (auto user = dyn_cast<tpu::Conv2DOp>(user_op)) {
      if (result == user.getInput()) {
        path += ".input";
      } else if (result == user.getFilter()) {
        path += ".weight";
      } else if (result == user.getBias()) {
        path += ".bias";
      }
    } else if (auto user = dyn_cast<tpu::LayerNormOp>(user_op)) {
      if (result == user.getInput()) {
        path += ".input";
      } else if (result == user.getWeight()) {
        path += ".weight";
      } else if (result == user.getBias()) {
        path += ".bias";
      }
    } else if (auto user = dyn_cast<tpu::RMSNormOp>(user_op)) {
      if (result == user.getInput()) {
        path += ".input";
      } else if (result == user.getGamma()) {
        path += ".gamma";
      }
    } else if (auto user = dyn_cast<tpu::GatherOp>(user_op)) {
      if (result == user.getInput()) {
        path += ".source";
      } else if (result == user.getIndices()) {
        path += ".index";
      }
    } else {
      int index = 0;
      for (int i = 0; i < user_op->getNumOperands(); i++) {
        if (user_op->getOperand(i) == result) {
          index = i;
          break;
        }
      }
      path += "." + std::to_string(index);
    }
    op.setPath(mlir::StringAttr::get(op.getContext(), path));
    return success();
  }
  bool shouldPrint(top::WeightOp op) const override { return false; }
};

class WeightReorderPass : public WeightReorderBase<WeightReorderPass> {
public:
  WeightReorderPass() {}
  void runOnOperation() override {
    if (!module::isState(module::State::TPU_LOWERED)) {
      llvm_unreachable("module should be tpu quantized");
    }
    auto modules = module::getAllModules();
    for (auto sub : *modules) {
      RewritePatternSet patterns(&getContext());
      if (module::isBM1684Family()) {
        populateWeightReorderBM1684Patterns(&patterns);
      } else if (module::isBM1684XFamily() || module::isBM1690Family()) {
        populateWeightReorderBM1684XPatterns(&patterns);
      } else if (module::isCV18xx()) {
        populateWeightReorderCV18xxPatterns(&patterns);
      }
      auto config = GreedyRewriteConfig();
      config.maxIterations = 1; // apply each pattern only once.
      applyPatternsAndFoldGreedily(sub, std::move(patterns), config);
      module::applyPatternOnce<WeightTypePattern>(sub);
    }
    module::updateModuleTypes();
    module::setState(module::State::TPU_REORDERED);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createWeightReorderPass() {
  return std::make_unique<WeightReorderPass>();
}
} // namespace tpu
} // namespace tpu_mlir
