//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "AddressAssign/BMAddressAssign.h"
#include "AddressAssign/CVAddressAssign.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;

namespace tpu_mlir {
namespace tpu {

extern void populateGlobalBufferBM168xPatterns(RewritePatternSet *patterns);

class ConcatFusePattern : public OpRewritePattern<tpu::ConcatOp> {
public:
  using OpRewritePattern<tpu::ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::ConcatOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getOnlyMerge()) {
      return failure();
    }
    if (op.getDoRelu()) {
      return failure();
    }
    if (module::isBM1684Family() &&
        module::isUniformQuantized(op.getOutput())) {
      // 1684 4N mode not support
      return failure();
    }
    auto shape = module::getShape(op.getOutput());
    int outer_dim = std::accumulate(shape.begin(), shape.begin() + op.getAxis(),
                                    1, std::multiplies<int64_t>());
    if (outer_dim != 1) {
      return failure();
    }
    for (auto in : op.getInputs()) {
      if (module::isWeight(in)) {
        return failure();
      }
      if (in.hasOneUse() == false) {
        return failure();
      }
      auto in_op = in.getDefiningOp();
      if (in_op == nullptr) {
        return failure();
      } else if (isa<tpu::ConcatOp>(in_op)) {
        return failure();
      } else if (auto rshape = dyn_cast<tpu::ReshapeOp>(in_op)) {
        auto in2 = rshape.getInput();
        if (in2.getDefiningOp() == nullptr || in2.hasOneUse() == false) {
          return failure();
        }
      } else if (auto sliceOp = dyn_cast<tpu::SliceOp>(in_op)) {
        auto p = sliceOp.parseParam();
        if (p.fusible) {
          return failure();
        }
      }
    }
    op.setOnlyMerge(true);
    return success();
  }
};

class AddressAssignPass : public AddressAssignBase<AddressAssignPass> {
public:
  AddressAssignPass() {}
  void runOnOperation() override {
    if (!module::isState(module::State::TPU_DIVIDED)) {
      llvm_unreachable("module should be divided");
    }
    module::removeUnusedOp();
    auto modules = module::getAllModules();
    for (auto s : *modules) {
      if (module::isCV18xx()) {
        CVAddressAssign addr_assign;
        addr_assign.assign(s, reuse_addr, merge_weight, compress_weight,
                           weight_map_file);
      } else {
        RewritePatternSet patterns(s.getContext());
        populateGlobalBufferBM168xPatterns(&patterns);
        patterns.add<ConcatFusePattern>(patterns.getContext());
        applyPatternsAndFoldGreedily(s, std::move(patterns));
        BMAddressAssign addr_assign;
        addr_assign.assign(s, reuse_addr);
      }
    }
    module::updateModuleTypes();
    module::setState(module::State::TPU_ADDRESSED);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAddressAssignPass() {
  return std::make_unique<AddressAssignPass>();
}
} // namespace tpu
} // namespace tpu_mlir
