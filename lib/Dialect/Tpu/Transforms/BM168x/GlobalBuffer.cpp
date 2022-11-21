//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/BMAddressAssign.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;

namespace tpu_mlir {

namespace bm168x {

class LSTMGlobalBuffer : public OpRewritePattern<tpu::LSTMOp> {
public:
  using OpRewritePattern<tpu::LSTMOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::LSTMOp lstmOp,
                                PatternRewriter &rewriter) const override {
    if (!lstmOp.buffer().getType().isa<mlir::NoneType>()) {
      return failure();
    }
    auto ctx = rewriter.getContext();
    auto attr = lstmOp.parseParam();
    auto type = Module::getStorageType(lstmOp.input());
    // add buffer
    int64_t buffer_size = attr.batch_size * attr.hidden_size * 5;
    std::vector<int64_t> buffer_shape = {5, attr.batch_size, attr.hidden_size};
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(lstmOp, buffer_type);
    lstmOp.buffer().replaceAllUsesWith(buffer);
    return success();
  }
};

void populateGlobalBufferPatterns(RewritePatternSet *patterns) {
  patterns->add<LSTMGlobalBuffer>(patterns->getContext());
}

} // namespace bm168x
} // namespace tpu_mlir
