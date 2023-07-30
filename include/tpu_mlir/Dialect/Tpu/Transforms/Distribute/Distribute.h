//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Traits/Traits.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <llvm/Support/Debug.h>

namespace tpu_mlir {
namespace tpu {
// ===================================
// helper functions
// ===================================
void distribute(PatternRewriter &rewriter, Operation *op_begin,
                Operation *op_end, tpu::DistributionPattern pattern);
bool isLargeMatMul(Operation *op);
bool isBinaryOp(Operation *op);
Operation *cloneOp(PatternRewriter &rewriter, Operation *op,
                   llvm::ArrayRef<int64_t> new_shape, llvm::StringRef suffix);
// erase op and its uers
void eraseForward(PatternRewriter &rewriter, Operation *op);

template <typename T> static void applyPatternOnce(ModuleOp m) {
  auto ctx = m.getContext();
  RewritePatternSet patterns(ctx);
  auto config = GreedyRewriteConfig();
  config.maxIterations = 1; // apply each pattern only once.
  patterns.add<T>(ctx);
  applyPatternsAndFoldGreedily(m, std::move(patterns), config);
}

// ===================================
// patterns for distribution
// ===================================
class MatMulSliceMerge : public OpRewritePattern<tpu::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::MatMulOp op,
                                PatternRewriter &rewriter) const override;
};

class MatMulTopK : public OpRewritePattern<tpu::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::MatMulOp op,
                                PatternRewriter &rewriter) const override;
};

template <typename T>
void splitByDevices(PatternRewriter &rewriter, tpu::DistributionBeginOp op,
                    int64_t num_devices);

// ===================================
// split module to multi modules for devices
// ===================================
void distributeModules(ModuleOp module, int64_t num_device);

} // namespace tpu
} // namespace tpu_mlir
