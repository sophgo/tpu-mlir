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
void distributeAfter(PatternRewriter &rewriter, Operation *op_begin,
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
template <typename MatMulTy>
class MatMulSliceMerge : public OpRewritePattern<MatMulTy> {
public:
  using OpRewritePattern<MatMulTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(MatMulTy op,
                                PatternRewriter &rewriter) const override;
};

template <typename MatMulTy>
class MatMulSliceMerge2 : public OpRewritePattern<MatMulTy> {
public:
  using OpRewritePattern<MatMulTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(MatMulTy op,
                                PatternRewriter &rewriter) const override;
};

template <typename MatMulTy>
class MatMulTopK : public OpRewritePattern<MatMulTy> {
public:
  using OpRewritePattern<MatMulTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(MatMulTy op,
                                PatternRewriter &rewriter) const override;
};

template <typename MatMulTy>
void sliceMergeSplit(MatMulTy mm0, PatternRewriter &rewriter, tpu::DistributionBeginOp op,
                    int64_t num_devices);

template <typename MatMulTy>
void sliceMerge2Split(MatMulTy mm_left, PatternRewriter &rewriter, tpu::DistributionBeginOp op,
                    int64_t num_devices);

template <typename MatMulTy>
void topKSplit(MatMulTy mm, PatternRewriter &rewriter, tpu::DistributionBeginOp op,
                    int64_t num_devices);

enum class DistributionEndMode {
  EndToUnknown = 0,
  EndToSum = 1,
  EndToTopK = 2,
};

static DistributionEndMode getEndMode(tpu::DistributionEndOp op) {
  switch (op.getPattern()) {
  case tpu::DistributionPattern::MatMulSliceMerge:
    return DistributionEndMode::EndToSum;
  case tpu::DistributionPattern::MatMulSliceMerge2:
    return DistributionEndMode::EndToSum;
  case tpu::DistributionPattern::MatMulTopK:
    return DistributionEndMode::EndToTopK;
  default:
    llvm_unreachable("Not Implemented");
  }
  return DistributionEndMode::EndToUnknown;
}

// ===================================
// split module to multi modules for devices
// ===================================
void distributeModules(ModuleOp module, int64_t num_device);

} // namespace tpu
} // namespace tpu_mlir
