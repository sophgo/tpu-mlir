//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Traits/Traits.h"
#include <llvm/Support/Debug.h>

namespace tpu_mlir {
namespace tpu {
// ===================================
// helper functions
// ===================================
void dump(Operation *op);

void distribute(PatternRewriter &rewriter, std::vector<Operation *> ops_begin,
                std::vector<Operation *> ops_end,
                tpu::DistributionPattern pattern,
                std::vector<int64_t> &begin_methods,
                std::vector<int64_t> &end_methods);
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
class AttentionSliceMerge2 : public OpRewritePattern<MatMulTy> {
public:
  using OpRewritePattern<MatMulTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(MatMulTy op,
                                PatternRewriter &rewriter) const override;
};

class MatMulSliceMerge3 : public OpRewritePattern<tpu::AddOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::AddOp op,
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
void sliceMergeSplit(MatMulTy mm0, PatternRewriter &rewriter,
                     tpu::DistributionBeginOp op, int64_t num_devices);

void sliceMerge2Split(PatternRewriter &rewriter, tpu::DistributionBeginOp op,
                      int64_t num_devices);

void sliceAttentionMerge2Split(PatternRewriter &rewriter,
                               tpu::DistributionBeginOp op,
                               int64_t num_devices);

template <typename MatMulTy>
void sliceAttentionMerge2Split(PatternRewriter &rewriter,
                               tpu::DistributionBeginOp op,
                               int64_t num_devices);

void sliceMerge3Split(PatternRewriter &rewriter, tpu::DistributionBeginOp op,
                      int64_t num_devices);

template <typename MatMulTy>
void topKSplit(MatMulTy mm, PatternRewriter &rewriter,
               tpu::DistributionBeginOp op, int64_t num_devices);

enum class DistributionEndMode {
  EndToUnknown = 0,
  EndToSum = 1,
  EndToTopK = 2,
  EndToConcat = 3
};

enum class DistributionBeginMethod {
  BeginFromUnknown = 0,
  BeginFromCopy = 1,
  BeginFromSplit = 2,
};

enum class DistributionEndMethod {
  EndToUnknown = 0,
  EndToSum = 1,
  EndToTopK = 2,
  EndToConcat = 3,
};

typedef std::shared_ptr<std::vector<int64_t>> begin_method_array_t;
typedef std::shared_ptr<std::vector<int64_t>> end_method_array_t;

static begin_method_array_t getBeginMethodArray(tpu::DistributionBeginOp op) {
  auto begin_methods = op.getBeginMethods();
  auto data = std::make_shared<std::vector<int64_t>>();
  for (auto it : llvm::enumerate(begin_methods)) {
    auto attr = it.value().dyn_cast<IntegerAttr>();
    if (attr) {
      data->push_back(attr.getInt());
    } else {
      llvm_unreachable("not DistributionBeginMethod type");
    }
  }
  return std::move(data);
}

static end_method_array_t getEndMethodArray(tpu::DistributionEndOp op) {
  auto end_methods = op.getEndMethods();
  auto data = std::make_shared<std::vector<int64_t>>();
  for (auto it : llvm::enumerate(end_methods)) {
    auto attr = it.value().dyn_cast<IntegerAttr>();
    if (attr) {
      data->push_back(attr.getInt());
    } else {
      llvm_unreachable("not DistributionEndMethod type");
    }
  }
  return std::move(data);
}

static DistributionEndMode getEndMode(tpu::DistributionEndOp op) {
  switch (op.getPattern()) {
  case tpu::DistributionPattern::MatMulSliceMerge:
  case tpu::DistributionPattern::MatMulSliceMerge2:
  case tpu::DistributionPattern::MatMulSliceMerge3:
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
